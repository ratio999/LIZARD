# cython: language_level=3
import random
import re
import warnings
import numpy as np
import sympy as sp
cimport numpy as cnp
cimport function_treeC
cimport gp_movesC
cimport real_functionC
cimport polya_treeC
import itertools
from tqdm import tqdm
import time
import sys
import os
from copy import deepcopy
from libc.math cimport exp, fmax, log, isfinite
from mpi4py import MPI
#from mpi4py cimport MPI as cMPI
#from mpi4py.libmpi cimport *
cdef str _DEFAULT_OPTIM = str("L-BFGS-B")

# Mirror the function_treeC warning filters here so users importing groveC
# directly (without first touching function_treeC) get the same quiet
# behaviour. Same rationale: every triggering arithmetic case is gated by
# is_valid downstream and converted into a hard reject; the warning is
# noise.
warnings.filterwarnings('ignore', message=r'invalid value encountered.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'divide by zero encountered.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'overflow encountered.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'underflow encountered.*', category=RuntimeWarning)

cdef inline double _safe_exp(double x) nogil:
    # mild guard for very negative x
    if x < -700.0:
        return 0.0
    elif x > 700.0:
        return 1.0e304  # not used directly as prob, but keep bounded
    return exp(x)

cdef inline double _accept_prob(double dlogpost, double T) nogil:
    # T <= 0 means greedy acceptance
    if T <= 0.0:
        return 1.0 if dlogpost >= 0.0 else 0.0
    return fmax(0.0, 1.0 if dlogpost >= 0.0 else _safe_exp(dlogpost / T))


cdef double _md_dataset_penalty(int k, int n, object sigma):
    """Phase MD: the BIC complexity term P_j >= 0 that _shortcut_bic
    charges dataset j — 0.5*k*log(n_j) with a given sigma, k+1 when sigma
    is profiled (sigma_j is None). Mirrors function_treeC._shortcut_bic's
    branches INCLUDING the Phase 4.3 ablation switch (review 2026-07-31:
    without the mirror, set_bic_penalty(False) + min/softmin decomposed
    L_j = S_j + P_j with a phantom P_j and mis-weighted the argmin by
    dataset size). Used by the min/softmin reductions to keep complexity
    SUMMED while the fit term is min-reduced (PLAN MR.4 D6)."""
    if n <= 0:
        return 0.0
    if not function_treeC.get_bic_penalty():
        return 0.0
    if sigma is None:
        return 0.5 * (k + 1) * log(<double> n)
    return 0.5 * k * log(<double> n)


cdef double _reduce_joint(list posts, list penalties, str combine,
                          double softmin_beta, object baselines):
    """Phase MD: reduce the per-dataset BIC vector [S_1..S_J] into ONE
    scalar energy (PLAN MR.4 D4/D5/D6).

    posts[j]     = S_j: the FULL per-dataset BIC (fit term minus P_j);
    penalties[j] = P_j >= 0 (from _md_dataset_penalty);
    baselines    = per-dataset constants B_j (list) or None (all 0) —
                   tau-independent, so subtracting them is a valid energy
                   shift (D6: for 'min' they make datasets commensurable).
                   NB B_j must be the incumbent's FIT term on dataset j,
                   i.e. L_j^inc = S_j^inc + P_j^inc — NOT its full BIC —
                   because the reduction shifts the fit part (review
                   2026-07-31 finding 6: full-BIC baselines leave
                   per-dataset +P_j^inc offsets inside the argmin).

      'sum'     -> sum_j S_j                       (DEFAULT: joint evidence)
      'min'     -> min_j (L_j - B_j) - sum_j P_j    with L_j = S_j + P_j
      'softmin' -> -(1/beta) log( (1/J) sum_j exp(-beta (L_j - B_j)) )
                   - sum_j P_j      (beta -> 0: mean; beta -> inf: min)

    Every branch is a genuine function of the tree (never of a proposal
    pair), so MH detailed balance holds for all modes; min-of-ALPHAS was
    rejected as invalid (D5). Any non-finite S_j yields -inf."""
    cdef int J = len(posts)
    cdef int j
    cdef double s, total, pen_sum, m, acc, b
    if J == 0:
        return -np.inf
    for j in range(J):
        s = <double> posts[j]
        if not isfinite(s):
            return -np.inf
    if combine == 'sum':
        total = 0.0
        for j in range(J):
            total += <double> posts[j]
        return total
    # min / softmin operate on the FIT part L_j = S_j + P_j, baseline-
    # shifted; the summed complexity charge is re-applied at the end.
    pen_sum = 0.0
    for j in range(J):
        pen_sum += <double> penalties[j]
    cdef list fit = []
    for j in range(J):
        b = <double> baselines[j] if baselines is not None else 0.0
        fit.append(<double> posts[j] + <double> penalties[j] - b)
    if combine == 'min':
        m = <double> fit[0]
        for j in range(1, J):
            if <double> fit[j] < m:
                m = <double> fit[j]
        return m - pen_sum
    if combine == 'softmin':
        # -(1/beta) * log( (1/J) * sum exp(-beta*fit_j) ), shifted by the
        # min for numerical stability.
        m = <double> fit[0]
        for j in range(1, J):
            if <double> fit[j] < m:
                m = <double> fit[j]
        acc = 0.0
        for j in range(J):
            acc += _safe_exp(-softmin_beta * (<double> fit[j] - m))
        return m - (1.0 / softmin_beta) * (log(acc) - log(<double> J)) \
               - pen_sum
    raise ValueError("_reduce_joint: unknown combine mode %r "
                     "(expected 'sum' | 'min' | 'softmin')" % (combine,))


cpdef double reduce_joint(list posts, list penalties, str combine,
                          double softmin_beta=8.0, object baselines=None):
    """Phase MD: Python-callable wrapper over _reduce_joint (tests and
    post-hoc joint-scoring scripts). POSITIONAL args across modules."""
    return _reduce_joint(posts, penalties, combine, softmin_beta,
                         baselines)


cpdef double md_dataset_penalty(int k, int n, object sigma=None):
    """Phase MD: Python-callable wrapper over _md_dataset_penalty."""
    return _md_dataset_penalty(k, n, sigma)

cdef bint _install_cached_theta(function_treeC.Tree cand, dict cached_nv):
    """Phase 2.7(f) cache hit: install the cached theta_hat into `cand`.

    Plain parameter Nodes via set_value; opaque atomic leaves via their
    Node.theta dicts (Phase 2.10 Stage 3a — before this, a cache hit
    restored only the plain parameters, so an opaque leaf silently kept
    its warm-start theta while the cache reported the optimized evidence:
    the installed tree and its score disagreed).

    Returns True iff `cand` carries ANY parameter-bearing leaf — the
    caller's cue to tree_update (the historical `if cand.params:` gate,
    extended to atomic leaves; atom-free behaviour is bit-identical)."""
    cdef function_treeC.Node par_node
    cdef function_treeC.Node leaf
    cdef str pname
    cdef bint has_any = False
    for par_node in cand.params:
        has_any = True
        pname = par_node.get_fun().get_description()
        if pname in cached_nv:
            par_node.set_value(<double> cached_nv[pname])
    for leaf in cand.leaves:
        if leaf.is_atomic():
            has_any = True
            for pname in leaf.leaf_param_names():
                if pname in cached_nv:
                    leaf.theta[pname] = float(cached_nv[pname])
    # Phase FX: flex-node internal thetas (same contract as opaque-leaf
    # theta above; slot-subtree parameter LEAVES are plain params and were
    # already restored by the cand.params pass). No-op on flex-free trees.
    cdef function_treeC.Node fx
    for fx in function_treeC.collect_flex_nodes(cand.get_root(), None):
        has_any = True
        for pname in fx.flex_names:
            if pname in cached_nv:
                fx.theta[pname] = float(cached_nv[pname])
    return has_any


# Phase 2.10 (final item): burn-in seeding of atom-bearing depth-1 trees.
# Opaque templates enter the burn-in enumeration as plain-string tokens
# '@atom:<symbol>', so every itertools.product sweep, work-item tuple and
# pickle path below stays untouched; _burn_in_primitive_node resolves the
# tokens to leaves at construction time. '@' cannot appear in a function
# string or parameter name, so tokens cannot collide.
_ATOM_MARK = "@atom:"


cpdef function_treeC.Node _burn_in_primitive_node(str prim_str, object domain,
                                                  dict atom_specs,
                                                  str prefix, int name_start):
    """Build a fresh leaf Node for one burn-in primitive token.

    'a0'             -> parameter leaf, value 1.0 (historical behaviour);
    '@atom:<symbol>' -> opaque leaf instantiated from atom_specs[<symbol>]
                        with internal names {prefix}{name_start},
                        {prefix}{name_start+1}, ... and theta at the
                        template's unit reference theta0 (the burn-in
                        scorer refits theta anyway);
    anything else    -> function leaf via from_string.

    `prefix`/`name_start` give callers DISJOINT name bands so two opaque
    leaves combined into one tree never share internal names (sharing
    would couple their values through name_values): stage 1 keeps
    candidates all-'a' (left band starts at 1 — 'a0' is the parameter
    primitive — right band starts past the left atom's k) because
    _combine_depth1 later renames right operands wholesale to 'b';
    mpi_worker's combiner renames nothing, so its primitive operand uses
    the 'c' band. Cross-module callers: POSITIONAL ARGS ONLY."""
    cdef real_functionC.AtomSpec spec
    cdef real_functionC.RealFunctionC rf
    cdef list names
    cdef int i
    cdef function_treeC.Node node
    if prim_str.startswith(_ATOM_MARK):
        if atom_specs is None or prim_str[len(_ATOM_MARK):] not in atom_specs:
            raise KeyError(
                "burn-in primitive %r has no matching AtomSpec" % prim_str)
        spec = <real_functionC.AtomSpec> atom_specs[prim_str[len(_ATOM_MARK):]]
        names = ["%s%d" % (prefix, name_start + i) for i in range(spec.k)]
        rf = real_functionC.instantiate_atom(spec, domain, names)
        return function_treeC.Node(rf)  # Node auto-inits theta to theta0
    node = function_treeC.Node(
        function=real_functionC.RealFunctionC.from_string(
            prim_str, domain, compute_grad=False))
    if prim_str == "a0":
        node.set_value(1.0)
    return node


def _resolve_structural_prior_spec(spec):
    """Phase 4.4 v2: rebuild a structural-prior callable from a
    declarative, PICKLABLE spec — ('module', 'builder', kwargs_dict).

    This is pickle's function-by-reference semantic made explicit: no
    code crosses any process boundary, only the (module, name, kwargs)
    triple; each process — the master AND every run_mpi_spawn worker —
    imports the named module locally and calls builder(**kwargs) on
    code it already has. The named module must therefore be importable
    in the worker processes (same cwd / sys.path as the master). The
    kernel stays decoupled from the experiment layer: the import is
    lazy and name-driven, nothing here references exp_constraints.
    """
    import importlib
    if (not isinstance(spec, (tuple, list)) or len(spec) != 3
            or not isinstance(spec[0], str)
            or not isinstance(spec[1], str)
            or not isinstance(spec[2], dict)):
        raise TypeError(
            "structural_prior_spec must be ('module', 'builder', "
            "kwargs_dict); got %r" % (spec,))
    mod = importlib.import_module(spec[0])
    builder = getattr(mod, spec[1], None)
    if builder is None:
        raise AttributeError(
            "structural_prior_spec: module %r has no builder %r"
            % (spec[0], spec[1]))
    cb = builder(**spec[2])
    if not callable(cb):
        raise TypeError(
            "structural_prior_spec: %s.%s(**kwargs) returned a "
            "non-callable %r" % (spec[0], spec[1], type(cb)))
    return cb


cdef class Grove:
    """
    A set of Trees evolving at a single temperature T (Metropolis-Hastings).
    Higher T → broader proposals; lower T → conservative proposals.

    Reproducibility
    ---------------
    Pass `seed` (int or None) to make the structural-MCMC trajectory
    bit-reproducible. With a fixed seed, two Grove instances constructed
    identically and run for the same number of steps will produce the same
    sequence of accepted trees and identical `move_stats()` output.

    The sole exception is the continuous-θ optimizer (`update_parameters`
    calls `scipy.optimize`), which is reproducible only up to BLAS/OpenMP
    non-determinism. For bit-exact runs, set `OMP_NUM_THREADS=1` and pin
    a single BLAS implementation (e.g. OpenBLAS single-threaded).
    """
    
    def __init__(self,
                 cnp.ndarray x,
                 cnp.ndarray[cnp.float64_t, ndim=1] y,
                 double T,  # accepts scalar or array; use T_arr[0]
                 int size,
                 int depth,
                 list funcs,
                 object sigma = 1.0,
                 object node_list=None,
                 int Npars=0,
                 object init_value=1.0,
                 list operations=None,
                 double base_temp=1.0,
                 double lambda_reg=0.00,
                 bint use_gradients=True,
                 bint burn_in=False,
                 int use_mpi=0,
                 bint use_spawn=False,
                 object polya=None,
                 object seed=None,
                 object g_tau=None,
                 bint adapt_moves=False,
                 double adapt_gamma0=1.0,
                 double adapt_t0=100.0,
                 double adapt_floor=0.02,
                 double adapt_logit_bound=4.0,
                 double adapt_target=0.5,
                 list atoms=None,
                 object structural_prior=None,
                 bint structural_prior_cache=True,
                 object structural_prior_spec=None,
                 object variables=None,
                 object flex_atoms=None,
                 object datasets=None,
                 str md_combine='sum',
                 double md_softmin_beta=8.0,
                 object md_baselines=None):

        # --- Flexible domain (flexible_domain_plan.md §3) -----------------
        # Resolve and install the variable registry BEFORE any leaf is
        # compiled. variables=None + 1-D x -> legacy ('x',), bit-identical;
        # variables=None + (d, n) x -> default ('x0', ..., 'x{d-1}');
        # an explicit tuple must match x's leading dimension. Strict (d, n)
        # layout only (coordinate ROWS — plan §6 Q1 ADOPTED).
        cdef tuple _vars
        cdef bint _flex
        cdef str _fs, _s_str
        if isinstance(variables, str):
            raise ValueError(
                "Grove: variables must be a sequence of names, not a "
                "string (got %r — a string would split into characters)"
                % (variables,))
        if x.ndim == 1:
            _vars = ('x',) if variables is None else tuple(variables)
            if len(_vars) != 1:
                raise ValueError(
                    "Grove: a 1-D x carries exactly one variable; got "
                    "variables=%r" % (variables,))
        elif x.ndim == 2:
            _vars = (tuple('x%d' % _i for _i in range(x.shape[0]))
                     if variables is None else tuple(variables))
            if len(_vars) != x.shape[0]:
                raise ValueError(
                    "Grove: x has shape (d, n) = %r — coordinate ROWS, "
                    "points along the last axis — but %d variable names "
                    "were given. Pass X.T if your data is (n, d)."
                    % (tuple([x.shape[0], x.shape[1]]), len(_vars)))
            if len(_vars) == 1:
                raise ValueError(
                    "Grove: a single-variable domain must use the 1-D "
                    "(n,) layout — pass x.ravel(). (A (1, n) array would "
                    "ride the d=1 code paths as a 2-D object; found by "
                    "the adversarial review.)")
        else:
            raise ValueError(
                "Grove: x must be (n,) or (d, n), got ndim=%d" % x.ndim)
        _flex = (x.ndim == 2) or (variables is not None)
        real_functionC.set_variables(_vars)

        # Store y as instance attribute for use in optimization and posterior computation
        self.data = np.asarray(y, dtype=np.float64)
        self.lambda_reg = lambda_reg  # default; can be changed in optimize()
        if operations is None:
            if len(_vars) > 1:
                raise ValueError(
                    "Grove: pass `operations` explicitly for a "
                    "multivariate domain — the legacy default includes "
                    "compose 'o', which is d=1-only "
                    "(flexible_domain_plan.md §0 D6 / §6 Q2)")
            operations = ["+", "-", "*", "/", "**", "o"]
        elif len(_vars) > 1 and "o" in operations:
            raise ValueError(
                "Grove: compose 'o' is not defined for multivariate "
                "domains (flexible_domain_plan.md §8.1 — standing item)")
        self.operations = operations
        # Flexible mode only (additive — legacy configs skip this): every
        # vocabulary string must be an expression in the REGISTERED
        # variables alone (catches typos like 'sin(x2)' at construction,
        # not as a silent parameter-bearing leaf mid-chain).
        if _flex:
            for _fs in funcs:
                for _s in sp.sympify(_fs).free_symbols:
                    _s_str = str(_s)
                    if _s_str not in _vars:
                        raise ValueError(
                            "Grove: vocabulary entry %r uses symbol %r, "
                            "which is not a registered variable %r"
                            % (_fs, _s_str, _vars))
        # Phase 2.10 Stage 3a: opaque-primitive templates. None/[] -> A = 0,
        # every move density reduces to the bit-identical legacy form.
        # Validated here so a mistyped entry fails at construction, not as
        # a cast error deep inside a move.
        if atoms:
            for _a in atoms:
                if not isinstance(_a, real_functionC.AtomSpec):
                    raise TypeError(
                        "Grove: atoms entries must be real_functionC."
                        "AtomSpec instances, got %r" % type(_a))
            if _flex:
                # Flexible domain: AtomSpec classification consults the
                # variable registry, and these specs may have been built
                # BEFORE this Grove installed it — re-derive each template
                # under the current registry (deterministic; a template
                # whose symbols no longer classify raises here, at
                # construction, instead of misbinding slots mid-chain).
                # Phase PP: re-derive from BASE name + pinned (the
                # extended self.name would double its pin suffix).
                atoms = [real_functionC.AtomSpec(_a.base_name, _a.template,
                                                 _a.pinned)
                         for _a in atoms]
            self.atoms = list(atoms)
        else:
            self.atoms = None
        # Phase FX (transparent atoms v1, 2026-07-28): flexible scaffolds.
        # None => bit-identical legacy (no extra move, no stats key, no
        # RNG change). v1 semantics: scaffolds enter trees by SEEDING only
        # (node_list / make_flex_node / the 'F' wire tag), are immortal
        # under the outer kernel, and their slots evolve via the dedicated
        # flex_slot_mutation move (9th physical move, self-inverse class).
        if flex_atoms:
            for _f in flex_atoms:
                if not isinstance(_f, real_functionC.FlexAtomSpec):
                    raise TypeError(
                        "Grove: flex_atoms entries must be real_functionC."
                        "FlexAtomSpec instances, got %r" % type(_f))
            if adapt_moves:
                raise ValueError(
                    "Grove: adapt_moves=True with flex_atoms is not "
                    "supported in v1 (the adaptive logit map covers the 8 "
                    "legacy moves only)")
            if burn_in or size <= 0:
                # size <= 0 flips self.burn_in on later in this ctor (the
                # legacy "let burn-in build the population" idiom) — the
                # same loophole the MD review closed for datasets
                # (2026-07-31 bit-identity review, LOW finding).
                raise ValueError(
                    "Grove: burn-in (burn_in=True OR size==0) with "
                    "flex_atoms is not supported in v1 (the burn-in "
                    "enumeration is legacy/opaque-only and its constant "
                    "folding is not flex-aware -- it would destroy or "
                    "crash on scaffolds)")
            if self.atoms is not None:
                for _f in flex_atoms:
                    for _a in self.atoms:
                        # Phase PP (review 2026-08-06): also reject the
                        # BASE name of a pinned atom — "Abeat" (flex) vs
                        # "Abeat[c1=...]" (pinned) share no registry key,
                        # but the near-collision is a config-confusion
                        # trap with no legitimate use.
                        if _a.name == _f.name or _a.base_name == _f.name:
                            raise ValueError(
                                "Grove: flex template %r collides with an "
                                "opaque atom name (or its pin-suffixed "
                                "base)" % _f.name)
            # Re-validate under the CURRENT variable registry (mirrors the
            # AtomSpec re-derivation above; deterministic).
            flex_atoms = [real_functionC.FlexAtomSpec(
                _f.name, _f.template, _f.slot_symbol) for _f in flex_atoms]
            self.flex_atoms = list(flex_atoms)
        else:
            self.flex_atoms = None
        # Phase 4.4: structural-constraint prior tilt (opt-in; None =
        # bit-identical legacy). Validated callable; see _sp().
        # v2 (2026-07-22): a declarative picklable spec may be passed
        # instead of a live callable; it is resolved HERE, in whatever
        # process constructs the Grove — which is exactly what lets
        # run_mpi_spawn workers rebuild the same tilt locally from the
        # shipped ctor args.
        if structural_prior_spec is not None:
            if structural_prior is not None:
                raise TypeError(
                    "Grove: pass structural_prior OR "
                    "structural_prior_spec, not both")
            structural_prior = _resolve_structural_prior_spec(
                structural_prior_spec)
        self.structural_prior_spec = structural_prior_spec
        if structural_prior is not None and not callable(structural_prior):
            raise TypeError("Grove: structural_prior must be callable "
                            "(tree -> float log-prior increment) or None")
        self.structural_prior = structural_prior
        self.structural_prior_cache = structural_prior_cache
        self._sp_cache = {}
        self.sigma = sigma if sigma is not None else 1.0
        # Phase 2.1.PM birth density g(theta) = N(0, g_tau^2).
        # Default to std(y) when caller doesn't pass one explicitly;
        # fall back to 1.0 if y is degenerate / empty.
        if g_tau is None:
            _y_arr = np.asarray(self.data)
            _std_y = float(_y_arr.std()) if _y_arr.size > 0 else 1.0
            self.g_tau = _std_y if _std_y > 0.0 else 1.0
        else:
            self.g_tau = float(g_tau)
            if self.g_tau <= 0.0:
                raise ValueError("Grove: g_tau must be > 0")
        self.funcs = funcs
        self.domain = real_functionC._domain_fixer(x)
        self.use_gradients = use_gradients
        self.method_str = 'L-BFGS-B' if self.use_gradients else 'Powell'
        self.depth = depth
        self.size = size
        self.use_mpi = use_mpi >= 2
        self.n_procs = use_mpi
        self.use_spawn = use_spawn
        self.T = float(T)
        self.base_temp = base_temp if base_temp > 0.0 else 1.0
        #self.trees = self._init_trees(node_list, Npars, init_value)
        if self.size > 0:
            self.root_posteriors = np.full((self.size,), np.nan, dtype=np.float64)
            self.burn_in = burn_in
            if burn_in:
                # Keep original behavior of the burn-in path: it sets up
                # `self.trees` itself, then re-initializes size below.
                self.size = 0
        else:
            self.root_posteriors=None
            self.burn_in = True

        self.acceptance_count = 0.0
        self.proposal_count = 0.0
        self.move_proposals = {
            "point_mutation":     0,
            "leaf_hoist":         0,
            "internal_hoist":     0,
            "add_node":           0,
            "split":              0,
            "subtree_mutation":   0,
            "subtree_crossover":  0,
            "etr_mutation":       0,
            "parameter_refresh":  0,
        }
        if self.flex_atoms is not None:
            # Phase FX: the stats key exists ONLY when the feature is on,
            # keeping move_stats()'s key set byte-identical for legacy
            # runs (test_phase23(f) asserts nonzero proposals per key).
            self.move_proposals["flex_slot_mutation"] = 0
        self.move_accepts = dict.fromkeys(self.move_proposals, 0)
        self.cache = {}
        self._am_states = {}
        # Phase 2.7(f) BIC topology cache (key: tree_get_description();
        # value: (log_evidence, dict(name_values))). _cache_y holds the
        # y reference the cache was built against; if `_step` sees a
        # different y, the cache is cleared.
        self._bic_cache = {}
        self._cache_y = None
        self.pr_prob = 0.02
        # --- Phase MD (multi-dataset joint evidence, 2026-07-31) ----------
        # datasets=None => bit-identical legacy (no extra branch taken on
        # any scoring path). datasets=[(x_1, y_1, s_1), ...] activates
        # JOINT scoring: every candidate is scored on EVERY dataset with
        # its own independently optimized theta_j, and the per-dataset
        # BIC vector is reduced by md_combine ('sum' DEFAULT = the joint
        # evidence; 'min'/'softmin' opt-in over baseline-shifted fit
        # terms with the complexity charge kept summed — PLAN MR.4
        # D4/D5/D6; md_baselines: per-dataset constants B_j or None).
        # CONTRACT: the positional (x, y) MUST equal the REFERENCE
        # dataset (datasets[0][:2]) — one source of truth; is_valid(
        # self.domain) keeps meaning "valid on the reference" and
        # self.sigma is forced to datasets[0][2]. Refusals: burn_in (D2:
        # five burn-in scoring paths are single-dataset, two inside MPI
        # workers); parameter_refresh (D1: PR accepts on ONE dataset's
        # raw likelihood — pr_prob is forced to 0.0 and the step itself
        # raises if re-enabled by hand). The BIC cache is REUSED with a
        # joint value shape: (joint_post, [nv_0 ... nv_{J-1}]).
        if datasets is not None:
            # Review 2026-07-31 (finding 2): test self.burn_in, not just
            # the ctor flag — size == 0 flips self.burn_in on at line
            # ~500 (the legacy "let burn-in build the population" idiom)
            # and would silently run the single-dataset burn-in on a
            # joint Grove.
            if burn_in or self.burn_in:
                raise ValueError(
                    "Grove: burn-in (burn_in=True OR size==0) with "
                    "datasets is not supported in v1 (every burn-in "
                    "scoring path is single-dataset; seed via node_list "
                    "or run single-dataset burn-in explicitly and pass "
                    "its winners — PLAN MR.4 D2)")
            if md_combine in ('min', 'softmin') and md_baselines is None:
                warnings.warn(
                    "Grove: md_combine=%r without md_baselines compares "
                    "RAW per-dataset fit terms — the largest/tightest "
                    "dataset dominates the argmin, not the worst FIT. "
                    "Pass per-dataset baselines B_j (the incumbent's FIT "
                    "term L_j = S_j + P_j on dataset j) to make datasets "
                    "commensurable (PLAN MR.4 D6)." % (md_combine,))
            if len(datasets) < 1:
                raise ValueError("Grove: datasets must hold >= 1 "
                                 "(x, y, sigma) triple")
            _md_list = []
            for _dj in datasets:
                if len(_dj) != 3:
                    raise ValueError(
                        "Grove: each dataset must be an (x, y, sigma) "
                        "triple, got length %d" % len(_dj))
                # Review 2026-07-31 (finding 4): x is COPIED (never a view
                # of the caller's array) and zero-scrubbed exactly like
                # the legacy domain — otherwise (a) datasets j>0 would
                # never be scrubbed, so an exact zero (e.g. t=0 at the
                # aligned peak) makes every 1/x-bearing candidate invalid
                # on that dataset only (support would depend on dataset
                # position), and (b) the legacy in-place scrub of the
                # positional x would leak through a shared view into the
                # caller's dataset arrays.
                _xj = np.array(_dj[0], dtype=np.float64, copy=True,
                               order='C')
                real_functionC._domain_fixer(_xj)
                _yj = np.ascontiguousarray(np.asarray(_dj[1],
                                                      dtype=np.float64))
                if _yj.ndim != 1:
                    raise ValueError("Grove: dataset y must be 1-D")
                if _yj.shape[0] == 0:
                    raise ValueError("Grove: dataset y must be non-empty")
                if _xj.shape[_xj.ndim - 1] != _yj.shape[0]:
                    raise ValueError(
                        "Grove: dataset x/y length mismatch (%d vs %d)"
                        % (_xj.shape[_xj.ndim - 1], _yj.shape[0]))
                _sj = _dj[2]
                if _sj is not None and hasattr(_sj, '__len__'):
                    _sj = np.ascontiguousarray(
                        np.asarray(_sj, dtype=np.float64)).ravel()
                    if _sj.shape[0] != _yj.shape[0]:
                        raise ValueError(
                            "Grove: dataset sigma length %d != n %d"
                            % (_sj.shape[0], _yj.shape[0]))
                _md_list.append((_xj, _yj, _sj))
            if not np.array_equal(np.asarray(x, dtype=np.float64),
                                  _md_list[0][0]):
                raise ValueError(
                    "Grove: with datasets, the positional x must BE the "
                    "reference dataset's x (datasets[0][0])")
            if not np.array_equal(np.asarray(y, dtype=np.float64),
                                  _md_list[0][1]):
                raise ValueError(
                    "Grove: with datasets, the positional y must BE the "
                    "reference dataset's y (datasets[0][1])")
            if md_combine not in ('sum', 'min', 'softmin'):
                raise ValueError(
                    "Grove: md_combine must be 'sum' | 'min' | "
                    "'softmin', got %r" % (md_combine,))
            if md_softmin_beta <= 0.0:
                raise ValueError("Grove: md_softmin_beta must be > 0")
            if md_baselines is not None:
                if len(md_baselines) != len(_md_list):
                    raise ValueError(
                        "Grove: md_baselines must have one constant per "
                        "dataset (%d != %d)"
                        % (len(md_baselines), len(_md_list)))
                md_baselines = [float(_b) for _b in md_baselines]
            self.datasets = _md_list
            self.md_combine = md_combine
            self.md_softmin_beta = md_softmin_beta
            self.md_baselines = md_baselines
            self.sigma = _md_list[0][2] if _md_list[0][2] is not None \
                else self.sigma
            self.pr_prob = 0.0          # D1: PR refused under datasets
        else:
            self.datasets = None
            self.md_combine = 'sum'
            self.md_softmin_beta = 8.0
            self.md_baselines = None
        # --- Phase 2.8: optional adaptive move-weight calibration ----------
        # OFF by default (adapt_moves=False) -> Grove behaves exactly as
        # before, using the fixed temperature-shaped weights in _choose_move.
        # When ON, six logical-group logits are updated by a diminishing-
        # adaptation (Robbins-Monro) rule (see _adapt_move) and mapped to the
        # eight physical move probabilities by _adapted_move_probs, which
        # preserves the birth/death pair equality (so the RJMCMC move-
        # selection correction stays exactly zero) and floors every
        # probability at `adapt_floor` (containment). Warm-started from the
        # fixed weights so toggling adaptation on is continuous.
        if not (0.0 <= adapt_floor < 1.0 / 8.0):
            raise ValueError("Grove: adapt_floor must lie in [0, 1/8)")
        if adapt_logit_bound <= 0.0:
            raise ValueError("Grove: adapt_logit_bound must be > 0")
        self.adapt_moves = adapt_moves
        self.adapt_gamma0 = adapt_gamma0
        self.adapt_t0 = adapt_t0 if adapt_t0 > 0.0 else 1.0
        self.adapt_floor = adapt_floor
        self.adapt_logit_bound = adapt_logit_bound
        self.adapt_target = adapt_target
        self._adapt_t = 0
        _tau0 = self.T / self.base_temp
        _f0 = 1.0 / (1.0 + _tau0)
        _g0 = _tau0 / (1.0 + _tau0)
        _pb0 = 0.15 * _f0 + 0.05
        _eta0 = np.log(np.array([
            0.30 * _f0 + 0.05,   # group 0: point_mutation
            _pb0,                # group 1: add_node / leaf_hoist
            _pb0,                # group 2: split / internal_hoist
            0.15 * _g0 + 0.10,   # group 3: subtree_mutation
            0.10 * _g0 + 0.05,   # group 4: subtree_crossover
            0.10 * _f0 + 0.05,   # group 5: etr_mutation
        ], dtype=np.float64))
        _eta0 = _eta0 - _eta0.mean()
        _eta0 = np.clip(_eta0, -self.adapt_logit_bound, self.adapt_logit_bound)
        self._move_logits = np.ascontiguousarray(_eta0, dtype=np.float64)
        # Phase 2.5(a): per-Grove RNGs. Both seeded from a single user
        # `seed` (None -> non-deterministic). _py_rng drives Python-side
        # random.* call sites; _np_rng drives NumPy call sites. Forest
        # will spawn child seeds via SeedSequence in 2.5(e); for now any
        # standalone Grove built with the same seed reproduces its own
        # trajectory bit-for-bit (modulo scipy/BLAS optimizer noise).
        self._py_rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        # Install once for any move calls reachable from burn-in / user
        # code before the first _step. _step / legacy_step re-install
        # before every step so multiple Groves in the same process keep
        # their trajectories independent.
        gp_movesC.set_rng(self._py_rng, self._np_rng)
        # function_treeC owns the tree-construction RNG used by build_tree
        # (called from Tree.__init__ in _init_trees, and from every
        # K-changing move that spawns a fresh subtree). Must be installed
        # BEFORE _init_trees / burn-in fires, otherwise build_tree picks
        # from Python's global random.* and determinism is lost.
        function_treeC.set_rng(self._py_rng, self._np_rng)
        if self.burn_in:
            # Run two-stage burn-in procedure to initialize trees
            # Stage 1: Generate all depth-1 combinations (primitive op primitive)
            depth1_candidates = self._get_depth1_candidates(y if 'y' in locals() else self.data, tree_cache=self.cache, optimize_params=True)
            
            # Stage 2: Combine top depth-1 trees to create depth-2 trees
            if len(depth1_candidates) > 0 and not self.use_mpi:
                self._burn_in_stage2(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    self.cache,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True
                )
            elif len(depth1_candidates) > 0 and self.use_mpi and not self.use_spawn:
                self._burn_in_stage2_parallel(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    self.cache,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True
                )
            
            elif len(depth1_candidates) > 0 and self.use_mpi and self.use_spawn:
                self._burn_in_stage2_parallel_spawn(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    self.cache,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True,
                    n_workers = self.n_procs
                )
            
            # Initialize trees list if not already done
            if not hasattr(self, 'trees') or self.trees is None:
                self.trees = [None] * self.size
                self.root_posteriors = np.full((self.size,), np.nan, dtype=np.float64)
        else:
            self.trees = self._init_trees(node_list, Npars, init_value)

        # Phase PP (review 2026-08-06 F4): pin_numbers CONSTANT leaves are
        # SCORING-ONLY in v1 — they sit outside the move vocabulary, so a
        # destructive move on one claims a nonzero reverse density the
        # kernel cannot realize (detailed-balance violation), and the 'L'
        # skel tag cannot carry the flag across the wire. Refuse them at
        # seeding, the FX-refusal pattern. Pure attribute reads on a
        # False-everywhere-legacy flag; runs once at construction.
        if node_list is not None and self.trees is not None:
            for _tt in self.trees:
                if _tt is None:
                    continue
                for _lf in _tt.get_leaves():
                    if _lf.get_fun() is not None \
                            and _lf.get_fun().pinned_const:
                        raise ValueError(
                            "Grove: node_list seeds contain a "
                            "pin_numbers CONSTANT leaf — such trees are "
                            "SCORING-ONLY in v1 (score them via "
                            "log_marginal_bic / score_dataset, or use a "
                            "pinned AtomSpec for evolvable pinned "
                            "structure; PLAN MR.5 D6)")

        # Optional Pólya tree prior. When None, _step behaves exactly as
        # before (no prior term, no adaptation).
        self.polya = polya
        # Phase 2.12 (MPI sampling): when polya_defer is True, accepted
        # construction paths are COLLECTED into _deferred_paths instead of
        # immediately reinforcing the shared Polya. The MPI master gathers
        # them each round and replays the same ordered list on every
        # replica (deterministic replay), keeping all Theta copies
        # bit-identical. Default False -> exact pre-2.12 behaviour.
        self.polya_defer = False
        self._deferred_paths = []
        
    def get_domain(self):
        return self.domain

    cdef _init_trees(self, list node_list, int Npars, object init_value):
        cdef list trees = [None] * self.size
        self.root_posteriors = [None] * self.size
        cdef int i
        for i in range(self.size):
            if node_list is not None and len(node_list)>i:
                    trees[i] = function_treeC.Tree(self.depth, self.funcs, self.domain,
                                               node_list[i], Npars, init_value, self.T,
                                               sigma=self.sigma, g_tau=self.g_tau)
            else:
                # Phase 2.13(f): initial trees must be built from the
                # kernel's own op vocabulary (support closure) — an
                # out-of-vocabulary op in the start state could never be
                # legally removed by any on-support move.
                trees[i] = function_treeC.Tree(self.depth, self.funcs, self.domain,
                                               Npars=Npars, value=init_value, T=self.T,
                                               sigma=self.sigma, g_tau=self.g_tau,
                                               operations=self.operations)


                if self.datasets is not None:
                    # Phase MD (review 2026-07-31, finding 1): random-init
                    # joint Groves must seed root_posteriors at the JOINT
                    # scale — the legacy single-dataset score below is
                    # incommensurable with joint candidate scores (the
                    # incumbent would be inflated by the missing (J-1)
                    # dataset terms and freeze acceptance).
                    self.root_posteriors[i] = self._score_cand_joint(
                        trees[i], True)
                    # Match the LEGACY ctor's cache discipline (it never
                    # writes the BIC cache): drop the entry the joint
                    # scorer just created, so J=1 trajectories remain
                    # EXACTLY reproducible against legacy groves (MD-a
                    # pins this; compute_posteriors repopulates on
                    # demand).
                    self._bic_cache.pop(
                        trees[i].tree_get_description(), None)
                else:
                    self.root_posteriors[i] = trees[i].log_marginal_bic(
                        self.data, optimize=True,
                        use_gradients=self.use_gradients,
                        method_str=self.method_str,
                            )
        return trees

    cpdef set_trees(self, list new_trees):
        self.trees = new_trees

    cpdef void set_sigmas(self, object new_sigma):
        for j in range(0, self.size):
            self.trees[j].set_sigma(new_sigma)

    cpdef set_single_tree(self, function_treeC.Tree tree, int i):
        self.trees[i] = tree

    cpdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=False):
        """
        Update/compute root posteriors. If full, compute entire tree; else superficial (root only).

        Phase MD: joint Groves score every tree with _score_cand_joint
        regardless of `full` (the full=True per-node posterior walk is a
        single-dataset legacy diagnostic with no joint meaning).
        """
        cdef int i
        if self.datasets is not None:
            for i in range(self.size):
                self.root_posteriors[i] = self._score_cand_joint(
                    self.trees[i], True)
            return
        if full:
            for i in range(self.size):
                self.trees[i].compute_posterior_tree(y) #controlla i metdi in functiontree
                self.root_posteriors[i] = self.trees[i].root.posterior
        else:
            for i in range(self.size):
                #print('FITNESS', self.trees[i])
                #BELOW IS DEPRECATED
                #self.root_posteriors[i] = self.trees[i].get_superficial_posterior(y, self.lambda_reg)
                if not self.trees[i].is_valid(self.domain):
                    self.root_posteriors[i] = -np.inf
                    continue

                self.root_posteriors[i] = self.trees[i].log_marginal_bic(
                    y, optimize=True,
                    use_gradients=self.use_gradients,
                    method_str=self.method_str,
                )
    cpdef best_index(self):
        return int(np.nanargmax(self.root_posteriors))

    cpdef best_tree(self):
        return self.trees[self.best_index()]

    cdef cnp.ndarray _compute_move_probs(self):
        """The 7-vector of physical move probabilities the kernel uses on the
        NEXT structural draw. Order matches the _choose_move dispatch:
        [point, leaf_hoist, internal_hoist, add, split, subtree, cross].

        When adapt_moves is False these are the fixed temperature-shaped
        weights; when True they come from _adapted_move_probs (Phase 2.8),
        which preserves the birth/death pair equality and the eps floor.
        Extracted so both _choose_move and the introspection getter
        get_move_probs() share one source of truth.
        """
        # normalize temperature to [0, +inf): higher → more disruptive moves
        cdef double tau = self.T / self.base_temp
        # Temperature shape factors:
        #   f -> 1 when cold (tau << 1), 0 when hot (tau >> 1)
        #   g -> 0 when cold,             1 when hot   (g == 1 - f)
        f = 1.0 / (1.0 + tau)
        g = tau / (1.0 + tau)

        # --- Birth/Death balanced pairs (RJMCMC move-selection term = 0) ---
        # The cross-move reverse pairs are now EXACT and one-to-one:
        #   add_node  <-> leaf_hoist      (death of an added leaf)
        #   split     <-> internal_hoist  (death of a split internal node)
        # The Green (1995) move-selection correction  log p(m'|T') - log p(m|T)
        # vanishes for each pair independently iff its forward and reverse
        # selection probabilities are equal:  p_add == p_leaf_hoist  AND
        # p_split == p_internal_hoist. Tie each pair to one expression so the
        # balance cannot silently drift. Splitting the old unified hoist into
        # two death moves also removes the former 2:1 birth:death proposal
        # asymmetry (now 2 births vs 2 deaths). DO NOT break a pair apart
        # without adding the selection term to _step's log_alpha.
        p_birth = 0.15 * f + 0.05
        p_add            = p_birth
        p_leaf_hoist     = p_birth   # exact reverse of add_node
        p_split          = p_birth
        p_internal_hoist = p_birth   # exact reverse of split

        # --- Self-inverse moves (m' == m): correction is 0 for ANY value ---
        # p(m|.) depends only on tau (constant across a grove), so these
        # weights are free to shape the kernel without breaking detailed
        # balance. Local refinement favored cold; global rebuild/swap hot.
        p_point   = 0.30 * f + 0.05
        p_subtree = 0.15 * g + 0.10
        p_cross   = 0.10 * g + 0.05
        p_etr     = 0.10 * f + 0.05   # cheap local ET rewrite; favored cold
        probs = np.array([p_point, p_leaf_hoist, p_internal_hoist,
                          p_add, p_split, p_subtree, p_cross, p_etr],
                        dtype=np.float64)

        probs = probs / probs.sum()
        if self.adapt_moves:
            # Phase 2.8: adaptive path overrides the fixed weights computed
            # above. _adapted_move_probs keeps the pair equality and floor.
            probs = self._adapted_move_probs()
        if self.flex_atoms is not None:
            # Phase FX: append flex_slot_mutation as a 9th physical move.
            # Scaling the 8-vector UNIFORMLY preserves both birth/death
            # pair equalities exactly (p_add==p_leaf_hoist,
            # p_split==p_internal_hoist), so every RJMCMC move-selection
            # term stays zero; the slot move is self-inverse-class (its
            # forward and reverse selection probabilities are equal by
            # construction), so any fixed weight is balance-safe. The
            # ctor forbids adapt_moves+flex, so the adapted branch above
            # never feeds this line.
            probs = np.concatenate(
                (probs * 0.80, np.array([0.20], dtype=np.float64)))
        return probs

    cpdef cnp.ndarray get_move_probs(self):
        """Phase 2.8 introspection: current physical move-probability vector
        (adapted if adapt_moves else the fixed temperature-shaped weights).
        Read-only; does not advance any RNG. Order:
        [point, leaf_hoist, internal_hoist, add, split, subtree, cross, etr]
        -- 8 entries; with flex_atoms set, flex_slot_mutation is appended
        as a 9th entry (Phase FX)."""
        return self._compute_move_probs()

    cdef _choose_move(self) :
        """
        Select a move and parameters based on temperature.
        Returns (name, kwargs_dict).
        """
        cdef cnp.ndarray probs = self._compute_move_probs()
        # Phase FX: arange sized to the vector (8 legacy / 9 with flex);
        # for the legacy 8-vector the np.choice call is argument-identical
        # to the historical form, so the RNG stream is unchanged.
        idx = int(self._np_rng.choice(np.arange(probs.shape[0]), p=probs))
        #random_depth = (tau > 0.7)  # deeper/remoter edits at higher T
        if idx == 0:
            return ("point_mutation", {"operations": self.operations})
        elif idx == 1:
            return ("leaf_hoist", {})
        elif idx == 2:
            return ("internal_hoist", {})
        elif idx == 3:
            return ("add_node", {"operations": self.operations})
        elif idx == 4:
            return ("split", {"operations": self.operations})
        elif idx == 5:
            return ("subtree_mutation", {})
        elif idx == 6:
            return ("subtree_crossover", {})
        elif idx == 7:
            return ("etr_mutation", {"operations": self.operations})
        else:
            return ("flex_slot_mutation", {"operations": self.operations})

    cdef cnp.ndarray _adapted_move_probs(self):
        """Map the 6 logical-group logits to the 8 physical move probabilities.

        Group -> physical layout (must match the _choose_move dispatch order
        [point, leaf_hoist, internal_hoist, add, split, subtree, cross, etr]):
            eta[0] -> point_mutation
            eta[1] -> leaf_hoist  AND  add_node      (shared: pair equality)
            eta[2] -> internal_hoist AND split       (shared: pair equality)
            eta[3] -> subtree_mutation
            eta[4] -> subtree_crossover
            eta[5] -> etr_mutation
        Because each birth/death pair shares a single logit, p_add == p_leaf
        and p_split == p_internal_hoist hold exactly, so the RJMCMC move-
        selection term log p(m'|T')/p(m|T) is identically zero. A uniform
        floor eps keeps every probability >= eps (containment).
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] eta = self._move_logits
        cdef cnp.ndarray[cnp.float64_t, ndim=1] l8 = np.array(
            [eta[0], eta[1], eta[2], eta[1], eta[2], eta[3], eta[4], eta[5]],
            dtype=np.float64)
        l8 = l8 - l8.max()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] b = np.exp(l8)
        b = b / b.sum()
        cdef double eps = self.adapt_floor
        return (1.0 - 8.0 * eps) * b + eps

    cdef int _move_group(self, str move_name):
        """Logical adaptation group (0..5) of a move, or -1 if the move is not
        part of the adaptive kernel (e.g. parameter_refresh)."""
        if move_name == "point_mutation":
            return 0
        elif move_name == "add_node" or move_name == "leaf_hoist":
            return 1
        elif move_name == "split" or move_name == "internal_hoist":
            return 2
        elif move_name == "subtree_mutation":
            return 3
        elif move_name == "subtree_crossover":
            return 4
        elif move_name == "etr_mutation":
            return 5
        return -1

    cdef void _adapt_move(self, str move_name, bint accepted):
        """One diminishing-adaptation update of the proposed move's group logit.

            eta_g <- clip( eta_g + gamma_t * (a - target),  -B, +B ),
            gamma_t = gamma0 / (t0 + t),   a = 1{accepted}.

        gamma_t -> 0 as t grows gives Diminishing Adaptation; the box [-B, B]
        together with the eps floor in _adapted_move_probs gives Containment.
        By Roberts & Rosenthal (2007) these two conditions make the adaptive
        chain ergodic for the target pi. No-op when adapt_moves is False or
        for non-adaptive moves (group -1).
        """
        if not self.adapt_moves:
            return
        cdef int g = self._move_group(move_name)
        if g < 0:
            return
        cdef double gamma = self.adapt_gamma0 / (self.adapt_t0 + <double> self._adapt_t)
        cdef double a = 1.0 if accepted else 0.0
        cdef double B = self.adapt_logit_bound
        cdef double val = self._move_logits[g] + gamma * (a - self.adapt_target)
        if val > B:
            val = B
        elif val < -B:
            val = -B
        self._move_logits[g] = val
        self._adapt_t += 1

    cdef tuple _propose(self, int i, int max_attempts=1) :
        """
        Build a proposal from tree i using a move drawn from the temperature-dependent kernel.
        Returns a deepcopy candidate (original tree remains unchanged).

        Phase 2.13(e): max_attempts MUST be 1. Retrying with a freshly drawn
        move on a validity failure creates a proposal mixture that the
        (lqf, lqr) bookkeeping does not account for (the returned densities
        describe only the successful attempt), silently breaking detailed
        balance. An invalid candidate is a hard REJECT (the None return path
        in _step), which is exact.
        """
        cdef function_treeC.Tree cand
        # Phase 2.13(b1): for subtree_crossover the donor child is PART of
        # the proposal (pair move). cand2 is the mutated donor deepcopy and
        # donor_idx its slot; both are None/-1 for every other move.
        cdef function_treeC.Tree cand2
        cdef str move_name
        cdef dict kwargs
        cdef int attempts = 0
        cdef int donor_idx = -1
        cdef double lqf, lqr

        while attempts < max_attempts:
            cand = deepcopy(self.trees[i])
            cand2 = None
            donor_idx = -1
            move_name, kwargs = self._choose_move()
            # Phase 2.10 Stage 3a: self.atoms rides along on EVERY
            # atom-aware move (point, the two hoists, add_node, split,
            # ETR) — always positionally (cpdef across cimport). Passing it
            # to a birth move but not its death partner (or vice versa)
            # breaks the M+1+A density pairing, so this block is the ONLY
            # place atoms enter the kernel. subtree_mutation and
            # subtree_crossover take no atoms by design (function-only
            # rebuild / pure relocation).
            if move_name == "point_mutation":
                lqf, lqr = gp_movesC.point_mutation(
                    cand, self.funcs, kwargs.get("operations", None),
                    None, self.atoms)
            elif move_name == "leaf_hoist":
                lqf, lqr = gp_movesC.leaf_hoist(
                    cand, self.funcs, self.operations, self.atoms)
            elif move_name == "internal_hoist":
                lqf, lqr = gp_movesC.internal_hoist(
                    cand, self.funcs, self.operations, self.atoms)
            elif move_name == "add_node":
                lqf, lqr = gp_movesC.add_node(
                    cand, self.funcs, kwargs.get("operations", None),
                    False, self.atoms)
            elif move_name == "split":
                lqf, lqr = gp_movesC.split(
                    cand, self.funcs, kwargs.get("operations", None),
                    self.atoms)
            elif move_name == "subtree_mutation":
                    #print(kwargs.get('random_depth', False))
                # Phase 2.13(f): pass the kernel's op vocabulary so the
                # rebuild samples from it (support closure + exact fwd/rev
                # build-factor cancellation).
                lqf, lqr = gp_movesC.subtree_mutation(cand, self.funcs,
                                                      self.operations)
            elif move_name == "subtree_crossover":
                # Phase 2.13(b1): pick a donor slot (not i); BOTH children
                # are kept and returned. Installing only slot i's child (the
                # pre-2.13 behaviour) made the reverse density zero while
                # the bookkeeping used pair-swap counts -> detailed balance
                # broken. _step installs the pair jointly on accept.
                if self.size > 1:
                    donor_idx = i
                    while donor_idx == i:
                        donor_idx = self._py_rng.randrange(self.size)
                    cand2 = deepcopy(self.trees[donor_idx])
                    lqf, lqr = gp_movesC.subtree_crossover(cand, cand2)
                else:
                    lqf, lqr = gp_movesC.point_mutation(
                        cand, self.funcs, self.operations,
                        None, self.atoms)
                    move_name = "point_mutation"
            elif move_name == "etr_mutation":
                lqf, lqr = gp_movesC.etr_mutation(
                    cand, self.funcs, kwargs.get("operations", None),
                    self.atoms)
            elif move_name == "flex_slot_mutation":
                # Phase FX: dedicated slot move — uniform flex pick x
                # uniform inner move on the slot subtree. Takes no atoms:
                # slots are plain trees by design (no opaque atoms, no
                # nesting). If the tree has no flex node the move returns
                # (0, -inf) -> hard reject, same convention as every
                # structurally-impossible move.
                lqf, lqr = gp_movesC.flex_slot_mutation(
                    cand, self.funcs, self.operations)
            else:
                lqf, lqr = gp_movesC.point_mutation(
                    cand, self.funcs, self.operations,
                    None, self.atoms)
                move_name = "point_mutation"

            if cand.is_valid(self.domain) and (
                    cand2 is None or cand2.is_valid(self.domain)):
                return (cand, lqf, lqr, move_name, cand2, donor_idx)
            attempts += 1
            #print(attempts)
        return (None, 0.0, 0.0, move_name, None, -1)
        #return deepcopy(self.trees[i])

    cdef _step_parameter_refresh(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        Phase 2.1.PR. Self-accepting parameter refresh on a uniformly chosen
        tree of this grove.

        Phase MD (D1): REFUSED under datasets — PR proposes theta and
        accepts on ONE (y, sigma)'s raw likelihood, which would leave the
        chain's accepted state inconsistent with the reported joint
        score. The ctor forces pr_prob = 0.0; reaching this method with
        datasets set means someone re-enabled it by hand.

        Flow
        ----
        1. Pick i uniformly in {0, ..., size-1}.
        2. If K(tree_i) == 0, count as a trivially-accepted no-op (PR on a
           parameter-free tree is the identity move).
        3. Look up / lazily create the per-topology AMState keyed by
           tree.tree_get_description() (D1, D6). If a cached state exists
           but its K disagrees with the current tree's K (e.g. a structural
           move was just accepted under the same description string -- which
           shouldn't happen with a canonical key, but we guard anyway),
           rebuild the AMState from scratch.
        4. Delegate to gp_movesC.parameter_refresh, which runs ONE step of
           the AM-tempered MH (D2 raw-likelihood scoring, D7 tempering by
           1/T) and returns (accepted, 0.0, 0.0). The (0, 0) is a sentinel
           meaning "PR has already accepted/rejected"; we MUST NOT re-score.
        5. On accept, refresh self.root_posteriors[i] from the new theta and
           bump counters.

        Why this is NOT in the structural _propose path
        -----------------------------------------------
        parameter_refresh is the only move with persistent state and the
        only move whose acceptance is decided internally. Folding it into
        _propose / outer-MH would either (a) double-evaluate the
        likelihood (the outer MH would re-score after PR already accepted,
        composing two MH kernels with the same target -- correct but no
        longer the AM kernel that PR was tuned for), or (b) silently
        sidestep the AM adaptation entirely. Keeping it on its own branch
        preserves the AM kernel as written.

        Why we do not feed (lqf, lqr) from PR into the outer MH
        -------------------------------------------------------
        PR's proposal is symmetric (Gaussian) -> log_q_rev - log_q_fwd = 0.
        The (0, 0) it returns is a convention, not a real proposal density
        for the outer MH. The accept/reject is already done.
        """
        if self.datasets is not None:
            raise ValueError(
                "Grove: parameter_refresh is not supported with datasets "
                "(PLAN MR.4 D1) — the ctor forces pr_prob = 0.0; do not "
                "re-enable it on a joint Grove.")
        if self.size == 0:
            return False
        cdef int i = self._py_rng.randrange(self.size)
        cdef function_treeC.Tree tree = self.trees[i]
        # Phase 2.10 Stage 3a: the coordinate names (and hence K) come from
        # pr_coord_names — the SAME derivation parameter_refresh uses
        # internally, so the AMState dimension can never desync from PR.
        # For atom-free trees this is bit-identical to the historical
        # [p.desc for p in tree.params] (duplicates included); for trees
        # with opaque leaves it is name_values' keys, which makes theta_psi
        # refreshable coordinates (DESIGN §5).
        cdef list names = gp_movesC.pr_coord_names(tree)
        cdef int K = len(names)

        # Counters: PR is always "proposed" (even no-op K=0 case).
        self.proposal_count += 1.0
        self.move_proposals["parameter_refresh"] = \
            self.move_proposals.get("parameter_refresh", 0) + 1

        if K == 0:
            # Identity move on a parameter-free tree. Count as accepted so
            # the per-move rate doesn't get artificially deflated by topologies
            # that PR cannot meaningfully act on.
            self.acceptance_count += 1.0
            self.move_accepts["parameter_refresh"] = \
                self.move_accepts.get("parameter_refresh", 0) + 1
            return True

        # Per-topology AM state (D1, D6). Key on canonical description --
        # the same key Phase 2.7(f) BIC cache will use.
        cdef str key = tree.tree_get_description()
        cdef gp_movesC.AMState am
        if key not in self._am_states:
            am = gp_movesC.AMState(names)
            self._am_states[key] = am
        else:
            am = <gp_movesC.AMState> self._am_states[key]
            if am.K != K:
                # Defensive: same description string but K mismatch.
                # Should not occur if tree_get_description is canonical;
                # rebuild to keep PR safe.
                am = gp_movesC.AMState(names)
                self._am_states[key] = am

        cdef bint accepted
        cdef double _lqf, _lqr  # sentinels (0, 0); not used downstream
        accepted, _lqf, _lqr = gp_movesC.parameter_refresh(
            tree, y, self.sigma, am, self.T)

        if accepted:
            # theta changed in place -> cached posterior is stale.
            # Refresh against the same scorer used by the structural path
            # so root_posteriors stays comparable across move types.
            '''self.root_posteriors[i] = tree.get_superficial_posterior(
                y, self.lambda_reg)
            self.acceptance_count += 1.0
            self.move_accepts["parameter_refresh"] = \
                self.move_accepts.get("parameter_refresh", 0) + 1
            # TODO[Phase 2.7 D8]: when the BIC cache lands, mark'''
            
            # self._bic_cache[key].dirty = True here so the next
            # structural-move scoring re-runs update_parameters from
            # the freshly accepted theta.
            # Phase 2.7: BIC is the score at the topology's MLE and is
            # independent of the current theta. PR diffuses theta around
            # theta_hat without changing the BIC of this topology, so the
            # cache stays valid — do NOT refresh root_posteriors here.
            # The next structural-move proposal will re-optimise the OLD
            # tree from the freshly accepted theta as a warm start (cheap)
            # via log_marginal_bic on the cache-fill path.
            self.acceptance_count += 1.0
            self.move_accepts["parameter_refresh"] = \
                self.move_accepts.get("parameter_refresh", 0) + 1
            return True
        return False

    cdef _step(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        One MH step on a uniformly chosen tree in this grove. Returns True
        if the proposal was accepted.

        Acceptance rule (Phase 2.2):
            log_alpha = beta * [(new_post + new_prior) - (old_post + old_prior)]
                    + (log_q_rev - log_q_fwd)
        with beta = 1/T tempering the target only. The proposal correction
        is applied at face value (see Phase 1.3, Phase 2.2 derivation).
        """
        # Phase 2.5: install this Grove's RNGs on gp_movesC for the
        # duration of this step. Required because multiple Groves in the
        # same process share gp_movesC's module-level _PY_RNG/_NP_RNG
        # (e.g. non-MPI Forest with several temperatures).
        gp_movesC.set_rng(self._py_rng, self._np_rng)
        function_treeC.set_rng(self._py_rng, self._np_rng)
                # Front-of-queue PR branch: PR is self-accepting (it runs its own
        # tempered MH internally with the AM kernel) and must NOT go
        # through the structural _propose / log_alpha path.
        if self._py_rng.random() < self.pr_prob:
            return self._step_parameter_refresh(y)
        cdef int i = self._py_rng.randrange(self.size)
        cdef function_treeC.Tree cand
        cdef function_treeC.Tree cand2
        cdef int donor_idx
        cdef double lqf, lqr
        cdef str move_name
        cdef tuple proposal = self._propose(i)


        cand     = <function_treeC.Tree> proposal[0]
        lqf      = <double> proposal[1]
        lqr      = <double> proposal[2]
        move_name = <str>   proposal[3]
        cand2    = <function_treeC.Tree> proposal[4]
        donor_idx = <int> proposal[5]

        self.proposal_count += 1.0
        self.move_proposals[move_name] = self.move_proposals.get(move_name, 0) + 1

        # Validity-failed proposal -> hard reject (already counted as proposed).
        if cand is None or not np.isfinite(lqr):
            self._adapt_move(move_name, False)
            return False

        # Phase 2.13(b1): within-grove crossover is a PAIR move — both
        # children accepted (and installed) or rejected together.
        if cand2 is not None:
            return self._step_pair_crossover(y, i, donor_idx, cand, cand2,
                                             lqf, lqr)

        # Phase 2.13(d): lazy cache-fill for the current tree. Groves built
        # from node_list (or any path that skipped scoring) hold NaN here;
        # without this fill old_post = NaN poisons log_alpha (every
        # comparison False) and the slot can never accept anything — a
        # silent stall. _score_post shares the validity gate + BIC cache
        # with the candidate-scoring path, so old and new are comparable.
        if np.isnan(self.root_posteriors[i]):
            self.root_posteriors[i] = self._score_post(self.trees[i], y, True)
        cdef double old_post = float(self.root_posteriors[i])

        # Refit theta on the candidate (BIC-style profile; Phase 2.7 will
        # formalise this into an explicit BIC scorer).
        #BELOW IS DEPRECATED
        '''if len(cand.param_values) > 0:
            cand.update_parameters(self.domain, y,
                                self.method_str, self.use_gradients)

        cdef double new_post = float(cand.get_superficial_posterior(
            y, self.lambda_reg))
        '''    
        cdef double new_post
        # Phase 2.10 Stage 3a: gate on name_values, not param_values — an
        # opaque-leaf tree can carry continuous parameters (theta_psi in
        # name_values) with ZERO plain parameter Nodes, and skipping the
        # optimizer for it would score the warm-start theta as if it were
        # the MLE. Atom-free trees: identical truthiness (every parameter
        # Node contributes a name), so legacy behaviour is unchanged.
        cdef bint optim_need = len(cand.name_values) > 0
        cdef str cand_key
        cdef object cached
        cdef dict cached_nv
        if self.datasets is not None:
            # Phase MD: JOINT scoring path — validity (strictest-wins),
            # cache and reduction all live inside _score_cand_joint. A
            # <= -1e290 return is the invalid/impossible convention:
            # hard-reject exactly like the legacy validity gate.
            new_post = self._score_cand_joint(cand, True)
            if new_post <= -1.0e290:
                self._adapt_move(move_name, False)
                return False
        else:
            if not cand.is_valid(self.domain):
                self._adapt_move(move_name, False)
                return False

            # Phase 2.7(f): BIC topology cache.
            # Reset cache when y reference changes between calls (different
            # dataset => stored log_evidence and theta_hat are invalid).
            if self._cache_y is not y:
                self._bic_cache.clear()
                self._cache_y = y

            cand_key = cand.tree_get_description()
            cached = self._bic_cache.get(cand_key)
            if cached is not None:
                # Hit: reuse stored log_evidence and install cached theta_hat
                # so the candidate inherits the MLE on accept (warm-starts
                # subsequent moves and keeps name_values consistent).
                # Stage 3a: _install_cached_theta also restores opaque-leaf
                # theta (plain-only before), and preserves the historical
                # tree_update gate.
                new_post  = <double> cached[0]
                cached_nv = <dict>   cached[1]
                if _install_cached_theta(cand, cached_nv):
                    cand.tree_update(cand.get_root())
            else:
                new_post = cand._log_marginal_bic(
                    y, optimize=optim_need,
                    use_gradients=self.use_gradients if optim_need else False,
                    method_str=self.method_str,
                )
                # NaN-leak fix (2026-07-15): do NOT cache penalty-scale posts
                # (< -1e90 needs mse/sigma^2 ~ 1e88 — no legitimate model).
                # Pre-fix these landings scored NaN and were never cached, so
                # the topology was RETRIED with a fresh warm start on the next
                # proposal; caching the now-finite penalty (with its junk
                # theta_hat) would permanently condemn a possibly-true
                # topology to a single bad Powell landing.
                if np.isfinite(new_post) and new_post > -1.0e90:
                    self._bic_cache[cand_key] = (new_post, dict(cand.name_values))

        if not np.isfinite(new_post):
            self._adapt_move(move_name, False)
            return False

        # Polya prior contributions (zero when self.polya is None).
        cdef double old_prior = 0.0
        cdef double new_prior = 0.0
        cdef list new_path = None
        cdef list old_path = None
        if self.polya is not None:
            # Phase 2.13(c): extract BOTH paths FIRST, then score both.
            # extract_path may auto-register a fresh symbol, growing every
            # Dirichlet (n_total += 1); interleaving extraction with scoring
            # evaluated old and new against DIFFERENT vocabularies whenever
            # the candidate introduced a new symbol, systematically
            # favoring the incumbent by ~(#unvisited steps)*log((n+1)/n).
            old_path = self.polya.extract_path(self.trees[i].get_root())
            new_path  = self.polya.extract_path(cand.get_root())
            old_prior = self.polya.log_prior(old_path)
            new_prior = self.polya.log_prior(new_path)

        # Phase 4.4: structural-prior tilt (0.0 unless structural_prior
        # is set — single branch inside _sp, so the default path is
        # unchanged). A prior term: lives ONLY in this comparison, never
        # in root_posteriors / the BIC cache.
        cdef double sp_old = self._sp(self.trees[i])
        cdef double sp_new = self._sp(cand)
        cdef double dtarget = ((new_post + new_prior + sp_new)
                               - (old_post + old_prior + sp_old))
        cdef double q_corr  = lqr - lqf
        cdef double log_alpha

        # Tempering: beta = 1/T multiplies dtarget ONLY. Greedy if T <= 0.
        if self.T <= 0.0:
            # Greedy acceptance on the target diff; q_corr is the tiebreaker
            # at exact equality. Strictly: accept iff target strictly improves.
            if dtarget > 0.0:
                log_alpha = 0.0
            elif dtarget == 0.0 and q_corr >= 0.0:
                log_alpha = 0.0
            else:
                log_alpha = -np.inf
        else:
            log_alpha = dtarget / self.T + q_corr

        cdef double u
        cdef bint accepted
        if log_alpha >= 0.0:
            accepted = True
        elif log_alpha == -np.inf:
            accepted = False
        else:
            u = self._py_rng.random()
            # u == 0 would give log(u) = -inf; accept (probability-zero edge).
            accepted = (u <= 0.0) or (np.log(u) < log_alpha)

        # Phase 2.8: feed the accept/reject outcome into the adaptive
        # move-weight update (no-op when adapt_moves is False).
        self._adapt_move(move_name, accepted)

        if accepted:
            self.trees[i] = cand
            self.root_posteriors[i] = new_post
            self.acceptance_count += 1.0
            self.move_accepts[move_name] = self.move_accepts.get(move_name, 0) + 1
            if self.polya is not None and new_path is not None:
                if self.polya_defer:
                    # Phase 2.12: collect for the master's deterministic
                    # replay instead of reinforcing immediately. Priors
                    # keep using the last-synced Theta (delayed adaptation;
                    # gamma_t -> 0 + containment still hold, Roberts &
                    # Rosenthal 2007).
                    self._deferred_paths.append(new_path)
                else:
                    self.polya.update_path(new_path)
            return True
        return False

    cdef bint _step_pair_crossover(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                   int i, int j,
                                   function_treeC.Tree candA,
                                   function_treeC.Tree candB,
                                   double lqf, double lqr):
        """Phase 2.13(b1): joint accept/reject for the within-grove
        subtree_crossover PAIR move.

        Pre-2.13 the move installed only slot i's child and discarded the
        donor's: from the new state the displaced subtree existed nowhere,
        so the true reverse density was zero while the bookkeeping used the
        pair-swap node counts — detailed balance broken (AUDIT 2.13(b)).

        Here BOTH children are proposed and installed together. Both slots
        live in this grove (same T), so the joint tempered target is
        pi_T(x_i) * pi_T(x_j) and

            log_alpha = [(U_i' - U_i) + (U_j' - U_j)] / T + (lqr - lqf),

        with U = BIC log-evidence + Polya log-prior. Slot-pair selection
        (i uniform, donor uniform, both (i,j)/(j,i) orderings) is equal
        forward and reverse and cancels; the node-selection counts live in
        (lqf, lqr) from subtree_crossover. Proposal correction untempered,
        as everywhere. Caller has already bumped proposal counters and
        hard-rejected non-finite lqr; _propose validated both children.
        """
        # Lazy fills (Phase 2.13(d)) for BOTH slots.
        if np.isnan(self.root_posteriors[i]):
            self.root_posteriors[i] = self._score_post(self.trees[i], y, True)
        if np.isnan(self.root_posteriors[j]):
            self.root_posteriors[j] = self._score_post(self.trees[j], y, True)
        cdef double old_post_i = float(self.root_posteriors[i])
        cdef double old_post_j = float(self.root_posteriors[j])

        cdef double new_post_i = self._score_post(candA, y, True)
        cdef double new_post_j = self._score_post(candB, y, True)
        if new_post_i <= -1.0e299 or new_post_j <= -1.0e299:
            self._adapt_move("subtree_crossover", False)
            return False

        cdef double old_prior_i = 0.0
        cdef double old_prior_j = 0.0
        cdef double new_prior_i = 0.0
        cdef double new_prior_j = 0.0
        cdef list new_path_i = None
        cdef list new_path_j = None
        cdef list old_path_i, old_path_j
        if self.polya is not None:
            # Phase 2.13(c) discipline: extract ALL paths, then score all.
            old_path_i = self.polya.extract_path(self.trees[i].get_root())
            old_path_j = self.polya.extract_path(self.trees[j].get_root())
            new_path_i = self.polya.extract_path(candA.get_root())
            new_path_j = self.polya.extract_path(candB.get_root())
            old_prior_i = self.polya.log_prior(old_path_i)
            old_prior_j = self.polya.log_prior(old_path_j)
            new_prior_i = self.polya.log_prior(new_path_i)
            new_prior_j = self.polya.log_prior(new_path_j)

        # Phase 4.4: structural-prior tilt terms (0.0 when feature off).
        cdef double dtarget = (
            (new_post_i + new_prior_i + self._sp(candA)
             + new_post_j + new_prior_j + self._sp(candB))
            - (old_post_i + old_prior_i + self._sp(self.trees[i])
               + old_post_j + old_prior_j + self._sp(self.trees[j])))
        cdef double q_corr = lqr - lqf
        cdef double log_alpha

        if self.T <= 0.0:
            # Greedy mode: same convention as the single-tree path.
            if dtarget > 0.0:
                log_alpha = 0.0
            elif dtarget == 0.0 and q_corr >= 0.0:
                log_alpha = 0.0
            else:
                log_alpha = -np.inf
        else:
            log_alpha = dtarget / self.T + q_corr

        cdef double u
        cdef bint accepted
        if log_alpha >= 0.0:
            accepted = True
        elif log_alpha == -np.inf:
            accepted = False
        else:
            u = self._py_rng.random()
            accepted = (u <= 0.0) or (np.log(u) < log_alpha)

        self._adapt_move("subtree_crossover", accepted)

        if accepted:
            self.trees[i] = candA
            self.trees[j] = candB
            self.root_posteriors[i] = new_post_i
            self.root_posteriors[j] = new_post_j
            self.acceptance_count += 1.0
            self.move_accepts["subtree_crossover"] = \
                self.move_accepts.get("subtree_crossover", 0) + 1
            if self.polya is not None and new_path_i is not None:
                # Within-grove crossover is mutation-family (R1): reinforce
                # BOTH accepted trees. Deferred under polya_defer (2.12).
                if self.polya_defer:
                    self._deferred_paths.append(new_path_i)
                    self._deferred_paths.append(new_path_j)
                else:
                    self.polya.update_path(new_path_i)
                    self.polya.update_path(new_path_j)
            return True
        return False

    cpdef list drain_deferred_paths(self):
        """Phase 2.12: return and clear the accepted construction paths
        collected while polya_defer was True. Order = acceptance order."""
        cdef list out = self._deferred_paths
        self._deferred_paths = []
        return out

    cdef double _score_cand_joint(self, function_treeC.Tree cand,
                                  bint optimize=True):
        """Phase MD: score `cand` JOINTLY over self.datasets.

        Flow (PLAN MR.4): strictest-wins validity at reference theta on
        EVERY dataset's x; description-keyed cache with the joint value
        shape (joint_post, [nv_0 .. nv_{J-1}]); on a miss, every dataset
        is fitted INDEPENDENTLY from the same cold start (the candidate's
        incoming theta — D7b: never the neighbouring dataset's fit) via
        Tree.score_dataset (positional; explicit x/y/sigma), the
        per-dataset penalties are recomputed by _md_dataset_penalty and
        the vector is reduced by _reduce_joint. The REFERENCE theta
        (dataset 0) is left installed on exit — the single-theta
        convention every other consumer of the tree relies on. Cache
        discipline: an entry is stored only when the joint AND every
        per-dataset post are sane (finite, > -1e90) — the NaN-leak
        contract applied PER DATASET.

        Returns the joint post, or -1.0e300 when any dataset's validity
        gate fails (the impossible-candidate convention of _score_post).
        """
        cdef int J = len(self.datasets)
        cdef int j
        cdef str cand_key
        cdef object cached
        cdef list posts, pens, nvs
        cdef dict start_nv
        cdef double post_j, joint
        cdef bint optim_need, cacheable
        cdef object xj, yj, sj
        cdef tuple res
        for j in range(J):
            if not cand.is_valid(self.datasets[j][0]):
                return -1.0e300
        cand_key = cand.tree_get_description()
        cached = self._bic_cache.get(cand_key)
        if cached is not None:
            if _install_cached_theta(cand, <dict> (<list> cached[1])[0]):
                cand.tree_update(cand.get_root())
            return <double> cached[0]
        posts = []; pens = []; nvs = []
        start_nv = dict(cand.name_values)
        optim_need = optimize and len(cand.name_values) > 0
        for j in range(J):
            xj = self.datasets[j][0]
            yj = self.datasets[j][1]
            sj = self.datasets[j][2]
            if j > 0 and _install_cached_theta(cand, start_nv):
                cand.tree_update(cand.get_root())
            # NB no <str> cast on method_str: groveC's language level
            # trips Cython's unicode->str coercion rule on an explicit
            # cast (the atoms-registration trap); the uncast object
            # converts at the call boundary exactly like the legacy
            # _log_marginal_bic call in _step.
            res = cand.score_dataset(
                xj, yj, sj, optim_need,
                self.use_gradients if optim_need else False,
                self.method_str)
            post_j = <double> res[0]
            posts.append(post_j)
            nvs.append(res[1])
            pens.append(_md_dataset_penalty(
                len(<dict> res[1]), (<cnp.ndarray> yj).shape[0], sj))
        # Reference theta FIRST, reduction second (review 2026-07-31,
        # finding 5): if the reduction raises (hand-reassigned
        # md_combine), the live tree is already back on the dataset-0
        # convention instead of carrying the last dataset's fit.
        if _install_cached_theta(cand, <dict> nvs[0]):
            cand.tree_update(cand.get_root())
        joint = _reduce_joint(posts, pens, self.md_combine,
                              self.md_softmin_beta, self.md_baselines)
        cacheable = np.isfinite(joint) and joint > -1.0e90
        for j in range(J):
            if not (np.isfinite(<double> posts[j])
                    and <double> posts[j] > -1.0e90):
                cacheable = False
        if cacheable:
            self._bic_cache[cand_key] = (joint, nvs)
        return joint

    cpdef object joint_thetas(self, function_treeC.Tree tree):
        """Phase MD introspection: the cached per-dataset theta list
        [nv_0 .. nv_{J-1}] for `tree`'s topology, or None when the
        topology has not been jointly scored (or the Grove is legacy).
        Hall-of-fame consumers use this to carry theta per dataset."""
        if self.datasets is None:
            return None
        cdef object cached = self._bic_cache.get(tree.tree_get_description())
        if cached is None:
            return None
        # Review 2026-07-31 (finding 2): DEEP-copy the per-dataset dicts —
        # a shallow list copy would hand callers live references into the
        # cache, and a consumer mutating one would silently change what
        # every later cache hit installs while the cached joint float
        # stayed stale (the Stage-3a installed-vs-reported bug class).
        return [dict(nv) for nv in <list> cached[1]]

    cdef double _score_post(self, function_treeC.Tree cand,
                            cnp.ndarray[cnp.float64_t, ndim=1] y,
                            bint optimize=True):
        """Score a candidate's BIC log-evidence, mirroring the scoring block
        of `_step`: validity gate + Phase 2.7(f) topology cache + optimize.

        Phase MD: joint Groves route through _score_cand_joint (same
        -1.0e300 impossible-candidate convention).

        Returns -1e300 for invalid / non-finite models (so any MH ratio that
        consumes it rejects). On a cache hit the cached theta_hat is
        re-installed on `cand` so an accept warm-starts subsequent moves.

        Used by Forest's cross-temperature swap / crossover (Phase 2.9) so
        trans-grove moves are scored exactly like within-grove ones.
        """
        cdef str cand_key
        cdef object cached
        cdef dict cached_nv
        cdef double new_post
        if self.datasets is not None:
            return self._score_cand_joint(cand, optimize)
        if not cand.is_valid(self.domain):
            return -1.0e300
        # Keep the per-topology cache coherent with the active y reference.
        if self._cache_y is not y:
            self._bic_cache.clear()
            self._cache_y = y
        cand_key = cand.tree_get_description()
        cached = self._bic_cache.get(cand_key)
        if cached is not None:
            new_post  = <double> cached[0]
            cached_nv = <dict>   cached[1]
            # Stage 3a: restores opaque-leaf theta too (see the helper).
            if _install_cached_theta(cand, cached_nv):
                cand.tree_update(cand.get_root())
            return new_post
        new_post = cand._log_marginal_bic(
            y, optimize=optimize,
            use_gradients=self.use_gradients if optimize else False,
            method_str=self.method_str,
        )
        if np.isfinite(new_post):
            # NaN-leak fix (2026-07-15): penalty-scale posts are returned
            # (correctly terrible) but NOT cached — see _step for why.
            if new_post > -1.0e90:
                self._bic_cache[cand_key] = (new_post,
                                             dict(cand.name_values))
            return new_post
        return -1.0e300

    cdef double _sp(self, function_treeC.Tree tree):
        """Phase 4.4: structural-constraint prior tilt for one tree.

        Returns the user callable's LOG-prior increment (encode -lam*C
        in the callable); 0.0 when the feature is off — one branch, so
        the default path is untouched. Evaluated once per candidate
        COMPARISON (the acceptance sums only): never inside the Powell
        objective, never folded into root_posteriors or the BIC cache,
        exactly like the Polya log-prior. Description-keyed caching
        (default) matches the structure-scale contract of constraints;
        set structural_prior_cache=False for theta-sensitive callables.
        Returns must be finite: a non-finite evaluation is treated as a
        constraint-evaluation failure and floored at -1e90 (soft
        reject) — bounded penalties keep the move graph irreducible,
        and the floor matches the magnitude the BIC-cache guard already
        treats as penalty-scale.
        """
        if self.structural_prior is None:
            return 0.0
        cdef object key, v
        cdef double out
        if self.structural_prior_cache:
            key = tree.tree_get_description()
            v = self._sp_cache.get(key)
            if v is not None:
                return <double> v
            out = <double> self.structural_prior(tree)
            if not np.isfinite(out) or out < -1.0e90:
                out = -1.0e90
            self._sp_cache[key] = out
            return out
        out = <double> self.structural_prior(tree)
        if not np.isfinite(out) or out < -1.0e90:
            out = -1.0e90
        return out

    cpdef dict move_stats(self):
        """Per-move (proposed, accepted, rate) summary. Feeds Phase 2.8."""
        cdef dict out = {}
        cdef str name
        cdef int n_p, n_a
        for name in self.move_proposals:
            n_p = int(self.move_proposals[name])
            n_a = int(self.move_accepts.get(name, 0))
            out[name] = {
                "proposed": n_p,
                "accepted": n_a,
                "rate": (float(n_a) / n_p) if n_p > 0 else 0.0,
            }
        return out

    cdef legacy_step(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        One MH step on a random tree in this grove.
        Returns True if accepted.
        """
        if self.size == 0:
            return False
        # Phase 2.5: install this Grove's RNGs on gp_movesC for the duration
        # of this step (same rationale as _step).
        gp_movesC.set_rng(self._py_rng, self._np_rng)
        function_treeC.set_rng(self._py_rng, self._np_rng)
        cdef int i = 0 #random.randrange(self.size)

        # ensure current posterior is available
        if np.isnan(self.root_posteriors[i]):
            self.root_posteriors[i] = self.trees[i].get_superficial_posterior(y, self.lambda_reg)

        cdef double old_post = float(self.root_posteriors[i])

        cdef function_treeC.Tree cand = self._propose(i)
        #print(cand.name_values)
        
        if len(cand.param_values)>0:
            cand.update_parameters(self.domain, y,
                 self.method_str, self.use_gradients)
        
        cdef double new_post = float(cand.get_superficial_posterior(y, self.lambda_reg))
        # Pólya prior contributions (zero when self.polya is None).
        cdef double old_prior = 0.0
        cdef double new_prior = 0.0
        cdef list new_path = None
        cdef list old_path = None
        if self.polya is not None:
            # Prior of the *current* tree must be recomputed every step
            # because Theta moves whenever any walker accepts.
            # Phase 2.13(c): extract both paths BEFORE scoring either
            # (auto-registration grows n_total; see _step).
            old_path = self.polya.extract_path(self.trees[i].get_root())
            new_path = self.polya.extract_path(cand.get_root())
            old_prior = self.polya.log_prior(old_path)
            new_prior = self.polya.log_prior(new_path)
        cdef double dlogpost = (new_post + new_prior) - (old_post + old_prior)

        cdef double acc_p = _accept_prob(dlogpost, self.T)
        self.proposal_count += 1.0
        if self._py_rng.random() < acc_p:
            self.trees[i] = cand
            self.root_posteriors[i] = new_post
            self.acceptance_count += 1.0
            if self.polya is not None and new_path is not None:
                # Reinforce accepted path with diminishing rate gamma_t = 1/(1+t).
                self.polya.update_path(new_path)
            return True
        return False

    cpdef run(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps):
        """
        Run n_steps MH updates within this grove.
        """
        cdef int t
        for t in range(n_steps):
            #print(t)
            self._step(y)

    cpdef acceptance_rate(self):
        if self.proposal_count == 0.0:
            return 0.0
        return self.acceptance_count / self.proposal_count

    
    cpdef burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                bint replace_all=True, bint optimize_params=True):
        # Phase MD (review 2026-07-31, finding 3): the ctor refusal only
        # covers construction — this public wrapper would silently score
        # every candidate on ONE dataset and REPLACE the joint population.
        if self.datasets is not None:
            raise ValueError(
                "Grove: burn_in_combinations is not supported with "
                "datasets (single-dataset scoring; PLAN MR.4 D2)")
        self._burn_in_combinations(y, replace_all=replace_all,
                                optimize_params=optimize_params)

    cdef _burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=True, bint optimize_params=True):
        """
        Systematic burn-in: generate all combinations of (func1 op func2) 
        for all operations and all pairs of functions (including parameters).
        
        Parameters
        ----------
        y : array
            Target data for posterior evaluation.
        replace_all : bool
            If True, replace all trees. If False, replace only if better.
        optimize_params : bool
            If True, optimize parameters for each candidate.
        """
      
        cdef list candidates = []
        cdef list all_primitives = []
        cdef str func_str, left_str, right_str, op
        cdef function_treeC.Tree tree
        cdef double post
        cdef int i, idx
        
        # Build full primitive set: functions + one parameter
        all_primitives = self.funcs.copy()
        all_primitives.append("a0")  # add a parameter primitive
        
        print(f"Burn-in: exploring {len(all_primitives)} squared times {len(self.operations)} = "
              f"{len(all_primitives)**2 * len(self.operations)} combinations")
        
        # Generate all (left, op, right) combinations
        for left_str, right_str in itertools.product(all_primitives, repeat=2):
            for op in self.operations:
                '''if op == 'o':  # composition: only makes sense when right is a function of x
                    # Skip if right is a parameter
                    if left_str == "a0" or right_str == "a0":
                        continue'''
                
                
                tree = function_treeC.Tree(
                    depth=1, 
                    functions=self.funcs, 
                    x=self.domain, 
                    Npars=0,  # will add params manually
                    value=1.0, 
                    T=self.T,
                    sigma=self.sigma,
                    g_tau=self.g_tau,
                )
                
                # Build left child
                if left_str == "a0":
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.left.set_value(1.0)
                else:
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(left_str, self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Build right child
                if right_str == "a0":
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a1", self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.right.set_value(1.0)
                else:
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(right_str, self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Assign operation and update
                tree.root.assign_operation(op)
                tree.tree_update(tree.root)
                
                # Validate
                if not tree.is_valid(self.domain):
                    continue
                # Optimize parameters if any exist
                '''if optimize_params and len(tree.params) > 0:
                    tree.update_parameters(self.domain, y,
                        self.method_str, self.use_gradients)'''
                    
                
                # Compute posterior
                post = tree._log_marginal_bic(y,optimize_params,
                        self.use_gradients if optimize_params else False,
                        self.method_str)
                if np.isfinite(post):
                    candidates.append((tree, post))
            
        
        if len(candidates) == 0:
            print("Warning: No valid burn-in candidates generated")
            return
        
        # Sort by posterior (descending = better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Burn-in: {len(candidates)} valid candidates generated")
        print(f"  Best 3 posteriors: {[c[1] for c in candidates[:3]]}")
        print(f"  Best 3 expressions: {[c[0].tree_get_description() for c in candidates[:3]]}")
        
        # Replace grove trees
        if replace_all:
            # Take top `size` candidates, cycling if fewer
            for i in range(self.size):
                idx = i % len(candidates)
                self.trees[i] = deepcopy(candidates[idx][0])
                self.root_posteriors[i] = candidates[idx][1]
        else:
            # Replace only if new candidate is better
            for i in range(min(self.size, len(candidates))):
                if np.isnan(self.root_posteriors[i]) or candidates[i][1] > self.root_posteriors[i]:
                    self.trees[i] = deepcopy(candidates[i][0])
                    self.root_posteriors[i] = candidates[i][1]
        
        print(f"Burn-in complete. Grove best posterior: {self.root_posteriors[self.best_index()]:.4f}")

    cpdef burn_in_depth2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                         bint replace_all=True, bint optimize_params=True):
        """
        Stage 2 burn-in: combine top depth-1 trees with primitives and each other.
        Call this AFTER burn_in_combinations to build depth-2 expressions.

        Phase MD (review 2026-07-31, finding 3): refused with datasets —
        single-dataset scoring would silently replace the joint
        population.
        
        Parameters
        ----------
        y : target data
        replace_all : if True, replace all grove trees with best depth-2 candidates
        optimize_params : if True, optimize parameters for each candidate
        
        """
        if self.datasets is not None:
            raise ValueError(
                "Grove: burn_in_depth2 is not supported with datasets "
                "(single-dataset scoring; PLAN MR.4 D2)")
        cdef dict tree_cache = {}
        # First, run stage 1 to get depth-1 candidates
        depth1_candidates = self._get_depth1_candidates(y, tree_cache, optimize_params)
        
        if len(depth1_candidates) == 0:
            print("Warning: No depth-1 candidates; cannot build depth-2")
            return
        #print(tree_cache)
        # Now combine them
        self._burn_in_stage2(y, depth1_candidates, tree_cache, replace_all, optimize_params)
    
    cdef tuple _burn_in_primitives(self):
        """Phase 2.10 (final): the burn-in token list — funcs + the 'a0'
        parameter primitive + one '@atom:<symbol>' token per registered
        template — and the {symbol: AtomSpec} registry
        _burn_in_primitive_node resolves them against (None when the Grove
        has no atoms: the historical list, bit-identical)."""
        cdef list prims = self.funcs.copy()
        prims.append("a0")
        cdef dict specs = None
        if self.atoms:
            specs = {}
            for _sp in self.atoms:
                specs[_sp.name] = _sp
                prims.append(_ATOM_MARK + _sp.name)
        return (prims, specs)

    cdef _get_depth1_candidates(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict tree_cache,
                                      bint optimize_params=True):
        """
        Build all depth-1 trees (primitive op primitive) and return sorted list.
        Reuses logic from _burn_in_combinations but returns candidates instead of replacing.
        """
        cdef list candidates = []
        cdef list all_primitives
        cdef dict atom_specs
        all_primitives, atom_specs = self._burn_in_primitives()
        cdef str left_str, right_str, op, primitive_expression
        cdef function_treeC.Tree tree, primitive_tree, cand
        cdef function_treeC.Node root_node
        cdef double post
        print(f"Stage 1: building {len(all_primitives)**2 * len(self.operations)} depth-1 trees")
        # Initialize candidates with primitive functions (depth-0 trees)
        # Phase 2.10 (final): construction goes through the token resolver —
        # identical to the old inline branches for 'a0'/functions, and it
        # also resolves '@atom:<symbol>' tokens to opaque leaves (names
        # a1..ak; 'a0' is the parameter primitive).
        for func_str in all_primitives:
            root_node = _burn_in_primitive_node(func_str, self.domain,
                                                atom_specs, 'a', 1)
            primitive_tree = function_treeC.Tree(
                depth=root_node.node_depth, functions=self.funcs, x=self.domain, node=root_node,
                Npars=len(root_node.function.params), value=1.0, T=self.T,
                sigma=self.sigma, g_tau=self.g_tau
            )
            primitive_tree.name_values = deepcopy(primitive_tree.name_values)
            
            #primitive_tree.tree_update(primitive_tree.root)
            primitive_expression = str(primitive_tree.root.function.expr)
            
            if primitive_tree.is_valid(self.domain) and primitive_expression not in tree_cache:
                
                post = primitive_tree._log_marginal_bic(y, optimize_params,
                        self.use_gradients if optimize_params else False, self.method_str)

                if np.isfinite(post) and primitive_expression not in tree_cache:
                    tree_cache[primitive_expression] = post
                
                    candidates.append((primitive_tree, post, 0))  # depth 0
                
        cdef function_treeC.Node root_left, root_right, root_tree
        cdef int right_start
        for left_str, right_str in itertools.product(all_primitives, repeat=2):
            if left_str=='a0' and right_str=='a0':
                continue
            for op in self.operations:

                # Phase 2.10 (final): both leaves via the token resolver.
                # The right leaf's name band starts past a left atom's k so
                # two opaque leaves in one pair never share internal names
                # (sharing would couple their values through name_values);
                # both bands stay in the 'a' namespace, preserving the
                # stage-2 invariant that depth-1 candidates are all-'a'
                # (_combine_depth1 renames right operands to 'b').
                root_left = _burn_in_primitive_node(left_str, self.domain,
                                                    atom_specs, 'a', 1)
                right_start = 1
                if left_str.startswith(_ATOM_MARK):
                    right_start = 1 + (<real_functionC.AtomSpec>
                        atom_specs[left_str[len(_ATOM_MARK):]]).k
                root_right = _burn_in_primitive_node(right_str, self.domain,
                                                     atom_specs, 'a',
                                                     right_start)
                #print(op, root_left.function.get_description(), root_right.function.get_description())
                root_tree = function_treeC.Node(
                    function=None,
                    left=deepcopy(root_left),
                    right=deepcopy(root_right),
                    symbol=op
                )
                #print('root built', root_tree.function.get_description())                
                tree = function_treeC.Tree(
                    depth=root_tree.depth(), functions=self.funcs, x=self.domain, node=root_tree,
                    Npars=len(root_tree.function.params), T=self.T,
                    sigma=self.sigma, g_tau=self.g_tau
                )
                tree.name_values = deepcopy(tree.name_values)
                print(f"  Just created tree: id(name_values)={id(tree.name_values.keys())}, content={tree.name_values}")

                if len(root_tree.function.params) > 1:
                    #print('# parameters > 1')

                    tree.replace_constants_with_pars()
                
                print(f"  Before update params: id(name_values)={id(tree.name_values.keys())}, content={tree.name_values}")
                #tree.tree_update(tree.root)
                #print(tree)
                expression = str(tree.root.function.expr)
                
                if expression in tree_cache or not tree.is_valid(self.domain):
                    continue
                #print(tree)
                
                post = tree._log_marginal_bic(y, optimize_params,
                        self.use_gradients if optimize_params else False, self.method_str)
                
                tree_cache[expression] = post
                
                if np.isfinite(post):
                    tree.name_values = deepcopy(tree.name_values)  
                    candidates.append((tree, post, 1)) # depth 1
                    print(f"  After append: id(name_values)={id(tree.name_values.keys())}, content={tree.name_values}")
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 1: {len(candidates)} valid depth-1 candidates")
        
        if len(candidates) > 0:
            print(f"  Top-3: {[(c[0].tree_get_description(), f'{c[1]:.3f}') for c in candidates[:3]]}")
        for can in candidates:
            cand = can[0]
            print(f'in get_depth1_candidates: id(dict)={id(cand.name_values)} {id(cand.name_values.keys())}, content={cand.name_values}')
        return candidates

    cdef _burn_in_stage2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                          list up2depth1_candidates,
                          dict tree_cache,
                          bint replace_all=True,
                          bint optimize_params=True
                          ):
        """
        Combine top depth-1 trees with primitives and with each other to build depth-2 trees.
        """
        cdef list candidates = []
        cdef list all_primitives
        cdef dict atom_specs
        all_primitives, atom_specs = self._burn_in_primitives()
        # Filter for only depth-1 candidates (third element == 1)
        
        depth1_candidates = [c for c in up2depth1_candidates if c[2] == 1]
        

        cdef int n_top = len(depth1_candidates)
        cdef int total_combinations = n_top * (n_top + len(all_primitives)) * len(self.operations)
        start_time = time.time()
        pbar = tqdm(total=total_combinations, 
                    desc="Stage 2",
                    unit=" combos",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        print(f"Stage 2: combining {n_top} depth-1 trees with {len(all_primitives)} primitives + themselves")
        print(f"  Estimated combinations: ~{n_top * (len(all_primitives) + n_top) * len(self.operations)}")
        print(f"  Estimated RAM occupancy of the grove: ~{n_top * (len(all_primitives) + n_top) * len(self.operations) * 160 / 1000000} MegaBytes")
        
        cdef str prim_str, op
        cdef function_treeC.Tree tree, d1_tree, d1_tree_2
        cdef double post
        cdef int i, j, idx
        cdef function_treeC.Node prim_node
        
        # 1) depth-1 op primitive (both directions)
        
        for i in range(n_top):
            for prim_str in all_primitives:
                # Phase 2.10 (final): token resolver ('a' band — the
                # combiner renames whichever side is the right operand).
                prim_node = _burn_in_primitive_node(prim_str, self.domain,
                                                    atom_specs, 'a', 1)
                for op in self.operations:
                    # left = depth-1, right = primitive
                    d1_tree = depth1_candidates[i][0]
                    tree = self._combine_depth1(d1_tree.root, prim_node, op)
                    expression = tree.tree_get_description()
                    
                    if tree is not None and tree.is_valid(self.domain) and expression not in tree_cache:
            
                        post = tree._log_marginal_bic(y, optimize_params,
                            self.use_gradients if optimize_params else False, self.method_str)

                        tree_cache[expression] = post
                        if np.isfinite(post):
                            candidates.append((tree, post))
                    
                    # left = primitive, right = depth-1
                    tree = self._combine_depth1(prim_node, d1_tree.root, op)
                    expression = tree.tree_get_description()
                    
                    if tree is not None and tree.is_valid(self.domain) and expression not in tree_cache:
                        
                        post = tree._log_marginal_bic(y, optimize_params,
                            self.use_gradients if optimize_params else False, self.method_str)
                        
                        tree_cache[expression] = post
                        if np.isfinite(post):
                            candidates.append((tree, post)) # depth 2
                    pbar.update(1)
                    pbar.set_postfix({"iter/s": f"{(pbar.n / (time.time() - start_time)):.1f}"})
        # 2) depth-1 op depth-1 (all pairs)
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    d1_tree = depth1_candidates[i][0]
                    d1_tree_2 = depth1_candidates[j][0]
                    tree = self._combine_depth1(d1_tree.root, d1_tree_2.root, op)
                    expression = tree.tree_get_description()
                    
                    if tree is not None and tree.is_valid(self.domain) and expression not in tree_cache:
                        
                        post = tree._log_marginal_bic(y, optimize_params,
                            self.use_gradients if optimize_params else False, self.method_str)
                        
                        tree_cache[expression] = post
                        if np.isfinite(post):
                            candidates.append((tree, post)) # depth 2
                    pbar.update(1)
                    pbar.set_postfix({"iter/s": f"{(pbar.n / (time.time() - start_time)):.1f}"})
        pbar.close()
        if len(candidates) == 0:
            print("Warning: No valid depth-2 candidates")
            return
        '''
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 2: {len(candidates)} valid depth-2 candidates")
        print(f"  Top-3: {[(c[0].tree_get_description(), f'{c[1]:.3f}') for c in candidates[:3]]}")
        '''
        # Replace grove trees
        # Append depth-1 candidates to depth-2 candidates
        for i in range(len(up2depth1_candidates)):
            candidates.append(up2depth1_candidates[i])
        
        # Re-sort the combined list (maybe delete?)
        candidates.sort(key=lambda x: x[1], reverse=True)
            
        if replace_all:
            if self.trees is None:
                self.trees = [None] * (len(candidates))
                self.root_posteriors = np.full((len(candidates),),
                     np.nan, dtype=np.float64)
                self.size = len(candidates)
                for i in range(self.size):
                    self.trees[i] = candidates[i][0]
                    self.root_posteriors[i] = candidates[i][1]
            else:
                for i in range(self.size):
                    idx = i % len(candidates)
                    self.trees[i] = candidates[idx][0]
                    self.root_posteriors[i] = candidates[idx][1]
        else:
            for i in range(min(self.size, len(candidates))):
                if np.isnan(self.root_posteriors[i]) or candidates[i][1] > self.root_posteriors[i]:
                    self.trees[i] = candidates[i][0]
                    self.root_posteriors[i] = candidates[i][1]
        
        print(f"Burn-in depth-2 complete. Best: {self.root_posteriors[self.best_index()]:.4f}")
    cdef _combine_depth1_trees(self, function_treeC.Tree left, function_treeC.Tree right, object op):
        """
        Combine depth-1 tree with a primitive (or vice versa).
        
        """
        
        # Left side
        cdef function_treeC.Tree left_tree = deepcopy(left)
        #cdef function_treeC.Node right_node = deepcopy(right)
        
        cdef function_treeC.Node right_copy = deepcopy(right.root)   # <-- FIX: deepcopy right too

        cdef function_treeC.Tree preliminary_tree = function_treeC.Tree(
            depth=right_copy.node_depth, functions=self.funcs,
            x=self.domain, node=right_copy,
            sigma=self.sigma, g_tau=self.g_tau)
        left_tree.tree_update(left_tree.root)
        preliminary_tree.tree_update(preliminary_tree.root)
    

        gp_movesC.rename_prefix(preliminary_tree)

        cdef function_treeC.Node root_node = function_treeC.Node(function=None,
                                            left=left_tree.root, right=preliminary_tree.root, 
                                            symbol=op )
        cdef function_treeC.Tree tree = function_treeC.Tree(depth=root_node.node_depth, functions=self.funcs,
                                    x=self.domain, node = root_node,
                                    Npars=len(root_node.function.params), T=self.T,
                                    sigma=self.sigma, g_tau=self.g_tau)
        tree.name_values = deepcopy(tree.name_values)
        gp_movesC.refresh_params(tree)
        #print(tree.name_values)
        del preliminary_tree
        return tree


    cdef _combine_depth1(self, function_treeC.Node left, function_treeC.Node right, object op):
        """
        Combine depth-1 tree with a primitive (or vice versa).
        
        """
        
        # Left side
        cdef function_treeC.Node left_node = deepcopy(left)
        #cdef function_treeC.Node right_node = deepcopy(right)
        cdef function_treeC.Node right_copy = deepcopy(right)   # <-- FIX: deepcopy right too
        cdef function_treeC.Tree preliminary_tree = function_treeC.Tree(depth = right_copy.node_depth, functions = self.funcs,
                x = self.domain, node = right_copy,
                sigma=self.sigma, g_tau=self.g_tau)
        gp_movesC.rename_prefix(preliminary_tree)
        cdef function_treeC.Node right_node = deepcopy(preliminary_tree.root)
        cdef function_treeC.Node root_node = function_treeC.Node(function=None,
                                            left=left_node, right=right_node, symbol=op )
        cdef function_treeC.Tree tree = function_treeC.Tree(depth=root_node.node_depth, functions=self.funcs,
                                    x=self.domain, node = root_node,
                                    Npars=len(root_node.function.params), value=1.0, T=self.T,
                                    sigma=self.sigma, g_tau=self.g_tau)
        gp_movesC.refresh_params(tree)
        #print(tree.name_values)
        del preliminary_tree
        return tree
    
    cpdef _burn_in_stage2_parallel(self, cnp.ndarray[cnp.float64_t, ndim=1] y, list up2depth1_candidates,
                            dict tree_cache, bint replace_all=True, bint optimize_params=True):
        """
        MPI-parallel version of _burn_in_stage2.
        """
        # Phase 2.10 (final): this rank-parallel variant reassembles its
        # candidates from DESCRIPTION STRINGS (from_string + _expr_to_tree
        # below), which cannot preserve opaque atomic leaves — an
        # instantiated atom description is indistinguishable from an
        # ordinary subexpression. Guard loudly instead of decomposing
        # silently; the serial (use_mpi=0) and spawn (use_spawn=True)
        # burn-ins both support atoms.
        if self.atoms:
            raise NotImplementedError(
                "_burn_in_stage2_parallel rebuilds candidates from "
                "description strings and cannot carry opaque atomic "
                "primitives; run the burn-in serially (use_mpi=0) or with "
                "use_spawn=True.")
        if real_functionC.n_variables() > 1:
            raise NotImplementedError(
                "_burn_in_stage2_parallel rebuilds candidates from "
                "description strings via _expr_to_tree (d=1-only, plan "
                "§0 D8); run the burn-in serially (use_mpi=0) or with "
                "use_spawn=True.")
        cdef object comm = MPI.COMM_WORLD
        cdef rank = int(comm.Get_rank())
        cdef n_procs = int(comm.Get_size())
        print('number of processes: ', n_procs)
        # Filter depth-1 candidates
        cdef list depth1_candidates = [c for c in up2depth1_candidates if c[2] == 1]
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        
        cdef int n_top = len(depth1_candidates)
        
        # ============================================================
        # BUILD ALL WORK ITEMS (list of tasks)
        # ============================================================
        cdef list work_items = []
        
        # Type 1: depth-1 op primitive (both directions)
        for i in range(n_top):
            for prim_str in all_primitives:
                for op in self.operations:
                    work_items.append(('d1_prim', i, prim_str, op))
        
        # Type 2: depth-1 op depth-1
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    work_items.append(('d1_d1', i, j, op))
        
        cdef int total_work = len(work_items)
        
        # ============================================================
        # DIVIDE WORK AMONG RANKS
        # ============================================================
        # Each rank gets a slice of work_items
        cdef int chunk_size = (total_work + n_procs - 1) // n_procs
        cdef int start_idx = rank * chunk_size
        cdef int end_idx = min(start_idx + chunk_size, total_work)
        cdef object my_work = work_items[start_idx:end_idx]
        
        if rank == 0:
            print(f"Stage 2: {n_top} depth-1 trees, {total_work} total combinations")
            print(f"  Distributed across {n_procs} MPI processes")
        
        # ============================================================
        # LOCAL PROGRESS BAR (only rank 0 shows it)
        # ============================================================
        cdef list local_candidates = []
        cdef dict local_cache = {}
        
        # Only rank 0 shows progress bar
        if rank == 0:
            pbar = tqdm(total=total_work, desc="Stage 2 (MPI)", unit=" combos")
        cdef function_treeC.Node prim_node
        cdef function_treeC.Tree tree, d1_tree, d1_tree_2
        #cdef int local_count, all_counts

        for idx, item in enumerate(my_work):
            if item[0] == 'd1_prim':
                _, i, prim_str, op = item
                prim_node = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(
                        prim_str, self.domain, compute_grad=False))
                d1_tree = depth1_candidates[i][0]
                
                # Left = depth-1, Right = primitive
                tree = self._combine_depth1(d1_tree.root, prim_node, op)
                self._process_candidate(tree, y, optimize_params, local_cache, local_candidates)
                
                # Right = depth-1, Left = primitive
                tree = self._combine_depth1(prim_node, d1_tree.root, op)
                self._process_candidate(tree, y, optimize_params, local_cache, local_candidates)
                
            elif item[0] == 'd1_d1':
                _, i, j, op = item
                d1_tree = depth1_candidates[i][0]
                d1_tree_2 = depth1_candidates[j][0]
                tree = self._combine_depth1(d1_tree.root, d1_tree_2.root, op)
                self._process_candidate(tree, y, optimize_params, local_cache, local_candidates)
            
            # ============================================================
            # UPDATE PROGRESS BAR (aggregate from all ranks)
            # ============================================================
            # Every N iterations, sync progress counts
            
            if (idx + 1) % 100 == 0 or idx == len(my_work) - 1:
                # Gather counts from all ranks to rank 0
                local_count = idx + 1
                all_counts = comm.gather(local_count, root=0)
                
                if rank == 0:
                    total_done = sum(all_counts)
                    pbar.n = total_done
                    pbar.refresh()
        
        if rank == 0:
            pbar.close()
        
        # ============================================================
        # GATHER ALL CANDIDATES TO RANK 0
        # ============================================================
        #cdef object all_candidates_list = comm.gather(local_candidates, root=0)
        cdef list all_cache_list = comm.gather(local_cache, root=0)
        cdef dict all_cache_dict = comm.gather(local_cache, root=0)[0]
        
        if rank==0:
            print(len(all_cache_dict))
            tree_cache.update(all_cache_dict)
        
        #cdef list all_candidates_list = [function_treeC._expr_to_tree(ex, self.domain, self.funcs) for ex in all_cache_list]
        if rank == 0:
            all_cache_list_of_tuples = [(k, v) for k, v in all_cache_dict.items()]
            #all_cache_list_of_tuples.sort(key=_bis2_parallel_helper, reverse=True)
            
            # Flatten the list of lists
            candidates = []
            for cache_tuple in all_cache_list_of_tuples:
                expr_str, post = cache_tuple[0], cache_tuple[1]
                
                tree = function_treeC._expr_to_tree(
                    real_functionC.RealFunctionC.from_string(
                        expr_str, self.domain, compute_grad=False).get_expr(), 
                    self.domain, self.funcs)
                candidates.append((tree, post))
            # Merge caches
            ''' 
            for cache_dict in all_cache_list:
            tree_cache.update(cache_dict)
            
            # Add depth-1 candidates
            for c in up2depth1_candidates:
                candidates.append(c)
            '''
            
            # Sort and assign to grove
            candidates.sort(key=_bis2_parallel_helper, reverse=True)
            
            # ... rest of the assignment logic (same as original)
            if replace_all:
                self.trees = [c[0] for c in candidates]
                self.root_posteriors = np.array([c[1] for c in candidates])
            
            print(f"Burn-in complete. Best: {self.root_posteriors[0]:.4f}")
            print(self.trees[0])
            print(self.trees[0].get_param_dicts()[1])
        
        # Broadcast final trees to all ranks if needed
        #self.trees = comm.bcast(self.trees if rank == 0 else None, root=0)
        #self.root_posteriors = comm.bcast(self.root_posteriors if rank == 0 else None, root=0)

       
    cdef _burn_in_stage2_parallel_spawn(self, 
                                    cnp.ndarray[cnp.float64_t, ndim=1] y,
                                    list up2depth1_candidates,
                                    dict tree_cache,
                                    bint replace_all=True,
                                    bint optimize_params=True,
                                    int n_workers=4):
        """
        Spawn MPI workers for parallel burn-in, then collect results.
        
        This function runs in a SINGLE process. It spawns workers,
        distributes work, collects results, and workers exit.
        No need to run with mpirun/mpiexec.
        
        Parameters
        ----------
        n_workers : int
            Number of worker processes to spawn (default: 4)
        """
        import subprocess
        from mpi4py import MPI
        from operator import itemgetter
        
        # =========================================================
        # PREPARE DATA
        # =========================================================
        cdef list depth1_candidates = [c for c in up2depth1_candidates if c[2] == 1]
        cdef list all_primitives
        cdef dict atom_specs
        all_primitives, atom_specs = self._burn_in_primitives()
        cdef int n_top = len(depth1_candidates)

        if n_top == 0:
            print("No depth-1 candidates to combine")
            return

        # Serialize depth-1 trees for workers.
        # Phase 2.10 (final): through the Stage 3b dual-payload wire — for
        # atom-free trees _tree_to_wire ships exactly the historical
        # (root expr, name_values, post) tuple; atom-bearing trees ship the
        # ('skel3b', skeleton, description) payload a raw expr cannot carry.
        cdef list  depth1_data = []
        cdef function_treeC.Tree tree
        cdef object post
        cdef int depth, i
        cdef object expr
        cdef int sum_of_misses = 0
        cdef dict names_params
        cdef str d1_desc
        for tree, post, depth in depth1_candidates:

            depth1_data.append(_tree_to_wire(tree, <double> post))
            d1_desc = tree.tree_get_description()
            if d1_desc not in tree_cache:
                print(f"Adding to cache: {d1_desc} with post {post}. Bug in line 917 if this is missing.")
                tree_cache[d1_desc] = post
        # Build ALL work items
        cdef list work_items = []
        cdef str prim_str, op
        for i in range(n_top):
            for prim_str in all_primitives:
                for op in self.operations:
                    work_items.append(('d1_prim', i, prim_str, op))
        
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    work_items.append(('d1_d1', i, j, op))
        
        cdef int total_work = len(work_items)
        print(f"Stage 2: {n_top} depth-1 trees × {len(all_primitives)} primitives × {len(self.operations)} ops")
        print(f"  Total combinations: {total_work}")
        print(f"  Spawning {n_workers} worker processes...")
        
        # =========================================================
        # SPAWN WORKERS
        # =========================================================
        # Path to worker script (same directory as this module)
        #cdef object worker_script = os.path.join(os.path.dirname(__file__), 'mpi_worker.py')
        
        # Spawn worker processes
        # MPI.COMM_SELF is the communicator for just this process
        intercomm = MPI.COMM_SELF.Spawn(
            sys.executable,  # Python interpreter
            args=["-c", "from mpi_worker import worker_main; worker_main()"],
            maxprocs=n_workers
        )
        
        # =========================================================
        # BROADCAST CONFIG TO WORKERS
        # =========================================================
        config = {
            'domain': self.domain,
            # Flexible domain: spawned workers are fresh processes — they
            # install this registry FIRST, before any wire rebuild
            # (plan §2.6). Legacy runs ship ('x',).
            'variables': real_functionC.get_variables(),
            'funcs': self.funcs,
            'operations': self.operations,
            'y': np.asarray(y),
            'lambda_reg': self.lambda_reg,
            'method_str': self.method_str,
            'use_gradients': self.use_gradients,
            'depth1_data': depth1_data,
            'all_primitives': all_primitives,
            'T': self.T,
            'tree_cache': tree_cache,
            # Phase 2.10 (final): templates for the workers — resolves the
            # '@atom:' tokens in all_primitives and rebuilds 'skel3b'
            # payloads in depth1_data. Flexible-domain fix (adversarial
            # review): ship (name, template) STRING PAIRS, not AtomSpec
            # objects — an AtomSpec unpickles via __reduce__ -> __init__
            # DURING the bcast, i.e. BEFORE the worker installs the
            # shipped registry, and template classification is
            # registry-dependent. The worker rebuilds AtomSpec AFTER
            # set_variables; for legacy ('x',) registries the rebuilt
            # objects are identical to what unpickling produced.
            # Phase PP: pinned specs ship (base_name, template, pinned)
            # TRIPLES; unpinned keep the historical pair byte-identically.
            'atoms': ([((_a.base_name, _a.template, dict(_a.pinned))
                        if _a.pinned else (_a.name, _a.template))
                       for _a in self.atoms]
                      if self.atoms else None),
        }

        intercomm.bcast(config, root=MPI.ROOT)
        
        # =========================================================
        # SCATTER WORK ITEMS TO WORKERS
        # =========================================================
        # Divide work items into n_workers chunks
        cdef int n_participants = n_workers + 1
        cdef int chunk_size = (total_work + n_participants - 1) // n_participants
        cdef list work_chunks = []
        
        for w in range(1, n_participants):
            start = w * chunk_size
            end = min(start + chunk_size, total_work)
            work_chunks.append(work_items[start:end])
        
        cdef list my_work_chunk = work_items[0:chunk_size]
        # Scatter chunks to workers
        intercomm.scatter(work_chunks, root=MPI.ROOT)
        
        print(f"  Work distributed. Waiting for workers...")
        
        cdef object pbar=tqdm(total=total_work, desc="Stage 2 (Spawned)", unit=" combos", 
                dynamic_ncols=True)
        
        cdef dict progress_received={w: 0 for w in range(n_workers)}
        cdef object last_update_time = time.time()
        cdef double timeout = 5.0  # seconds - if no updates for 5s, assume workers done
        
        cdef str itm_type
        cdef int indx
        cdef function_treeC.Node prim_node
        cdef function_treeC.Tree d1_tree, d1_tree_2
        cdef list candidates=[]
        
        for indx, itm in enumerate(my_work_chunk):
            
            itm_type = itm[0]
            if itm_type == 'd1_prim':

                _, i, prim_str, op = itm
                # Phase 2.10 (final): token resolver ('a' band; the
                # combiner renames right operands to 'b').
                prim_node = _burn_in_primitive_node(prim_str, self.domain,
                                                    atom_specs, 'a', 1)

                d1_tree = depth1_candidates[i][0]
                if d1_tree is None:
                    continue
                
                tree = self._combine_depth1(d1_tree.root, prim_node, op)
                self._process_candidate(tree, y,
                        True, tree_cache, candidates)
                
                if op not in ['+', '*'] and not (op=='o' and prim_node.bis_param):  # if not commutative, also do primitive op depth-1
                    tree = self._combine_depth1(prim_node, d1_tree.root, op)
                    
                    self._process_candidate(tree, y,
                        True, tree_cache, candidates)
                    
            
            elif itm_type == 'd1_d1':
                _, i, j, op = itm
                #print('prior to assignment: ', id(depth1_candidates[i][0].name_values.keys()), id(depth1_candidates[j][0].name_values.keys()))
                d1_tree = depth1_candidates[i][0]
                d1_tree_2 = depth1_candidates[j][0]

                if d1_tree is None or d1_tree_2 is None:
                    continue
                #print('prior to call: ', id(d1_tree.name_values.keys()), id(d1_tree_2.name_values.keys()))
                tree = self._combine_depth1_trees(d1_tree, d1_tree_2, op)
                #print('combined tree: ', tree.name_values)
                self._process_candidate(tree, y, True, tree_cache, candidates)

            if indx % 50 ==0:
                pbar.update(50)
                pbar.refresh()
                
                for source in range(n_workers):
                    status = MPI.Status()
                    if intercomm.Iprobe(source=source, tag=99, status = status):
                        count=intercomm.recv(source=source, tag=99, status=status)
                        delta = count - progress_received[source]
                        progress_received[source] = count
                        pbar.update(delta)
                        pbar.refresh()
                        sys.stdout.flush()
                        last_update_time = time.time()
                        got_update = True
        
        while True:
            got_update = False
            
            # Non-blocking check for incoming messages
            for source in range(n_workers):
                status = MPI.Status()
                if intercomm.Iprobe(source=source, tag=99, status=status):
                    # Progress update from worker
                    count = intercomm.recv(source=source, tag=99)
                    delta = count - progress_received[source]
                    progress_received[source] = count
                    pbar.update(delta)
                    pbar.refresh()
                    sys.stdout.flush()
                    last_update_time = time.time()
                    got_update = True
            
            # Check for timeout - if no updates for X seconds, workers likely done
            elapsed_since_update = time.time() - last_update_time
            
            if elapsed_since_update > timeout and sum(progress_received.values()) > 0:
                # No updates for a while and we've received some progress = done
                break
            
            time.sleep(0.05)  # Short sleep to avoid busy-waiting
        
        pbar.close()
        sys.stdout.flush()  

       

        cdef list all_return_list
        #print('Gathering trees')
        try:
            all_return_list = intercomm.gather(None, root=MPI.ROOT)
            print(f'Spawner: gather succeeded, got {len(all_return_list)} result lists')
        except Exception as e:
            print(f'Spawner: gather failed with error: {e}')
            import traceback
            traceback.print_exc()
            all_return_list = [[] for _ in range(n_workers)]
        sys.stdout.flush()
        
        # =========================================================
        # DISCONNECT WORKERS
        # =========================================================
        #intercomm.Disconnect()
        print(f"  Workers finished and disconnected.")
        
        # =========================================================
        # PROCESS RESULTS
        # =========================================================
        #cdef list candidates = []
        cdef list worker_results
        cdef function_treeC.Node param
        # Flatten results from all workers
        #cdef int jj=0
        #cdef bint ok
        cdef object pbar1=tqdm(total=len(all_return_list), desc="Growing Grove", unit=" trees", 
                dynamic_ncols=True)
        cdef str skel_desc
        for worker_results in all_return_list:
            for expr, names_params, post in worker_results:
                # Phase 2.10 (final): atom-bearing worker results arrive as
                # ('skel3b', skeleton, description) payloads — rebuild them
                # through the Stage 3b wire (fidelity-checked inside; a
                # failed rebuild is simply dropped, mirroring the legacy
                # mismatch `continue` below).
                if (isinstance(expr, tuple) and len(expr) == 3
                        and expr[0] == 'skel3b'):
                    tree = _tree_from_wire(expr, names_params, self.domain,
                                           self.funcs, self.sigma,
                                           self.g_tau, atom_specs)
                    if tree is None:
                        print('Warning: skeleton wire rebuild failed; '
                              'dropping candidate')
                        continue
                    skel_desc = tree.tree_get_description()
                    if skel_desc not in tree_cache:
                        candidates.append((tree, post))
                        tree_cache[skel_desc] = post
                        pbar1.update(1)
                        pbar1.refresh()
                    continue
                # Rebuild tree from expression
                #print(jj, ' expr is: ', str(expr))
                tree = function_treeC._expr_to_tree(expr, self.domain, self.funcs)
                
                #print(tree)
                #print(tree.name_values)
                descr = tree.tree_get_description()
                print(descr, str(expr))
                if str(expr) not in tree_cache:
                    
                    if tree is not None and names_params and len(names_params) > 0:
                        for param in tree.params:
                            #print(param.function.expr.is_Number)
                            if not param.function.expr.is_Number and param.get_description() in names_params.keys():
                                param.set_value(names_params[param.get_description()])
                        
                        #ok = tree.replace_constants_with_pars(names_params)
                        descr = str(tree.root.function.expr)
                        if descr != str(expr):
                            print('Warning: expression mismatch after rebuilding tree')
                            print('Original expr: ', str(expr))
                            print('Rebuilt expr: ', tree.tree_get_description())
                            print(tree)
                            continue
                        candidates.append((tree, post))
                        tree_cache[str(expr)] = post
                        pbar1.update(1)
                        pbar1.refresh()
                    #jj += 1
        pbar.close()
        print(f"  Received {len(candidates)} valid candidates from workers")
        
        # Add depth-1 candidates
        for c in up2depth1_candidates:
            candidates.append((c[0], c[1]))
        
        # Sort by posterior (best first)
        candidates.sort(key=itemgetter(1), reverse=True)
        
        # =========================================================
        # ASSIGN TO GROVE
        # =========================================================
        if replace_all:
            self.size = len(candidates)
            self.trees = [c[0] for c in candidates]
            self.root_posteriors = np.array([c[1] for c in candidates], dtype=np.float64)
        else:
            for i in range(min(self.size, len(candidates))):
                if np.isnan(self.root_posteriors[i]) or candidates[i][1] > self.root_posteriors[i]:
                    self.trees[i] = candidates[i][0]
                    self.root_posteriors[i] = candidates[i][1]
        
        print(f"Burn-in complete. Best posterior: {self.root_posteriors[0]:.4f}")
        print(f"  Best expression: {self.trees[0].tree_get_description()} with params {self.trees[0].get_param_dicts()[1]}")



    cdef _process_candidate(self, function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                        bint optimize_params, dict local_cache, list local_candidates):
        """Helper to process a single candidate tree."""
        if tree is None:
            return
        cdef str expression = tree.tree_get_description()
        if expression in local_cache:
            return
        if not tree.is_valid(self.domain):
            return
        
        if optimize_params and len(tree.params) > 0:
            tree.update_parameters(self.domain, y,
                                self.method_str, self.use_gradients)
        
        #cdef dict names_values = tree.name_values
        tree.replace_constants_with_pars()
        #cdef double post = tree.get_superficial_posterior(y, self.lambda_reg)
        cdef double post = tree.root._log_marginal_bic(y)
        local_cache[expression] = post
        
        if np.isfinite(post):
            local_candidates.append((tree, post))


# ---------------------------------------------------------------------------
# Phase 2.12 (MPI sampling): tree wire format
# ---------------------------------------------------------------------------
# The wire is a 3-tuple (payload, name_values, post) — trees are cdef
# classes (not picklable); payloads + dicts of floats are. Two payload
# forms (Phase 2.10 Stage 3b):
#   * raw sympy expr : the HISTORICAL atom-free format (rebuild via
#     function_treeC._expr_to_tree) — kept for the spawn BURN-IN wire,
#     bit-identical;
#   * ('skel3b', skeleton, description) — a structural skeleton
#     (function_treeC.tree_to_skel) that preserves the exact tree shape
#     and each opaque leaf's (atom_symbol, internal names). A raw expr
#     CANNOT carry an opaque leaf: a0*sin(a1*x) is a sympy Mul and
#     _expr_to_tree would decompose it into an ordinary subtree.
# Wire-fidelity fix (2026-07-24): the SAMPLING wire (run_mpi_spawn +
# mpi_worker._sampler_*) now ships EVERY tree as 'skel3b', via
# force_skel=True. A raw expr canonicalizes on construction (x*x*x ->
# x**3, x + x -> 2*x); when the sampling vocabulary cannot re-express
# the folded form (the depth-3 / Feynman suites exclude '**'), the
# round-trip gate below rejected the rebuild and the master silently
# kept its stale initial trees (population starvation — see
# PROMPT_DEV_WIRE_FIDELITY.md). The skeleton round-trip is shape-exact,
# never invokes sympy, and its gate compares against the SHIPPED
# description. Burn-in producers keep the legacy dual format
# (force_skel=False default) — certified bit-identical.
# Values travel in name_values for both forms. A rebuild whose description
# does not round-trip is reported as None and the caller must treat the
# exchange as REJECTED (self-loop; target-preserving). Both cpdef since
# Stage 3b: mpi_worker delegates here (ONE wire implementation) and the
# unit tests drive them directly — cross-module calls POSITIONAL ONLY.
cpdef tuple _tree_to_wire(function_treeC.Tree tree, double post,
                          bint force_skel=False):
    cdef function_treeC.Node lf
    cdef bint has_atomic = False
    if not force_skel:
        for lf in tree.leaves:
            if lf.is_atomic():
                has_atomic = True
                break
    # Flexible domain (plan §0 D8): multivariate trees ALWAYS ship as the
    # exact-shape skeleton — the raw-expr payload is rebuilt by
    # _expr_to_tree, whose function-application branch composes (d=1-only).
    # Atom-free d=1 trees keep the HISTORICAL raw-expr payload bytes; the
    # SAMPLING wire forces the skeleton at any d (wire-fidelity fix).
    # Phase FX: a flex-bearing tree ALWAYS ships as skeleton — its expr is
    # an unexpanded sympy Function head that _expr_to_tree cannot rebuild.
    # tree_has_flex is a plain walk returning False on legacy trees (no
    # RNG, no payload change).
    if (force_skel or has_atomic or real_functionC.n_variables() > 1
            or function_treeC.tree_has_flex(tree.root)):
        return (('skel3b',
                 function_treeC.tree_to_skel(tree.root),
                 tree.tree_get_description()),
                dict(tree.name_values) if tree.name_values else {},
                float(post))
    return (tree.root.function.expr,
            dict(tree.name_values) if tree.name_values else {},
            float(post))


cpdef function_treeC.Tree _tree_from_wire(object expr, dict name_values,
                                          object domain, list funcs,
                                          object sigma, double g_tau,
                                          object atom_specs=None):
    cdef function_treeC.Tree tree
    cdef function_treeC.Node param
    cdef function_treeC.Node root
    cdef str desc
    cdef bint is_skel = (isinstance(expr, tuple) and len(expr) == 3
                         and expr[0] == 'skel3b')
    if is_skel:
        # Stage 3b skeleton payload: exact-shape rebuild, opaque leaves
        # reconstructed as units from their templates. `atom_specs` maps
        # atom_symbol -> AtomSpec; a missing template (or any rebuild
        # error) is a REJECT (None), the same convention as below.
        try:
            root = function_treeC.node_from_skel(
                expr[1], np.ascontiguousarray(domain, dtype=np.float64),
                <dict> atom_specs if atom_specs is not None else None)
            tree = function_treeC.Tree(0, list(funcs), domain, node=root)
        except Exception:
            return None
        if name_values:
            # Plain params AND opaque theta — same installer as the BIC
            # cache hits.
            _install_cached_theta(tree, name_values)
        tree.set_sigma(sigma)
        tree.set_g_tau(g_tau)
        tree.tree_update(tree.get_root())
        if tree.tree_get_description() != <str> expr[2]:
            return None
        return tree
    try:
        tree = function_treeC._expr_to_tree(expr, domain, funcs)
    except Exception:
        return None
    if tree is None:
        return None
    if name_values:
        for param in tree.params:
            desc = param.get_fun().get_description()
            if desc in name_values:
                param.set_value(<double> name_values[desc])
    tree.set_sigma(sigma)
    tree.set_g_tau(g_tau)
    tree.tree_update(tree.get_root())
    # Round-trip fidelity check (same guard the burn-in spawn path uses).
    if tree.tree_get_description() != str(expr):
        return None
    return tree


cdef class Forest:
    """
    Evolutionary Monte Carlo population of Groves at different temperatures
    (Phase 2.9). Operators:
      - mutation  : within-grove MH        (run_all -> Grove.run -> _step)
      - exchange  : cross-temperature swap (attempt_swaps)
      - crossover : trans-grove subtree crossover (attempt_crossovers)

    Shared Pólya prior (R1)
    -----------------------
    ONE PolyaTree is built once and handed to EVERY grove, so the tempered
    population is a canonical ladder of a single self-consistent posterior.
    It is NEVER frozen: it keeps evolving through `update_path`'s diminishing
    adaptation (gamma_t = 1/(1+t)), reinforced by every grove's accepted
    WITHIN-grove moves. The trans-grove swap and trans-grove crossover are
    prior-preserving relabelings and do NOT reinforce.

    Acceptance (R1)
    ---------------
    With a shared Theta the general per-grove-prior MH ratio reduces to the
    clean tempered form using U(x) = post(x) + log pi_Polya(x | Theta):
      swap      : log_alpha = (1/Ta - 1/Tb) * (U_b - U_a)
      crossover : log_alpha = (U_a' - U_a)/Ta + (U_b' - U_b)/Tb + (lqr - lqf)

    Pair selection (R3)
    -------------------
    ONE random consecutive temperature pair per call, for both swap and
    crossover (random-scan, reversible per step).
    """

    def __init__(self,
                 cnp.ndarray x,
                 cnp.ndarray[cnp.float64_t, ndim=1] y,
                 cnp.ndarray[cnp.float64_t, ndim=1] temperatures,
                 int size_per_grove,
                 int depth,
                 list funcs,
                 object sigma=1.0,
                 object node_lists=None,
                 int Npars=0,
                 object init_value=1.0,
                 list operations=None,
                 double base_temp=1.0,
                 double lambda_reg=0.0,
                 bint use_gradients=True,
                 bint burn_in=False,
                 int n_procs=0,
                 object seed=None,
                 object g_tau=None,
                 object polya=None,
                 double alpha_0=1.0,
                 bint adapt_moves=False,
                 list atoms=None,
                 bint freeze_vocab=False,
                 object structural_prior=None,
                 bint structural_prior_cache=True,
                 object structural_prior_spec=None,
                 object variables=None,
                 object flex_atoms=None,
                 object datasets=None,
                 str md_combine='sum',
                 double md_softmin_beta=8.0,
                 object md_baselines=None):
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.groves = []
        self.n_procs = n_procs
        self.use_mpi = self.n_procs >= 2
        # Phase 4.4 v2: mutually exclusive tilt inputs (fail fast, with
        # a Forest-labeled message; each Grove would also catch it).
        # Checked BEFORE the registry install below so a bad call cannot
        # mutate the process-global variable registry.
        if structural_prior_spec is not None and structural_prior is not None:
            raise TypeError("Forest: pass structural_prior OR "
                            "structural_prior_spec, not both")
        # --- Flexible domain (flexible_domain_plan.md §3): resolve the
        # variable registry BEFORE the Pólya/atoms handling below (both are
        # registry-dependent) — same rules as Grove.__init__ (which then
        # re-installs the identical tuple, idempotently).
        cdef tuple _vars
        cdef bint _flex
        if isinstance(variables, str):
            raise ValueError(
                "Forest: variables must be a sequence of names, not a "
                "string (got %r — a string would split into characters)"
                % (variables,))
        if x.ndim == 1:
            _vars = ('x',) if variables is None else tuple(variables)
            if len(_vars) != 1:
                raise ValueError(
                    "Forest: a 1-D x carries exactly one variable; got "
                    "variables=%r" % (variables,))
        elif x.ndim == 2:
            _vars = (tuple('x%d' % _vi for _vi in range(x.shape[0]))
                     if variables is None else tuple(variables))
            if len(_vars) != x.shape[0]:
                raise ValueError(
                    "Forest: x has shape (d, n) = %r — coordinate ROWS, "
                    "points along the last axis — but %d variable names "
                    "were given. Pass X.T if your data is (n, d)."
                    % (tuple([x.shape[0], x.shape[1]]), len(_vars)))
            if len(_vars) == 1:
                raise ValueError(
                    "Forest: a single-variable domain must use the 1-D "
                    "(n,) layout — pass x.ravel(). (A (1, n) array would "
                    "ride the d=1 code paths as a 2-D object; found by "
                    "the adversarial review.)")
        else:
            raise ValueError(
                "Forest: x must be (n,) or (d, n), got ndim=%d" % x.ndim)
        _flex = (x.ndim == 2) or (variables is not None)
        real_functionC.set_variables(_vars)
        if operations is None:
            if len(_vars) > 1:
                raise ValueError(
                    "Forest: pass `operations` explicitly for a "
                    "multivariate domain — the legacy default includes "
                    "compose 'o', which is d=1-only "
                    "(flexible_domain_plan.md §0 D6 / §6 Q2)")
            operations = ["+", "-", "*", "/", "**", "o"]
        elif len(_vars) > 1 and "o" in operations:
            raise ValueError(
                "Forest: compose 'o' is not defined for multivariate "
                "domains (flexible_domain_plan.md §8.1 — standing item)")
        cdef int G = self.temperatures.shape[0]
        cdef int g
        cdef int cap
        cdef list param_names
        # --- Shared Pólya prior (R1) ---------------------------------------
        # Build ONCE and hand the SAME instance to every grove. When the
        # caller supplies `polya=`, respect it (they own vocabulary
        # registration). Otherwise pre-register the parameter symbols that
        # build_tree / crossover may emit (a0.., b0..), else extract_path
        # silently drops those leaves from the prior.
        if polya is None:
            if freeze_vocab:
                # Phase 2.10 Stage 3c — the FULL vocabulary freeze (settles
                # Phase 2.3 / DESIGN §9 item 4). The vocabulary is CLOSED
                # at construction: operations + function leaves + the ONE
                # canonical psi_const parameter cell (unify_params —
                # parameter NAMES are bound labels whose indices new_par
                # ratchets without bound, so no finite name pool covers
                # them) + the atom templates registered below. Nothing
                # unknown can then appear during sampling, so auto_register
                # is OFF: any out-of-vocabulary symbol raises the guarded
                # KeyError instead of silently growing the prior. The prior
                # is one fixed measure from t = 0, and MPI path indices are
                # replica-stable by construction. (DP mode composes via a
                # caller-supplied polya built with remainder_mass > 0 and
                # unify_params=True.)
                polya = polya_treeC.PolyaTree(
                    operations, list(funcs), alpha_0=alpha_0,
                    auto_register=False, strict_ops=True,
                    remainder_mass=0.0, unify_params=True)
            else:
                # Legacy (default): per-name parameter cells + mid-run
                # auto-registration — bit-identical historical behaviour.
                polya = polya_treeC.PolyaTree(
                    operations, list(funcs), alpha_0=alpha_0,
                    auto_register=True, strict_ops=True)
                cap = Npars if Npars > 0 else 8
                param_names = [f"a{k}" for k in range(cap)] + \
                              [f"b{k}" for k in range(cap)]
                polya.add_primitives(param_names)
        self.polya = polya
        # --- Phase 2.10 Stage 3a: opaque-primitive templates ----------------
        # Register each atom's canonical SYMBOL in the shared Polya HERE,
        # once, before any sampling — so the opaque cells exist from t = 0
        # in every grove's view of the one shared prior and never enter via
        # mid-run auto_register (registration invariance, THEORY_2_13.md
        # §5.3; in DP mode this is the carve-from-remainder with polya's
        # default mass — pass a pre-registered `polya=` to control the
        # per-atom G_Psi mass). A caller-supplied polya that already
        # registered the symbols is respected (add_primitive is idempotent
        # on known symbols). NB: add_primitive is cpdef across a cimport
        # boundary — the mass argument MUST stay positional.
        if atoms:
            # Untyped dispatch on purpose: groveC compiles under a different
            # language_level than polya_treeC (which pins level 3), so a
            # C-level call with a typed-str argument trips Cython's
            # str/unicode coercion rule. A Python-level call is resolved at
            # runtime and is identical in behaviour (this runs once, at
            # construction).
            for _spec_obj in atoms:
                if not isinstance(_spec_obj, real_functionC.AtomSpec):
                    raise TypeError(
                        "Forest: atoms entries must be real_functionC."
                        "AtomSpec instances, got %r" % type(_spec_obj))
            if _flex:
                # Flexible domain: re-derive each template under the
                # registry installed above (AtomSpec classification is
                # registry-dependent; specs may pre-date this Forest).
                # Phase PP: from BASE name + pinned (suffix would double).
                atoms = [real_functionC.AtomSpec(_spec_obj.base_name,
                                                 _spec_obj.template,
                                                 _spec_obj.pinned)
                         for _spec_obj in atoms]
            _polya_any = self.polya
            for _spec_obj in atoms:
                _polya_any.add_primitive(_spec_obj.name, None)
            self.atoms = list(atoms)
        else:
            self.atoms = None
        # Phase FX (transparent atoms v1): validate + register each flex
        # template head in the ONE shared Polya, exactly like the opaque
        # atoms above (registration invariance: cells exist from t = 0,
        # never enter via mid-run auto_register). Same untyped dispatch /
        # positional-mass rules. Each Grove re-validates its own copy
        # under the current registry.
        if flex_atoms:
            for _spec_obj in flex_atoms:
                if not isinstance(_spec_obj, real_functionC.FlexAtomSpec):
                    raise TypeError(
                        "Forest: flex_atoms entries must be real_functionC."
                        "FlexAtomSpec instances, got %r" % type(_spec_obj))
            if adapt_moves:
                # Fail HERE, before the shared-Polya registration below,
                # so a rejected call cannot leave stray cells behind
                # (each Grove would also raise, but only after mutation).
                raise ValueError(
                    "Forest: adapt_moves=True with flex_atoms is not "
                    "supported in v1 (the adaptive logit map covers the 8 "
                    "legacy moves only)")
            if burn_in:
                raise ValueError(
                    "Forest: burn_in=True with flex_atoms is not "
                    "supported in v1 (burn-in constant folding is not "
                    "flex-aware)")
            flex_atoms = [real_functionC.FlexAtomSpec(
                _spec_obj.name, _spec_obj.template, _spec_obj.slot_symbol)
                for _spec_obj in flex_atoms]
            _polya_any = self.polya
            for _spec_obj in flex_atoms:
                _polya_any.add_primitive(_spec_obj.name, None)
            self.flex_atoms = list(flex_atoms)
        else:
            self.flex_atoms = None
        # --- Forest-level RNGs ---------------------------------------------
        # Fixes the previously-missing _py_rng (the old swap referenced it and
        # would AttributeError) and drives the R3 random consecutive-pair
        # selection. Independent of the per-grove RNGs, all reproducibly
        # derived from a single top-level seed via SeedSequence.
        seed_seq = np.random.SeedSequence(seed)
        child_seeds = seed_seq.spawn(G + 1)
        self._py_rng = random.Random(int(child_seeds[G].generate_state(1)[0]))
        self._np_rng = np.random.default_rng(child_seeds[G])
        for g in range(G):
            nodes_g = None
            if node_lists is not None:
                nodes_g = node_lists[g]
            self.groves.append(
                Grove(x, y, float(self.temperatures[g]), size_per_grove,
                      depth, funcs, sigma=sigma, node_list=nodes_g,
                      Npars=Npars, init_value=init_value,
                      operations=operations, base_temp=base_temp,
                      lambda_reg=lambda_reg, use_gradients=use_gradients,
                      burn_in=burn_in, use_mpi=n_procs,
                      polya=self.polya,
                      seed=int(child_seeds[g].generate_state(1)[0]),
                      g_tau=g_tau, adapt_moves=adapt_moves,
                      atoms=self.atoms,
                      structural_prior=structural_prior,
                      structural_prior_cache=structural_prior_cache,
                      structural_prior_spec=structural_prior_spec,
                      # Legacy stays on the variables=None Grove path
                      # (bit-identical: no funcs validation, no atom
                      # re-derivation); flexible mode passes the resolved
                      # tuple explicitly.
                      variables=(_vars if _flex else None),
                      flex_atoms=self.flex_atoms,
                      # Phase MD: every Grove validates its own copy; the
                      # rungs share one datasets list (read-only in the
                      # kernel) so the tempered ladder scores ONE joint
                      # posterior.
                      datasets=datasets,
                      md_combine=md_combine,
                      md_softmin_beta=md_softmin_beta,
                      md_baselines=md_baselines)
            )
        # Phase MD: stash for the spawn refusal below (Groves hold the
        # validated copies; Forest only needs the flag).
        self.datasets = datasets
        # Phase 2.12 (MPI sampling): stash the Grove constructor arguments so
        # run_mpi_spawn can ship them to spawned workers, which rebuild
        # structurally identical Groves in their own processes. Purely
        # additive; nothing else reads this.
        # --- Phase 4.1 (PT ladder tuning): per-consecutive-pair monitors ---
        # attempts/accepts for the swap and trans-grove crossover moves,
        # indexed by pair g = (T_g, T_{g+1}). Pure observability: no RNG
        # consumption, no behavioural change; read via ladder_stats().
        self._swap_attempts = np.zeros(max(G - 1, 0), dtype=np.int64)
        self._swap_accepts = np.zeros(max(G - 1, 0), dtype=np.int64)
        self._xover_attempts = np.zeros(max(G - 1, 0), dtype=np.int64)
        self._xover_accepts = np.zeros(max(G - 1, 0), dtype=np.int64)
        self._ctor = {
            'x': np.asarray(x, dtype=np.float64),
            'size': size_per_grove,
            'depth': depth,
            'funcs': list(funcs),
            'operations': list(operations),
            'sigma': sigma,
            'Npars': Npars,
            'init_value': init_value,
            'base_temp': base_temp,
            'lambda_reg': lambda_reg,
            'use_gradients': use_gradients,
            'g_tau': g_tau,
            'adapt_moves': adapt_moves,
            # Phase 4.4 v2: the declarative tilt spec + cache flag
            # travel to spawned workers (the raw callable never does).
            'structural_prior_spec': structural_prior_spec,
            'structural_prior_cache': structural_prior_cache,
            # Flexible domain: spawned sampler workers are fresh processes
            # — they must install THIS registry before any wire rebuild or
            # Grove construction (plan §2.6).
            'variables': _vars,
        }

    cpdef run_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps_per_grove):
        """Mutation phase: run within-grove MH in each grove independently."""
        cdef Grove gr
        for gr in self.groves:
            gr.run(y, n_steps_per_grove)

    cpdef compute_posteriors_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=False):
        cdef Grove gr
        for gr in self.groves:
            gr.compute_posteriors(y, full=full)

    cdef bint _attempt_swap_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """Exchange move between consecutive groves g_left, g_left+1 at the
        same slot i. U = post + shared-Polya prior; with shared Theta the
        acceptance is the clean (1/Ta - 1/Tb)(U_b - U_a). Does NOT reinforce.
        """
        cdef Grove A = self.groves[g_left]
        cdef Grove B = self.groves[g_left + 1]
        cdef int sizeA = A.size
        cdef int sizeB = B.size
        if sizeA == 0 or sizeB == 0:
            return False
        cdef double Ta = A.T
        cdef double Tb = B.T
        # Both greedy: parallel tempering has no well-defined swap; skip.
        if Ta <= 0.0 or Tb <= 0.0:
            return False
        # Phase 4.1 monitor: a genuine pair proposal is being made.
        self._swap_attempts[g_left] += 1
        cdef int i = self._py_rng.randrange(min(sizeA, sizeB))
        if A.root_posteriors[i] is None or np.isnan(A.root_posteriors[i]):
            A.root_posteriors[i] = A._score_post(A.trees[i], y, False)
        if B.root_posteriors[i] is None or np.isnan(B.root_posteriors[i]):
            B.root_posteriors[i] = B._score_post(B.trees[i], y, False)
        cdef double postA = float(A.root_posteriors[i])
        cdef double postB = float(B.root_posteriors[i])
        cdef double priorA = 0.0
        cdef double priorB = 0.0
        cdef list pathA, pathB
        if self.polya is not None:
            # Phase 2.13(c): extract both paths before scoring either.
            pathA = self.polya.extract_path(A.trees[i].get_root())
            pathB = self.polya.extract_path(B.trees[i].get_root())
            priorA = self.polya.log_prior(pathA)
            priorB = self.polya.log_prior(pathB)
        # Phase 4.4: the structural tilt is part of each chain's target
        # (whole-posterior tempering), so it belongs in the swap U's.
        cdef double Ua = postA + priorA + A._sp(A.trees[i])
        cdef double Ub = postB + priorB + B._sp(B.trees[i])
        cdef double delta = (1.0 / Ta - 1.0 / Tb) * (Ub - Ua)
        cdef double acc_p = _safe_exp(delta)
        if self._py_rng.random() < min(1.0, acc_p):
            tmp_tree = A.trees[i]; A.trees[i] = B.trees[i]; B.trees[i] = tmp_tree
            tmp_post = A.root_posteriors[i]
            A.root_posteriors[i] = B.root_posteriors[i]
            B.root_posteriors[i] = tmp_post
            self._swap_accepts[g_left] += 1
            return True
        return False

    cpdef int attempt_swaps(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """R3: attempt ONE swap on a random consecutive pair. Returns #accepted."""
        cdef int G = len(self.groves)
        if G <= 1:
            return 0
        cdef int g = self._py_rng.randrange(G - 1)
        if self._attempt_swap_pair(g, y):
            return 1
        return 0

    cdef bint _attempt_crossover_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """Trans-grove subtree crossover between consecutive groves g_left,
        g_left+1 (slots i, j independent uniform). Reuses
        gp_movesC.subtree_crossover on deepcopies of both parents: child i
        lands in grove A(@Ta), child j in grove B(@Tb). Acceptance uses
        U = post + shared prior:
            log_alpha = (Ua'-Ua)/Ta + (Ub'-Ub)/Tb + (lqr - lqf).
        Prior-preserving relabeling: does NOT reinforce the Polya.
        """
        cdef Grove A = self.groves[g_left]
        cdef Grove B = self.groves[g_left + 1]
        cdef int sizeA = A.size
        cdef int sizeB = B.size
        if sizeA == 0 or sizeB == 0:
            return False
        cdef double Ta = A.T
        cdef double Tb = B.T
        if Ta <= 0.0 or Tb <= 0.0:
            return False
        # Phase 4.1 monitor: a genuine pair proposal is being made (early
        # child-validity failures below count as rejected attempts).
        self._xover_attempts[g_left] += 1
        cdef int i = self._py_rng.randrange(sizeA)
        cdef int j = self._py_rng.randrange(sizeB)
        if A.root_posteriors[i] is None or np.isnan(A.root_posteriors[i]):
            A.root_posteriors[i] = A._score_post(A.trees[i], y, False)
        if B.root_posteriors[j] is None or np.isnan(B.root_posteriors[j]):
            B.root_posteriors[j] = B._score_post(B.trees[j], y, False)
        cdef double postA_old = float(A.root_posteriors[i])
        cdef double postB_old = float(B.root_posteriors[j])
        # Phase 2.13(c): EXTRACT all four construction paths before scoring
        # ANY prior. The children can introduce fresh parameter labels
        # (auto-registration grows every Dirichlet), and old priors scored
        # against the pre-growth vocabulary are not comparable to new priors
        # scored after it.
        cdef double priorA_old = 0.0
        cdef double priorB_old = 0.0
        cdef double priorA_new = 0.0
        cdef double priorB_new = 0.0
        cdef list pathA_old = None
        cdef list pathB_old = None
        cdef list pathA_new, pathB_new
        if self.polya is not None:
            pathA_old = self.polya.extract_path(A.trees[i].get_root())
            pathB_old = self.polya.extract_path(B.trees[j].get_root())
        # Install the Forest RNG for the crossover move (mirrors Grove._step,
        # which re-installs before each step) so the population trajectory is
        # reproducible from the Forest seed.
        gp_movesC.set_rng(self._py_rng, self._np_rng)
        function_treeC.set_rng(self._py_rng, self._np_rng)
        cdef function_treeC.Tree childA = deepcopy(A.trees[i])
        cdef function_treeC.Tree childB = deepcopy(B.trees[j])
        cdef double lqf, lqr
        lqf, lqr = gp_movesC.subtree_crossover(childA, childB)
        if not np.isfinite(lqf) or not np.isfinite(lqr):
            return False
        cdef double postA_new = A._score_post(childA, y, True)
        cdef double postB_new = B._score_post(childB, y, True)
        if postA_new <= -1.0e299 or postB_new <= -1.0e299:
            return False
        if self.polya is not None:
            pathA_new = self.polya.extract_path(childA.get_root())
            pathB_new = self.polya.extract_path(childB.get_root())
            # All extractions done: score everything on the same vocabulary.
            priorA_old = self.polya.log_prior(pathA_old)
            priorB_old = self.polya.log_prior(pathB_old)
            priorA_new = self.polya.log_prior(pathA_new)
            priorB_new = self.polya.log_prior(pathB_new)
        # Phase 4.4: structural-prior tilt in each U (0.0 when off).
        cdef double Ua_old = postA_old + priorA_old + A._sp(A.trees[i])
        cdef double Ub_old = postB_old + priorB_old + B._sp(B.trees[j])
        cdef double Ua_new = postA_new + priorA_new + A._sp(childA)
        cdef double Ub_new = postB_new + priorB_new + B._sp(childB)
        cdef double log_alpha = ((Ua_new - Ua_old) / Ta
                                 + (Ub_new - Ub_old) / Tb
                                 + (lqr - lqf))
        cdef bint accept
        cdef double u
        if log_alpha >= 0.0:
            accept = True
        else:
            u = self._py_rng.random()
            accept = (u <= 0.0) or (np.log(u) < log_alpha)
        if not accept:
            return False
        A.trees[i] = childA
        B.trees[j] = childB
        A.root_posteriors[i] = postA_new
        B.root_posteriors[j] = postB_new
        self._xover_accepts[g_left] += 1
        return True

    cpdef int attempt_crossovers(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """R3: attempt ONE trans-grove crossover on a random consecutive pair.
        Returns #accepted (0 or 1)."""
        cdef int G = len(self.groves)
        if G <= 1:
            return 0
        cdef int g = self._py_rng.randrange(G - 1)
        if self._attempt_crossover_pair(g, y):
            return 1
        return 0

    cpdef dict ladder_stats(self):
        """Phase 4.1 (PT ladder tuning): per-consecutive-pair swap and
        trans-grove-crossover attempt/accept counters and rates. Pair g is
        (T_g, T_{g+1}); rates are NaN where nothing was attempted. Pure
        read-out of the monitors — the PLAN 4.1 target band for swap rates
        is 0.2-0.4 per pair."""
        cdef int G = len(self.groves)
        cdef list temps = [float(self.temperatures[g]) for g in range(G)]
        cdef list s_att = [int(v) for v in self._swap_attempts]
        cdef list s_acc = [int(v) for v in self._swap_accepts]
        cdef list x_att = [int(v) for v in self._xover_attempts]
        cdef list x_acc = [int(v) for v in self._xover_accepts]
        return {
            'temperatures': temps,
            'swap_attempts': s_att,
            'swap_accepts': s_acc,
            'swap_rates': [ (s_acc[g] / s_att[g]) if s_att[g] > 0 else float('nan')
                            for g in range(len(s_att)) ],
            'xover_attempts': x_att,
            'xover_accepts': x_acc,
            'xover_rates': [ (x_acc[g] / x_att[g]) if x_att[g] > 0 else float('nan')
                             for g in range(len(x_att)) ],
        }

    # ------------------------------------------------------------------
    # Phase 2.12: MPI-parallel EMC sampling loop (spawn model)
    # ------------------------------------------------------------------
    cpdef dict run_mpi_spawn(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                             int n_rounds, int n_steps_per_round,
                             object seed=None,
                             object max_tree_depth=None,
                             object subtree_rebuild_max_height=None):
        """Run the EMC loop with ONE spawned MPI worker per temperature rung.

        Mirrors the serial usage pattern
            for r in range(n_rounds):
                forest.run_all(y, n_steps_per_round)
                forest.attempt_swaps(y)
                forest.attempt_crossovers(y)
        but the mutation phase (the expensive part: BIC scoring + scipy
        optimize per step) runs CONCURRENTLY, one worker process per grove.
        Burn-in is untouched: this parallelizes only the sampling loop.

        Design (all rigor-critical pieces):
          * PERSISTENT workers: each worker builds one Grove once (from the
            same ctor args + the master's CURRENT trees, shipped in the
            startup config) and owns it for the whole run. No per-round
            grove serialization.
          * SHARED Polya by deterministic replay: workers run with
            polya_defer=True (accepted paths collected, not applied). Each
            round the master gathers all paths (ordered by rank), replays
            them into ITS Polya, then broadcasts the same ordered list;
            every worker replays it identically -> all Theta replicas stay
            bit-identical. Within-round priors use the last-synced Theta
            (delayed adaptation: gamma_t -> 0 and containment still hold).
            The master's vocabulary is stabilized up-front so ('kind', idx)
            tuples mean the same thing on every replica: under the Stage 3c
            freeze (Forest(freeze_vocab=True)) it is CLOSED by construction
            (psi_const cell, auto_register off); in legacy mode a generous
            parameter-name pool is pre-registered (best-effort).
          * Exchange + crossover are computed ON THE MASTER with the synced
            Theta and the master RNG, using exactly the serial acceptance
            forms (_attempt_swap_pair / _attempt_crossover_pair math).
            Trees cross the wire as (payload, name_values, post): payload
            is ALWAYS a ('skel3b', skeleton, description) structural
            payload on this wire (wire-fidelity fix 2026-07-24 — the
            historical raw-sympy-expr payload canonicalized on
            construction, e.g. x*x*x -> x**3, and whenever the
            vocabulary could not re-express the folded form the
            round-trip gate discarded the worker's tree, silently
            starving the master's population; the skeleton round-trip
            is shape-exact and sympy-free). Any rebuild-fidelity
            failure still turns the exchange into a REJECT (self-loop;
            target-preserving).
          * Control flow is bcast-driven (every worker sees every command;
            only the addressed rank acts on point-to-point pieces), so no
            interleaving can deadlock.

        Returns a stats dict. On exit the master's self.groves hold the
        final trees/posteriors and self.polya the final Theta, so all
        downstream code (best_tree, ...) works unchanged.
        """
        cdef int G = len(self.groves)
        if G < 2:
            raise ValueError("run_mpi_spawn needs >= 2 temperature rungs")
        # Phase 4.4 guard (v2-relaxed): a RAW-callable tilt cannot cross
        # the spawn boundary (workers rebuild their Groves from
        # picklable ctor args only), so the workers' in-round _step
        # moves would silently sample WITHOUT the tilt — refusing beats
        # a partial tilt. A declarative structural_prior_spec CAN
        # cross: it ships in _ctor and every worker's Grove ctor
        # resolves it locally (the AtomSpec pattern), so spec-built
        # tilts pass through. NB spawn users should prefer constraints
        # whose value is determined by the tree DESCRIPTION alone
        # (symbolic-resolvable, or theta_policy='reference' builders):
        # per-process caches make theta-dependent fallbacks replica-
        # dependent.
        for _g_obj in self.groves:
            if ((<Grove> _g_obj).structural_prior is not None
                    and (<Grove> _g_obj).structural_prior_spec is None):
                raise NotImplementedError(
                    "structural_prior (raw callable) is not supported "
                    "under run_mpi_spawn: the callable cannot be "
                    "shipped to spawned workers. Pass "
                    "structural_prior_spec=('module', 'builder', "
                    "kwargs) instead (see exp_constraints.spec), or "
                    "use the serial EMC loop.")
        # Phase FX v1: flex scaffolds do not cross the spawn boundary yet
        # (workers would need the FlexAtomSpec registry in their startup
        # config plus 'F'-aware wire configs on both spawn wires). The
        # serial EMC loop fully supports them. Refusing beats silently
        # dropping scaffolds at the wire.
        if self.flex_atoms:
            raise NotImplementedError(
                "flex_atoms is not supported under run_mpi_spawn in v1: "
                "use the serial EMC loop (run_all / attempt_swaps / "
                "attempt_crossovers).")
        # Phase MD v1 (D8): joint datasets do not cross the spawn boundary
        # (workers would need all J datasets in the startup payload and a
        # joint-aware in-round scorer). The serial EMC loop fully
        # supports them. Refusing beats silently scoring one dataset.
        if self.datasets is not None:
            raise NotImplementedError(
                "datasets (multi-dataset joint evidence) is not supported "
                "under run_mpi_spawn in v1: use the serial EMC loop "
                "(run_all / attempt_swaps / attempt_crossovers).")
        # Phase 2.10 Stage 3b: opaque primitives cross the wire as
        # ('skel3b', ...) payloads (see _tree_to_wire). The master-side
        # rebuilds below need the template registry; workers receive the
        # same atoms through the startup config and build their own.
        cdef dict atom_specs = None
        if self.atoms:
            atom_specs = {}
            for _spec_obj in self.atoms:
                atom_specs[_spec_obj.name] = _spec_obj
        cdef Grove g0 = self.groves[0]

        # --- Freeze the Polya vocabulary so path indices are stable ------
        # LEGACY (per-name) vocabularies: pre-register a generous fixed
        # parameter-name pool BEFORE the state snapshot, so workers rarely
        # auto-register mid-round and the ('kind', idx) mapping stays
        # replica-stable (best-effort: new_par can in principle exceed any
        # fixed pool). Phase 2.10 Stage 3c: under unify_params the
        # vocabulary is CLOSED (psi_const covers every parameter leaf), the
        # mapping is exactly stable with no pool at all — and registering
        # names would only shift the symmetric prior — so skip.
        cdef list param_pool = [f"a{k}" for k in range(32)] + \
                               [f"b{k}" for k in range(32)]
        if self.polya is not None and not self.polya.unify_params:
            self.polya.add_primitives(param_pool)

        # --- Per-worker seeds (reproducible from `seed`) ------------------
        # NB: use `ss` (untyped object) here, NOT `s` -- `s` is declared
        # `cdef int` below and Cython hoists that to function scope, so
        # iterating SeedSequence objects into `s` raises "an integer is
        # required".
        seed_seq = np.random.SeedSequence(seed)
        worker_seed_children = seed_seq.spawn(G)
        cdef list worker_seeds = [
            int(ss.generate_state(1)[0]) for ss in worker_seed_children]

        # --- Ship the master's CURRENT trees so workers start exactly here
        cdef list init_trees = []      # per grove: list of wire tuples
        cdef Grove gr
        cdef int gidx, s
        cdef function_treeC.Tree t
        for gidx in range(G):
            gr = self.groves[gidx]
            wires = []
            for s in range(gr.size):
                t = gr.trees[s]
                wires.append(_tree_to_wire(
                    t, float(gr.root_posteriors[s])
                    if np.isfinite(gr.root_posteriors[s]) else -1.0e300,
                    True))
            init_trees.append(wires)

        # --- Spawn -------------------------------------------------------
        intercomm = MPI.COMM_SELF.Spawn(
            sys.executable,
            args=["-c", "from mpi_worker import sampler_main; sampler_main()"],
            maxprocs=G)

        cdef dict config = {
            'ctor': self._ctor,
            'y': np.asarray(y, dtype=np.float64),
            'temperatures': np.asarray(self.temperatures, dtype=np.float64),
            'worker_seeds': worker_seeds,
            'polya_state': (self.polya.get_state()
                            if self.polya is not None else None),
            'init_trees': init_trees,
            'max_tree_depth': max_tree_depth,
            'subtree_rebuild_max_height': subtree_rebuild_max_height,
            # Phase 2.10 Stage 3b: ship the templates (AtomSpec is
            # picklable via __reduce__). Workers pass them to their Grove
            # (kernel moves) and to every wire rebuild. The atom SYMBOLS
            # are already registered in polya_state's vocabulary, so the
            # ('kind', idx) path mapping is identical on every replica.
            # Flexible-domain fix (adversarial review): (name, template)
            # STRING PAIRS on the wire — see the burn-in config note; the
            # worker rebuilds AtomSpec AFTER installing the registry.
            # Phase PP: pinned specs ship (base_name, template, pinned)
            # TRIPLES; unpinned keep the historical pair byte-identically.
            'atoms': ([((_a.base_name, _a.template, dict(_a.pinned))
                        if _a.pinned else (_a.name, _a.template))
                       for _a in self.atoms]
                      if self.atoms else None),
        }
        intercomm.bcast(config, root=MPI.ROOT)

        # --- Master-side accumulators -------------------------------------
        # Each round is a pure COLLECTIVE cycle, exactly like
        # _burn_in_stage2_parallel_spawn: bcast a command, GATHER every
        # worker's trees (+ its accepted Polya paths), do the deterministic
        # replay and one swap / one crossover LOCALLY on the wires, then
        # SCATTER the (possibly patched) trees plus the global path list back.
        # No point-to-point send/recv, so no message-ordering can deadlock.
        cdef int n_swap_acc = 0
        cdef int n_x_acc = 0
        cdef int n_paths_total = 0
        cdef int rnd, gsel, i, j
        cdef list replies, all_paths, all_wires, outgoing
        cdef object wireA, wireB
        cdef function_treeC.Tree treeA, treeB
        cdef double Ua, Ub, delta, log_alpha, lqf, lqr
        cdef double postA_new, postB_new
        cdef double priorA_old, priorB_old, priorA_new, priorB_new
        cdef double u
        cdef Grove A, B
        cdef object domain = g0.domain
        cdef list funcs = self._ctor['funcs']
        # Effective sigma: Grove.__init__ maps sigma=None -> 1.0, and the
        # master's live trees carry THAT value; wire rebuilds must match.
        cdef object sigma = g0.sigma
        cdef double g_tau = float(g0.g_tau)

        try:
            for rnd in range(n_rounds):
                # ---- 1. mutation phase (parallel) + gather all trees ----
                intercomm.bcast(('run', n_steps_per_round), root=MPI.ROOT)
                replies = intercomm.gather(None, root=MPI.ROOT)
                # reply g = (deferred_paths_g, [wire per slot])
                all_paths = []
                all_wires = [None] * G
                for gidx in range(G):
                    all_paths.extend(replies[gidx][0])
                    all_wires[gidx] = list(replies[gidx][1])
                n_paths_total += len(all_paths)

                # ---- 2. Polya deterministic replay on the master --------
                # Workers replay the same list (shipped in the scatter
                # below), so both advance the shared urn identically.
                if self.polya is not None and all_paths:
                    self.polya.replay_paths(all_paths)

                # Local swap + crossover are guarded so a compute error can
                # never skip the scatter that keeps the ranks in lockstep.
                try:
                    # ---- 3. one swap attempt (mirrors ------------------
                    #         _attempt_swap_pair) -------------------------
                    gsel = self._py_rng.randrange(G - 1)
                    A = self.groves[gsel]
                    B = self.groves[gsel + 1]
                    if (A.T > 0.0 and B.T > 0.0
                            and all_wires[gsel] and all_wires[gsel + 1]):
                        i = self._py_rng.randrange(min(A.size, B.size))
                        wireA = all_wires[gsel][i]
                        wireB = all_wires[gsel + 1][i]
                        treeA = _tree_from_wire(wireA[0], wireA[1],
                                                domain, funcs, sigma, g_tau,
                                                atom_specs)
                        treeB = _tree_from_wire(wireB[0], wireB[1],
                                                domain, funcs, sigma, g_tau,
                                                atom_specs)
                        if treeA is not None and treeB is not None:
                            priorA_old = 0.0
                            priorB_old = 0.0
                            if self.polya is not None:
                                # Phase 2.13(c): extract both, then score both.
                                _pA = self.polya.extract_path(treeA.get_root())
                                _pB = self.polya.extract_path(treeB.get_root())
                                priorA_old = self.polya.log_prior(_pA)
                                priorB_old = self.polya.log_prior(_pB)
                            # Phase 4.4: structural tilt in the U's (0.0
                            # when off; unreachable with the tilt ON under
                            # the v1 spawn guard — kept correct for v2).
                            Ua = (<double> wireA[2]) + priorA_old + A._sp(treeA)
                            Ub = (<double> wireB[2]) + priorB_old + B._sp(treeB)
                            delta = (1.0 / A.T - 1.0 / B.T) * (Ub - Ua)
                            if self._py_rng.random() < min(1.0, _safe_exp(delta)):
                                # A swap is just exchanging the two wire tuples.
                                all_wires[gsel][i] = wireB
                                all_wires[gsel + 1][i] = wireA
                                n_swap_acc += 1

                    # ---- 4. one crossover attempt (mirrors --------------
                    #         _attempt_crossover_pair) --------------------
                    gsel = self._py_rng.randrange(G - 1)
                    A = self.groves[gsel]
                    B = self.groves[gsel + 1]
                    if (A.T > 0.0 and B.T > 0.0
                            and all_wires[gsel] and all_wires[gsel + 1]):
                        i = self._py_rng.randrange(A.size)
                        j = self._py_rng.randrange(B.size)
                        wireA = all_wires[gsel][i]
                        wireB = all_wires[gsel + 1][j]
                        treeA = _tree_from_wire(wireA[0], wireA[1],
                                                domain, funcs, sigma, g_tau,
                                                atom_specs)
                        treeB = _tree_from_wire(wireB[0], wireB[1],
                                                domain, funcs, sigma, g_tau,
                                                atom_specs)
                        if treeA is not None and treeB is not None:
                            priorA_old = 0.0
                            priorB_old = 0.0
                            # Phase 2.13(c): capture OLD paths BEFORE the
                            # in-place crossover mutates treeA/treeB; score
                            # all four priors only after every extraction.
                            # (Vocabulary is frozen up-front in MPI mode —
                            # 32 pre-registered names — precisely so master-
                            # side auto-registration, which would desync the
                            # worker replicas, cannot fire in practice.)
                            _pA = None
                            _pB = None
                            if self.polya is not None:
                                _pA = self.polya.extract_path(treeA.get_root())
                                _pB = self.polya.extract_path(treeB.get_root())
                            # Phase 4.4: OLD structural tilts BEFORE the
                            # in-place crossover mutates treeA/treeB (0.0
                            # when off; unreachable with the tilt ON under
                            # the v1 spawn guard — kept correct for v2).
                            sp_mA_old = A._sp(treeA)
                            sp_mB_old = B._sp(treeB)
                            # Same RNG discipline as the serial path.
                            gp_movesC.set_rng(self._py_rng, self._np_rng)
                            function_treeC.set_rng(self._py_rng, self._np_rng)
                            lqf, lqr = gp_movesC.subtree_crossover(treeA, treeB)
                            if np.isfinite(lqf) and np.isfinite(lqr):
                                postA_new = A._score_post(treeA, y, True)
                                postB_new = B._score_post(treeB, y, True)
                                if postA_new > -1.0e299 and postB_new > -1.0e299:
                                    priorA_new = 0.0
                                    priorB_new = 0.0
                                    if self.polya is not None:
                                        _pAn = self.polya.extract_path(treeA.get_root())
                                        _pBn = self.polya.extract_path(treeB.get_root())
                                        priorA_old = self.polya.log_prior(_pA)
                                        priorB_old = self.polya.log_prior(_pB)
                                        priorA_new = self.polya.log_prior(_pAn)
                                        priorB_new = self.polya.log_prior(_pBn)
                                    # Phase 4.4: treeA/treeB are the CHILDREN
                                    # here, so A._sp(treeA) is the new tilt.
                                    log_alpha = (
                                        (postA_new + priorA_new + A._sp(treeA)
                                         - (<double> wireA[2]) - priorA_old
                                         - sp_mA_old) / A.T
                                        + (postB_new + priorB_new + B._sp(treeB)
                                           - (<double> wireB[2]) - priorB_old
                                           - sp_mB_old) / B.T
                                        + (lqr - lqf))
                                    u = self._py_rng.random()
                                    if log_alpha >= 0.0 or (u > 0.0 and
                                                            np.log(u) < log_alpha):
                                        # Round-trip check the CHILD wires on
                                        # the master BEFORE installing:
                                        # crossover can produce expressions
                                        # that do not survive _expr_to_tree
                                        # (the burn-in sees the same and
                                        # skips). The workers rebuild the
                                        # identical wire, so a failure here
                                        # would desync a worker. Print + skip,
                                        # no exceptions (a skip is a reject).
                                        wireA = _tree_to_wire(treeA, postA_new,
                                                              True)
                                        wireB = _tree_to_wire(treeB, postB_new,
                                                              True)
                                        if (_tree_from_wire(wireA[0], wireA[1],
                                                            domain, funcs, sigma,
                                                            g_tau, atom_specs)
                                                is None or
                                            _tree_from_wire(wireB[0], wireB[1],
                                                            domain, funcs, sigma,
                                                            g_tau, atom_specs)
                                                is None):
                                            print("[run_mpi_spawn] crossover "
                                                  "child failed wire round-trip; "
                                                  "skipping exchange (round %d)"
                                                  % rnd)
                                        else:
                                            all_wires[gsel][i] = wireA
                                            all_wires[gsel + 1][j] = wireB
                                            n_x_acc += 1
                except Exception as _cerr:
                    print("[run_mpi_spawn] round %d compute error: %s"
                          % (rnd, _cerr))

                # ---- 5. scatter updated trees + the global path list ----
                # outgoing[g] -> worker g: (its trees, all accepted paths).
                outgoing = [(all_wires[gidx], all_paths) for gidx in range(G)]
                intercomm.scatter(outgoing, root=MPI.ROOT)
        finally:
            # ---- collect final states + shut workers down ---------------
            intercomm.bcast(('stop',), root=MPI.ROOT)
            final_states = intercomm.gather(None, root=MPI.ROOT)
            for gidx in range(G):
                gr = self.groves[gidx]
                wires = final_states[gidx]
                for s in range(min(gr.size, len(wires))):
                    t = _tree_from_wire(wires[s][0], wires[s][1],
                                        domain, funcs, sigma, g_tau,
                                        atom_specs)
                    if t is not None:
                        gr.trees[s] = t
                        gr.root_posteriors[s] = wires[s][2]
            intercomm.Disconnect()

        return {
            'rounds': n_rounds,
            'swap_accepted': n_swap_acc,
            'crossover_accepted': n_x_acc,
            'polya_paths_replayed': n_paths_total,
        }

    cpdef function_treeC.Tree best_tree(self):
        """Best (highest BIC posterior) tree across all groves. Caller should
        have populated posteriors (run_all / compute_posteriors_all)."""
        cdef double best_val = -1e300
        cdef function_treeC.Tree best = None
        cdef Grove gr
        cdef int idx
        cdef object val
        for gr in self.groves:
            idx = gr.best_index()
            val = gr.root_posteriors[idx]
            if val is not None and val > best_val:
                best_val = val
                best = gr.trees[idx]
        return best


cdef _bis2_parallel_helper(object x):
        return x[1]
