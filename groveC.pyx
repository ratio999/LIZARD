# cython: language_level=3
import random
import warnings
import numpy as np
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
from libc.math cimport exp, fmax
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
                 cnp.ndarray[cnp.float64_t, ndim=1] x,
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
                 double adapt_target=0.5):

        # Memory view - safer alternative to raw pointers
        cdef cnp.float64_t[::1] x_view = x
        # Store y as instance attribute for use in optimization and posterior computation
        self.data = np.asarray(y, dtype=np.float64)
        self.lambda_reg = lambda_reg  # default; can be changed in optimize()
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
        self.operations = operations
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
            "parameter_refresh":  0,
        }
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
        # --- Phase 2.8: optional adaptive move-weight calibration ----------
        # OFF by default (adapt_moves=False) -> Grove behaves exactly as
        # before, using the fixed temperature-shaped weights in _choose_move.
        # When ON, five logical-group logits are updated by a diminishing-
        # adaptation (Robbins-Monro) rule (see _adapt_move) and mapped to the
        # seven physical move probabilities by _adapted_move_probs, which
        # preserves the birth/death pair equality (so the RJMCMC move-
        # selection correction stays exactly zero) and floors every
        # probability at `adapt_floor` (containment). Warm-started from the
        # fixed weights so toggling adaptation on is continuous.
        if not (0.0 <= adapt_floor < 1.0 / 7.0):
            raise ValueError("Grove: adapt_floor must lie in [0, 1/7)")
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

        # Optional Pólya tree prior. When None, _step behaves exactly as
        # before (no prior term, no adaptation).
        self.polya = polya
        
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
                trees[i] = function_treeC.Tree(self.depth, self.funcs, self.domain,
                                               Npars=Npars, value=init_value, T=self.T,
                                               sigma=self.sigma, g_tau=self.g_tau)


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
        """
        cdef int i
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
        probs = np.array([p_point, p_leaf_hoist, p_internal_hoist,
                          p_add, p_split, p_subtree, p_cross],
                        dtype=np.float64)

        probs = probs / probs.sum()
        if self.adapt_moves:
            # Phase 2.8: adaptive path overrides the fixed weights computed
            # above. _adapted_move_probs keeps the pair equality and floor.
            probs = self._adapted_move_probs()
        return probs

    cpdef cnp.ndarray get_move_probs(self):
        """Phase 2.8 introspection: current physical move-probability 7-vector
        (adapted if adapt_moves else the fixed temperature-shaped weights).
        Read-only; does not advance any RNG. Order:
        [point, leaf_hoist, internal_hoist, add, split, subtree, cross]."""
        return self._compute_move_probs()

    cdef _choose_move(self) :
        """
        Select a move and parameters based on temperature.
        Returns (name, kwargs_dict).
        """
        cdef cnp.ndarray probs = self._compute_move_probs()
        idx = int(self._np_rng.choice(np.arange(7), p=probs))
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
        else:
            return ("subtree_crossover", {})

    cdef cnp.ndarray _adapted_move_probs(self):
        """Map the 5 logical-group logits to the 7 physical move probabilities.

        Group -> physical layout (must match the _choose_move dispatch order
        [point, leaf_hoist, internal_hoist, add, split, subtree, cross]):
            eta[0] -> point_mutation
            eta[1] -> leaf_hoist  AND  add_node      (shared: pair equality)
            eta[2] -> internal_hoist AND split       (shared: pair equality)
            eta[3] -> subtree_mutation
            eta[4] -> subtree_crossover
        Because each birth/death pair shares a single logit, p_add == p_leaf
        and p_split == p_internal_hoist hold exactly, so the RJMCMC move-
        selection term log p(m'|T')/p(m|T) is identically zero. A uniform
        floor eps keeps every probability >= eps (containment).
        """
        cdef cnp.ndarray[cnp.float64_t, ndim=1] eta = self._move_logits
        cdef cnp.ndarray[cnp.float64_t, ndim=1] l7 = np.array(
            [eta[0], eta[1], eta[2], eta[1], eta[2], eta[3], eta[4]],
            dtype=np.float64)
        l7 = l7 - l7.max()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] b = np.exp(l7)
        b = b / b.sum()
        cdef double eps = self.adapt_floor
        return (1.0 - 7.0 * eps) * b + eps

    cdef int _move_group(self, str move_name):
        """Logical adaptation group (0..4) of a move, or -1 if the move is not
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

    cdef tuple _propose(self, int i, int max_attempts=2) :
        """
        Build a proposal from tree i using a move drawn from the temperature-dependent kernel.
        Returns a deepcopy candidate (original tree remains unchanged).
        """
        cdef function_treeC.Tree cand 
        cdef str move_name
        cdef dict kwargs
        cdef int attempts = 0
        cdef int donor_idx
        cdef double lqf, lqr

        while attempts < max_attempts:
            cand = deepcopy(self.trees[i])
            move_name, kwargs = self._choose_move()
            if move_name == "point_mutation":
                lqf, lqr = gp_movesC.point_mutation(
                    cand, self.funcs, kwargs.get("operations", None))
            elif move_name == "leaf_hoist":
                lqf, lqr = gp_movesC.leaf_hoist(
                    cand, self.funcs, self.operations)
            elif move_name == "internal_hoist":
                lqf, lqr = gp_movesC.internal_hoist(
                    cand, self.funcs, self.operations)
            elif move_name == "add_node":
                lqf, lqr = gp_movesC.add_node(
                    cand, self.funcs, kwargs.get("operations", None))
            elif move_name == "split":
                lqf, lqr = gp_movesC.split(
                    cand, self.funcs, kwargs.get("operations", None))
            elif move_name == "subtree_mutation":
                    #print(kwargs.get('random_depth', False))
                lqf, lqr = gp_movesC.subtree_mutation(cand, self.funcs)
                                        #kwargs.get("random_depth", False)) USED ONLY IF LEGACY SUBTREE
            elif move_name == "subtree_crossover":
                # pick a donor tree (not i), deep-copied to keep original intact
                if self.size > 1:
                    donor_idx = i
                    while donor_idx == i:
                        donor_idx = self._py_rng.randrange(self.size)
                    donor = deepcopy(self.trees[donor_idx])
                    lqf, lqr = gp_movesC.subtree_crossover(cand, donor)
                else:
                    lqf, lqr = gp_movesC.point_mutation(
                        cand, self.funcs, self.operations)
                    move_name = "point_mutation"
            else:
                lqf, lqr = gp_movesC.point_mutation(
                    cand, self.funcs, self.operations)
                move_name = "point_mutation"

            if cand.is_valid(self.domain):
                
                return (cand, lqf, lqr, move_name)
            attempts += 1
            #print(attempts)
        return (None, 0.0, 0.0, move_name)
        #return deepcopy(self.trees[i])

    cdef _step_parameter_refresh(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        Phase 2.1.PR. Self-accepting parameter refresh on a uniformly chosen
        tree of this grove.

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
        if self.size == 0:
            return False
        cdef int i = self._py_rng.randrange(self.size)
        cdef function_treeC.Tree tree = self.trees[i]
        cdef int K = len(tree.params)

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
        cdef list names
        if key not in self._am_states:
            names = [p.get_fun().get_description() for p in tree.params]
            am = gp_movesC.AMState(names)
            self._am_states[key] = am
        else:
            am = <gp_movesC.AMState> self._am_states[key]
            if am.K != K:
                # Defensive: same description string but K mismatch.
                # Should not occur if tree_get_description is canonical;
                # rebuild to keep PR safe.
                names = [p.get_fun().get_description() for p in tree.params]
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
        cdef double lqf, lqr
        cdef str move_name
        cdef tuple proposal = self._propose(i)

        
        cand     = <function_treeC.Tree> proposal[0]
        lqf      = <double> proposal[1]
        lqr      = <double> proposal[2]
        move_name = <str>   proposal[3]

        self.proposal_count += 1.0
        self.move_proposals[move_name] = self.move_proposals.get(move_name, 0) + 1

        # Validity-failed proposal -> hard reject (already counted as proposed).
        if cand is None or not np.isfinite(lqr):
            self._adapt_move(move_name, False)
            return False

        # Cache-fill for current tree if needed.
        '''if np.isnan(self.root_posteriors[i]):
            #nelow is deprecated
            #self.root_posteriors[i] = self.trees[i].get_superficial_posterior(
            #    y, self.lambda_reg)
            self.root_posteriors[i] = self.trees[i].log_marginal_bic(
                y, optimize=True,
                use_gradients=self.use_gradients,
                method_str=self.method_str,
            )'''
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
        cdef bint optim_need = len(cand.param_values) > 0
        cdef str cand_key
        cdef object cached
        cdef dict cached_nv
        cdef function_treeC.Node par_node
        cdef str pname
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
            new_post  = <double> cached[0]
            cached_nv = <dict>   cached[1]
            for par_node in cand.params:
                pname = par_node.get_fun().get_description()
                if pname in cached_nv:
                    par_node.set_value(<double> cached_nv[pname])
            if cand.params:
                cand.tree_update(cand.get_root())
        else:
            new_post = cand._log_marginal_bic(
                y, optimize=optim_need,
                use_gradients=self.use_gradients if optim_need else False,
                method_str=self.method_str,
            )
            if np.isfinite(new_post):
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
            old_path = self.polya.extract_path(self.trees[i].get_root())
            old_prior = self.polya.log_prior(old_path)
            new_path  = self.polya.extract_path(cand.get_root())
            new_prior = self.polya.log_prior(new_path)

        cdef double dtarget = (new_post + new_prior) - (old_post + old_prior)
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
                self.polya.update_path(new_path)
            return True
        return False

    cdef double _score_post(self, function_treeC.Tree cand,
                            cnp.ndarray[cnp.float64_t, ndim=1] y,
                            bint optimize=True):
        """Score a candidate's BIC log-evidence, mirroring the scoring block
        of `_step`: validity gate + Phase 2.7(f) topology cache + optimize.

        Returns -1e300 for invalid / non-finite models (so any MH ratio that
        consumes it rejects). On a cache hit the cached theta_hat is
        re-installed on `cand` so an accept warm-starts subsequent moves.

        Used by Forest's cross-temperature swap / crossover (Phase 2.9) so
        trans-grove moves are scored exactly like within-grove ones.
        """
        cdef str cand_key
        cdef object cached
        cdef dict cached_nv
        cdef function_treeC.Node par_node
        cdef str pname
        cdef double new_post
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
            for par_node in cand.params:
                pname = par_node.get_fun().get_description()
                if pname in cached_nv:
                    par_node.set_value(<double> cached_nv[pname])
            if cand.params:
                cand.tree_update(cand.get_root())
            return new_post
        new_post = cand._log_marginal_bic(
            y, optimize=optimize,
            use_gradients=self.use_gradients if optimize else False,
            method_str=self.method_str,
        )
        if np.isfinite(new_post):
            self._bic_cache[cand_key] = (new_post, dict(cand.name_values))
            return new_post
        return -1.0e300

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
        if self.polya is not None:
            # Prior of the *current* tree must be recomputed every step
            # because Theta moves whenever any walker accepts.
            old_prior = self.polya.log_prior(
                self.polya.extract_path(self.trees[i].get_root())
            )
            new_path = self.polya.extract_path(cand.get_root())
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
        
        Parameters
        ----------
        y : target data
        replace_all : if True, replace all grove trees with best depth-2 candidates
        optimize_params : if True, optimize parameters for each candidate
        
        """
        cdef dict tree_cache = {}
        # First, run stage 1 to get depth-1 candidates
        depth1_candidates = self._get_depth1_candidates(y, tree_cache, optimize_params)
        
        if len(depth1_candidates) == 0:
            print("Warning: No depth-1 candidates; cannot build depth-2")
            return
        #print(tree_cache)
        # Now combine them
        self._burn_in_stage2(y, depth1_candidates, tree_cache, replace_all, optimize_params)
    
    cdef _get_depth1_candidates(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict tree_cache,
                                      bint optimize_params=True):
        """
        Build all depth-1 trees (primitive op primitive) and return sorted list.
        Reuses logic from _burn_in_combinations but returns candidates instead of replacing.
        """
        cdef list candidates = []
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        cdef str left_str, right_str, op, primitive_expression
        cdef function_treeC.Tree tree, primitive_tree, cand
        cdef function_treeC.Node root_node
        cdef double post
        print(f"Stage 1: building {len(all_primitives)**2 * len(self.operations)} depth-1 trees")
        # Initialize candidates with primitive functions (depth-0 trees)
        for func_str in all_primitives:
            if func_str == "a0":
            
                root_node= function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                    )
                root_node.set_value(1.0)
            
            else:
                root_node = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(func_str, self.domain, compute_grad = False),
                    )
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
        for left_str, right_str in itertools.product(all_primitives, repeat=2):
            if left_str=='a0' and right_str=='a0':
                continue
            for op in self.operations:
                
                if left_str == "a0":
                    root_left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                    )
                    root_left.set_value(1.0)
                else:
                    root_left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(left_str, self.domain, compute_grad = False),
                    )
                
                # Build right child
                if right_str == "a0":
                    root_right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                    )
                    root_right.set_value(1.0)
                else:
                    root_right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(right_str, self.domain, compute_grad = False),
                    )
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
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
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
                prim_node = function_treeC.Node(function=real_functionC.RealFunctionC.from_string(prim_str, self.domain, compute_grad = False))
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
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        cdef int n_top = len(depth1_candidates)
        
        if n_top == 0:
            print("No depth-1 candidates to combine")
            return
        
        # Serialize depth-1 trees for workers
        cdef list  depth1_data = []
        cdef function_treeC.Tree tree
        cdef object post
        cdef int depth, i
        cdef object expr
        cdef int sum_of_misses = 0
        cdef dict names_params
        for tree, post, depth in depth1_candidates:

            expr = tree.root.function.expr
            names_params = tree.name_values if tree.name_values else {}
            depth1_data.append((expr, names_params, post))
            if str(expr) not in tree_cache:
                print(f"Adding to cache: {expr} with post {post}. Bug in line 917 if this is missing.")
                tree_cache[str(expr)] = post
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
            'funcs': self.funcs,
            'operations': self.operations,
            'y': np.asarray(y),
            'lambda_reg': self.lambda_reg,
            'method_str': self.method_str,
            'use_gradients': self.use_gradients,
            'depth1_data': depth1_data,
            'all_primitives': all_primitives,
            'T': self.T,
            'tree_cache': tree_cache
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
                prim_node = function_treeC.Node(
                    function = real_functionC.RealFunctionC.from_string(
                        prim_str, self.domain, compute_grad=False
                    ))
                
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
        for worker_results in all_return_list:
            for expr, names_params, post in worker_results:
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
                 cnp.ndarray[cnp.float64_t, ndim=1] x,
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
                 bint adapt_moves=False):
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.groves = []
        self.n_procs = n_procs
        self.use_mpi = self.n_procs >= 2
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
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
            polya = polya_treeC.PolyaTree(
                operations, list(funcs), alpha_0=alpha_0,
                auto_register=True, strict_ops=True)
            cap = Npars if Npars > 0 else 8
            param_names = [f"a{k}" for k in range(cap)] + \
                          [f"b{k}" for k in range(cap)]
            polya.add_primitives(param_names)
        self.polya = polya
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
                      g_tau=g_tau, adapt_moves=adapt_moves)
            )

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
        cdef int i = self._py_rng.randrange(min(sizeA, sizeB))
        if A.root_posteriors[i] is None or np.isnan(A.root_posteriors[i]):
            A.root_posteriors[i] = A._score_post(A.trees[i], y, False)
        if B.root_posteriors[i] is None or np.isnan(B.root_posteriors[i]):
            B.root_posteriors[i] = B._score_post(B.trees[i], y, False)
        cdef double postA = float(A.root_posteriors[i])
        cdef double postB = float(B.root_posteriors[i])
        cdef double priorA = 0.0
        cdef double priorB = 0.0
        if self.polya is not None:
            priorA = self.polya.log_prior(self.polya.extract_path(A.trees[i].get_root()))
            priorB = self.polya.log_prior(self.polya.extract_path(B.trees[i].get_root()))
        cdef double Ua = postA + priorA
        cdef double Ub = postB + priorB
        cdef double delta = (1.0 / Ta - 1.0 / Tb) * (Ub - Ua)
        cdef double acc_p = _safe_exp(delta)
        if self._py_rng.random() < min(1.0, acc_p):
            tmp_tree = A.trees[i]; A.trees[i] = B.trees[i]; B.trees[i] = tmp_tree
            tmp_post = A.root_posteriors[i]
            A.root_posteriors[i] = B.root_posteriors[i]
            B.root_posteriors[i] = tmp_post
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
        cdef int i = self._py_rng.randrange(sizeA)
        cdef int j = self._py_rng.randrange(sizeB)
        if A.root_posteriors[i] is None or np.isnan(A.root_posteriors[i]):
            A.root_posteriors[i] = A._score_post(A.trees[i], y, False)
        if B.root_posteriors[j] is None or np.isnan(B.root_posteriors[j]):
            B.root_posteriors[j] = B._score_post(B.trees[j], y, False)
        cdef double postA_old = float(A.root_posteriors[i])
        cdef double postB_old = float(B.root_posteriors[j])
        cdef double priorA_old = 0.0
        cdef double priorB_old = 0.0
        if self.polya is not None:
            priorA_old = self.polya.log_prior(self.polya.extract_path(A.trees[i].get_root()))
            priorB_old = self.polya.log_prior(self.polya.extract_path(B.trees[j].get_root()))
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
        cdef double priorA_new = 0.0
        cdef double priorB_new = 0.0
        if self.polya is not None:
            priorA_new = self.polya.log_prior(self.polya.extract_path(childA.get_root()))
            priorB_new = self.polya.log_prior(self.polya.extract_path(childB.get_root()))
        cdef double Ua_old = postA_old + priorA_old
        cdef double Ub_old = postB_old + priorB_old
        cdef double Ua_new = postA_new + priorA_new
        cdef double Ub_new = postB_new + priorB_new
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
