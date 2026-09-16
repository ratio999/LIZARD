# cython: language_level=3
"""
Pólya tree prior over GP construction choices.

Two kinds of decision nodes:
  - 'op'   : choose an operation symbol for an internal GP node
  - 'prim' : choose a primitive (or parameter) for a GP leaf

Construction path representation:
  A path is a list of (kind, choice_idx) tuples obtained by a pre-order
  DFS traversal of a GP tree:
      visit(node):
          if node.is_leaf():   record ('prim', idx)
          else:                record ('op',   idx); visit(left); visit(right)

The Pólya tree branches according to (kind, choice_idx) pairs. Each
PolyaNode holds Dirichlet parameters over its own choice set and is
created lazily on first visit.
"""
cimport numpy as cnp
import numpy as np
cimport function_treeC
from libc.math cimport log

cdef list _OPERATIONS = ["+", "-", "*", "/", "o", "**"]

'''The reserved vocabulary key of the SINGLE canonical parameter cell
psi_const, used under unify_params=True. Per
DESIGN_2_10_PRIMITIVES.md §1 the parameter leaf is one atom of Psi_1:
parameter NAMES (a0, b3, ...) are bound labels chosen deterministically
by new_par and carry no prior mass of their own. '<' cannot appear in a
leaf description or in a sane atom symbol, so the token cannot
collide.'''
PARAM_CELL_TOKEN = "<param>"

# ---------------------------------------------------------------------------
# PolyaNode
# ---------------------------------------------------------------------------
cdef class PolyaNode:
    """A single decision node in the Pólya tree."""

    def __init__(self, str kind, int n_choices, int depth, double alpha_0=1.0):
        self.kind = kind
        self.n_choices = n_choices
        self.depth = depth
        self.alphas = np.ones(n_choices, dtype=np.float64) * alpha_0
        self.alpha_total = float(n_choices) * alpha_0
        '''Base mass of the unregistered remainder cell, set by PolyaTree
        after construction in dp_mode; 0.0 otherwise. It is EXCLUDED from
        alpha_total by restriction-by-neutrality — the predictive
        normaliser runs over REGISTERED cells only (THEORY_2_13.md §5.4)
        — so log_prob, sample_choice and update need no changes.'''
        self.alpha_rem = 0.0
        # Children indexed by (kind, choice_idx). Use a dict so we can have
        # both 'op' and 'prim' continuations from the same decision point.
        self.children = {}

    cdef double log_prob(self, int choice_idx):
        """log P(choice = k | this node) under current Dirichlet mean."""
        return log(self.alphas[choice_idx]) - log(self.alpha_total)

    cdef int sample_choice(self):
        """Sample a choice index from current Dirichlet-Multinomial mean."""
        cdef cnp.ndarray probs = self.alphas / self.alpha_total
        return int(np.random.choice(self.n_choices, p=probs))

    cdef void update(self, int choice_idx, double gamma):
        """Pólya urn update: alpha_k += gamma."""
        self.alphas[choice_idx] += gamma
        self.alpha_total += gamma


# ---------------------------------------------------------------------------
# PolyaTree
# ---------------------------------------------------------------------------
cdef class PolyaTree:
    """
    Pólya tree prior over GP construction choices.

    Parameters
    ----------
    operations : list of str
        Operation symbols, e.g. ['+', '-', '*', '/', '**', 'o'].
    primitives : list of str
        Primitive descriptions, e.g. ['x', 'sin(x)', 'exp(x)', ...].
    alpha_0 : float
        Initial Dirichlet concentration (symmetric).
    """

    def __init__(self, list operations, list primitives, double alpha_0=1.0,
                 bint auto_register=True, bint strict_ops=True,
                 double remainder_mass=0.0, bint unify_params=False):
        '''unify_params=True maps EVERY plain parameter leaf onto the
        single canonical psi_const cell PARAM_CELL_TOKEN, appended to
        the vocabulary here, instead of one cell per parameter NAME.

        Why: names are bound labels, and new_par ratchets their indices
        upward without bound — births after deaths raise the max index,
        and only crossover compaction resets it — so no finite name pool
        covers the reachable names, and per-name cells would weight
        alpha-equivalent trees differently.

        With unify_params the vocabulary {ops} + {function leaves} +
        {psi_const} + {atom templates} is FINITE and CLOSED, so
        auto_register can be False and the prior is one fixed measure
        from t = 0 (the DESIGN §9 item 4 freeze).'''
        cdef list prims = list(primitives)
        self.unify_params = unify_params
        if unify_params:
            if PARAM_CELL_TOKEN in prims:
                raise ValueError(
                    "PolyaTree: %r is the reserved canonical parameter "
                    "cell and may not appear in `primitives`"
                    % PARAM_CELL_TOKEN)
            prims.append(PARAM_CELL_TOKEN)
        self.operations = _OPERATIONS
        self.primitives = prims
        self.alpha_0 = alpha_0
        self.t = 0
        self.n_ops = len(self.operations)
        self.n_prims = len(self.primitives)
        # Lookups
        self._op_to_idx = {op: i for i, op in enumerate(self.operations)}
        self._prim_to_idx = {p: i for i, p in enumerate(self.primitives)}
        # Root: a "virtual" decision over n_ops + n_prims (op vs prim + which one).
        self.auto_register = auto_register
        self.strict_ops = strict_ops
        '''Adaptation freeze is off by default. Set the public attribute
        from Python after warm-up to enter the frozen regime; it is
        deliberately not a constructor argument, to avoid call-site
        churn.'''
        self.adapt_frozen = False
        '''remainder_mass > 0 activates DP mode (THEORY_2_13.md §5). The
        node-level prior is then the projection of ONE DP(c, G0) with
        total base mass

            c = (n_ops + n_prims) * alpha_0 + remainder_mass,

        onto {registered cells} + {remainder R}. base_alphas holds the
        per-cell base masses used to initialise fresh nodes and rem_base
        the unallocated remainder; add_primitive CARVES mass out of the
        remainder, conserving c, instead of appending new mass.
        remainder_mass == 0.0 keeps the simple non-DP behaviour.'''
        if remainder_mass < 0.0:
            raise ValueError("PolyaTree: remainder_mass must be >= 0")
        self.dp_mode = remainder_mass > 0.0
        self.rem_base = remainder_mass
        self.base_alphas = [alpha_0] * (self.n_ops + self.n_prims)
        self.base_reg_total = alpha_0 * (self.n_ops + self.n_prims)
        # psi_const's flat index (symmetric alpha_0 mass like every other
        # initially registered atom, DESIGN §2). -1 when not unified.
        self._param_idx = self._prim_to_idx[PARAM_CELL_TOKEN] \
            if unify_params else -1
        self.root = PolyaNode(
            'root', self.n_ops + self.n_prims, depth=0, alpha_0=alpha_0
        )
        self.root.alpha_rem = self.rem_base

    # ------------------------------------------------------------------
    # Tree navigation (lazy construction)
    # ------------------------------------------------------------------
    cdef PolyaNode _ensure_child(self, PolyaNode parent, str kind, int choice_idx):
        """Create child PolyaNode if absent. Mutates the tree.

        Children always inherit the FULL combined op+prim space, so a path
        that alternates kinds (as produced by pre-order DFS over a GP tree)
        traverses without kind mismatches.
        """
        cdef tuple key = (kind, choice_idx)
        cdef PolyaNode child
        cdef int n_total
        if key not in parent.children:
            n_total = self.n_ops + self.n_prims
            child = PolyaNode('mixed', n_total, depth=parent.depth + 1,
                              alpha_0=self.alpha_0)
            if self.dp_mode:
                # Fresh nodes inherit the CURRENT base masses (registered
                # primitives may carry custom carved masses) and the current
                # unallocated remainder — this is what keeps every node a
                # projection of the same DP(c, G0).
                child.alphas = np.asarray(self.base_alphas, dtype=np.float64).copy()
                child.alpha_total = self.base_reg_total
                child.alpha_rem = self.rem_base
            parent.children[key] = child
        return parent.children[key]

    cdef PolyaNode _get_child_or_none(self, PolyaNode parent, str kind, int choice_idx):
        """Read-only child lookup. Returns None if absent."""
        cdef tuple key = (kind, choice_idx)
        return parent.children.get(key, None)

    cdef int _root_index(self, str kind, int choice_idx):
        """Map (kind, idx) onto the flat root Dirichlet index."""
        if kind == 'op':
            return choice_idx
        return self.n_ops + choice_idx

    # ------------------------------------------------------------------
    # Prior evaluation (READ-ONLY)
    # ------------------------------------------------------------------
    cpdef double log_prior(self, list construction_path):
        """
        Compute log pi(T | Theta) for a tree with given construction path.
        cpdef so tests can probe the prior on hand-built paths without
        constructing GP trees.

        Pure function of the current Pólya tree: does NOT create new nodes.
        Steps that fall in unvisited subtrees contribute -log(n_total) under
        the symmetric default Dirichlet, where n_total = n_ops + n_prims.
        """
        cdef double log_p = 0.0
        cdef PolyaNode current = self.root
        cdef PolyaNode nxt
        cdef int idx, flat_idx
        cdef int n_total = self.n_ops + self.n_prims
        cdef str kind
        cdef tuple step

        for step in construction_path:
            kind = <str> step[0]
            idx = <int> step[1]
            flat_idx = self._root_index(kind, idx)

            if current is None:
                if self.dp_mode:
                    # Unvisited region, DP mode: base-measure predictive
                    # G0(cell | registered) — registered cells may carry
                    # heterogeneous carved masses. Reduces to -log(n_total)
                    # exactly when all base masses are equal.
                    log_p += (log(<double> self.base_alphas[flat_idx])
                              - log(self.base_reg_total))
                else:
                    # Legacy: symmetric default over the combined space.
                    log_p += -log(<double> n_total)
                continue

            log_p += current.log_prob(flat_idx)
            nxt = self._get_child_or_none(current, kind, idx)
            current = nxt  # may be None: subsequent steps use defaults

        return log_p

    # ------------------------------------------------------------------
    # Posterior update (call only on accepted MH steps)
    # ------------------------------------------------------------------
    cdef void update_path(self, list construction_path):
        """
        Reinforce path with diminishing learning rate gamma_t = 1/(1+t).
        Call AFTER an accepted MH step.
        """
        # In the frozen regime reinforcement halts and Theta stays at
        # Theta-bar. One C bool branch, and update_path only runs on
        # accepted moves anyway, so the cost is unmeasurable.
        if self.adapt_frozen:
            return
        cdef double gamma = 1.0 / (1.0 + <double> self.t)
        cdef PolyaNode current = self.root
        cdef int idx, flat_idx
        cdef str kind
        cdef tuple step

        for step in construction_path:
            kind = <str> step[0]
            idx = <int> step[1]
            flat_idx = self._root_index(kind, idx)
            current.update(flat_idx, gamma)
            current = self._ensure_child(current, kind, idx)

        self.t += 1

    # ------------------------------------------------------------------
    # Path extraction from a GP tree
    # ------------------------------------------------------------------
    cpdef list extract_path(self, function_treeC.Node node):
        """
        Pre-order DFS traversal of a GP (sub)tree. Returns a list of
        (kind, choice_idx) tuples suitable for `log_prior` / `update_path`.

        Unknown ops/primitives are silently dropped (e.g. parameter symbols
        not registered in `primitives`).
        """
        cdef list path = []
        self._extract_recursive(node, path)
        return path

    # ------------------------------------------------------------------
    # Vocabulary growth
    # ------------------------------------------------------------------
    cpdef int add_primitive(self, str desc, object mass=None):
        """
        Register a new primitive description in the Pólya vocabulary.
        Returns the (possibly pre-existing) primitive index.

        NON-DP mode (dp_mode False): grows EVERY node's Dirichlet by
        appending a fresh alpha_0 slot, so total mass GROWS and every
        registration perturbs every symbol's predictive. This is the
        historical behaviour, kept bit-exact for existing tests and users.

        DP mode (THEORY_2_13.md §5): registration is a
        MASS-CONSERVING partition refinement. The new cell's base mass
        (`mass`, default alpha_0) is CARVED out of the remainder cell at
        every node: alpha_rem -= mass, new slot = mass, so the total base
        mass c = alpha_total + alpha_rem is invariant at every node
        (Lemma 5.2), registrations commute (Theorem 5.3), and the prior is
        one fixed measure rather than a registration-history-indexed
        family. Raises ValueError when the remainder is exhausted.

        NB (cross-module callers): cpdef + cimport + KEYWORD args is the
        known NameError trap — pass `mass` positionally from other modules.
        """
        cdef int idx
        cdef double new_mass
        # The psi_const token is reserved: silently returning its index
        # would let an atom template named like it merge into the
        # parameter cell.
        if self.unify_params and desc == PARAM_CELL_TOKEN:
            raise ValueError(
                "add_primitive: %r is the reserved canonical parameter "
                "cell" % desc)
        if desc in self._prim_to_idx:
            return self._prim_to_idx[desc]
        if self.dp_mode:
            new_mass = self.alpha_0 if mass is None else <double> float(mass)
            if new_mass <= 0.0:
                raise ValueError(
                    "add_primitive: mass must be > 0 (got %r)" % mass)
            if new_mass > self.rem_base * (1.0 + 1e-12):
                raise ValueError(
                    "add_primitive: remainder exhausted — requested mass "
                    "%g > unallocated remainder %g. Construct the PolyaTree "
                    "with a larger remainder_mass, or register with a "
                    "smaller per-primitive mass." % (new_mass, self.rem_base))
        else:
            if mass is not None:
                raise ValueError(
                    "add_primitive: custom mass requires DP mode "
                    "(construct with remainder_mass > 0)")
            new_mass = self.alpha_0
        idx = self.n_prims
        self.primitives.append(desc)
        self._prim_to_idx[desc] = idx
        self.n_prims += 1
        self.base_alphas.append(new_mass)
        self.base_reg_total += new_mass
        if self.dp_mode:
            self.rem_base -= new_mass
        # Walk all nodes and append the new cell (carving in dp_mode).
        self._grow_all_nodes(new_mass)
        return idx

    cdef void _grow_all_nodes(self, double new_mass):
        """BFS: append one `new_mass` slot to every node's Dirichlet.
        In dp_mode the same mass is simultaneously removed from the node's
        remainder cell, conserving alpha_total + alpha_rem per node."""
        cdef list stack = [self.root]
        cdef PolyaNode node
        cdef cnp.ndarray new_alphas
        while stack:
            node = stack.pop()
            new_alphas = np.empty(node.n_choices + 1, dtype=np.float64)
            new_alphas[:node.n_choices] = node.alphas
            new_alphas[node.n_choices] = new_mass
            node.alphas = new_alphas
            node.n_choices += 1
            node.alpha_total += new_mass
            if self.dp_mode:
                node.alpha_rem -= new_mass
            for child in node.children.values():
                stack.append(child)

    cpdef add_primitives(self, list descs):
        """Bulk register primitives. Returns a list of their indices."""
        cdef list out = []
        cdef str d
        for d in descs:
            out.append(self.add_primitive(d))
        return out

    cdef void _legacy_extract_recursive(self, function_treeC.Node node, list path):
        cdef str op, desc
        cdef int idx
        if node is None:
            return
        if node.is_leaf():
            # psi_const unification; see _extract_recursive.
            if self.unify_params and node.is_param():
                path.append(('prim', self._param_idx))
                return
            '''vocab_symbol equals the description for plain leaves. For
            opaque atomic leaves it is instead the template's canonical
            symbol: one vocabulary cell per template, NOT per
            instantiation, since instantiated descriptions carry
            tree-specific parameter names that must never enter the
            vocabulary.'''
            desc = node.vocab_symbol()
            idx = self._prim_to_idx.get(desc, -1)
            if idx >= 0:
                path.append(('prim', idx))
            return
        # Template cell, then the slot subtrie. This walker's silent-drop
        # style is preserved for the cell itself.
        if node.is_flex():
            idx = self._prim_to_idx.get(node.flex_spec.name, -1)
            if idx >= 0:
                path.append(('prim', idx))
            # Stay in THIS walker for the slot (silent-drop convention
            # throughout; delegating to the strict walker could raise,
            # contradicting this legacy path's contract). NB dropping an
            # unregistered head changes the path SHAPE, not just one
            # cell -- acceptable only because this walker has no callers.
            self._legacy_extract_recursive(node.get_left(), path)
            return
        op = node.get_op()
        idx = self._op_to_idx.get(op, -1)
        if idx >= 0:
            path.append(('op', idx))
        self._extract_recursive(node.get_left(), path)
        self._extract_recursive(node.get_right(), path)

    cdef void _extract_recursive(self, function_treeC.Node node, list path):
            """No silent drops here. Unknown leaves auto-register by
            default, or raise; unknown ops raise unless explicitly opted
            out."""
            cdef str op, desc
            cdef int idx
            if node is None:
                return
            if node.is_leaf():
                '''Under unify_params every PLAIN parameter leaf is the
                single psi_const cell, since the name is a bound label
                with no prior mass of its own. Opaque leaves keep their
                template cells via vocab_symbol below.'''
                if self.unify_params and node.is_param():
                    path.append(('prim', self._param_idx))
                    return
                # Canonical vocabulary key: the atom template symbol for
                # opaque leaves, the description otherwise.
                desc = node.vocab_symbol()
                idx = self._prim_to_idx.get(desc, -1)
                if idx < 0:
                    if self.auto_register:
                        # Side effect: grows every node's Dirichlet by one
                        # slot. The invariance property makes this
                        # well-defined.
                        idx = self.add_primitive(desc)
                    else:
                        raise KeyError(
                            f"PolyaTree: unknown leaf primitive '{desc}'. "
                            f"Either pre-register via add_primitive(), enable "
                            f"auto_register=True, or check the move that "
                            f"introduced this symbol."
                        )
                path.append(('prim', idx))
                return
            '''ONE cell for the template head, then the slot's
            construction path continues in the child subtrie under that
            cell, which makes slot-specific reinforcement automatic. The
            frozen interior emits with probability 1 and contributes no
            steps. Unreachable on trees without scaffolds.'''
            if node.is_flex():
                desc = node.flex_spec.name
                idx = self._prim_to_idx.get(desc, -1)
                if idx < 0:
                    if self.auto_register:
                        idx = self.add_primitive(desc)
                    else:
                        raise KeyError(
                            f"PolyaTree: unknown flex template '{desc}'. "
                            f"Pre-register it (Forest does this at "
                            f"construction) or enable auto_register."
                        )
                path.append(('prim', idx))
                '''One cell for the template head, then EVERY slot's
                construction path continues in the child subtrie, in
                declaration order (a single slot is the historical
                left-descent, bit-identical).'''
                if node.flex_slots is not None:
                    for _slot in node.flex_slots:
                        self._extract_recursive(_slot, path)
                    return
                self._extract_recursive(node.get_left(), path)
                return
            op = node.get_op()
            idx = self._op_to_idx.get(op, -1)
            if idx < 0:
                if self.strict_ops:
                    raise KeyError(
                        f"PolyaTree: unknown operation '{op}'. The op vocabulary "
                        f"is supposed to be locked at construction; an unknown op "
                        f"here means the move set produces an op outside "
                        f"`operations`. Pass strict_ops=False to silence (NOT "
                        f"recommended)."
                    )
                # Legacy silent-drop path (strict_ops=False only).
            else:
                path.append(('op', idx))
            self._extract_recursive(node.get_left(), path)
            self._extract_recursive(node.get_right(), path)

    cpdef double log_prior_from_tree(self, function_treeC.Node root_node):
        """log pi(T | Theta) computed directly from a GP tree root node."""
        cdef list path = self.extract_path(root_node)
        return self.log_prior(path)

    # ------------------------------------------------------------------
    # Sampling a fresh construction path (Pólya-guided)
    # ------------------------------------------------------------------
    cpdef list sample_construction_path(self, int max_steps=64):
        """
        Sample a construction path purely from the Pólya tree, without
        regard to GP structural constraints. Useful for debugging.
        Returns a list of (kind, choice_idx) tuples.
        """
        cdef list path = []
        cdef PolyaNode current = self.root
        cdef int raw_idx, idx, step
        cdef str kind

        for step in range(max_steps):
            raw_idx = current.sample_choice()
            # Every node uses the unified flat index space:
            #   raw_idx in [0, n_ops)            -> ('op',   raw_idx)
            #   raw_idx in [n_ops, n_ops+n_prims)-> ('prim', raw_idx-n_ops)
            if raw_idx < self.n_ops:
                kind = 'op'
                idx = raw_idx
            else:
                kind = 'prim'
                idx = raw_idx - self.n_ops
            path.append((kind, idx))
            current = self._ensure_child(current, kind, idx)

        return path

        # ------------------------------------------------------------------
    # Explicit vocabulary-coverage check, used by the tests.
    # ------------------------------------------------------------------
    cpdef assert_covers(self, function_treeC.Node root_node):
        """
        Walk `root_node` and raise KeyError on the first leaf description or
        op symbol that is absent from this Pólya tree's vocabulary.

        This is the no-silent-drops guarantee promised by the plan: tests
        should call `polya.assert_covers(tree.get_root())` after every move
        to catch any vocabulary leak before it silently corrupts the prior.

        Note: this is a *read-only* check. It does NOT call add_primitive()
        — that's `extract_path`'s job under auto_register=True.
        """
        self._assert_covers_recursive(root_node)

    cdef _assert_covers_recursive(self, function_treeC.Node node):
        cdef str op, desc
        if node is None:
            return
        if node.is_leaf():
            # Under unify_params a plain parameter leaf is always covered
            # by the psi_const cell.
            if self.unify_params and node.is_param():
                return
            # Check the canonical vocabulary key.
            desc = node.vocab_symbol()
            if desc not in self._prim_to_idx:
                raise KeyError(
                    f"PolyaTree.assert_covers: leaf primitive '{desc}' "
                    f"is not in the vocabulary."
                )
            return
        # The template head must be a registered cell; the slot is
        # checked like any other subtree.
        if node.is_flex():
            if node.flex_spec.name not in self._prim_to_idx:
                raise KeyError(
                    f"PolyaTree.assert_covers: flex template "
                    f"'{node.flex_spec.name}' is not in the vocabulary."
                )
            self._assert_covers_recursive(node.get_left())
            return
        op = node.get_op()
        if op not in self._op_to_idx:
            raise KeyError(
                f"PolyaTree.assert_covers: operation '{op}' is not in the "
                f"vocabulary."
            )
        self._assert_covers_recursive(node.get_left())
        self._assert_covers_recursive(node.get_right())

    # ------------------------------------------------------------------
    # Picklable state snapshot / restore, used by MPI sampling.
    #
    # The MPI-parallel EMC loop keeps one PolyaTree REPLICA per worker
    # process and keeps them bit-identical to the master's by (a) shipping
    # the master's full state once at startup via get_state()/set_state()
    # and (b) replaying the same ordered list of accepted construction
    # paths on every copy after each round (deterministic replay). Both
    # methods are pure-Python-object walks over the public fields, so the
    # returned state contains only dict/list/str/int/float/ndarray — all
    # picklable by mpi4py's default (pickle-based) communication.
    # ------------------------------------------------------------------
    cpdef dict get_state(self):
        """Deep, picklable snapshot of the whole Pólya tree (vocabulary,
        urn counter t, and every node's Dirichlet alphas + children),
        including the DP-mode fields."""
        return {
            'operations': list(self.operations),
            'primitives': list(self.primitives),
            'alpha_0': float(self.alpha_0),
            't': int(self.t),
            'auto_register': bool(self.auto_register),
            'strict_ops': bool(self.strict_ops),
            'root': _node_to_state(self.root),
            'dp_mode': bool(self.dp_mode),
            'rem_base': float(self.rem_base),
            'base_alphas': list(self.base_alphas),
            # psi_const unification travels with the state so MPI
            # replicas map parameter leaves identically.
            'unify_params': bool(self.unify_params),
            # The frozen-regime flag travels too; older snapshots
            # restore with the False default in set_state.
            'adapt_frozen': bool(self.adapt_frozen),
        }

    cpdef void set_state(self, dict state) except *:
        """Restore a snapshot produced by get_state(). Overwrites the
        vocabulary, counter and the entire node tree in place. DP-mode
        fields are read with safe defaults, so older snapshots restore
        as ordinary non-DP trees."""
        self.operations = list(state['operations'])
        self.primitives = list(state['primitives'])
        self.alpha_0 = <double> state['alpha_0']
        self.t = <int> state['t']
        self.auto_register = <bint> state['auto_register']
        self.strict_ops = <bint> state['strict_ops']
        # Safe default: older snapshots lack this key.
        self.adapt_frozen = <bint> state.get('adapt_frozen', False)
        self.n_ops = len(self.operations)
        self.n_prims = len(self.primitives)
        self._op_to_idx = {op: i for i, op in enumerate(self.operations)}
        self._prim_to_idx = {p: i for i, p in enumerate(self.primitives)}
        self.dp_mode = <bint> state.get('dp_mode', False)
        self.rem_base = <double> state.get('rem_base', 0.0)
        self.base_alphas = list(state.get(
            'base_alphas',
            [self.alpha_0] * (self.n_ops + self.n_prims)))
        self.base_reg_total = float(sum(self.base_alphas))
        # Safe default False; when unification was ever enabled the
        # token is already inside state['primitives'].
        self.unify_params = <bint> state.get('unify_params', False)
        self._param_idx = self._prim_to_idx.get(PARAM_CELL_TOKEN, -1) \
            if self.unify_params else -1
        self.root = _node_from_state(state['root'])

    cpdef void replay_paths(self, list paths) except *:
        """Replay a list of accepted construction paths (each a list of
        (kind, idx) tuples) through update_path IN ORDER. Used by the MPI
        deterministic-replay sync: every replica replays the identical
        ordered list, so all copies stay bit-identical."""
        cdef list p
        for p in paths:
            self.update_path(p)


cdef dict _node_to_state(PolyaNode node):
    """Recursive picklable snapshot of one PolyaNode."""
    return {
        'kind': node.kind,
        'n_choices': int(node.n_choices),
        'depth': int(node.depth),
        'alphas': np.asarray(node.alphas, dtype=np.float64).copy(),
        'alpha_total': float(node.alpha_total),
        'alpha_rem': float(node.alpha_rem),
        'children': {key: _node_to_state(child)
                     for key, child in node.children.items()},
    }


cdef PolyaNode _node_from_state(dict s):
    """Rebuild a PolyaNode (and its subtree) from _node_to_state output.
    'alpha_rem' read with a legacy-safe default (pre-2.10 snapshots)."""
    cdef PolyaNode node = PolyaNode(
        <str> s['kind'], <int> s['n_choices'], <int> s['depth'], alpha_0=1.0)
    node.alphas = np.asarray(s['alphas'], dtype=np.float64).copy()
    node.alpha_total = <double> s['alpha_total']
    node.alpha_rem = <double> s.get('alpha_rem', 0.0)
    node.children = {key: _node_from_state(child)
                     for key, child in (<dict> s['children']).items()}
    return node
