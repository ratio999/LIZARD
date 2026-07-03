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
                 bint auto_register=True, bint strict_ops=True):
        self.operations = _OPERATIONS
        self.primitives = list(primitives)
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
        self.root = PolyaNode(
            'root', self.n_ops + self.n_prims, depth=0, alpha_0=alpha_0
        )

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
    cdef double log_prior(self, list construction_path):
        """
        Compute log pi(T | Theta) for a tree with given construction path.

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
                # Unvisited region: symmetric default over the combined space.
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
    cpdef int add_primitive(self, str desc):
        """
        Register a new primitive description in the Pólya vocabulary.
        Grows EVERY existing Pólya node's Dirichlet by appending a fresh
        alpha_0 slot at the end of the prim block. Existing children stay
        addressable because their root_index = n_ops + choice_idx and we
        only append (we don't shift).
        Returns the (possibly pre-existing) primitive index.
        """
        cdef int idx
        if desc in self._prim_to_idx:
            return self._prim_to_idx[desc]
        idx = self.n_prims
        self.primitives.append(desc)
        self._prim_to_idx[desc] = idx
        self.n_prims += 1
        # Walk all nodes and append a fresh alpha_0 slot.
        self._grow_all_nodes()
        return idx

    cdef void _grow_all_nodes(self):
        """BFS: append one alpha_0 slot to every node's Dirichlet."""
        cdef list stack = [self.root]
        cdef PolyaNode node
        cdef cnp.ndarray new_alphas
        while stack:
            node = stack.pop()
            new_alphas = np.empty(node.n_choices + 1, dtype=np.float64)
            new_alphas[:node.n_choices] = node.alphas
            new_alphas[node.n_choices] = self.alpha_0
            node.alphas = new_alphas
            node.n_choices += 1
            node.alpha_total += self.alpha_0
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
            desc = node.get_fun().get_description()
            idx = self._prim_to_idx.get(desc, -1)
            if idx >= 0:
                path.append(('prim', idx))
            return
        op = node.get_op()
        idx = self._op_to_idx.get(op, -1)
        if idx >= 0:
            path.append(('op', idx))
        self._extract_recursive(node.get_left(), path)
        self._extract_recursive(node.get_right(), path)

    cdef void _extract_recursive(self, function_treeC.Node node, list path):
            """Phase 2.3: no silent drops. Unknown leaves auto-register (default)
            or raise; unknown ops raise unless explicitly opted out."""
            cdef str op, desc
            cdef int idx
            if node is None:
                return
            if node.is_leaf():
                desc = node.get_fun().get_description()
                idx = self._prim_to_idx.get(desc, -1)
                if idx < 0:
                    if self.auto_register:
                        # Side effect: grows every node's Dirichlet by one slot.
                        # Phase 1.5 invariance ensures this is well-defined.
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
    # Phase 2.3: explicit vocabulary-coverage check (for tests)
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
            desc = node.get_fun().get_description()
            if desc not in self._prim_to_idx:
                raise KeyError(
                    f"PolyaTree.assert_covers: leaf primitive '{desc}' "
                    f"is not in the vocabulary."
                )
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
    # Phase 2.12 (MPI sampling): picklable state snapshot / restore.
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
        urn counter t, and every node's Dirichlet alphas + children)."""
        return {
            'operations': list(self.operations),
            'primitives': list(self.primitives),
            'alpha_0': float(self.alpha_0),
            't': int(self.t),
            'auto_register': bool(self.auto_register),
            'strict_ops': bool(self.strict_ops),
            'root': _node_to_state(self.root),
        }

    cpdef void set_state(self, dict state) except *:
        """Restore a snapshot produced by get_state(). Overwrites the
        vocabulary, counter and the entire node tree in place."""
        self.operations = list(state['operations'])
        self.primitives = list(state['primitives'])
        self.alpha_0 = <double> state['alpha_0']
        self.t = <int> state['t']
        self.auto_register = <bint> state['auto_register']
        self.strict_ops = <bint> state['strict_ops']
        self.n_ops = len(self.operations)
        self.n_prims = len(self.primitives)
        self._op_to_idx = {op: i for i, op in enumerate(self.operations)}
        self._prim_to_idx = {p: i for i, p in enumerate(self.primitives)}
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
        'children': {key: _node_to_state(child)
                     for key, child in node.children.items()},
    }


cdef PolyaNode _node_from_state(dict s):
    """Rebuild a PolyaNode (and its subtree) from _node_to_state output."""
    cdef PolyaNode node = PolyaNode(
        <str> s['kind'], <int> s['n_choices'], <int> s['depth'], alpha_0=1.0)
    node.alphas = np.asarray(s['alphas'], dtype=np.float64).copy()
    node.alpha_total = <double> s['alpha_total']
    node.children = {key: _node_from_state(child)
                     for key, child in (<dict> s['children']).items()}
    return node
