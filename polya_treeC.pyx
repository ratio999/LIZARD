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

    cpdef double log_prob(self, int choice_idx):
        """log P(choice = k | this node) under current Dirichlet mean."""
        return log(self.alphas[choice_idx]) - log(self.alpha_total)

    cpdef int sample_choice(self):
        """Sample a choice index from current Dirichlet-Multinomial mean."""
        cdef cnp.ndarray probs = self.alphas / self.alpha_total
        return int(np.random.choice(self.n_choices, p=probs))

    cpdef void update(self, int choice_idx, double gamma):
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

    def __init__(self, list operations, list primitives, double alpha_0=1.0):
        self.operations = list(operations)
        self.primitives = list(primitives)
        self.alpha_0 = alpha_0
        self.t = 0
        self.n_ops = len(self.operations)
        self.n_prims = len(self.primitives)
        # Lookups
        self._op_to_idx = {op: i for i, op in enumerate(self.operations)}
        self._prim_to_idx = {p: i for i, p in enumerate(self.primitives)}
        # Root: a "virtual" decision over n_ops + n_prims (op vs prim + which one).
        self.root = PolyaNode(
            'root', self.n_ops + self.n_prims, depth=0, alpha_0=alpha_0
        )

    # ------------------------------------------------------------------
    # Tree navigation (lazy construction)
    # ------------------------------------------------------------------
    cdef PolyaNode _ensure_child(self, PolyaNode parent, str kind, int choice_idx):
        """Create child PolyaNode if absent. Mutates the tree."""
        cdef tuple key = (kind, choice_idx)
        cdef PolyaNode child
        cdef int n
        if key not in parent.children:
            n = self.n_ops if kind == 'op' else self.n_prims
            child = PolyaNode(kind, n, depth=parent.depth + 1,
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
    cpdef double log_prior(self, list construction_path):
        """
        Compute log pi(T | Theta) for a tree with given construction path.

        Pure function of the current Pólya tree: does NOT create new nodes.
        Steps that fall in unvisited subtrees contribute -log(n_kind) under
        the symmetric default Dirichlet (since alpha_k / sum = 1/n_kind).
        """
        cdef double log_p = 0.0
        cdef PolyaNode current = self.root
        cdef PolyaNode nxt
        cdef int idx, root_idx, n
        cdef str kind
        cdef tuple step

        for step in construction_path:
            kind = <str> step[0]
            idx = <int> step[1]

            if current is None:
                # We are walking an unvisited region: symmetric default.
                n = self.n_ops if kind == 'op' else self.n_prims
                log_p += -log(<double> n)
                continue

            if current is self.root:
                root_idx = self._root_index(kind, idx)
                log_p += current.log_prob(root_idx)
            else:
                if kind != current.kind:
                    n = self.n_ops if kind == 'op' else self.n_prims
                    log_p += -log(<double> n)
                else:
                    log_p += current.log_prob(idx)

            nxt = self._get_child_or_none(current, kind, idx)
            current = nxt  # may be None: subsequent steps use defaults

        return log_p

    # ------------------------------------------------------------------
    # Posterior update (call only on accepted MH steps)
    # ------------------------------------------------------------------
    cpdef void update_path(self, list construction_path):
        """
        Reinforce path with diminishing learning rate gamma_t = 1/(1+t).
        Call AFTER an accepted MH step.
        """
        cdef double gamma = 1.0 / (1.0 + <double> self.t)
        cdef PolyaNode current = self.root
        cdef int idx, root_idx
        cdef str kind
        cdef tuple step

        for step in construction_path:
            kind = <str> step[0]
            idx = <int> step[1]

            if current is self.root:
                root_idx = self._root_index(kind, idx)
                current.update(root_idx, gamma)
            else:
                if kind != current.kind:
                    # Defensive: structural mismatch, abort further updates.
                    break
                current.update(idx, gamma)

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
        Grows the root Dirichlet by appending a fresh alpha_0 slot.
        Existing 'prim' children of the root are unaffected because their
        root_index is `n_ops + choice_idx` and we only append at the end.
        Returns the (possibly pre-existing) primitive index.
        """
        cdef int idx
        cdef cnp.ndarray new_alphas
        if desc in self._prim_to_idx:
            return self._prim_to_idx[desc]
        idx = self.n_prims
        self.primitives.append(desc)
        self._prim_to_idx[desc] = idx
        self.n_prims += 1
        # Grow root.alphas by one slot at value alpha_0.
        new_alphas = np.empty(self.root.n_choices + 1, dtype=np.float64)
        new_alphas[:self.root.n_choices] = self.root.alphas
        new_alphas[self.root.n_choices] = self.alpha_0
        self.root.alphas = new_alphas
        self.root.n_choices += 1
        self.root.alpha_total += self.alpha_0
        return idx

    cpdef add_primitives(self, list descs):
        """Bulk register primitives. Returns a list of their indices."""
        cdef list out = []
        cdef str d
        for d in descs:
            out.append(self.add_primitive(d))
        return out

    cdef void _extract_recursive(self, function_treeC.Node node, list path):
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
            if current is self.root:
                if raw_idx < self.n_ops:
                    kind = 'op'
                    idx = raw_idx
                else:
                    kind = 'prim'
                    idx = raw_idx - self.n_ops
            else:
                kind = current.kind
                idx = raw_idx
            path.append((kind, idx))
            current = self._ensure_child(current, kind, idx)

        return path
