import random
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
cimport function_treeC
#from function_treeC import build_tree
cimport real_functionC
cimport numpy as cnp


# --- Phase 2.5: per-Grove RNG plumbing ------------------------------------
# _PY_RNG / _NP_RNG are module-level handles read by every move that
# needs randomness. Defaulting to the bare `random` and `np.random`
# modules keeps the pre-2.5 behaviour bit-exact when no caller invokes
# set_rng (the module-level functions are just methods on the global
# random.Random / numpy default Generator). Grove.__init__ calls
# set_rng(self._py_rng, self._np_rng) at construction; tests can call
# set_rng directly with their own seeded handles.
_PY_RNG = random
_NP_RNG = np.random

cpdef void set_rng(object py_rng, object np_rng) except *:
    """Install the (Python, NumPy) RNG handles consumed by every move.

    Both arguments must expose the same call surface as `random` /
    `np.random` respectively (i.e. `random.Random` / `numpy.random.Generator`
    instances are the canonical choices). Pass `random` / `np.random`
    themselves to restore the unseeded global behaviour.
    """
    global _PY_RNG, _NP_RNG
    if py_rng is None or np_rng is None:
        raise ValueError("set_rng: py_rng and np_rng must both be non-None")
    _PY_RNG = py_rng
    _NP_RNG = np_rng

cpdef tuple get_rng():
    return (_PY_RNG, _NP_RNG)


# --- Phase 2.1.PM: birth density g for new parameters ---------------------
# g(theta) = Normal(0, _G_TAU^2). Set per-run via set_g_tau(std(y)).
# Tree-level plumbing of tau (analogous to sigma) is deferred; this module
# global is the temporary single source of truth.

cdef double _KIND_CROSS_PROB = 0.5      # mixture weight for sub-move B (BIRTH/DEATH)

cdef inline double _sample_g(double g_tau):
    """Draw theta ~ N(0, _G_TAU^2)."""
    return _PY_RNG.gauss(0.0, g_tau)

cdef inline double _log_g(double theta, double g_tau):
    """log N(theta; 0, _G_TAU^2)."""
    cdef double t2 = g_tau * g_tau
    return -0.5 * np.log(2.0 * np.pi * t2) - 0.5 * theta * theta / t2


# --- Phase 2.1.PR: Adaptive-Metropolis parameter refresh -------------------
# Constants per plan decisions D5/D7.
cdef double _AM_TARGET_ACCEPT_LARGE_K = 0.234   # Roberts-Gelman-Gilks 1997
cdef double _AM_TARGET_ACCEPT_K1      = 0.44
cdef double _AM_KAPPA                 = 0.6     # Robbins-Monro decay  gamma_t = t^{-kappa}
cdef double _AM_REL_SCALE             = 0.1     # cold-start: |theta_i| * rel
cdef double _AM_EPS                   = 1e-6    # cold-start floor
cdef double _AM_JITTER                = 1e-8  

# --- Phase 2.1.K: add_node / hoist RJMCMC bookkeeping ---------------------
cdef double _ADD_NODE_DEPTH_RHO  = 0.5    # truncated-geometric over depths
cdef int    _ADD_NODE_DEPTH_MAX  = 4      # truncation; reverse hoists of
                                          # subtrees deeper than this give
                                          # log_q_rev = -inf (off-support).
cdef double _ADD_NODE_PARAM_PROB = 0.3    # per-leaf Bernoulli prob to be a
                                          # parameter leaf in the new subtree

# ---------------------------------------------------------------------------
# Helpers for proposal-density bookkeeping
# ---------------------------------------------------------------------------

cdef list _flatten_nodes(function_treeC.Node node, list out):
    """Pre-order DFS into `out`, returns `out`. Lists every node exactly once."""
    if node is None:
        return out
    out.append(node)
    if not node.is_leaf():
        _flatten_nodes(node.left,  out)
        _flatten_nodes(node.right, out)
    return out

cpdef list flatten_nodes(function_treeC.Node node, list out=None):
    """Wrapper around _flatten_nodes that creates the output list."""
    if out is None:
        out = []
    return _flatten_nodes(node, out)


cdef function_treeC.Node _select_uniform_node(function_treeC.Tree tree):
    """Pick one node of `tree` uniformly at random over all nodes."""
    cdef list all_nodes = _flatten_nodes(tree.root, [])
    return all_nodes[_PY_RNG.randrange(len(all_nodes))]


cdef bint _is_perfect_binary(function_treeC.Node node, int h):
    """True iff `node` is a perfect binary tree of height exactly h.

    h == 0 -> node must be a leaf.
    h  > 0 -> node must be internal with both children perfect of height h-1.
    """
    if node is None:
        return False
    if h == 0:
        return node.is_leaf()
    if node.is_leaf():
        return False
    if node.left is None or node.right is None:
        return False
    return _is_perfect_binary(node.left, h - 1) and _is_perfect_binary(node.right, h - 1)

cdef tuple _sample_truncated_geometric(double rho, int d_max):
    """Sample d ~ truncated Geom(rho) on {1,...,d_max}.
    Returns (d, log_p_d(d))."""
    cdef double Z = (1.0 - rho ** d_max) / (1.0 - rho)
    cdef double u = _PY_RNG.random()
    cdef double cdf = 0.0
    cdef double pmf
    cdef int d
    for d in range(1, d_max + 1):
        pmf = (rho ** (d - 1)) / Z
        cdf += pmf
        if u < cdf:
            return (d, np.log(pmf))
    return (d_max, np.log((rho ** (d_max - 1)) / Z))


cdef double _truncated_geometric_log_pmf(int d, double rho, int d_max):
    """log p(d) under truncated Geom on {1,...,d_max}. -inf if out of range."""
    if d < 1 or d > d_max:
        return -np.inf
    cdef double Z = (1.0 - rho ** d_max) / (1.0 - rho)
    return (d - 1) * np.log(rho) - np.log(Z)


cdef int _count_nodes(function_treeC.Node node):
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + _count_nodes(node.left) + _count_nodes(node.right)


cdef double _eval_build_density(function_treeC.Node S, int d,
                                int M, int O, double p_par, double g_tau):
    """log q_build(S | d) for a perfect-binary subtree of height d.
    Caller must verify _is_perfect_binary(S, d) before calling.

    Per-leaf Bernoulli(p_par) for parameter-vs-function, uniform from F
    for function leaves, uniform from O for ops, and log g(theta_i) per
    parameter leaf evaluated at its current value."""
    cdef int n_leaves = 1 << d           # 2^d
    cdef int n_int    = n_leaves - 1     # 2^d - 1
    cdef list param_leaves = _collect_parameter_leaves(S, [])
    cdef int K_S = len(param_leaves)
    cdef double log_g_sum = 0.0
    cdef function_treeC.Node leaf
    for leaf in param_leaves:
        log_g_sum += _log_g(<double> leaf.value, g_tau)
    return (
        (<double> K_S) * np.log(p_par) + log_g_sum
        + (<double> (n_leaves - K_S)) * (np.log(1.0 - p_par) - np.log(<double> M))
        - (<double> n_int) * np.log(<double> O)
    )


cdef function_treeC.Node _build_subtree_with_params(
        int d, list functions, list operations, object domain,
        double p_par, list existing_param_names, double g_tau):
    """Build a perfect-binary subtree of height d with per-leaf Bernoulli
    parameter assignment. Parameter names extend `existing_param_names`."""
    cdef function_treeC.Node root = function_treeC.build_tree(d, functions, domain)
    cdef list leaves = root.get_leaves(leaves_list=[])
    cdef list pool = list(existing_param_names)
    cdef function_treeC.Node leaf
    cdef str new_name
    for leaf in leaves:
        if _PY_RNG.random() < p_par:
            new_name = function_treeC.new_par(pool, "a")
            pool.append(new_name)
            leaf.function = real_functionC.RealFunctionC.from_string(
                new_name, domain, compute_grad=False)
            leaf.bis_param = leaf.is_param()
            # Phase 2.1.PM: birth from g = N(0, tau^2). Same draw is what
            # _eval_build_density(...) reads back via leaf.value.
            leaf.set_value(_sample_g(g_tau))
    function_treeC._post_order_update(root)
    return root

cdef list _collect_parameter_leaves(function_treeC.Node node, list out):
    """Pre-order DFS collecting parameter (constant-leaf) Nodes into `out`."""
    if node is None:
        return out
    if node.is_leaf():
        if node.is_param():
            out.append(node)
        return out
    _collect_parameter_leaves(node.left,  out)
    _collect_parameter_leaves(node.right, out)
    return out

cdef int _count_selectable(function_treeC.Node node):
    """Count nodes that are valid selection targets for subtree_mutation:
    every node that is NOT a parameter leaf."""
    if node is None:
        return 0
    if node.is_leaf():
        return 0 if node.is_param() else 1
    return 1 + _count_selectable(node.left) + _count_selectable(node.right)

@cython.boundscheck(True)
cdef tuple _subtree_mutation(function_treeC.Tree tree, list functions):
    """
    Replace a uniformly-selected node's subtree with a freshly-built subtree
    of the same height. Mutates `tree` in place. The caller is responsible
    for deepcopying `tree` BEFORE calling this if the original must survive.

    Parameter recycling (v1.5):
        If the OLD subtree contains K parameter leaves with names {a_{i_1}, ...,
        a_{i_K}} and values {v_1, ..., v_K}, the NEW subtree built here is a
        fresh perfect-binary tree of height d_max with Npars=0; we then choose
        K leaf positions uniformly at random and inject the recycled
        (name, value) parameters in name-sorted order. This keeps the
        symbol vocabulary fixed (no Polya tree hookup needed for this move).

    Returns
    -------
    (log_q_fwd, log_q_rev) : (double, double)
        Forward and reverse log proposal densities, up to a common additive
        constant that cancels in the MH ratio. The Metropolis-Hastings
        acceptance ratio in groveC._step must include the term
            log_q_rev - log_q_fwd
        in addition to the (tempered) posterior + prior contributions.
        IMPORTANT: tempering applies ONLY to (posterior + prior); the
        proposal-correction bracket above is NOT scaled by 1/T.

    Mathematical content (uniform-selection + parameter-recycling)
    --------------------------------------------------------------
        q(T'|T) = (1/|T|)  * |F|^(-(2^d - K)) * |Omega|^(-(2^d - 1))
                          * C(2^d, K)^(-1)
        q(T|T') = (1/|T'|) * |F|^(-(2^d - K)) * |Omega|^(-(2^d - 1))
                          * C(2^d, K)^(-1)
    so on the support (S perfect-binary of height d_max, K matches):
        log q(T|T') - log q(T'|T) = log|T| - log|T'|.
    Off support, log_q_rev = -inf (forced reject); irreducibility is
    recovered by mixing with other moves.

    Restrictions of this v1.5 implementation:
      * random_depth removed (was not p_md-symmetric).
      * No NEW parameter symbols introduced; K of the tree is preserved by
        this move (it can change via add_node / hoist_mutation).
    """
    cdef function_treeC.Node v, parent_node, new_subtree, old_subtree, n
    cdef function_treeC.Node target_leaf, parent_of_leaf, new_param_node, src
    cdef list old_param_nodes, old_param_nodes_sorted, new_leaves, chosen_slots
    cdef int d_max, size_T, size_Tp, K, n_leaves, i
    cdef double log_q_fwd = 0.0
    cdef double log_q_rev = 0.0
    if tree.root.is_leaf():
        #print('actually addnode')
        
        return add_node(tree, functions)
    parent_node = None
    cdef bint is_par = True
    # ---- 1. Uniform node selection (root allowed) ------------------------
    # Reject only parameter leaves (the build/recycle math is undefined for
    # them); the root is a legal target -- step 5 handles its splice. The
    # previous loop used `parent_node is None` as the exit condition, which
    # silently excluded the root and produced an infinite loop on any tree
    # whose only non-parameter node is the root (e.g. `a0 + a1`).
    v = _select_uniform_node(tree)
    while v.is_param():
        v = _select_uniform_node(tree)
    parent_node = tree.root.find_parent(v)   # None iff v is tree.root
    size_T = _count_selectable(tree.root)

    # ---- 2. Inspect OLD subtree -----------------------------------------
    old_subtree = v
    d_max = v.depth()                         # height of the subtree at v
    old_param_nodes = _collect_parameter_leaves(old_subtree, [])
    K = len(old_param_nodes)

    # ---- 3. Build a fresh non-parametric subtree of height d_max --------
    new_subtree = function_treeC.build_tree(d_max, functions, tree.domain)

    # ---- 4. Recycle K parameters into uniformly chosen leaf positions ---
    if K > 0:
        new_leaves = new_subtree.get_leaves(leaves_list=[])
        n_leaves = len(new_leaves)
        # K <= 2^d_max == n_leaves always (build_tree returns perfect binary).
        chosen_slots = sorted(_PY_RNG.sample(range(n_leaves), K))
        old_param_nodes_sorted = sorted(
            old_param_nodes,
            key=lambda n: n.get_fun().get_description(),
        )
        for i in range(K):
            target_leaf = new_leaves[chosen_slots[i]]
            src = old_param_nodes_sorted[i]
            new_param_node = function_treeC.Node(
                function=real_functionC.RealFunctionC.from_string(
                    src.function.get_description(),
                    tree.domain,
                    compute_grad=False,
                ),
                value=src.value,
            )
            if target_leaf is new_subtree:
                # d_max == 0 case: the "subtree" was a single leaf.
                new_subtree = new_param_node
            else:
                parent_of_leaf = new_subtree.find_parent(target_leaf)
                if parent_of_leaf.get_left() is target_leaf:
                    parent_of_leaf.assign_left(new_param_node)
                else:
                    parent_of_leaf.assign_right(new_param_node)
        # Rebuild internal functions of new_subtree bottom-up so that
        # ancestor functions reflect the recycled leaves (assign_left/right
        # only swap pointers; they do not call update_func).
        function_treeC._post_order_update(new_subtree)

    # ---- 5. Splice -------------------------------------------------------
    if parent_node is None:
        tree.root = new_subtree
        tree.tree_update(tree.root)
    else:
        if parent_node.get_left() is v:
            parent_node.assign_left(new_subtree)
        elif parent_node.get_right() is v:
            parent_node.assign_right(new_subtree)
        tree.tree_update(parent_node)
    '''print("root.op_symbol =", tree.root.op_symbol)
    print("root.left.descr =", tree.tree_get_description())
    print("root.right.descr =", tree.tree_get_description())
    print("root.descr      =", tree.tree_get_description())'''
    # ---- 6. MH proposal-density correction ------------------------------
    size_Tp = _count_selectable(tree.root)
    # Selection-density term (always finite):
    log_q_fwd += -np.log(<double> size_T)
    log_q_rev += -np.log(<double> size_Tp)

    # Build-density support: S must be perfect-binary of height d_max.
    # Within support, the shape, K-subset, names+values factors cancel
    # (K is preserved by recycling; names+values are observed, not sampled).
    if not _is_perfect_binary(old_subtree, d_max):
        log_q_rev = -np.inf

    return log_q_fwd, log_q_rev

@cython.boundscheck(True)
cpdef tuple subtree_mutation(function_treeC.Tree tree, list functions=None):
    if functions is None:
        functions = tree.functions
    return _subtree_mutation(tree, functions)

@cython.boundscheck(True) 
cpdef legacy_subtree_mutation( function_treeC.Tree tree, functions, random_depth=False):
    """Substitutes a subtree with a random-generated new subtree: old version (but works)"""
    if tree.root.is_leaf():
        #print('actually addnode')
        add_node(tree, functions, random_depth=random_depth)
        return
    parent_node = None
    #print('starting while')
    while parent_node is None:
        node_to_replace = tree.root.select_random_node()
        parent_node = tree.root.find_parent(node_to_replace)
    cdef str parent_op = parent_node.get_op()
    if not random_depth:
        max_depth = node_to_replace.depth()
    else:
        #max_depth = max(1, int(random.expovariate(1/2)))
        max_depth = _PY_RNG.choice(list(np.arange(1, node_to_replace.depth()+2)))
    cdef function_treeC.Node new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)

    if parent_node is None:
        tree.root = new_subtree
    elif parent_node.get_left() == node_to_replace:
        parent_node.assign_left(new_subtree)
        
    elif parent_node.get_right() == node_to_replace:
        parent_node.assign_right(new_subtree)
        
    #print('op_new: ', new_subtree.function.get_description(),new_subtree.depth(), new_subtree.is_leaf())
    
    tree.tree_update(parent_node)
    #print('nuova descr: ', tree.tree_get_description())

@cython.boundscheck(True)
cpdef legacy_subtree_crossover( function_treeC.Tree parent1, function_treeC.Tree parent2):
    """Crossover between two subtrees from different trees"""
    #cdef function_treeC.Node node1 = parent1.root.select_random_node()
    #cdef function_treeC.Node node2 = parent2.root.select_random_node()
    #cdef function_treeC.Node parent_node = parent1.root.find_parent(node1)
    #print("Prima del crossover:", node1.function.get_description(), node2.function.get_description())
    cdef function_treeC.Tree  old2 = parent2
    rename_prefix(old2)
    
    cdef function_treeC.Node node1 = parent1.root.select_random_node()
    cdef function_treeC.Node node2 = old2.root.select_random_node()
    cdef function_treeC.Node parent_node = parent1.root.find_parent(node1)
    
    if parent_node is None:
        parent1.root = node2
    elif parent_node.left == node1:
        parent_node.left = node2
    elif parent_node.right == node1:
        parent_node.right = node2
    parent1.tree_update(parent_node)
    
    refresh_params(parent1)

# ---------------------------------------------------------------------------
# Phase 2.1.K v4: Symmetric subtree_crossover (RJMCMC-aware)
# ---------------------------------------------------------------------------
@cython.boundscheck(True)
cdef tuple _subtree_crossover(function_treeC.Tree parent1,
                              function_treeC.Tree parent2):
    """
    subtree_crossover v2 (Phase 2.1.K). SYMMETRIC structural swap.

    Pick v ~ Uniform(N(T_a)) and w ~ Uniform(N(T_b)). Let S_v, S_w be the
    subtrees rooted at v, w. The proposal is the deterministic swap
        T_a -> T_a' = T_a[v <- S_w]
        T_b -> T_b' = T_b[w <- S_v].

    Why no build density / no Jacobian / no g(.)
    --------------------------------------------
    Crossover relocates existing material; it does NOT sample any new
    structure, and does NOT draw any new parameter values. Therefore the
    proposal correction reduces to the node-count ratio derived in the
    Phase 2.1.K v4 plan:
        log q_fwd = - log |N(T_a)|  - log |N(T_b)|
        log q_rev = - log |N(T_a')| - log |N(T_b')|
        log q_rev - log q_fwd = log[ |N(T_a)| |N(T_b)|
                                     / ( |N(T_a')| |N(T_b')| ) ]
    Re-applying crossover to (root-of-S_w in T_a', root-of-S_v in T_b')
    restores (T_a, T_b), so this swap is its own inverse on the chosen
    pair, which is what makes the reverse density well-defined.

    Parameter-name handling
    -----------------------
    BEFORE the swap, we move T_b's parameters into the disjoint 'b'
    namespace via rename_prefix(parent2, 'b'). After the swap, T_a' and
    T_b' each contain a mix of 'a' and 'b' labels. refresh_params on
    each tree re-canonicalizes every 'b' to a fresh 'a' index past the
    tree's current max. Renaming is a deterministic bijection on label
    space and therefore contributes 0 to the proposal correction.

    NOTE on selection-mix balance: same TODO[move-pair-balance] as the
    other K-changing moves. Phase 2.8.

    SIDE EFFECTS: parent1 AND parent2 are both mutated in place. The
    caller (typically Grove._step) must deepcopy if originals are needed.

    Returns (log_q_fwd, log_q_rev).
    """
    cdef function_treeC.Node v, w, parent_v, parent_w
    cdef list nodes_a, nodes_b
    cdef int n_int_a, n_int_b, n_int_a_p, n_int_b_p
    cdef double log_q_fwd, log_q_rev

    if parent1.root is None or parent2.root is None:
        return (0.0, 0.0)

    # ---- 1. Disjoint label spaces ---------------------------------------
    # Every parameter currently in parent2 becomes 'b<k>'. parent1 keeps
    # its 'a<k>'. After the swap the post-refresh step will resolve any
    # 'b' that landed in either tree to fresh 'a' indices.
    rename_prefix(parent2, 'b')

    # ---- 2. Uniform node selection in each tree -------------------------
    nodes_a = [n for n in _flatten_nodes(parent1.root, []) if not n.is_leaf()]
    nodes_b = [n for n in _flatten_nodes(parent2.root, []) if not n.is_leaf()]
    n_int_a = len(nodes_a)
    n_int_b = len(nodes_b)
    if n_int_a == 0 or n_int_b == 0:
        return (0.0, 0.0)
    v = nodes_a[_PY_RNG.randrange(n_int_a)]
    w = nodes_b[_PY_RNG.randrange(n_int_b)]

    parent_v = parent1.root.find_parent(v)   # None iff v is parent1.root
    parent_w = parent2.root.find_parent(w)   # None iff w is parent2.root

    # ---- 3. Perform the swap --------------------------------------------
    # v and w live in disjoint trees, so there is no aliasing risk: we
    # can rewire parent1 first, then parent2, in either order.
    if parent_v is None:
        parent1.root = w
    else:
        if parent_v.get_left() is v:
            parent_v.assign_left(w)
        else:
            parent_v.assign_right(w)

    if parent_w is None:
        parent2.root = v
    else:
        if parent_w.get_left() is w:
            parent_w.assign_left(v)
        else:
            parent_w.assign_right(v)

    # ---- 4. Rebuild both trees & refresh parameter labels ---------------
    # tree_update walks UP from the splice point recomputing functions
    # and refreshing leaves/params/name_values. After the swap both trees
    # need it; passing the (now updated) parent slot is the natural entry.
    if parent_v is None:
        parent1.tree_update(parent1.root)
    else:
        parent1.tree_update(parent_v)
    if parent_w is None:
        parent2.tree_update(parent2.root)
    else:
        parent2.tree_update(parent_w)

    # Re-canonicalize 'b<k>' -> 'a<new>' in each tree, continuing past the
    # tree's current max 'a' index. refresh_params is a no-op when no 'b'
    # labels remain, so calling on both trees is always safe.
    #refresh_params(parent1)
    #refresh_params(parent2)
    _compact_param_labels(parent1)
    _compact_param_labels(parent2)
    # ---- 5. Proposal-density correction ---------------------------------
    n_int_a_p = sum(1 for n in _flatten_nodes(parent1.root, []) if not n.is_leaf())
    n_int_b_p = sum(1 for n in _flatten_nodes(parent2.root, []) if not n.is_leaf())
    log_q_fwd = -np.log(<double> n_int_a) - np.log(<double> n_int_b)
    if n_int_a_p <= 0 or n_int_b_p <= 0:
        # Defensive: should be impossible since both swapped-in subtrees
        # were internal nodes themselves.
        log_q_rev = -np.inf
    else:
        log_q_rev = -np.log(<double> n_int_a_p) - np.log(<double> n_int_b_p)


    return (log_q_fwd, log_q_rev)


@cython.boundscheck(True)
cpdef tuple subtree_crossover(function_treeC.Tree parent1,
                              function_treeC.Tree parent2):
    """Phase 2.1.K v4 public entry point. Returns (log_q_fwd, log_q_rev).

    SYMMETRIC: both trees are mutated in place. The caller (typically
    Grove._step) must deepcopy parent1 and parent2 BEFORE calling if
    the originals must be preserved for the MH reject path.
    """
    return _subtree_crossover(parent1, parent2)

cpdef rename_prefix(function_treeC.Tree tree, str new_prefix = 'b'):
    '''Renames parameters of a tree using a new prefix'''
    cdef function_treeC.Node node
    
    
    for node in tree.params:
        old_rep = node.function.get_description()
        param_suffix = old_rep[1:]
        node.function = real_functionC.RealFunctionC.from_string(f"{new_prefix}{param_suffix}", tree.domain, compute_grad = False)
    tree.tree_update(tree.root)
    
cdef _compact_param_labels(function_treeC.Tree tree, str prefix='a'):
    """Re-label every parameter in `tree` to a contiguous
    `<prefix>0`, `<prefix>1`, ... sequence in tree.params order.

    Why this exists
    ---------------
    The legacy refresh_params only renames 'b<k>' -> 'a<max_a + 1 + ...>',
    which leaves gaps when the swap removed some 'a' indices (e.g. a swap
    that drops the subtree containing a0 leaves {a2, b0} -> {a2, a3}
    instead of {a0, a1}). _compact_param_labels rebuilds the index space
    from scratch every time, so the resulting label set is always
    contiguous starting at 0.

    Why it is RJMCMC-safe
    ---------------------
    Renaming is a deterministic bijection on label space; the prior factor
    p_Polya(T) and the likelihood L(y | T) depend on the tree's labelled
    structure only through value lookup, which is rebuilt by tree_update
    before any score is computed. The rename therefore contributes 0 to
    the MH proposal correction.

    Repeat names (e.g. a tree of the form a0 + a0) are preserved as a
    single shared label by going through the rename_map.
    """
    cdef function_treeC.Node node
    cdef list params = list(tree.params)
    cdef dict rename_map = {}
    cdef int counter = 0
    cdef str old_name, new_name
    # Pass 1: build the map in tree.params order. tree.params has already
    # been canonicalised by name via sort_params during tree_update, so
    # the resulting indexing is deterministic given the structure.
    for node in params:
        old_name = node.function.get_description()
        if old_name not in rename_map:
            rename_map[old_name] = f"{prefix}{counter}"
            counter += 1
    # Pass 2: apply. Reading node.function.get_description() inside the
    # loop is safe even with renames-in-flight, because each Node owns
    # its own RealFunctionC and renaming one Node never aliases another.
    for node in params:
        old_name = node.function.get_description()
        new_name = rename_map[old_name]
        if new_name == old_name:
            continue
        node.function = real_functionC.RealFunctionC.from_string(
            new_name, tree.domain, compute_grad=False)
    # Rebuild internal functions bottom-up + refresh tree metadata.
    function_treeC._post_order_update(tree.root)
    tree.update_tree_description()
    tree.leaves = tree.get_leaves()
    tree.params = tree.get_params()
    tree.param_values, tree.name_values, tree.penalty = tree.get_param_values()


cdef refresh_params(function_treeC.Tree tree):
    '''Conforms all parameters to one prefix'''
    
    cdef list params = [tree.params[j].get_fun().get_description() for j in range (0, len(tree.params))]
    cdef dict rename_map = {}
    cdef int max_index = -1
    for param in params:
        if param[0] == 'a' and param[1:].isdigit():  
            max_index = max(max_index, int(param[1:]))
    cdef int counter = max_index + 1    
    for param in params:
        if param[0] =='b' :
            rename_map[param] = f"a{counter}"
            counter += 1
    
    cdef function_treeC.Node node
    for node in tree.params:
        param_name = node.function.get_description()
        if param_name in rename_map:
            parent_node = tree.root.find_parent(node)
            old_node = node
            node.function = real_functionC.RealFunctionC.from_string(rename_map[param_name], tree.domain, compute_grad = False)
            node.update_func()
            if parent_node is None:
    
                tree.root = node
            elif parent_node.get_left() == old_node:
                parent_node.assign_left(node)
        
            elif parent_node.get_right() == old_node:
                parent_node.assign_right(node)
            
            tree.tree_update(parent_node)

cpdef rename_prefix_downwards(function_treeC.Tree tree, str new_prefix = 'b'):
    '''Renames parameters of a tree using a new prefix'''
    cdef function_treeC.Node node
    
    for node in tree.params:
        old_rep = node.function.get_description()
        param_suffix = old_rep[1:]
        node.function = real_functionC.RealFunctionC.from_string(f"{new_prefix}{param_suffix}", tree.domain, compute_grad = False)
    # Bottom-up rebuild all intermediate functions, then refresh metadata
    function_treeC._post_order_update(tree.root)
    tree.update_tree_description()
    tree.leaves = tree.get_leaves()
    tree.params = tree.get_params()
    tree.param_values, tree.name_values, tree.penalty = tree.get_param_values()

cdef refresh_params_downwards(function_treeC.Tree tree):
    '''Conforms all parameters to one prefix'''
    
    cdef list params = [tree.params[j].get_fun().get_description() for j in range (0, len(tree.params))]
    cdef dict rename_map = {}
    cdef int max_index = -1
    for param in params:
        if param[0] == 'a' and param[1:].isdigit():  
            max_index = max(max_index, int(param[1:]))
    cdef int counter = max_index + 1    
    for param in params:
        if param[0] =='b' :
            rename_map[param] = f"a{counter}"
            counter += 1
    
    if not rename_map:
        return
    
    cdef function_treeC.Node node
    for node in tree.params:
        param_name = node.function.get_description()
        if param_name in rename_map:
            node.function = real_functionC.RealFunctionC.from_string(rename_map[param_name], tree.domain, compute_grad = False)
    
    # Bottom-up rebuild all intermediate functions, then refresh metadata
    function_treeC._post_order_update(tree.root)
    tree.update_tree_description()
    tree.leaves = tree.get_leaves()
    tree.params = tree.get_params()
    tree.param_values, tree.name_values, tree.penalty = tree.get_param_values()

@cython.boundscheck(True)
cdef tuple _point_mutation(function_treeC.Tree tree, list functions, list operations):
    """
    point_mutation v2: RJMCMC-aware label resampling at one node.

    Selection
    ---------
    Uniform over ALL nodes of `tree` (size |N(T)|). Since point_mutation
    never adds or removes nodes, |N(T')| = |N(T)| and the 1/|N(T)| factor
    cancels between forward and reverse.

    Branching on the chosen node v
    ------------------------------
      * v internal: kind-preserving op resample, uniform from `operations`
        (size O). Symmetric -> contributes (0, 0).
      * v function leaf:
          - prob (1 - p_cross): sub-move A, pick uniform from F (size M).
            Symmetric. (0, 0).
          - prob p_cross: sub-move B, case (c) BIRTH. Replace with a fresh
            parameter leaf a_new, value theta_new ~ g = N(0, tau^2).
              log_q_fwd contribution =  log g(theta_new)
              log_q_rev contribution = -log M
      * v parameter leaf:
          - prob (1 - p_cross): identity. Parameter relabel/refresh is
            handled by the dedicated parameter-refresh move (Phase 2.1.PR).
          - prob p_cross: sub-move B, case (d) DEATH. Pick a new function
            uniform from F; the parameter value theta_j is discarded.
              log_q_fwd contribution = -log M
              log_q_rev contribution =  log g(theta_j)

    The selection factor 1/|N(T)| and the mixture weight log p are common
    to forward and reverse and therefore omitted.

    Returns
    -------
    (log_q_fwd, log_q_rev) : (double, double)
        Consumed by groveC._step:
            log_alpha = beta * (delta target) + (log_q_rev - log_q_fwd)
        beta = 1/T tempers ONLY the (posterior + prior) term.
    """
    cdef function_treeC.Node v, parent
    cdef int O = len(operations)
    cdef int M = len(functions)
    cdef double log_q_fwd = 0.0
    cdef double log_q_rev = 0.0
    cdef str new_func_str, new_name
    #cdef int new_idx
    cdef bint do_cross
    cdef double g_tau = tree.g_tau
    if tree.root is None:
        return (0.0, 0.0)
    if O <= 0 or M <= 0:
        return (0.0, 0.0)

    # 1. Uniform node selection.
    v = _select_uniform_node(tree)
    parent = tree.root.find_parent(v)

    # 2. Internal node -> op resample (symmetric).
    if not v.is_leaf():
        v.op_symbol = _PY_RNG.choice(operations)
        v.update_func()
        if parent is None:
            tree.tree_update(tree.root)
        else:
            tree.tree_update(parent)
        return (0.0, 0.0)

    # 3. Leaf node. Decide kind-cross vs kind-preserving.
    do_cross = (_PY_RNG.random() < _KIND_CROSS_PROB)

    if not do_cross:
        # Sub-move A.
        if v.is_param():
            # Identity for parameter leaves (refresh is its own move).
            return (0.0, 0.0)
        # Function leaf: pick uniform from F. Symmetric.
        new_func_str = _PY_RNG.choice(functions)
        v.function = real_functionC.RealFunctionC.from_string(
            new_func_str, tree.domain, compute_grad=False)
        v.bis_param = v.is_param()
        if parent is None:
            tree.tree_update(tree.root)
        else:
            tree.tree_update(parent)
        return (0.0, 0.0)

    # 4. Sub-move B: kind-crossing.
    cdef double theta_dying
    cdef double theta_new
    if v.is_param():
        # Case (d): DEATH of theta_j. Replace with a function leaf.
        # Capture the dying value BEFORE swapping the leaf, since the
        # reverse move is a BIRTH at exactly this value: log q_rev = log g(theta_j).
        theta_dying = <double> v.value
        new_func_str = _PY_RNG.choice(functions)
        v.function = real_functionC.RealFunctionC.from_string(
            new_func_str, tree.domain, compute_grad=False)
        v.bis_param = v.is_param()
        if parent is None:
            tree.tree_update(tree.root)
        else:
            tree.tree_update(parent)
        log_q_fwd += -np.log(<double> M)
        log_q_rev += _log_g(theta_dying, g_tau)
        return (log_q_fwd, log_q_rev)

    # Function leaf -> parameter leaf: case (c) BIRTH of theta_new.
    cdef list existing_names = [p.get_fun().get_description() for p in tree.params]
    new_name = function_treeC.new_par(existing_names, "a")
    v.function = real_functionC.RealFunctionC.from_string(
        new_name, tree.domain, compute_grad=False)
    # set_value gates on bis_param; flip flag BEFORE assigning the value.
    v.bis_param = v.is_param()
    # Phase 2.1.PM: draw theta_new ~ g and store the SAME draw as the leaf
    # value AND as the argument to log g(.) in log_q_fwd.
    theta_new = _sample_g(g_tau)
    v.set_value(theta_new)
    if parent is None:
        tree.tree_update(tree.root)
    else:
        tree.tree_update(parent)
    log_q_fwd += _log_g(theta_new, g_tau)
    log_q_rev += -np.log(<double> M)
    return (log_q_fwd, log_q_rev)


@cython.boundscheck(True)
cpdef tuple point_mutation(function_treeC.Tree tree, list functions= None, list operations=None,
                           function_treeC.Node node_to_mutate=None):
    """Public entry point. The legacy `node_to_mutate` keyword is kept for
    backward compatibility but is IGNORED in v2: selection must be uniform
    over all nodes for the proposal-density bookkeeping to be correct."""
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    if functions is None:
        functions = tree.functions
    return _point_mutation(tree, functions, list(operations))



@cython.boundscheck(True)
cpdef legacy_point_mutation(function_treeC.Tree tree, functions, operations=None, function_treeC.Node node_to_mutate=None):
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    
    if node_to_mutate is None:
        node_to_mutate = tree.root.select_random_node() 
    
    cdef str new_func_str
    cdef function_treeC.Node parent = tree.root.find_parent(node_to_mutate)
    if node_to_mutate.is_leaf():
        new_func_str= _PY_RNG.choice(functions)
        node_to_mutate.function = real_functionC.RealFunctionC.from_string(new_func_str, tree.domain, compute_grad = False)
    else:
        node_to_mutate.op_symbol = _PY_RNG.choice(operations)
        node_to_mutate.update_func()
    '''
    if parent is None:
      
        tree.root = node_to_mutate
    elif parent.left == node_to_mutate:
        parent.left = node_to_mutate
    elif parent.right == node_to_mutate:
        parent.right = node_to_mutate
    '''
    # Aggiorna la struttura dell'albero
    if parent is None:
        tree.tree_update(tree.root)
    else:
        tree.tree_update(parent)
    
@cython.boundscheck(True)
@cython.wraparound(True)
cpdef legacy_hoist_mutation(function_treeC.Tree tree, function_treeC.Node node_to_lift = None):
    """Replaces subtree with one of its subtrees"""
 
    cdef function_treeC.Node subtree, parent_node, granpa
   
    if not tree.root.is_leaf():
        check=False
        if node_to_lift is None:
            while not check:
                subtree = tree.root.select_random_node()
                parent_node = tree.root.find_parent(subtree)
            
                check = parent_node is not None
        else:
            subtree = node_to_lift
            parent_node = tree.root.find_parent(subtree)
            if parent_node is None:
                pass
        #parent_node = subtree
        if not subtree.is_leaf():
            parent_node.assign_left(subtree.left)
            parent_node.assign_right(subtree.right)
            parent_node.assign_operation(subtree.get_op())
            tree.tree_update(parent_node)

        else:
            granpa = tree.root.find_parent(parent_node)
            if granpa is not None:

                if granpa.left == parent_node:
                    granpa.assign_left(subtree)
                elif granpa.right == parent_node:
                    granpa.assign_right(subtree)
                tree.tree_update(granpa) 
            else:

                tree.root.assign_left(None)
                tree.root.assign_right(None)
                tree.root.assign_operation(None)
                tree.root = function_treeC.Node(subtree.get_fun())
                tree.tree_update(tree.root)
            #parent_node.assign_left(None)
            #parent_node.assign_right(None)
            #parent_node.assign_operation(None)
            #print(subtree.get_fun().get_description())
            #parent_node = function_treeC.Node(subtree.get_fun())
            #print(parent_node.get_fun().get_description()) 
            #granpa = tree.root.find_parent(parent_node)
            #if granpa is not None:
            #    if granpa.get_left() == parent_node
              
        '''
        if not subtree.is_leaf():
            #print('subtree node: ', subtree.get_fun().get_description(), 'left:', subtree.left.get_fun().get_description(),
            # 'right: ', subtree.right.get_fun().get_description())
            choice = _PY_RNG.choice([True, False])
            if choice:
                try:
                    parent_node.assign_left(subtree.get_left())
                except: 
                    parent_node.assign_right(subtree.get_right())
            else:
                try:
                    parent_node.assign_right(subtree.get_right())
                except:
                    parent_node.assign_left(subtree.get_left())
            
            tree.tree_update(parent_node)
        '''
        '''
        elif subtree.is_leaf():
            parent_node.assign_left(None)
            parent_node.assign_right(None)
            parent_node.assign_operation(None)
            parent_node = subtree
        '''
           
            #print('new parent node: ', parent_node.get_fun().get_description(), 'left: ', parent_node.left.get_fun().get_description(),
            # 'right: ', parent_node.right.get_fun().get_description())
    
@cython.boundscheck(True)
@cython.wraparound(True)
cdef tuple _hoist_mutation(function_treeC.Tree tree, list functions, list operations):
    """
    hoist_mutation v2 (Phase 2.1.K). Lift a uniformly-chosen non-root node v;
    v takes its parent's place; the sibling subtree is dropped.

    Reverse pairing
    ---------------
      * If v is a LEAF: reverse is add_node (wrap v with the dropped sibling).
        Both densities computable; on-support iff sibling is perfect-binary
        of depth <= _ADD_NODE_DEPTH_MAX.
      * If v is INTERNAL: no inverse move in the current move set; the kernel
        from T_new back to T has zero density via add_node. Set
        log_q_rev = -inf so the move is always rejected.
        TODO[split-move]: introduce a split move so internal hoists become
        productive. Phase 2.1.K v3 or Phase 2.8.

    Returns (log_q_fwd, log_q_rev).

    NOTE on selection-mix balance: same TODO[move-pair-balance] as add_node.
    """
    cdef function_treeC.Node v, parent, sibling, granpa
    cdef list non_root_nodes
    cdef int n_nodes_T,  M, O, d_S
    cdef double log_q_fwd = 0.0
    cdef double log_q_rev = 0.0
    cdef bint v_is_leaf

    if tree.root is None or tree.root.is_leaf():
        return (0.0, 0.0)
    M = len(functions)
    O = len(operations)

    # ---- 1. uniform pick of non-root node v ----
    non_root_nodes = _flatten_nodes(tree.root, [])
    non_root_nodes = non_root_nodes[1:]   # drop the root
    n_nodes_T = len(non_root_nodes) + 1   # |N(T)|; hoist selects from |N(T)|-1
    if n_nodes_T <= 1:
        return (0.0, 0.0)
    v = non_root_nodes[_PY_RNG.randrange(n_nodes_T - 1)]
    parent = tree.root.find_parent(v)     # never None: v != root
    v_is_leaf = v.is_leaf()
    if parent.get_left() is v:
        sibling = parent.right
    else:
        sibling = parent.left

    # ---- 2. perform the lift (parent <- v) ----
    granpa = tree.root.find_parent(parent)
    if granpa is None:
        # parent IS the root; v becomes the new root.
        tree.root.assign_left(None)
        tree.root.assign_right(None)
        tree.root.assign_operation(None)
        tree.root = v
        tree.tree_update(tree.root)
    else:
        if granpa.left is parent:
            granpa.assign_left(v)
        else:
            granpa.assign_right(v)
        tree.tree_update(granpa)

    # ---- 3. proposal densities ----
    log_q_fwd = -np.log(<double> (n_nodes_T - 1))

    cdef int n_nodes_Tnew, n_leaves_Tnew
    cdef double log_select_rev
    d_S = sibling.depth()
    if d_S < 1 or d_S > _ADD_NODE_DEPTH_MAX or not _is_perfect_binary(sibling, d_S):
        log_q_rev = -np.inf
        return (log_q_fwd, log_q_rev)

    cdef list internal_Tnew
    cdef int n_int_Tnew
    
    if v_is_leaf:
        n_leaves_Tnew = len(tree.root.get_leaves(leaves_list=[]))
        if n_leaves_Tnew <= 0:
            log_q_rev = -np.inf
            return (log_q_fwd, log_q_rev)
        log_select_rev = -np.log(<double> n_leaves_Tnew)
    else:
        # Reverse is split: selects from internal(T').
        internal_Tnew = [n for n in _flatten_nodes(tree.root, [])
                         if not n.is_leaf()]
        n_int_Tnew = len(internal_Tnew)
        if n_int_Tnew <= 0:
            return (log_q_fwd, -np.inf)
        log_select_rev = -np.log(<double> n_int_Tnew)

    cdef double g_tau = tree.g_tau
    log_q_rev = (log_select_rev
                 - np.log(<double> O)
                 + _truncated_geometric_log_pmf(d_S, _ADD_NODE_DEPTH_RHO,
                                                _ADD_NODE_DEPTH_MAX)
                 - np.log(2.0)
                 + _eval_build_density(sibling, d_S, M, O, _ADD_NODE_PARAM_PROB, g_tau))

    return (log_q_fwd, log_q_rev)


@cython.boundscheck(True)
@cython.wraparound(True)
cpdef tuple hoist_mutation(function_treeC.Tree tree, list functions=None, list operations=None):
    """Phase 2.1.K public entry point. Returns (log_q_fwd, log_q_rev).
    `functions`/`operations` are required to compute the reverse-add_node
    density; they default to the standard sets if omitted (legacy callers)."""
    if functions is None:
        functions = tree.functions
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    return _hoist_mutation(tree, list(functions), list(operations))

#@cython.boundscheck(False)
#@cython.wraparound(False)
'''
def size_fair_mutation(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Effettua una mutazione preservando la dimensione dell'albero, sostituendo una parte con un'altra di dimensioni simili."""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef int subtree_size = node_to_replace.compute_subtree_size()
    cdef function_treeC.Node new_subtree = generate_subtree_with_size(subtree_size, functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right
    tree.tree_update(node_to_replace)
'''

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_with_subtree(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Replaces a subtree with a random generated subtree GONNA BE DELETED PROBABLY"""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef function_treeC.Node new_subtree = function_treeC.build_tree(_PY_RNG.randint(1, 3), functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right
'''
@cython.boundscheck(True)
cpdef add_node(function_treeC.Tree tree, functions, operations):
    cdef function_treeC.Node node_to_expand = tree.root.select_random_leaf()
    cdef function_treeC.Node parent = tree.root.find_parent(node_to_expand)
    cdef str new_func_str = random.choice(functions)
    cdef function_treeC.Node cache = node_to_expand
    x = tree.domain
    op = random.choice(operations)
    cdef function_treeC.Node new_node = function_treeC.build_tree(random.choice(list(np.arange(1, tree.depth))), functions, x)
    if random.choice([True, False]):
        node_to_expand.assign_right(node_to_expand)
        node_to_expand.assign_left(new_node)
    else:
        node_to_expand.assign_left(node_to_expand)
        node_to_expand.assign_right(new_node)
    node_to_expand.assign_operation(op)
    if parent is None:
        tree.root = node_to_expand
    elif parent.get_left() == node_to_expand:
        parent.assign_left(node_to_expand)
    elif parent.get_right() ==node_to_expand:
        parent.assign_right(node_to_expand)
    
    #print('check5')
    #print(node_to_expand.get_fun().get_description())
    #print(parent.get_fun().get_description())
    parent.update_func()
    #print(parent.get_fun().get_description())
    #print('check6')
    #print(node_to_expand.is_leaf())
    #print(tree.root.find_parent(node_to_expand).update_func())
    tree.tree_update(parent)

'''

@cython.boundscheck(True)  # Per migliorare le prestazioni
cpdef legacy_add_node( function_treeC.Tree tree, functions, operations=None, random_depth=False):
    """Adds a node from a random location in an existing tree
    NOT ONLY DEPRECATED: THIS IS ACTUALLY WRONG. DO NOT USE.
    """
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    cdef str op = _PY_RNG.choice(operations)
    if not tree.root.is_leaf():
        parent_node = None
        while parent_node is None:
            node_to_replace = tree.root.select_random_leaf()
            parent_node = tree.root.find_parent(node_to_replace)
        parent_op = parent_node.get_op()
        if not random_depth:
            #max_depth = tree.root.depth()
            max_depth = 2
        else:
            max_depth = _PY_RNG.choice(list(np.arange(1, tree.root.depth()+1)))
        new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)
        if _PY_RNG.choice([True, False]):
            node_to_replace.assign_left(node_to_replace)
            node_to_replace.assign_right(new_subtree)
        else:
            node_to_replace.assign_right(node_to_replace)
            node_to_replace.assign_left(new_subtree)

        node_to_replace.assign_operation(op)
       

        if parent_node is None:
            tree.root = new_subtree
        elif parent_node.get_left() == node_to_replace:
            parent_node.assign_left(new_subtree)
        
        elif parent_node.get_right() == node_to_replace:
            parent_node.assign_right(new_subtree)
 
        node_to_replace.update_func()  
    #print('op_new: ', new_subtree.function.get_description(),new_subtree.depth(), new_subtree.is_leaf())
    
        tree.tree_update(parent_node)
    #print('nuova descr: ', tree.tree_get_description())
    else:
        parent_node = function_treeC.Node(function=tree.root.get_fun())
        max_depth = _PY_RNG.choice([1,2])
        new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)
        if _PY_RNG.choice([True, False]):
            tree.root.assign_right(parent_node)
            tree.root.assign_left(new_subtree)
        else:
            tree.root.assign_left(parent_node)
            tree.root.assign_right(new_subtree)
       
        tree.root.assign_operation(op)
        #print(tree.root.get_fun().get_description(), parent_node.is_leaf())
        
        #print(tree.root.get_fun().get_description(), tree.root.is_leaf(), parent_node.is_leaf())
        #print(parent_node, tree.root)
        tree.tree_update(parent_node)

@cython.boundscheck(True)
cdef tuple _add_node(function_treeC.Tree tree, list functions, list operations):
    """
    add_node v2 (Phase 2.1.K). Wraps a uniformly-chosen LEAF L of T with
    op(L, S) or op(S, L), where:
      * op  ~ Uniform(operations),                 (size O)
      * d   ~ TruncatedGeom(rho) on {1,...,d_max}, (depth of S)
      * S   = perfect-binary subtree of height d,
              each leaf ind. parameter w.p. p_par (else uniform from F),
              ops uniform from operations,
              parameter values from g (placeholder),
      * side ~ Uniform({L, R}).                    (1/2 each)

    Reverse move = leaf-hoist of L on T' (Phase 2.1 hoist).

    Returns (log_q_fwd, log_q_rev). On-support by construction.
    K(T') = K(T) + K_S, where K_S is the number of parameter leaves in S.

    NOTE on selection probabilities (mixture-balance term):
    The exact RJMCMC ratio also needs log p_select(hoist) - log p_select(add)
    from _choose_move. For Phase 2.1 we assume p_add = p_hoist (term = 0);
    Phase 2.8 (move-rate calibration) will revisit. TODO[move-pair-balance].
    """
    cdef list leaves = tree.root.get_leaves(leaves_list=[])
    cdef int n_leaves_T = len(leaves)
    cdef int M = len(functions)
    cdef int O = len(operations)
    cdef function_treeC.Node L, parent, new_internal, S
    cdef str op
    cdef int d, n_nodes_Tp
    cdef double log_p_d
    cdef double log_q_fwd = 0.0
    cdef double log_q_rev = 0.0
    cdef list existing_names = [p.get_fun().get_description() for p in tree.params]
    cdef bint side_left
    cdef double g_tau = tree.g_tau

    if n_leaves_T == 0 or M <= 0 or O <= 0:
        return (0.0, 0.0)

    # ---- 1. pick leaf, op, depth, side ----
    #parent = None
    
    L = tree.root.select_random_leaf()
    parent = tree.root.find_parent(L)
    #L = leaves[random.randrange(n_leaves_T)]
    op = _PY_RNG.choice(operations)
    d, log_p_d = _sample_truncated_geometric(_ADD_NODE_DEPTH_RHO, _ADD_NODE_DEPTH_MAX)
    side_left = (_PY_RNG.random() < 0.5)

    # ---- 2. build subtree S ----
    S = _build_subtree_with_params(d, functions, operations, tree.domain,
                                   _ADD_NODE_PARAM_PROB, existing_names, g_tau)

    # ---- 3. wrap L ----
    parent = tree.root.find_parent(L)   # None iff L is the root (single-node T)
    
    new_internal = function_treeC.Node()
    #print('new internal', new_internal.get_fun().get_description())  # placeholder
    if side_left:
        new_internal.assign_left(L, brute_force=True)
        new_internal.assign_right(S, brute_force=True)
    else:
        new_internal.assign_left(S, brute_force=True)
        new_internal.assign_right(L, brute_force=True)
    new_internal.assign_operation(op, brute_force=True)
    new_internal.update_func()

    if parent is None:
        tree.root = new_internal
        tree.tree_update(tree.root)
    else:
        if parent.get_left() is L:
            parent.assign_left(new_internal)
        else:
            parent.assign_right(new_internal)
        tree.tree_update(parent)

    # ---- 4. proposal densities ----
    n_nodes_Tp = _count_nodes(tree.root)
    
    log_q_fwd = (-np.log(<double> n_leaves_T)
                 - np.log(<double> O)
                 + log_p_d
                 - np.log(2.0)
                 + _eval_build_density(S, d, M, O, _ADD_NODE_PARAM_PROB, g_tau))

    # Reverse: leaf-hoist of L on T'. Uniform over non-root nodes of T'.
    if n_nodes_Tp <= 1:
        log_q_rev = -np.inf
    else:
        log_q_rev = -np.log(<double> (n_nodes_Tp - 1))

    return (log_q_fwd, log_q_rev)


@cython.boundscheck(True)
cpdef tuple add_node(function_treeC.Tree tree, list functions=None, list operations=None,
                     random_depth=False):
    """Phase 2.1.K public entry point.
    Legacy `random_depth` kept for back-compat; ignored (depth is sampled
    from the truncated-geometric distribution by design)."""
    if functions is None:
        functions = tree.functions
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    return _add_node(tree, functions, operations)

# ---------------------------------------------------------------------------
# Phase 2.1.K v3: Split move (inverse of internal hoist_mutation)
# ---------------------------------------------------------------------------
@cython.boundscheck(True)
cdef tuple _split(function_treeC.Tree tree, list functions, list operations):
    """
    Split move: paired inverse of the internal-node branch of hoist_mutation.

    Selection
    ---------
    Pick u ~ Uniform(N(T)) (size n = |N(T)|). Wrap u with a fresh internal
    node carrying:
      * op    ~ Uniform(operations),                       (size O)
      * d     ~ TruncatedGeom(rho) on {1,...,d_max},
      * S     = perfect-binary subtree of height d, built by
                _build_subtree_with_params (per-leaf parameter Bernoulli),
      * side  ~ Uniform({L, R}).                           (1/2 each)

    Difference from add_node
    ------------------------
    add_node selects from leaves(T); split selects from N(T). On a leaf u
    the two moves are structurally identical. We keep them separate (Option
    A from the design discussion) so that each has a single, clean reverse:
        add_node  <-> hoist (leaf branch)
        split     <-> hoist (internal branch)
    The leaf-overlap inflates the effective Birth proposal weight by a
    constant; this is harmless for correctness (changes effective mixture
    weights, not the stationary distribution) and will be folded into the
    Phase 2.8 calibration.

    Returns
    -------
    (log_q_fwd, log_q_rev) : (double, double)
        On-support by construction (S is built from the proposal kernel
        itself).

    NOTE on selection-mix balance: same TODO[move-pair-balance] as
    add_node and hoist. Phase 2.8.
    """
    cdef list all_nodes = _flatten_nodes(tree.root, [])
    cdef list internal_nodes = [n for n in all_nodes if not n.is_leaf()]
    cdef int n_int_T = len(internal_nodes)
    cdef int M = len(functions)
    cdef int O = len(operations)
    cdef function_treeC.Node u, parent, new_internal, S
    cdef str op
    cdef int d, n_nodes_Tp
    cdef double log_p_d
    cdef double log_q_fwd = 0.0
    cdef double log_q_rev = 0.0
    cdef list existing_names = [p.get_fun().get_description() for p in tree.params]
    cdef bint side_left
    cdef double g_tau = tree.g_tau

    if n_int_T == 0 or M <= 0 or O <= 0:
        return (0.0, 0.0)

    # ---- 1. pick wrap target u, op, depth, side ----
    u = internal_nodes[_PY_RNG.randrange(n_int_T)]
    op = _PY_RNG.choice(operations)
    d, log_p_d = _sample_truncated_geometric(_ADD_NODE_DEPTH_RHO,
                                             _ADD_NODE_DEPTH_MAX)
    side_left = (_PY_RNG.random() < 0.5)

    # ---- 2. build subtree S ----
    S = _build_subtree_with_params(d, functions, operations, tree.domain,
                                   _ADD_NODE_PARAM_PROB, existing_names, g_tau)

    # ---- 3. wrap u with a fresh internal node ----
    parent = tree.root.find_parent(u)   # None iff u is the root
    new_internal = function_treeC.Node()  # placeholder
    if side_left:
        new_internal.assign_left(u)
        new_internal.assign_right(S)
    else:
        new_internal.assign_left(S)
        new_internal.assign_right(u)
    new_internal.assign_operation(op)
    new_internal.update_func()

    if parent is None:
        tree.root = new_internal
        tree.tree_update(tree.root)
    else:
        if parent.get_left() is u:
            parent.assign_left(new_internal)
        else:
            parent.assign_right(new_internal)
        tree.tree_update(parent)

    # ---- 4. proposal densities ----
    n_nodes_Tp = _count_nodes(tree.root)
    
    log_q_fwd = (-np.log(<double> n_int_T)
                 - np.log(<double> O)
                 + log_p_d
                 - np.log(2.0)
                 + _eval_build_density(S, d, M, O, _ADD_NODE_PARAM_PROB, g_tau))

    # Reverse: hoist of (the former) u on T'. The chosen node in the
    # reverse is the new_internal's child that corresponds to u; selecting
    # it requires picking it uniformly from the (n_nodes_Tp - 1) non-root
    # nodes of T'. Hoist's reverse density is just the selection term.
    if n_nodes_Tp <= 1:
        log_q_rev = -np.inf
    else:
        log_q_rev = -np.log(<double> (n_nodes_Tp - 1))

    return (log_q_fwd, log_q_rev)


@cython.boundscheck(True)
cpdef tuple split(function_treeC.Tree tree, list functions=None, list operations=None):
    """Phase 2.1.K v3 public entry point. Returns (log_q_fwd, log_q_rev).

    Paired inverse of the internal-node branch of hoist_mutation. See
    `_split` docstring for full mathematical content.
    """
    if functions is None:
        functions = tree.functions
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    return _split(tree, functions, operations)

def generate_subtree_with_size(int size, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Generates a random subtree with chosen size"""
    depth = size // 2  #  profondità approssimativa
    cdef function_treeC.Tree tree = function_treeC.Tree(depth, functions, x)
    return tree.root


cpdef root_replacement(function_treeC.Tree tree):
    cdef int depth = tree.depth
    cdef object x = tree.domain
    cdef list funcs = tree.funcs
    cdef function_treeC.Tree new_tree = function_treeC.Tree(depth, funcs, x)
    tree = new_tree
    tree.tree_update(tree.root) #RIGUARDA QUESTO METODO
'''
TO BE WORKED ON
cpdef prune_node(function_treeC.Tree tree, function_treeC.Node node_to_prune = None):
    if node_to_prune is None:
       hoist_mutation(tree, tree.functions)
       pass

    cdef function_treeC.Node parent = tree.root.find_parent(node_to_prune)  
    if parent is not None:
        if parent.right == node_to_prune:
            hoist_mutation(tree, parent.left)
        else:
            hoist_mutation(tree, parent.right)
    else:
        pass
        #point_mutation(tree, node_to_prune)
'''

cdef class AMState:
    """
    Per-topology Adaptive Metropolis state for parameter_refresh.

    Two-layer Robbins-Monro (Andrieu-Thoms 2008, Algo 4):
      Layer 1 (global scale):  log lambda_{t+1} = log lambda_t
                                + gamma_t (alpha_obs - alpha*)
      Layer 2 (covariance):    mu, C updated from theta visits
                                Sigma_t = lambda^2 (C + jitter I)

    Cold-start (n_visits < K+1): diagonal multiplicative scale
        Sigma = diag(max(eps, |theta_i| * rel))^2

    Diminishing adaptation (gamma_t -> 0) preserves ergodicity
    (Roberts-Rosenthal 2007).
    """
    # NB: cdef classes do NOT support __slots__; attributes are declared
    # below as cdef public so they are reachable from Python and from
    # other Cython modules (parameter_refresh reads .K and .param_names).

    def __init__(self, list param_names, object alpha_star=None,
                 double kappa=_AM_KAPPA, double rel=_AM_REL_SCALE,
                 double eps=_AM_EPS, double jitter=_AM_JITTER):
        # alpha_star is `object` (not `double`) because we need to accept
        # None as the sentinel meaning "pick the dim-dependent default".
        self.param_names = tuple(param_names)
        self.K = len(self.param_names)
        self.mu = np.zeros(self.K, dtype=np.float64)
        self.C = np.eye(self.K, dtype=np.float64) * (rel * rel)
        self.log_lambda = 0.0
        self.n_visits = 0
        self.n_accepts = 0
        if alpha_star is None:
            self.alpha_star = _AM_TARGET_ACCEPT_K1 if self.K == 1 \
                              else _AM_TARGET_ACCEPT_LARGE_K
        else:
            self.alpha_star = float(alpha_star)
        self.kappa = float(kappa)
        self.rel = float(rel)
        self.eps = float(eps)
        self.jitter = float(jitter)

    cdef double _gamma(self):
        # gamma_t = (n_visits + 1)^(-kappa); +1 keeps gamma_0 finite.
        return (self.n_visits + 1) ** (-self.kappa)

    cdef cnp.ndarray _propose(self, cnp.ndarray theta):
        # theta is the current K-vector; returns a candidate K-vector.
        # Cold-start branch uses a diagonal multiplicative scale; warm
        # branch draws from N(0, lambda^2 (C + jitter I)) via Cholesky.
        cdef cnp.ndarray scale
        if self.n_visits < self.K + 1:
            scale = np.maximum(self.eps, np.abs(theta) * self.rel)
            return theta + _NP_RNG.normal(0.0, scale)
        cdef cnp.ndarray cov, L
        cov = np.exp(2.0 * self.log_lambda) * (self.C + self.jitter * np.eye(self.K))
        try:
            L = np.linalg.cholesky(cov)
            return theta + L @ _NP_RNG.standard_normal(self.K)
        except np.linalg.LinAlgError:
            # Defensive fallback: use the diagonal of cov if C drifted
            # non-PSD (shouldn't happen with the jitter, but cheap to guard).
            scale = np.maximum(self.eps, np.sqrt(np.diag(cov)))
            return theta + _NP_RNG.normal(0.0, scale)
    
    cpdef cnp.ndarray propose(self, cnp.ndarray theta):
        return self._propose(theta)

    cdef void _update(self, cnp.ndarray theta_after, bint accepted, double alpha_obs):
        # Robbins-Monro updates on (log_lambda, mu, C). Called on EVERY
        # step (accept or reject) so the adaptation actually diminishes.
        cdef double g = self._gamma()
        cdef cnp.ndarray d_old, d_new
        self.log_lambda += g * (alpha_obs - self.alpha_star)
        d_old = theta_after - self.mu
        self.mu = self.mu + g * d_old
        d_new = theta_after - self.mu
        self.C = self.C + g * (np.outer(d_old, d_new) - self.C)
        # Numerical symmetrization to keep C exactly symmetric in float64.
        self.C = 0.5 * (self.C + self.C.T)
        self.n_visits += 1
        if accepted:
            self.n_accepts += 1

    cpdef void update(self, cnp.ndarray theta_after, bint accepted, double alpha_obs):
        self._update(theta_after, accepted, alpha_obs)

cdef tuple _parameter_refresh(function_treeC.Tree tree,
                              cnp.ndarray[cnp.float64_t, ndim=1] y,
                              double sigma,
                              AMState am_state,
                              double T):
    """
    Phase 2.1.PR. Adaptive Metropolis on theta with topology fixed.
    Self-contained MH: scores on raw Gaussian likelihood (D2),
    tempered by 1/T (D7). Symmetric AM proposal => no proposal
    correction => log_q_fwd = log_q_rev = 0.

    Mutates `tree` in place ON ACCEPT. Updates am_state in place
    (Robbins-Monro) regardless of accept/reject (required for
    diminishing-adaptation guarantee).

    Returns (accepted, 0.0, 0.0). Caller MUST NOT recompute MH from
    the (0,0) tuple - PR has already accepted/rejected internally.

    SHOULD BE SAFE BUT CHECK THAT THIS FUNCTION LEAVES TREES UNCHANGED
    BEFORE _apply_param_values AND THAT IT CHANGES THE SPECIFIED TREE ONLY!
    """
    cdef int K = len(tree.params)
    if K == 0:
        return (True, 0.0, 0.0)

    if am_state is None or am_state.K != K:
        # Topology changed under our feet; caller must rebuild AMState.
        return (False, 0.0, 0.0)

    cdef list names = list(am_state.param_names)
    cdef int j
    cdef cnp.ndarray[cnp.float64_t, ndim=1] theta_old = np.empty(K, dtype=np.float64)
    cdef dict cur_nv = tree.name_values
    for j in range(K):
        if names[j] not in cur_nv:
            return (False, 0.0, 0.0)
        theta_old[j] = cur_nv[names[j]]

    cdef cnp.ndarray[cnp.float64_t, ndim=1] theta_new = np.asarray(
        am_state._propose(theta_old), dtype=np.float64)

    cdef dict cand_nv = dict(cur_nv)
    for j in range(K):
        cand_nv[names[j]] = float(theta_new[j])

    cdef double logL_old = tree._log_likelihood_at(y, cur_nv, sigma)
    cdef double logL_new = tree._log_likelihood_at(y, cand_nv, sigma)
    
    cdef double log_acc
    if not np.isfinite(logL_new):
        log_acc = -np.inf
    elif T <= 0.0:
        log_acc = 0.0 if logL_new >= logL_old else -np.inf
    else:
        log_acc = (logL_new - logL_old) / T

    cdef double alpha_obs
    if log_acc >= 0.0:
        alpha_obs = 1.0
    elif log_acc <= -700.0:
        alpha_obs = 0.0
    else:
        alpha_obs = float(np.exp(log_acc))

    cdef bint accepted = (_PY_RNG.random() < alpha_obs)

    if accepted:
        tree._apply_param_values(cand_nv)
        am_state._update(theta_new, True, alpha_obs)
    else:
        am_state._update(theta_old, False, alpha_obs)

    return (accepted, 0.0, 0.0)

cpdef tuple parameter_refresh(function_treeC.Tree tree,
                              cnp.ndarray[cnp.float64_t, ndim=1] y,
                              double sigma,
                              AMState am_state,
                              double T):
    return _parameter_refresh(tree, y, sigma, am_state, T)