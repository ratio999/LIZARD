import random
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
cimport function_treeC
#from function_treeC import build_tree
cimport real_functionC
cimport numpy as cnp


cdef double _LOG_G_PLACEHOLDER = 0.0
cdef double _KIND_CROSS_PROB = 0.5      # mixture weight for sub-move B
cdef double _NEW_PARAM_INIT_VALUE = 1.0  # TODO[g-placeholder]: draw from g


# --- Phase 2.1.PR: Adaptive-Metropolis parameter refresh -------------------
# Constants per plan decisions D5/D7.
cdef double _AM_TARGET_ACCEPT_LARGE_K = 0.234   # Roberts-Gelman-Gilks 1997
cdef double _AM_TARGET_ACCEPT_K1      = 0.44
cdef double _AM_KAPPA                 = 0.6     # Robbins-Monro decay  gamma_t = t^{-kappa}
cdef double _AM_REL_SCALE             = 0.1     # cold-start: |theta_i| * rel
cdef double _AM_EPS                   = 1e-6    # cold-start floor
cdef double _AM_JITTER                = 1e-8  



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


cdef function_treeC.Node _select_uniform_node(function_treeC.Tree tree):
    """Pick one node of `tree` uniformly at random over all nodes."""
    cdef list all_nodes = _flatten_nodes(tree.root, [])
    return all_nodes[random.randrange(len(all_nodes))]


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
        add_node(tree, functions)
        return
    parent_node = None
    cdef bint is_par = True
    # ---- 1. Uniform node selection (root allowed) ------------------------
    
    while parent_node is None:
        v = _select_uniform_node(tree)
        is_par = v.is_param()
        if not is_par:
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
        chosen_slots = sorted(random.sample(range(n_leaves), K))
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
cpdef tuple subtree_mutation(function_treeC.Tree tree, list functions):
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
        max_depth = random.choice(list(np.arange(1, node_to_replace.depth()+2)))
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
cpdef subtree_crossover( function_treeC.Tree parent1, function_treeC.Tree parent2):
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

cpdef rename_prefix(function_treeC.Tree tree, str new_prefix = 'b'):
    '''Renames parameters of a tree using a new prefix'''
    cdef function_treeC.Node node
    
    
    for node in tree.params:
        old_rep = node.function.get_description()
        param_suffix = old_rep[1:]
        node.function = real_functionC.RealFunctionC.from_string(f"{new_prefix}{param_suffix}", tree.domain, compute_grad = False)
    tree.tree_update(tree.root)
    
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
            parameter leaf a_new, value = _NEW_PARAM_INIT_VALUE.
              log_q_fwd contribution =  _LOG_G_PLACEHOLDER
              log_q_rev contribution = -log M
      * v parameter leaf:
          - prob (1 - p_cross): identity. Parameter relabel/refresh is
            handled by the dedicated parameter-refresh move (Phase 2.1.PR).
          - prob p_cross: sub-move B, case (d) DEATH. Pick a new function
            uniform from F; the parameter value is discarded.
              log_q_fwd contribution = -log M
              log_q_rev contribution =  _LOG_G_PLACEHOLDER

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

    if tree.root is None:
        return (0.0, 0.0)
    if O <= 0 or M <= 0:
        return (0.0, 0.0)

    # 1. Uniform node selection.
    v = _select_uniform_node(tree)
    parent = tree.root.find_parent(v)

    # 2. Internal node -> op resample (symmetric).
    if not v.is_leaf():
        v.op_symbol = random.choice(operations)
        v.update_func()
        if parent is None:
            tree.tree_update(tree.root)
        else:
            tree.tree_update(parent)
        return (0.0, 0.0)

    # 3. Leaf node. Decide kind-cross vs kind-preserving.
    do_cross = (random.random() < _KIND_CROSS_PROB)

    if not do_cross:
        # Sub-move A.
        if v.is_param():
            # Identity for parameter leaves (refresh is its own move).
            return (0.0, 0.0)
        # Function leaf: pick uniform from F. Symmetric.
        new_func_str = random.choice(functions)
        v.function = real_functionC.RealFunctionC.from_string(
            new_func_str, tree.domain, compute_grad=False)
        v.bis_param = v.is_param()
        if parent is None:
            tree.tree_update(tree.root)
        else:
            tree.tree_update(parent)
        return (0.0, 0.0)

    # 4. Sub-move B: kind-crossing.
    if v.is_param():
        # Case (d): DEATH of theta_j. Replace with a function leaf.
        # TODO[g-placeholder]: replace _LOG_G_PLACEHOLDER with log g(v.value).
        new_func_str = random.choice(functions)
        v.function = real_functionC.RealFunctionC.from_string(
            new_func_str, tree.domain, compute_grad=False)
        v.bis_param = v.is_param()
        if parent is None:
            tree.tree_update(tree.root)
        else:
            tree.tree_update(parent)
        log_q_fwd += -np.log(<double> M)
        log_q_rev += _LOG_G_PLACEHOLDER
        return (log_q_fwd, log_q_rev)

    # Function leaf -> parameter leaf: case (c) BIRTH of theta_new.
    cdef list existing_names = [p.get_fun().get_description() for p in tree.params]
    new_name = function_treeC.new_par(existing_names, "a")
    v.function = real_functionC.RealFunctionC.from_string(
        new_name, tree.domain, compute_grad=False)
    # set_value gates on bis_param; flip flag BEFORE assigning the value.
    v.bis_param = v.is_param()
    # TODO[g-placeholder]: draw v_init ~ g and use the SAME draw both as
    # the stored value here AND as the argument to log g(.) below.
    v.set_value(_NEW_PARAM_INIT_VALUE)
    if parent is None:
        tree.tree_update(tree.root)
    else:
        tree.tree_update(parent)
    log_q_fwd += _LOG_G_PLACEHOLDER
    log_q_rev += -np.log(<double> M)
    return (log_q_fwd, log_q_rev)


@cython.boundscheck(True)
cpdef tuple point_mutation(function_treeC.Tree tree, functions, operations=None,
                           function_treeC.Node node_to_mutate=None):
    """Public entry point. The legacy `node_to_mutate` keyword is kept for
    backward compatibility but is IGNORED in v2: selection must be uniform
    over all nodes for the proposal-density bookkeeping to be correct."""
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
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
        new_func_str= random.choice(functions)
        node_to_mutate.function = real_functionC.RealFunctionC.from_string(new_func_str, tree.domain, compute_grad = False)
    else:
        node_to_mutate.op_symbol = random.choice(operations)
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
cpdef hoist_mutation(function_treeC.Tree tree, function_treeC.Node node_to_lift = None):
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
            choice = random.choice([True, False])
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
    cdef function_treeC.Node new_subtree = function_treeC.build_tree(random.randint(1, 3), functions, x)
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
cpdef add_node( function_treeC.Tree tree, functions, operations=None, random_depth=False):
    """Adds a node from a random location in an existing tree"""
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    cdef str op = random.choice(operations)
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
            max_depth = random.choice(list(np.arange(1, tree.root.depth()+1)))
        new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)
        if random.choice([True, False]):
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
        max_depth = random.choice([1,2])
        new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)
        if random.choice([True, False]):
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

cpdef prune_node(function_treeC.Tree tree, function_treeC.Node node_to_prune = None):
    if node_to_prune is None:
       hoist_mutation(tree)
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
            return theta + np.random.normal(0.0, scale)
        cdef cnp.ndarray cov, L
        cov = np.exp(2.0 * self.log_lambda) * (self.C + self.jitter * np.eye(self.K))
        try:
            L = np.linalg.cholesky(cov)
            return theta + L @ np.random.standard_normal(self.K)
        except np.linalg.LinAlgError:
            # Defensive fallback: use the diagonal of cov if C drifted
            # non-PSD (shouldn't happen with the jitter, but cheap to guard).
            scale = np.maximum(self.eps, np.sqrt(np.diag(cov)))
            return theta + np.random.normal(0.0, scale)
    
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

    cdef bint accepted = (random.random() < alpha_obs)

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