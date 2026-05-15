# mpi_worker.py
"""
MPI worker script for parallel burn-in.
Spawned by the main process, receives work, computes, sends results back.
"""
from mpi4py import MPI
import numpy as np
cimport numpy as cnp
import sys
import os
from copy import deepcopy
# Add the module path
from sympy import sympify, symbols
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
cimport function_treeC
cimport real_functionC
cimport gp_movesC
cdef function_treeC.Tree _helper(object expr, cnp.ndarray[cnp.float64_t, ndim=1] domain, 
            list funcs, dict names_params):
    
    cdef function_treeC.Tree tree
    cdef function_treeC.Node param

    tree = function_treeC._expr_to_tree(expr, domain, funcs)
    #tree.
    
    if tree is not None and names_params and len(names_params) > 0:
        for param in tree.params:
            if not param.function.expr.is_Number:
                param.set_value(names_params[param.get_description()])
    return tree



cdef void _worker_main():
    """
    Worker entry point. Communicates with parent (spawner) process.
    """
    # Get the intercommunicator to the parent
    cdef object parent_comm = MPI.Comm.Get_parent()
    
    if parent_comm == MPI.COMM_NULL:
        print("Error: This script must be spawned by MPI, not run directly.")
        sys.exit(1)
    
    # Get my rank among workers
    cdef object worker_comm = MPI.COMM_WORLD
    cdef int rank = worker_comm.Get_rank()
    cdef int n_workers = worker_comm.Get_size()
    
    # =========================================================
    # RECEIVE SHARED DATA FROM PARENT
    # =========================================================
    # Parent broadcasts the configuration to all workers
    cdef dict config = parent_comm.bcast(None, root=0)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] domain = config['domain']
    cdef list funcs = config['funcs']
    cdef list operations = config['operations']
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y = config['y']
    cdef double lambda_reg = config['lambda_reg']
    cdef str method_str = config['method_str']
    cdef bint use_gradients = config['use_gradients']
    cdef list depth1_data = config['depth1_data']  # List of (expression, params, posterior)
    cdef list all_primitives = config['all_primitives']
    cdef double T = config['T']
    cdef dict local_cache = config['tree_cache']
    
    # =========================================================
    # RECEIVE MY WORK ITEMS
    # =========================================================
    # Parent scatters work items to each worker
    cdef object my_work = parent_comm.scatter(None, root=0)
    
    # =========================================================
    # REBUILD DEPTH-1 TREES FROM SERIALIZED DATA
    # =========================================================
    cdef list depth1_trees = []
    cdef int i=0
    cdef object expr
    cdef dict names_params
    cdef object post
    cdef function_treeC.Tree tree, d1_tree, d1_tree_2

    for expr, names_params, post in depth1_data:    
        #print(i, expr, names_params)
        tree = _helper(expr, domain, funcs, names_params)    
        depth1_trees.append((tree, post))
        if tree.tree_get_description() not in local_cache:
            print(f'[Worker {rank}] Cache miss for depth-1 tree: {tree.tree_get_description()}')
            local_cache[tree.tree_get_description()] = post
        if tree.tree_get_description() != str(expr):
            print(f'[Worker {rank}] Warning: Reconstructed tree description "{tree.tree_get_description()}" does not match original expression string "{expr}".')
    # =========================================================
    # PROCESS MY WORK ITEMS
    # =========================================================
    cdef list local_results = []  # (expression, params_list, posterior)
    
    cdef object item
    cdef object item_type
    cdef function_treeC.Node prim_node
    cdef str prim_str, op
    cdef tuple result
    cdef int idx

    for idx, item in enumerate(my_work):
        item_type = item[0]
        
        
        if item_type == 'd1_prim':
            _, i, prim_str, op = item
            
            # Create primitive node
            prim_node = function_treeC.Node(
                function=real_functionC.RealFunctionC.from_string(
                    prim_str, domain, compute_grad=False))
            
            d1_tree = depth1_trees[i][0]
            if d1_tree is None:
                continue
            
            # Direction 1: depth-1 op primitive
            tree = _combine_nodes(d1_tree.root, prim_node, op, domain, funcs, T)
            result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
            expr_string = tree.tree_get_description()
            if result is not None:
                local_results.append(result)
                local_cache[expr_string] = result[2]  # Cache the posterior 
            # Direction 2: primitive op depth-1
            tree = _combine_nodes(prim_node, d1_tree.root, op, domain, funcs, T)
            result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
            expr_string = tree.tree_get_description()
            if result is not None:
                local_results.append(result)
                local_cache[expr_string] = result[2]  # Cache the posterior        
        elif item_type == 'd1_d1':
            _, i, j, op = item
            
            d1_tree = depth1_trees[i][0]
            d1_tree_2 = depth1_trees[j][0]
            
            if d1_tree is None or d1_tree_2 is None:
                continue
            
            tree = _combine_nodes(d1_tree.root, d1_tree_2.root, op, domain, funcs, T)
            result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
            expr_string = tree.tree_get_description()
            if result is not None:
                local_results.append(result)
                local_cache[expr_string] = result[2]  # Cache the posterior
        if (idx +1) % 50 == 0:
            parent_comm.isend(idx+1, dest=0, tag=99)
        #print(idx, rank)
        #sys.stdout.flush()
    # =========================================================
    # SEND RESULTS BACK TO PARENT
    # =========================================================
    worker_comm.Barrier()
    print(f'[Worker {rank}] About to gather {len(local_results)} results...')
    sys.stdout.flush()
    
    try:
        parent_comm.gather(local_results, root=0)
        print(f'[Worker {rank}] Gather completed successfully!')
    except Exception as e:
        print(f'[Worker {rank}] Gather failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
    
    # =========================================================
    # DISCONNECT AND EXIT
    # =========================================================
    #worker_comm.Disconnect()
    parent_comm.Disconnect()
    print(f'[Worker {rank}] disconnected.')

cdef function_treeC.Tree _combine_nodes(function_treeC.Node left_node, function_treeC.Node right_node, object op,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, list funcs, double T):
    """Combine two nodes with an operation into a new tree."""
    
    
    cdef function_treeC.Node left_copy = deepcopy(left_node)
    cdef function_treeC.Node right_copy = deepcopy(right_node)
    #cdef function_treeC.Tree preliminary_tree = function_treeC.Tree(
	#depth = right_copy.node_depth, functions = funcs,
	#x = domain, node = right_copy)
    
    #gp_movesC.rename_prefix(preliminary_tree)
  
    cdef function_treeC.Node root_node = function_treeC.Node(
        function=None,
        left=left_copy,
        right=right_copy,
        symbol=op
    )
    
    cdef function_treeC.Tree tree = function_treeC.Tree(
        depth=root_node.node_depth,
        functions=funcs,
        x=domain,
        node=root_node,
        Npars=len(root_node.function.params),
        value=1.0,
        T=T
    )
    gp_movesC.refresh_params(tree)
    #del preliminary_tree
    return tree


cdef tuple _evaluate_tree(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, double lambda_reg, 
                    str method_str, bint use_gradients, double T):
    """Evaluate a tree and return serializable result or None."""
    if tree is None:
        return None
    
    #cdef object expression = tree.root.function.expr
    
    if not tree.is_valid(domain):
        return None
    
    # Optimize parameters if needed
    if len(tree.params) > 0:
        tree.update_parameters(domain, y, method_str, use_gradients)
    
    #tree.replace_constants_with_pars()
    tree.replace_constants_with_pars()
    cdef double post = tree.root._log_marginal_bic(y)
    
    if not np.isfinite(post):
        return None
    cdef object expression = tree.tree_get_description()
    try:
        
        # Let sympify auto-detect and create symbols
        expression = sympify(expression)
    except:
        expression = tree.root.function.expr
        pass
    # Return serializable tuple
    return (expression, tree.name_values if tree.name_values else {}, post)

def worker_main():
    _worker_main()

if __name__ == "__main__":
    worker_main()
