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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
cimport function_treeC
cimport real_functionC


def worker_main():
    """
    Worker entry point. Communicates with parent (spawner) process.
    """
    # Get the intercommunicator to the parent
    parent_comm = MPI.Comm.Get_parent()
    
    if parent_comm == MPI.COMM_NULL:
        print("Error: This script must be spawned by MPI, not run directly.")
        sys.exit(1)
    
    # Get my rank among workers
    worker_comm = MPI.COMM_WORLD
    rank = worker_comm.Get_rank()
    n_workers = worker_comm.Get_size()
    
    # =========================================================
    # RECEIVE SHARED DATA FROM PARENT
    # =========================================================
    # Parent broadcasts the configuration to all workers
    config = parent_comm.bcast(None, root=0)
    
    domain = config['domain']
    funcs = config['funcs']
    operations = config['operations']
    y = config['y']
    lambda_reg = config['lambda_reg']
    method_str = config['method_str']
    use_gradients = config['use_gradients']
    depth1_data = config['depth1_data']  # List of (expression, params, posterior)
    all_primitives = config['all_primitives']
    T = config['T']
    
    # =========================================================
    # RECEIVE MY WORK ITEMS
    # =========================================================
    # Parent scatters work items to each worker
    my_work = parent_comm.scatter(None, root=0)
    
    # =========================================================
    # REBUILD DEPTH-1 TREES FROM SERIALIZED DATA
    # =========================================================
    depth1_trees = []
    for expr, params, post in depth1_data:
        tree = function_treeC._expr_to_tree(expr, domain, funcs)
        if tree is not None and params:
            tree.params = list(params)
        depth1_trees.append((tree, post))
    
    # =========================================================
    # PROCESS MY WORK ITEMS
    # =========================================================
    local_results = []  # (expression, params_list, posterior)
    
    for item in my_work:
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
            if result is not None:
                local_results.append(result)
            
            # Direction 2: primitive op depth-1
            tree = _combine_nodes(prim_node, d1_tree.root, op, domain, funcs, T)
            result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
            if result is not None:
                local_results.append(result)
        
        elif item_type == 'd1_d1':
            _, i, j, op = item
            
            d1_tree = depth1_trees[i][0]
            d1_tree_2 = depth1_trees[j][0]
            
            if d1_tree is None or d1_tree_2 is None:
                continue
            
            tree = _combine_nodes(d1_tree.root, d1_tree_2.root, op, domain, funcs, T)
            result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
            if result is not None:
                local_results.append(result)
    
    # =========================================================
    # SEND RESULTS BACK TO PARENT
    # =========================================================
    parent_comm.gather(local_results, root=0)
    
    # =========================================================
    # DISCONNECT AND EXIT
    # =========================================================
    parent_comm.Disconnect()


cdef function_treeC.Tree _combine_nodes(function_treeC.Node left_node, function_treeC.Node right_node, object op,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, list funcs, double T):
    """Combine two nodes with an operation into a new tree."""
    
    
    cdef function_treeC.Node left_copy = deepcopy(left_node)
    cdef function_treeC.Node right_copy = deepcopy(right_node)
    
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
    return tree


cdef tuple _evaluate_tree(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, double lambda_reg, 
                    str method_str, bint use_gradients, double T):
    """Evaluate a tree and return serializable result or None."""
    if tree is None:
        return None
    
    cdef str expression = tree.tree_get_description()
    
    if not tree.is_valid(domain):
        return None
    
    # Optimize parameters if needed
    if len(tree.params) > 0:
        tree.update_parameters(domain, y, lambda_reg, method_str, use_gradients)
    
    cdef object post = tree.get_superficial_posterior(y)
    
    if not np.isfinite(post):
        return None
    
    # Return serializable tuple
    return (expression, list(tree.params) if tree.params else [], post)


if __name__ == "__main__":
    worker_main()