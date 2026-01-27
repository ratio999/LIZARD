cimport numpy as cnp
cimport function_treeC
cimport real_functionC

cdef function_treeC.Tree _helper(object expr, cnp.ndarray[cnp.float64_t, ndim=1] domain, 
            list funcs, dict names_params)

cdef function_treeC.Tree _combine_nodes(function_treeC.Node left_node, function_treeC.Node right_node, object op,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, list funcs, double T)

cdef tuple _evaluate_tree(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, double lambda_reg, 
                    str method_str, bint use_gradients, double T)
