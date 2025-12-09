cimport numpy as cnp
import numpy as np
cimport function_treeC
cimport real_functionC

cdef Polya(function_treeC.Tree previous_tree, list function_list, list operation_list):
    
    
cdef Dirichlet(function_treeC.Tree tree, list function_list, list operation_list):
    cdef double[:] u_k = dirichlet_params 
    cdef double[:] probs = np.random.dirichlet(u_k)
    return probs
