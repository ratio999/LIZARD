cimport numpy as cnp
cimport function_treeC

cdef calculate_likelihood(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] x, 
    cnp.ndarray[cnp.float64_t, ndim=1] y)