cimport function_treeC
cimport numpy as cnp

cdef calculate_posterior(function_treeC.Tree tree,cnp.ndarray[cnp.float64_t, ndim=1] data)