cimport numpy as cnp
cimport function_treeC

cpdef calculate_likelihood(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y)