cimport function_treeC
cimport numpy as cnp

cdef class MCMCSampler:
    cdef object current_posterior
    cdef function_treeC.Tree current_tree
    cdef cnp.float64_t temperature

    cdef ste(self)

    cpdef run(self, n_steps)

    cpdef get_best_tree(self)
