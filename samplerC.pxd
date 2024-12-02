cimport function_treeC
cimport numpy as cnp

cdef class MCMCSampler:
    cdef object current_posterior
    cdef dict current_trees
    cdef cnp.float64_t temperature

    cdef step(self)

    cpdef run(self, n_steps)

    cpdef get_best_tree(self)
