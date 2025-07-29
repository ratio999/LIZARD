import numpy as np
cimport numpy as cnp
cimport function_treeC

cdef class Grove:
    cdef list operations
    cdef int[:] depth
    cdef list funcs
    cdef object x
    cdef list nodes
    cdef int n_pars
    cdef double[:] T
    cdef int size
    cdef object root_posteriors
    cdef list trees

    cpdef set_trees(self, list new_trees)

    cpdef set_single_tree(self, function_treeC.Tree tree, int i)

    cdef grow_grove(self)

    cdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    
    cdef posterior_ranked_trees(self)

    cdef posterior_ranked_neighbours(self, int index)
