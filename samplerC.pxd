cimport numpy as cnp
cimport forestC
cdef class MCMCSampler:
    cdef forestC.Forest forest
    cdef float[:] current_posterior
    cdef object data
    
    cdef propose_new_trees(self)

    cdef step(self)

    cdef _run(self, n_steps)

    cpdef get_best_tree(self)
