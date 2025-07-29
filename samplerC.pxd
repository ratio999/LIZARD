cimport numpy as cnp
cimport groveC
#cimport function_treeC
cdef class MCMCWalker:
    cdef groveC.Grove grove
    cdef bint parallel_bint
    cdef object data
    cdef int max_iter
    cdef bint verbose
    cdef list funcs
    cdef object operations
    cdef dict standard_probabilities
    cdef dict gp_probabilities
    cdef dict op_range


    cdef print_best_tree(self)    
    cdef evolve(self, int i)
    
    cpdef parallel_evolve(self)
    
    cdef _run(self, n_steps)

    cpdef get_best_tree(self)
