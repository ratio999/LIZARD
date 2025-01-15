cimport numpy as cnp
cimport groveC
#cimport function_treeC
cdef class MCMCWalker:
    cdef groveC.Grove grove
    cdef bint parallel_bint
    cdef object data
    
    cdef evolve(self, int i)

    cpdef parallel_evolve(self)
    
    cdef _run(self, n_steps)

    cpdef get_best_tree(self)
