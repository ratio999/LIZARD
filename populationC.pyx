import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport real_functionC
cimport gp_movesC
import sympy as sp

cdef class Population:
    def __init__(self, object T, list functions, dict params=None, int pop_size=100, object members=None):
        cdef int i
        if np.isinstance(T, np.ndarray):
            self.parallel_tempering = True
        else:
            self.parallel_tempering = False
            
        if members is None:
            for i in range(pop_size):
                if self.parallel_tempering:
                    self.members[i] = tree(T[i], functions, params...)
                else:
                    self.members[i] = tree(T, functions, params...)
        self.generation=0

     


