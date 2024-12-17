import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport real_functionC
cimport gp_movesC
import sympy as sp

cdef class Population:
    def __init__(self, object T, list functions, dict params=None, int pop_size=100):
        cdef bint parallel_tempering
        if np.isinstance(T, np.ndarray):
            parallel_tempering = True
        else:
            parallel_tempering = False
        for i in range(pop_size):
            self.members[i] = tree(T, functions, params...)


