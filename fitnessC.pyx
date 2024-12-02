cimport numpy as cnp
import numpy as np
cimport function_treeC
cimport real_functionC

cdef calculate_likelihood(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] x, 
    cnp.ndarray[cnp.float64_t, ndim=1] y):
    # Valuta il modello rappresentato dall'albero sui dati
    cdef cnp.float64_t out = 0.0

    return out