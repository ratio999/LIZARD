cimport numpy as cnp
import numpy as np
cimport function_treeC
cimport real_functionC

cdef calculate_posterior(function_treeC.Tree tree):
    # Valuta il modello rappresentato dall'albero sui dati
    cdef cnp.float64_t out = 0.0

    return out