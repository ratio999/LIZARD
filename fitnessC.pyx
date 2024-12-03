cimport numpy as cnp
import numpy as np
cimport function_treeC
cimport real_functionC

cpdef calculate_likelihood(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y):
    cdef cnp.float64_t mse
    cdef cnp.ndarray[cnp.float64_t, ndim=1] values = tree.evaluate()
    #values[np.where(np.isnan(values))] = np.inf
    try:
        mse = np.nanmean((values - y)**2)
    except:
        mse = np.inf
    #cdef cnp.float64_t out = 0.0

    return mse
