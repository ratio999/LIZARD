import numpy as np
cimport numpy as cnp

cdef extern from "numerical_backend/marginalize_source.c":
    double marginalize_posterior(double *posterior_values, int n, int num_samples)

def marginalize(cnp.ndarray[cnp.double_t, ndim=1] posterior_values, int n):
    cdef int num_samples = posterior_values.shape[0]
    cdef double result = marginalize_posterior(&posterior_values[0], n, num_samples)
    return result
