# gsl_optimizerC.pxd
cimport numpy as cnp
cimport gsl_wrapper
cdef double objective(double *params, void *data)
cpdef optimize()