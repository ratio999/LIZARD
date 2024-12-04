# gsl_optimizerC.pxd
cimport numpy as cnp
cimport gsl_wrapper
cdef extern from "math.h":
    double objective(double *params, void *params_struct) noexcept
cpdef optimize(f, params, data)