# gsl_optimizerC.pxd
cimport numpy as cnp
cimport gsl_wrapper
cdef extern from "math.h":
    double objective(gsl_wrapper.gsl_vector *params, void *params_struct) noexcept
cpdef optimize(f, params, data)