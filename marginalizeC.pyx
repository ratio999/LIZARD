import numpy as np
cimport numpy as cnp

cdef extern from "numerical_backend/marginalize_source.c":
    double marginalize_posterior(double *posterior_values, int n, int num_samples)

def marginalize(cnp.ndarray[cnp.double_t, ndim=1] posterior_values, int n):
    cdef int num_samples = posterior_values.shape[0]
    cdef double result = marginalize_posterior(&posterior_values[0], n, num_samples)
    return result

cdef extern from "numerical_backend/static_vegas.c":
    void static_vegas(double regn[], int ndim, double (*fxn)(double[], double), int init, unsigned long ncall, int itmx, int nprn, double *tgral, double *sd, double *chi2a)

def vegas(cnp.ndarray[cnp.float64_t, ndim=1] regn, int ndimm, 
          callable fxn, int init, unsigned long ncall, int itmx,
          int nprn, cnp.ndarray[cnp.float64_t, ndim=1] tgral,
          cnp.ndarray[cnp.float64_t, ndim=1] sd, cnp.ndarray[cnp.float64_t, ndim=1] chi2a):
    
    cdef double fxn_c(double regn[], double weight) nogil:
        cdef cnp.ndarray[cnp.float64_t, ndim=1] regn_array = np.array(regn, dtype=np.float64)
        return fxn(regn_array, weight)

    cdef double *regn_ptr <double *> regn.data
    cdef double *tgral_ptr = <double *> tgral.data
    cdef double *sd_ptr = <double *> sd.data
    cdef double *chi2a_ptr = <double *> chi2a.data

    vegas(regn_ptr, ndimm, fxn_c, init, call, itmx, nprn, 
          tgral_ptr, sd_ptr, chi2a_ptr)

    return (tgral_ptr, sd_ptr, chi2a_ptr)
