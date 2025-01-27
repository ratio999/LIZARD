import numpy as np
cimport numpy as cnp
import ctypes as ct
cimport ctypes as cct

cdef extern from "numerical_backend/marginalize_source.c":
    double marginalize_posterior(double *posterior_values, int n, int num_samples)
cdef extern from "numerical_backend/static.c":
    int vegas(double *regn, int ndim, double (*fxn)(double*, double), 
               int init, unsigned long ncall, int itmx, int nprn,
               double *tgral, double *sd, double *chi2a)



def marginalize(cnp.ndarray[cnp.double_t, ndim=1] posterior_values, int n):
    cdef int num_samples = posterior_values.shape[0]
    cdef double result = marginalize_posterior(&posterior_values[0], n, num_samples)
    return result
'''
cdef extern from "numerical_backend/static_vegas.c":
    void static_vegas(double regn[], int ndim, double (*fxn)(double[], double), int init, unsigned long ncall, int itmx, int nprn, double *tgral, double *sd, double *chi2a)
'''


def vegas(double[:] regn, int ndimm, 
          fxn, int init, unsigned long ncall, int itmx,
          int nprn, cnp.ndarray[cnp.double_t, ndim=1] tgral,
          cnp.ndarray[cnp.double_t, ndim=1] sd, cnp.ndarray[cnp.double_t, ndim=1] chi2a):
    
    '''
    cdef double *regn_ptr = <double *> regn.data
    cdef double *tgral_ptr = <double *> tgral.data
    cdef double *sd_ptr = <double *> sd.data
    cdef double *chi2a_ptr = <double *> chi2a.data
    '''
    #cdef double (*fxn_c_ptr)(double *, double) 
    #fxn_c_ptr= <double (*)(double *, double)> fxn
    
    #cdef PyObject* py_fxn = <PyObject*> fxn
    #cdef double (*cfunc)(double[:] arr, double weight)
    c_func_type = ct.CFUNCTYPE(ct.c_double, ct.POINTER(ct.c_double), ct.c_double)
    c_func = c_func_type(fxn)

    static_vegas(&regn[0], ndimm, c_func, init, ncall, itmx, nprn, 
          &tgral[0], &sd[0], &chi2a[0])

    return (tgral, sd, chi2a)
