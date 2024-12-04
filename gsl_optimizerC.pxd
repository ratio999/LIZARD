# gsl_optimizer.pxd
cdef extern from "gsl/gsl_multimin.h":
    ctypedef struct gsl_multimin_fminimizer
    ctypedef struct gsl_multimin_function

    # Dichiarazione della funzione di minimizzazione Nelder-Mead
    gsl_multimin_fminimizer *gsl_multimin_fminimizer_alloc(int n)
    void gsl_multimin_fminimizer_free(gsl_multimin_fminimizer *s)
    int gsl_multimin_fminimizer_minimum(gsl_multimin_fminimizer *s)

    # Dichiarazione della funzione obiettivo
    int gsl_multimin_function_fdf(cdef gsl_multimin_function *f, cdef double *x)