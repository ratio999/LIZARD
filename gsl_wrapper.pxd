cdef extern from "gsl/gsl_multimin.h":
    ctypedef struct gsl_multimin_function:
        int n
        void *params
        double (*f)(double *, void *)

    ctypedef struct gsl_multimin_fminimizer:
        pass
    
    ctypedef struct gsl_multimin_fminimizer_type:
        pass

    gsl_multimin_fminimizer *gsl_multimin_fminimizer_alloc(const gsl_multimin_fminimizer_type *T, int n)
    void gsl_multimin_fminimizer_free(gsl_multimin_fminimizer *s)
    int gsl_multimin_fminimizer_set(gsl_multimin_fminimizer *s, gsl_multimin_function *f, double *x, double *step_size)
    int gsl_multimin_fminimizer_iterate(gsl_multimin_fminimizer *s)
    double gsl_multimin_fminimizer_size(const gsl_multimin_fminimizer *s)

    cdef const gsl_multimin_fminimizer_type *gsl_multimin_fminimizer_nmsimplex2