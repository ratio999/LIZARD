cdef extern from "gsl/gsl_vector.h":
    # Declare the type of gsl_vector_double as a struct
    ctypedef struct gsl_vector_double:
        pass

    # Declare the GSL functions that manipulate gsl_vector_double
    cdef gsl_vector_double *gsl_vector_double_alloc(int n)  # Allocate a gsl_vector_double
    cdef void gsl_vector_double_set(gsl_vector_double *v, int i, double x)  # Set an element in gsl_vector_double
    cdef double gsl_vector_double_get(gsl_vector_double *v, int i)  # Get an element from gsl_vector_double
    cdef void gsl_vector_double_free(gsl_vector_double *v)  # Free memory allocated for gsl_vector_double


cdef extern from "gsl/gsl_multimin.h":
    ctypedef struct gsl_multimin_function:
        int n
        void *params
        double (*f)(gsl_vector_double *, void *)

    ctypedef struct gsl_multimin_fminimizer:
        pass
    
    ctypedef struct gsl_multimin_fminimizer_type:
        pass

    gsl_multimin_fminimizer *gsl_multimin_fminimizer_alloc(const gsl_multimin_fminimizer_type *T, int n)
    void gsl_multimin_fminimizer_free(gsl_multimin_fminimizer *s)
    #int gsl_multimin_fminimizer_set(gsl_multimin_fminimizer *s, gsl_multimin_function *f, double *x, double *step_size)
    int gsl_multimin_fminimizer_iterate(gsl_multimin_fminimizer *s)
    double gsl_multimin_fminimizer_size(const gsl_multimin_fminimizer *s)
    cdef int gsl_multimin_fminimizer_set(gsl_multimin_fminimizer *s, const gsl_multimin_function *f,
                                         const gsl_vector_double *x, const gsl_vector_double *step_size)

    cdef const gsl_multimin_fminimizer_type *gsl_multimin_fminimizer_nmsimplex2