cimport gsl_wrapper
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np
cimport real_functionC

cdef double objective(const gsl_wrapper.gsl_vector *params, void *params_struct) noexcept:
    cdef double result
    cdef int n = 3  # Number of parameters for optimization
    result = 0.0
    for i in range(n):
        result += gsl_wrapper.gsl_vector_get(params, i) ** 2  # Example: sum of squares
    return result

cpdef optimize(f, params, data):
    cdef int n = len(params)  # Number of variables
    #cdef gsl_wrapper.gsl_vector_double* x = [0.0, 0.0]  # Initial guess
    #cdef gsl_wrapper.gsl_vector_double* step = [0.1, 0.1]  # Initial step size
    cdef gsl_wrapper.gsl_vector *x
    cdef gsl_wrapper.gsl_vector *step
    cdef gsl_wrapper.gsl_multimin_function func
    cdef gsl_wrapper.gsl_multimin_fminimizer *s

    x = gsl_wrapper.gsl_vector_alloc(n)  # Allocate gsl_vector_double for x
    step = gsl_wrapper.gsl_vector_alloc(n)  # Allocate gsl_vector_double for step

    for i in range(n):
        gsl_wrapper.gsl_vector_set(x, i, 0.0)  # Set initial guess to 0.0
        gsl_wrapper.gsl_vector_set(step, i, 0.1)  # Set initial step size to 0.1

    # Set up the GSL function
    func.n = n
    func.params = NULL
    func.f = objective

    # Allocate the minimizer
    s = gsl_wrapper.gsl_multimin_fminimizer_alloc(
        gsl_wrapper.gsl_multimin_fminimizer_nmsimplex2, n)

    try:
        gsl_wrapper.gsl_multimin_fminimizer_set(s, &func, &x[0], &step[0])

        # Perform iterations
        for i in range(100):  # Maximum iterations
            gsl_wrapper.gsl_multimin_fminimizer_iterate(s)
            size = gsl_wrapper.gsl_multimin_fminimizer_size(s)
            print(f"Iteration {i}: x = ({x[0]}, {x[1]}), size = {size}")

            if size < 1e-6:  # Convergence criterion
                break
    finally:
        gsl_wrapper.gsl_multimin_fminimizer_free(s)
