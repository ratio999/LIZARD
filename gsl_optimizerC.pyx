cimport gsl_wrapper
from libc.stdlib cimport malloc, free

cdef double objective(double *params, void *data):
    # Simple quadratic function: (x-2)^2 + (y-3)^2
    return (params[0] - 2.0)**2 + (params[1] - 3.0)**2

cpdef optimize():
    cdef int n = 2  # Number of variables
    cdef double x = [0.0, 0.0]  # Initial guess
    cdef double step = [0.1, 0.1]  # Initial step size
    cdef gsl_wrapper.gsl_multimin_function func
    cdef gsl_wrapper.gsl_multimin_fminimizer *s

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
