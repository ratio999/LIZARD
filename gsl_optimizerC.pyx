cimport gsl_wrapper
from libc.stdlib cimport malloc, free
cimport numpy as cnp
import numpy as np
cimport real_functionC

cdef class ObjectiveParams:
    cdef real_functionC.RealFunctionC model
    cdef double[:,:] data  # Assuming data is a 2D array of (x, y) pairs


    def __init__(self, real_functionC.RealFunctionC model, cnp.ndarray[cnp.float64_t, ndim=1] data):
        self.model = model
        self.data = data

cdef double objective(const gsl_wrapper.gsl_vector *params, void *params_struct) noexcept:
    cdef ObjectiveParams obj_params = <ObjectiveParams>params_struct
    cdef real_functionC.RealFunctionC model = obj_params.model
    cdef double[:,:] data = obj_params.data
    cdef list param_keys = model.params
    cdef double result
    cdef int n = params.size 
    cdef int i
    cdef double error = 0.0
    cdef double prediction, x, y

    for i in range(n):
        model.params[param_keys[i]] = gsl_wrapper.gsl_vector_get(params, i) 

    for i in range(data.shape[0]):
        x = data[i, 0]
        y = data[i, 1]
        prediction = model.evaluate(x)
        error += (y-prediction)**2

    return error

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
            print(f"Iteration {i}: x = ({gsl_wrapper.gsl_vector_get(x,0)}, {gsl_wrapper.gsl_vector_get(x,1)}), size = {size}")

            if size < 1e-6:  # Convergence criterion
                break
    finally:
        gsl_wrapper.gsl_multimin_fminimizer_free(s)
