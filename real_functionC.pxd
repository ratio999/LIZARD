cimport numpy as cnp
import sympy as sp



cdef class RealFunctionC:
    cdef object func  # Funzione callable in Python
    cdef object expr, symbols # Espressione simbolica
    cdef bint regular  # Flag per la regolarizzazione
    cdef object domain  # Non utilizzato, puoi rimuoverlo se non serve
    cdef str description  # Descrizione della funzione
    cdef list params
    cdef list params_strs
    cdef object prior
    cdef list variables 
    cdef object cfit
    cdef object cfit_expr
    cdef evaluate(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params = ?)   
    cdef set_prior(self)
    cdef classify_symbols(self)
    cdef fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=?, 
    object x=?, double lambda_reg=?)
    cdef posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
            dict params=?, object fitness=?, object prior=?,
            object sigma=?, object x=?, double lambda_reg=?)
 
    cdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, double lambda_reg, str method_str=?)
    cdef sort_params_list(self, list pars)
    cdef get_par_string(self)
    cdef gradient(self, cnp.ndarray[cnp.float64_t, ndim=1] y, object x=?, dict params=?, 
    double lambda_reg=?)
    cpdef get_domain(self) 
    
    cpdef is_param(self) 
    
    #cpdef  evaluate(self,  cnp.ndarray[cnp.float32_t, ndim=1] x)
    
    #cpdef mutate_parameters(self)

    cpdef  derivative(self, x, h=?, dict params=?)

    #cpdef fitness(self, x, y_true)
    cdef _arr_param_expr(self, double[:] x, object x_sym)
    
    cdef _cfit(self, double[:] x, double[:] y, object x_sym)

    cdef set_cfit(self, double[:] x, double[:] y, object x_sym)

    cdef double _cfunc(self, double *x, double w, int ndim) noexcept
                    
    cpdef vegas_call(self, double[:] params_domain, int init, int n_samples,
                     int n_steps, int nprn)


cdef double _wrapper(void *self_ptr, double *x, double w, int ndim) noexcept
