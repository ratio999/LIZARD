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
    cdef list variables 
    cdef evaluate(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params = ?)   
    cdef classify_symbols(self)
    cdef fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=?, 
    object x=?, double lambda_reg=?)
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

    
