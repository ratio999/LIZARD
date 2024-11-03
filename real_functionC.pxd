cimport numpy as cnp
import sympy as sp

cdef class RealFunctionC:
    cdef object func  # Funzione callable in Python
    cdef object expr  # Espressione simbolica
    cdef bint regular  # Flag per la regolarizzazione
    cdef object domain  # Non utilizzato, puoi rimuoverlo se non serve
    cdef str description  # Descrizione della funzione
    cdef list params

    
 
    
    cpdef  evaluate(self,  x)
    
    cpdef mutate_parameters(self)

    cpdef  derivative(self, x, h=?)

    cpdef fitness(self, x, y_true)

    
