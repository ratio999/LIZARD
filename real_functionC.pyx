import numpy as np
cimport numpy as cnp
import sympy as sp
cimport sympy as csp

import re
from libc.stdlib cimport malloc, free  # Not needed so far 

cdef class RealFunctionC:

    def __init__(self, object func, object expr, cnp.ndarray[cnp.float32_t, ndim=1] domain, list params=None):
        """Initialization"""
        self.func = func
        self.expr = expr 
        self.description = str(expr)  # Needed for printing 
        self.regular = False  # 100% useless 
        self.params = params
        self.domain = domain 
        self.symbols = self.expr.free_symbols
        self.params, self.variables = self.classify_symbols()      
        
        if not isinstance(domain, (np.ndarray)):  # is domain one single element? 
            if self.is_defined(domain):
                self.description = str(expr)
                self.regular = True
            else:
                self.description = None
               
        else:
            self.description = str(expr)
        self.regular = True
    

    def get_expr(self):
        return(self.expr)


    cdef classify_symbols(self):
        
        cdef set params = set()
        cdef set variables = set()

        for s in self.symbols:
            if re.match(r"^[a-zA-Z]\d+$", str(s)):
                params.add(s)
            elif re.match(r"[a-zA-Z]$", str(s)):
                variables.add(s)

        return params, variables

    #def?
    cpdef get_domain(self):
        return self.domain


    #cpdef?
    def evaluate(self,   x):
        """Evaluate function on domain"""
 
        cdef int n = x.size

        cdef cnp.ndarray[cnp.float32_t, ndim=1] results = np.zeros(n, dtype=np.float32)
        cdef object substituted_expr

        cdef object x_sym = sp.symbols('x')
      
        # Substituting x values  
        for i in range(n):
            substituted_expr = self.expr.subs(x_sym, x[i])

            if substituted_expr.is_real and substituted_expr.is_finite:
                results[i] = float(substituted_expr)  
            else:
                results[i] = np.nan  

        return results
    #cdef?
    cpdef is_param(self):
        """Is function a constant?"""
        if len(self.get_variables())==0:
            return True
        cdef cnp.ndarray[cnp.float32_t, ndim=1] y
        
    
        y = self.evaluate(self.domain)
        return np.all(np.isclose(y, y[0]))
        
    # Use as less as possible: loop on check_single_value is bottleneck --> NEEDS FIX! 
    #ONLY DEF SUPPORTS INNER DEFs!
    def is_defined(self, cnp.ndarray[cnp.float32_t, ndim=1] x):
        """ Verify if function is defined on domain"""
        cdef int n = x.size
        cdef list results = []

        cdef object x_sym = sp.symbols('x')
        def check_single_value(cnp.float64_t x_val):
            try:
                expr_eval = self.expr.subs(x_sym, x_val)
                return expr_eval.is_real and expr_eval.is_finite
            except ValueError as e:
                return False

        if isinstance(x, np.ndarray):
            for i in range(n):
                results.append(check_single_value(x[i])) 

            return np.array(results, dtype=np.bool_)
        else:
            return check_single_value(x)
    #needs improvement. useless so far. cdef?.
    cpdef derivative(self,  x,  h=1e-5):
        """Compute derivative"""
        return (self.evaluate(x + h) - self.evaluate(x - h)) / (2 * h)

    def __call__(self, cnp.ndarray[cnp.float32_t, ndim=1] x):
        """Allows standard calling"""
        return self.evaluate(x)
    # Very important: keep it Python func    
    @classmethod
    def from_string(cls, expr_str,  cnp.ndarray[cnp.float32_t, ndim=1] x):
        """Creates function class from string"""
       
      
        
        cdef object x_sym = sp.symbols('x') 
        cdef object expr
        
        expr = sp.sympify(expr_str)
        def func(cnp.float32_t input_x):
            return np.array(sp.lambdify(x_sym, expr)(input_x), dtype=np.float32)
        
        return cls(func, expr, x) 

    def get_description(self):
        """Access str(self.expr) directly --> redundancy with str(get_expr(self))"""
        return self.description

    def __add__(self, RealFunctionC other):
        """Addition of two RealFunctions."""
        cdef object new_expr = self.expr + other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) + other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __sub__(self, RealFunctionC other):
        """Subtraction of two RealFunctions."""
        cdef object new_expr = self.expr - other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) - other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __mul__(self, RealFunctionC other):
        """Multiplication of two RealFunctions."""
        cdef object new_expr = self.expr * other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) * other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __truediv__(self, RealFunctionC other):
        """Division of two RealFunctions with protection against division by zero."""
        cdef object new_expr = self.expr / other.expr

        def safe_div(cnp.ndarray[cnp.float32_t, ndim=1] x):
            cdef object numerator = self.evaluate(x)
            cdef object denominator = other.evaluate(x)
            return np.where(denominator == 0, np.inf, numerator / denominator)

        return RealFunctionC(safe_div, new_expr, self.domain)

    def compose(self, RealFunctionC other):
        """Composition of two RealFunctions (f1(f2(x)))."""
        cdef object x_sym = sp.symbols('x')
        cdef object new_expr = self.expr.subs(x_sym, other.expr)

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            cdef object is_defined_mask = other.is_defined(x)
            cdef object evaluated_inner = other.evaluate(x)
            cdef object result = np.zeros_like(x)
            result[:] = np.nan
            result[is_defined_mask] = self.evaluate(evaluated_inner[is_defined_mask])
            return result

        return RealFunctionC(new_func, new_expr, self.domain)

    def __pow__(self, RealFunctionC other):
        """Exponentiation of a RealFunction."""
        cdef object new_expr = self.expr ** other.expr

        def new_func(cnp.ndarray[cnp.float32_t, ndim=1] x):
            return self.evaluate(x) ** other.evaluate(x)

        return RealFunctionC(new_func, new_expr, self.domain)

    def get_params(self):
        """Access function parameters from Python"""
        return self.params

    def get_variables(self):
        """Access function variables from Python"""
        return self.variables

    def get_symbols(self):
        """Access function symbols from Python"""
        return self.symbols
    


cdef is_parameter(str name):
    """Is name a parameter declaration?"""
    return bool(re.fullmatch(r'[a-zA-Z]\d+', name))

