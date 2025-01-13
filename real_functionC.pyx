import numpy as np
cimport numpy as cnp
import sympy as sp
#cimport sympy as csp
from scipy.optimize import minimize
import re
#from libc.stdlib cimport malloc, free  # Not needed so far 
from cython.cimports.libc.string import memcpy
def safe_convert(value):
        try:
            return np.float64(value)
        except (TypeError, ValueError):
            #print('!\n!\n|\n!\n!')
            return np.nan

cdef class RealFunctionC:

    def __init__(self, object func, object expr, cnp.ndarray[cnp.float64_t, ndim=1] domain, list params=None):
        """Initialization"""
        self.func = func
        self.expr = expr 
        self.description = str(expr)  # Needed for printing 
        self.regular = False  # 100% useless 
        self.params = params
        self.domain = domain 
        self.symbols = self.expr.free_symbols
        self.params, self.variables = self.classify_symbols()      
        self.params_strs = self.get_par_string()         
        if not isinstance(domain, (np.ndarray)):  # is domain one single element? 
            if self.is_defined(domain):
                self.description = str(expr)
                self.regular = True
              
            else:
                self.description = None
               
        else:
            self.description = str(expr)
        self.regular = True
    
    cdef get_par_string(self):
        return [str(i) for i in self.params]

    def get_expr(self):
        return(self.expr)


    cdef classify_symbols(self):
        
        cdef list params = []
        cdef list variables = []

        for s in self.symbols:
            if re.match(r"^[a-zA-Z]\d+$", str(s)):
                params.append(s)
            elif re.match(r"[a-zA-Z]$", str(s)):
                variables.append(s)

        return params, variables

    #def?
    cpdef get_domain(self):
        return self.domain

    cdef evaluate(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
        if params is None:
            params = {}
        
        return self.func(x, params=params)
    
    def get_eval(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
        return self.evaluate(x, params)

    cdef posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                   dict params=None, object fitness=None, object prior=None, 
                   object sigma=1.00, object x = None, double lambda_reg = 0.0):
        if fitness is None:
            fitness = self.fitness(y, params, x, lambda_reg)
        if x is None:
            x = self.domain

        try:
            return -fitness/(2 * sigma**2) - prior  
        except:
            return -np.inf

    cdef fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=None, object x = None, double lambda_reg=0.0): 
        if x is None:
            x = self.domain
        try:
            #if self.is_param():
                #print(np.nanmean((y - self.evaluate(self.domain, params))**2))
            return np.nanmean((y - self.evaluate(x,  params))**2) + lambda_reg * np.sum(np.array(list(params.values()))**2)
        except:
            return np.inf

    cdef gradient(self, cnp.ndarray[cnp.float64_t, ndim=1] y, object x = None, dict params=None, double lambda_reg = 0.0):
        if x is None:
            x=self.domain
        cdef double[:] y_pred = self.evaluate(x, params)
        cdef double[:] residuals = y_pred - y
        cdef double grad_a0, grad_a1, grad_a2
        grad_a0 = 2 * np.mean(residuals * x**2)
        grad_a1 = 2 * np.mean(residuals *x)
        grad_a2 = 2 * np.mean(residuals)
        

        cdef object par_values = np.array(list(params.values()))
        return 2*par_values + 2*lambda_reg*par_values

    def objective(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data, double lambda_reg):
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self.fitness( y_data,  params_dict, x_data, lambda_reg)

    def grad(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data, double lambda_reg):   
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self.gradient( y_data, x_data, params_dict, lambda_reg)

    cdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data, dict input_pars, double lambda_reg, str method_str='L-BFGS-B'):
        cdef double[:] initial_params = np.array(list(input_pars.values()))
        cdef object result = minimize(
                fun=self.objective,
                x0=initial_params,
                jac=self.grad,
                args=(x_data, y_data, lambda_reg),
                method=method_str,
                options={'disp': True}
                )
        #self.params = dict(zip(self.param_str, result.x))
        return result

    cdef sort_params_list(self, list pars):

        try:
            self.params = sorted(self.params, key=lambda n: pars.index(n))
            print('sorting went fine')
            self.params_strs = self.get_par_string()         
        except:
            print('Issues: ', self.params, pars)
        

    #cdef?
    cpdef is_param(self):
        """Is function a constant?"""
        if len(self.variables)==0:
            return True
        ''''
        cdf cnp.ndarray[cnp.float64_t, ndim=1] y
        
    
        y = self.evaluate(self.domain)
        return np.all(np.isclose(y, y[0]))
        '''
    
    # Use as less as possible: loop on check_single_value is bottleneck --> NEEDS FIX! 
    #ONLY DEF SUPPORTS INNER DEFs!
    def is_defined(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params = None):
        if params is None:
            params = {}
        try:
            y = self.evaluate(x, params)
            vector_safe_convert = np.vectorize(safe_convert)
            y_safe = vector_safe_convert(y)
            return np.isfinite(y_safe) & np.isreal(y_safe)
        except Exception as e:
            return np.full_like(x, False, dtype=np.bool_)
    
    #needs improvement. useless so far. cdef?.
    cpdef derivative(self,  x,  h=1e-5, dict params = None):
        """Compute derivative"""
        if params is None:
            params = {}

        return (self.evaluate(x + h, params) - self.evaluate(x - h, params)) / (2 * h)

    def __call__(self, cnp.ndarray[cnp.float64_t, ndim=1] x):
        """Allows standard calling"""
        return self.evaluate(x)
    # Very important: keep it Python func    
    @classmethod
    def from_string(cls, expr_str,  cnp.ndarray[cnp.float64_t, ndim=1] x):
        """Creates function class from string"""
       
      
        
        cdef object x_sym = sp.symbols('x') 
        cdef object expr
        
        expr = sp.sympify(expr_str)
        def func(cnp.ndarray[cnp.float64_t, ndim=1] input_x, dict params = None):
            if params is None:
                params = {}
            expr_with_params = expr
            for param_name, param_value in params.items():
                expr_with_params = expr_with_params.subs(sp.symbols(param_name), param_value)
            result = sp.lambdify(x_sym, expr_with_params, "numpy")(input_x)
            
            try: 
                vector_safe_convert = np.vectorize(safe_convert)
                result_safe = vector_safe_convert(result)
                return np.array(result_safe, dtype=np.float64)
            except:
                return np.full_like(input_x, np.nan, dtype=np.float64)
           
        return cls(func, expr, x) 

    def get_description(self):
        """Access str(self.expr) directly --> redundancy with str(get_expr(self))"""
        return self.description

    def __add__(self, RealFunctionC other):
        """Addition of two RealFunctions."""
        cdef object new_expr = self.expr + other.expr
        
        def new_func(cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
            return self.evaluate(x, params) + other.evaluate(x, params)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __sub__(self, RealFunctionC other):
        """Subtraction of two RealFunctions."""
        cdef object new_expr = self.expr - other.expr

        def new_func(cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
            return self.evaluate(x, params) - other.evaluate(x, params)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __mul__(self, RealFunctionC other):
        """Multiplication of two RealFunctions."""
        cdef object new_expr = self.expr * other.expr

        def new_func(cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
            return self.evaluate(x, params) * other.evaluate(x, params)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __truediv__(self, RealFunctionC other):
        """Division of two RealFunctions with protection against division by zero."""
        cdef object new_expr = self.expr / other.expr

        def safe_div(cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
            cdef object numerator = self.evaluate(x, params)
            cdef object denominator = other.evaluate(x, params)
            return np.where(denominator == 0, np.inf, numerator / denominator)

        return RealFunctionC(safe_div, new_expr, self.domain)

    def compose(self, RealFunctionC other):
        """Composition of two RealFunctions (f1(f2(x)))."""
        cdef object x_sym = sp.symbols('x')
        cdef object new_expr = self.expr.subs(x_sym, other.expr)
        if self.is_param():
            new_expr = self.expr * other.expr
        else:
            new_expr = self.expr.subs(x_sym, other.expr)
 
            
        def new_func(cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
            if self.is_param(): 
                return self.evaluate(x, params) * other.evaluate(x, params)

            cdef object is_defined_mask = other.is_defined(x, params)
            cdef object evaluated_inner = other.evaluate(x, params)
            cdef object result = np.zeros_like(x)
            result[:] = np.nan
            try:
                result[is_defined_mask] = self.evaluate(evaluated_inner[is_defined_mask], params)
            except:
                pass
            return result

        return RealFunctionC(new_func, new_expr, self.domain)

    def __pow__(self, RealFunctionC other):
        """Exponentiation of a RealFunction."""
        cdef object new_expr = self.expr ** other.expr

        def new_func(cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
            return self.evaluate(x, params) ** other.evaluate(x, params)

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

