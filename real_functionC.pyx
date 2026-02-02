import numpy as np
cimport numpy as cnp
import sympy as sp
from wrapperC cimport a_vegas
from cpython.ref cimport Py_INCREF, Py_DECREF
#cimport sympy as csp
from utils.pointer_utils cimport python_object_to_void_ptr, void_ptr_to_python_object, void_ptr_to_rf
from scipy.optimize import minimize
import re
#from libc.stdlib cimport malloc, free  # Not needed so far 
from cython.cimports.libc.string import memcpy
from copy import copy, deepcopy
def safe_convert(value):
        try:
            return np.float64(value)
        except (TypeError, ValueError):
            #print('!\n!\n|\n!\n!')
            return np.nan

cdef class RealFunctionC:

    def __init__(self, object func, object expr, cnp.ndarray[cnp.float64_t, ndim=1] domain, list params=None, mode='Time', bint compute_grad = False, double constant_value=1.0):
        """Initialization"""
        self.func = func
        self.mode = mode 
        self.expr = expr 
        self.description = str(expr)  # Needed for printing 
        self.regular = False  # 100% useless 
        self.params = params
        self.domain = _domain_fixer(domain) 
        self.symbols = self.expr.free_symbols
        self.params, self.variables = self.classify_symbols()      
        self.params_strs = self.get_par_string()         
        self.prior = None
        if self.is_param():
            self.constant_value = constant_value
        #self.is_differentiable = compute_grad
        else:
            self.constant_value = None
        self.grad_funcs = self._build_gradient_functions() if compute_grad else None
        #self.fit = self.fitness()
        if not isinstance(domain, (np.ndarray)):  # is domain one single element? 
            if self.is_defined(domain):
                self.description = str(expr)
                self.regular = True
              
            else:
                self.description = None
               
        else:
            self.description = str(expr)
        self.regular = True
        self.cfit = None
        self.cfit_expr = None
    
    
        
    
    def __copy__(self):
        cdef object copy_func = lambda input_x, params: self.func(input_x, params)
        return RealFunctionC(copy_func, self.expr, copy(self.domain), copy(self.params))

    def __deepcopy__(self, memo2):
        cdef object copy_func = lambda input_x, params: self.func(input_x, params)
        cdef RealFunctionC copy_rf = RealFunctionC(
                copy_func, deepcopy(self.expr, memo2), deepcopy(self.domain, memo2),
                deepcopy(self.params, memo2)
                )
        memo2[id(self)] = copy_rf
        return copy_rf

    cdef _update_symbols(self):
        self.symbols = self.expr.free_symbols
        self.params, self.variables = self.classify_symbols()
    
    cdef _arr_param_expr(self, double[:] x, object x_sym):
        
        cdef object expr_with_params = copy.copy(self.expr)
        
        return [expr_with_params.subs(x_sym, x[i]) 
                for i in range(0, x.shape[0])]

    cdef _cfit(self, double[:] x, double[:] y, object x_sym):
        
        cdef list expr_list = self._arr_param_expr(x, x_sym)  
        
        cdef object _cfit_expr = -0.5 * (y[0] - expr_list[0])**2
        for i in range(1, x.shape[0]):
            _cfit_expr -= 0.5 * (y[i] - expr_list[i])**2

        cdef object _cfit = sp.lambdify(self.params,
                    _cfit_expr, "numpy")

        return _cfit, _cfit_expr

    cdef set_cfit(self, double[:] x, double[:] y, object x_sym):
        self.cfit, self.cfit_expr = self._cfit(x, y, x_sym)
        
    cdef double _cfunc(self, double *x, double w, int ndim) noexcept:
        cdef double[:] args = np.array([x[i] for i in range (0, ndim) ])
        return self.cfit(*args)
    

    cpdef vegas_call(self, double[:] params_domain, int init, int n_samples,
                     int n_steps, int nprn):
        cdef void* ptr = python_object_to_void_ptr(self)
        cdef double[:] tgral = np.array([0.0])
        cdef double[:] sd =  np.array([0.0])
        cdef double[:] chi2a = np.array([0.0])
        
        cdef int ndim = int(np.shape(params_domain)[0] / 2)
        cdef long idum = -12345
        a_vegas(ptr, &params_domain[0], ndim, _wrapper, init, n_samples, n_steps, 
              nprn, &tgral[0], &sd[0], &chi2a[0], idum)

        return (*tgral, *sd, *chi2a)

    

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

    cdef set_prior(self):
        self.prior = 0.0

    #START FROM HERE TO INCORPORATE PRIOR DEFINITIONS FROM ANOTHER MODULE

    cpdef get_domain(self):
        return self.domain
    
    cdef time_to_frequency(self, double[:] x, object data):
        cdef double dt = x[1]-x[0]
        cdef double fs = 1/dt
        cdef double[:] f = np.fft.fftfreq(x.shape[0], dt)
        cdef complex[:] dft = np.fft.fft(data) * dt
        return f, dft
        
    cdef bint is_valid(self, object x=None, double max_abs=1e12):
                #print('checking validity...')
        cdef object yhat
        with np.errstate(all='ignore'):
            try:
                yhat = self.evaluate(self.domain if x is None else x)
                #print('yhat: ', yhat)
            except Exception:
                return False
        #print('1-passed')
        yhat = np.asarray(yhat)
        if yhat.size == 0:
            return False
        #print('2-passed')
        if not np.all(np.isfinite(yhat)):
            #print(yhat)
            return False
        #print('3-passed')
        if np.any(np.abs(yhat) > max_abs):
            return False
        #print('is_valid passed for expression:', self.get_description())
        return True


    cdef object evaluate(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
        #print(params)
        if params is None:
            #print('params is NONE')
            params = {}
        #print('params in eval ', params)
        #print('in evaluate: ', self.func(x, params=params))
        if self.is_param():
            return np.full_like(x, self.constant_value, dtype=np.float64).ravel()
        cdef object out = self.func(x, params=params)
        
        try:
            vector_safe_convert = np.vectorize(safe_convert)
            out_safe = vector_safe_convert(out)
            return np.ascontiguousarray(out_safe, dtype=np.float64).ravel()
        except:
            return np.full_like(x, np.nan, dtype=np.float64)
        #return self.func(x, params=params)
    
    def get_eval(self, cnp.ndarray[cnp.float64_t, ndim=1] x, dict params=None):
        return self.evaluate(x, params)

    cdef posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                   dict params=None, object fitness=None, object prior=None, 
                   object sigma=1.00, object x = None,  double penalty=0.0):
        if fitness is None:
            #print('FITNESS IS NONE')
            
            if x is None:
                x = self.domain
            fitness = self.posterior_fitness(y, params, x, penalty)
        
        
        if prior is None:
            prior = 0.0
        #print('fitness is: ', fitness, ' and prior is: ', prior)
        try:
            return -fitness/(2 * sigma**2) - prior  
        except:
            return -np.inf

    cdef posterior_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=None, object x = None, double penalty=0.0): 
        if x is None:
            x = self.domain
        
        cdef double mse = np.nanmean((y - self.evaluate(x, params))**2)
        
        # Fast path: skip regularization when lambda is 0
        
        
        
        return mse + penalty
        
    
    cdef fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=None, object x = None): 
        #print(x)
        if x is None:
            x = self.domain
        try:
            #if self.is_param():
            #print('params in fitness', params, self.evaluate(x,params))
                #print(np.nanmean((y - self.evaluate(self.domain, params))**2))
            return np.nanmean((y - self.evaluate(x,  params))**2) #+ lambda_reg * np.sum(np.array(list(params.values()))**2)
        except:
            return np.inf
    


    cdef dict _build_gradient_functions(self):
        """
        Compute analytical gradients symbolically using SymPy.
        Returns dict: {param_name: lambdified_gradient_function}
        """
        if len(self.params) == 0:
            return {}
        
        cdef dict grad_dict = {}
        cdef object x_sym = sp.symbols('x')
        cdef object param_sym, grad_expr, grad_func
        cdef RealFunctionC diff_obj
        for param_sym in self.params:
            # Symbolic differentiation: d(expr)/d(param)
            grad_expr = sp.diff(self.expr, param_sym)
            # Include all params + x as arguments
            diff_obj = RealFunctionC.from_string(str(grad_expr), self.domain, compute_grad=False)
            #print('Gradient expression for param', str(param_sym), ':', grad_expr, ' in ', self.expr)
            if diff_obj.is_valid(self.domain):    
                all_syms = [x_sym] + list(self.params)
                grad_func = sp.lambdify(all_syms, grad_expr, "numpy")
                
                grad_dict[str(param_sym)] = grad_func

        return grad_dict

    cdef _symbolic_gradient(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                  object x=None, dict params=None):
        """
        Compute analytical gradient of MSE loss w.r.t. parameters.
        Uses chain rule: dLoss/dParam = dLoss/dPred * dPred/dParam
        """
        if x is None:
            x = self.domain
        if params is None or len(params) == 0:
            return np.array([], dtype=np.float64)
        
        # Predictions and residuals
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_pred = self.evaluate(x, params)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] residuals = y_pred - y
        cdef int n = len(y)
        
        # Gradient array: one element per parameter
        cdef cnp.ndarray[cnp.float64_t, ndim=1] grad = np.zeros(len(params), dtype=np.float64)
        cdef int i = 0
        cdef str param_name
        cdef object grad_func
        cdef cnp.ndarray[cnp.float64_t, ndim=1] dpred_dparam
        
        for param_name in self.params_strs:
            if param_name not in self.grad_funcs:
                # Fallback: finite differences if symbolic gradient not available
                grad[i] = self._finite_diff_gradient(param_name, params, x, y)
            else:
                grad_func = self.grad_funcs[param_name]
                
                # Evaluate d(f)/d(param) at each x point
                # grad_func expects (x, a0, a1, ...) where params are in order
                param_values = [params[p] for p in self.params_strs]
                
                try:
                    dpred_dparam = np.asarray(grad_func(x, *param_values), dtype=np.float64)
                    
                    # Chain rule: dLoss/dParam = (2/n) * sum(residuals * dpred/dparam)
                    grad[i] = (2.0 / n) * np.nansum(residuals * dpred_dparam)
                    
                    # Add regularization term: d(lambda*param^2)/d(param) = 2*lambda*param
                    #grad[i] += 2.0 * lambda_reg * params[param_name]
                    
                except Exception as e:
                    # If evaluation fails, use finite differences
                    grad[i] = self._finite_diff_gradient(param_name, params, x, y)
            
            i += 1
        
        return grad

    cdef _symbolic_gradient_parallel(self, cnp.ndarray[cnp.float64_t, ndim=1] y, object x=None, 
              dict params=None):
        if x is None:
            x = self.domain
        if params is None or len(params) == 0:
            return np.array([], dtype=np.float64)
        
        # Build gradient functions if not already done
        if not hasattr(self, 'grad_funcs') or self.grad_funcs is None:
            self.grad_funcs = self._build_gradient_functions()
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_pred = self.evaluate(x, params)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] residuals = y_pred - y
        cdef int n = len(y)
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1] grad = np.zeros(len(params), dtype=np.float64)
        cdef list param_values = [params[p] for p in self.params_strs]
        cdef int i
        cdef object grad_func
        cdef cnp.ndarray[cnp.float64_t, ndim=1] dpred_dparam
        
        # Compute gradient for each parameter
        for i in range(len(self.params_strs)):
            grad_func = self.grad_funcs.get(self.params_strs[i])
            
            if grad_func is not None:
                # Evaluate d(f)/d(param) - this is where NumPy vectorization helps
                dpred_dparam = np.asarray(grad_func(x, *param_values), dtype=np.float64)
                
                # Chain rule: dLoss/dParam = (2/n) * sum(residuals * dpred/dparam)
                grad[i] = (2.0 / n) * np.nansum(residuals * dpred_dparam)
                
                # Add L2 regularization term
                #grad[i] += 2.0 * lambda_reg * params[self.params_strs[i]]
            else:
                # Fallback: finite differences
                grad[i] = 0.0  # or implement finite diff
        
        return grad

    cdef double _finite_diff_gradient(self, str param_name, dict params, 
                                       cnp.ndarray[cnp.float64_t, ndim=1] x,
                                       cnp.ndarray[cnp.float64_t, ndim=1] y,
                                       double h=1e-7):
        """Finite difference fallback for gradient computation."""
        cdef dict params_plus = params.copy()
        cdef dict params_minus = params.copy()
        
        params_plus[param_name] = params[param_name] + h
        params_minus[param_name] = params[param_name] - h
        
        cdef double f_plus = self.fitness(y, params_plus, x)
        cdef double f_minus = self.fitness(y, params_minus, x)
        
        return (f_plus - f_minus) / (2.0 * h)
    
    #deprecated
    cdef gradient(self, cnp.ndarray[cnp.float64_t, ndim=1] y, object x = None, dict params=None):
        if x is None:
            x=self.domain
        if params is None or len(params)==0:
            return np.array([], dtype=np.float64)
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_pred = np.ascontiguousarray(self.evaluate(x, params), dtype=np.float64).ravel()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_true = np.ascontiguousarray(y, dtype=np.float64).ravel()
        cdef cnp.ndarray[cnp.float64_t, ndim=1] residuals = y_pred - y_true
        cdef cnp.ndarray[cnp.float64_t, ndim=1] grad_a0, grad_a1, grad_a2
        #grad_a0 = 2 * np.mean(residuals * x**2)
        #grad_a1 = 2 * np.mean(residuals *x)
        #grad_a2 = 2 * np.mean(residuals)
        

        cdef cnp.ndarray[cnp.float64_t, ndim=1] par_values = np.ascontiguousarray(list(params.values()), dtype=np.float64)
        return 2*par_values 

    def objective(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data):
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self.fitness( y_data,  params_dict, x_data)
    

    def grad(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data):   
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self.gradient( y_data, x_data, params_dict)
    
    def symb_grad(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data):   
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self._symbolic_gradient( y_data, x_data, params_dict)
    
    cdef _optimize_with_grad(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, 
              cnp.ndarray[cnp.float64_t, ndim=1] y_data, 
              dict input_pars,  str method_str='L-BFGS-B'):
    
        if len(input_pars) == 0:
            return None  # No parameters to optimize
        
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1] initial_params = np.ascontiguousarray(
            list(input_pars.values()), dtype=np.float64)
        
        # Use gradient-based method if gradients available
        if self.grad_funcs is None:
            try:
                self.grad_funcs = self._build_gradient_functions()
                
            except:
                return self.optimize(x_data, y_data, input_pars, method_str='Powell')
        
        cdef cnp.ndarray[cnp.float64_t, ndim=1] test_grad = self.symb_grad(initial_params, x_data, y_data)
        #print('Done computing gradients')
        use_gradients=False
        
        if len(self.grad_funcs) != initial_params.shape[0] or np.any(~np.isfinite(test_grad)) or np.allclose(test_grad, 0.0):    
            return self.optimize(x_data, y_data, input_pars, method_str='Powell')
        else:
            use_gradients=True

        cdef object result = minimize(
            fun=self.objective,
            x0=initial_params,
            jac=self.symb_grad if use_gradients else None,
            args=(x_data, y_data),
            method=method_str if use_gradients else 'Powell',
            options={'disp': False}
        )
        #if use_gradients:
            #print('Optimization with gradients succeeded.')
        #else:
            #print('Optimization without gradients (Powell) succeeded.')
        
        return result
    
    
    
    cdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str='Powell'):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] initial_params = np.ascontiguousarray(list(input_pars.values()), dtype=np.float64)
        #cdef double[:] initial_params = np.array(list(input_pars.values()))
        cdef object result = minimize(
                fun=self.objective,
                x0=initial_params,
                jac=self.grad,
                args=(x_data, y_data),
                method=method_str,
                options={'disp': False}
                )
        #self.params = dict(zip(self.param_str, result.x))
        
        return result

    cdef sort_params_list(self, list pars):

        try:
            try:
                self.params = sorted(self.params, key=lambda n: pars.index(n))
                #print('sorting went fine')
                self.params_strs = self.get_par_string()
            except:
                self._update_symbols()         
                self.params = sorted(self.params, key=lambda n: pars.index(n))
                #print('sorting went fine')
                self.params_strs = self.get_par_string()
        except:
            print('Issues: ', self.params, pars, self.description)
        

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
    
    # Use as less as possible: loop on check_single_value is bottleneck 
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
    def from_string(cls, expr_str,  cnp.ndarray[cnp.float64_t, ndim=1] x, compute_grad=True):
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
           
        return cls(func, expr, x, compute_grad=compute_grad) 

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

cdef double _wrapper(void *self_ptr, double *x, double w, int ndim) noexcept:
    cdef RealFunctionC realf = <RealFunctionC>self_ptr
    return realf._cfunc(x, w, ndim)
    

cdef cnp.ndarray[cnp.float64_t, ndim=1] _domain_fixer( 
                    cnp.ndarray[cnp.float64_t, ndim=1] domain):
        """Ensure domain is a 1D numpy array of float64."""
        #cdef double[:] arrview = np.ascontiguousarray(domain, dtype=np.float64).ravel()
        cdef Py_ssize_t i, N = domain.shape[0]
        for i in range(N):
            if domain[i] == 0.0:
                domain[i] = 1e-10     
        return domain