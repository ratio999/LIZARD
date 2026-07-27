import numpy as np
cimport numpy as cnp
import sympy as sp
import warnings

from cpython.ref cimport Py_INCREF, Py_DECREF
#cimport sympy as csp
from utils.pointer_utils cimport python_object_to_void_ptr, void_ptr_to_python_object, void_ptr_to_rf
from scipy.optimize import minimize, OptimizeResult
from libc.math cimport isfinite as _c_isfinite

# NaN-leak fix (2026-07-15): finite MSE penalty for models with any
# non-finite prediction — see the comment block above posterior_fitness.
cdef double _MSE_PENALTY = 1.0e100
import re
#from libc.stdlib cimport malloc, free  # Not needed so far 
from cython.cimports.libc.string import memcpy
from copy import copy, deepcopy

cdef object _LOG = np.log
cdef double _PI = np.pi
cdef double _NAN = np.nan

# ---------------------------------------------------------------------------
# Flexible domain (flexible_domain_plan.md §2.1) — module-level variable
# registry: the independent variables of every expression built through this
# module. Default ('x',) IS the legacy configuration — descriptions, cache
# keys and Pólya cells are unchanged unless a caller registers a different
# tuple (Grove/Forest `variables=` argument). Module-global exactly like the
# RNG handles in gp_movesC/function_treeC: one configuration per process,
# installed at construction time; changing it mid-run is unsupported
# (compiled leaf closures capture it at from_string time).
# Layout contract (plan §0 D2): d = len(registry) == 1 keeps the historical
# (n,) domain; d > 1 uses ONE C-contiguous float64 array of shape (d, n) —
# coordinate ROWS, points along the LAST axis; row i is registry[i].
# ---------------------------------------------------------------------------
_VARIABLES = ('x',)
_VAR_SET = frozenset(_VARIABLES)
_VAR_SYMS = [sp.Symbol('x')]

# Kernel parameter-name bands: 'a' (parameter_parser / new_par fallback),
# 'b' (burn-in rename buffer), 'c' (MPI combiner). Variable names must not
# collide with names these bands can mint.
_RESERVED_PARAM_BANDS = ('a', 'b', 'c')


cpdef void set_variables(tuple names) except *:
    """Install the independent-variable name registry (flexible-domain
    opt-in). Cross-module callers: POSITIONAL ARGS ONLY (cpdef rule)."""
    global _VARIABLES, _VAR_SET, _VAR_SYMS
    cdef str nm, band
    cdef object obj
    cdef list syms = []
    if names is None or len(names) == 0:
        raise ValueError("set_variables: need at least one variable name")
    if len(set(names)) != len(names):
        raise ValueError("set_variables: duplicate names in %r" % (names,))
    for nm in names:
        if not isinstance(nm, str) or len(nm) == 0:
            raise ValueError(
                "set_variables: names must be non-empty strings, got %r"
                % (nm,))
        if nm == 'y':
            raise ValueError(
                "set_variables: 'y' is reserved for the observation vector")
        for band in _RESERVED_PARAM_BANDS:
            if re.fullmatch(band + r'\d+', nm):
                raise ValueError(
                    "set_variables: %r collides with the kernel parameter "
                    "band %r (band + digits are mintable parameter names)"
                    % (nm, band))
        try:
            obj = sp.sympify(nm)
        except Exception:
            raise ValueError(
                "set_variables: %r is not sympifiable" % (nm,))
        if not isinstance(obj, sp.Symbol):
            raise ValueError(
                "set_variables: %r does not sympify to a plain Symbol "
                "(it shadows a sympy function or constant)" % (nm,))
        syms.append(obj)
    _VARIABLES = tuple(names)
    _VAR_SET = frozenset(names)
    _VAR_SYMS = syms


cpdef tuple get_variables():
    """Current variable-name registry (POSITIONAL calls across cimport)."""
    return _VARIABLES


cpdef int n_variables():
    """Registry dimension d (POSITIONAL calls across cimport)."""
    return len(_VARIABLES)


cpdef bint is_variable_name(object name):
    """Registry membership test (POSITIONAL calls across cimport).

    `object`-typed on purpose: callers compiled under language_level 2
    (function_treeC) pass f-string/unicode names, which cannot implicitly
    coerce to a `str`-typed parameter there."""
    return name in _VAR_SET

def safe_convert(value):
        try:
            return np.float64(value)
        except (TypeError, ValueError):
            #print('!\n!\n|\n!\n!')
            return np.nan


cdef _to_float64_array(object out, cnp.ndarray like):
    """Vectorised replacement for ``np.vectorize(safe_convert)``.

    The old per-element Python loop ran on every evaluate() call. For the
    common case (real numeric output) this is a single C-level cast. Only
    complex / object dtypes fall back to the element-wise safe_convert,
    preserving the original NaN-on-complex behaviour exactly.
    """
    cdef object arr = np.asarray(out)
    if arr.dtype == np.float64:
        return np.ascontiguousarray(arr, dtype=np.float64).ravel()
    if arr.dtype.kind in ('f', 'i', 'u', 'b'):
        return np.ascontiguousarray(arr, dtype=np.float64).ravel()
    # Complex or object dtype: replicate safe_convert element-wise (rare).
    cdef object vector_safe_convert = np.vectorize(safe_convert)
    return np.ascontiguousarray(vector_safe_convert(arr), dtype=np.float64).ravel()


cdef inline object _flat_fill(object x, double value):
    """(n,)-shaped constant fill for a domain of shape (n,) or (d, n).

    Flexible-domain invariant (plan §1.1): every leaf output is a 1-D
    length-n vector with n = x.shape[-1]. The historical
    ``np.full_like(x, ...)`` breaks that at d > 1 — it returns (d, n),
    which then broadcasts wrongly (or "successfully") into the residuals.
    On (n,) input this is value-identical to full_like(...).ravel()."""
    return np.full(x.shape[-1], value, dtype=np.float64)

cdef class RealFunctionC:

    def __init__(self, object func, object expr, cnp.ndarray domain, list params=None, mode='Time', bint compute_grad = False, double constant_value=1.0):
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
        # Phase 2.10 Stage 2: opaque-leaf metadata. Plain constructions carry
        # None; ONLY instantiate_atom (and rename_leaf_params on its output)
        # sets these after construction.
        self.atom_symbol = None
        self.theta0 = None
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
        cdef RealFunctionC copy_rf = RealFunctionC(copy_func, self.expr, copy(self.domain), copy(self.params))
        # Phase 2.10 Stage 2: opaque-leaf identity travels with the function.
        copy_rf.atom_symbol = self.atom_symbol
        copy_rf.theta0 = None if self.theta0 is None else dict(self.theta0)
        return copy_rf

    def __deepcopy__(self, memo2):
        cdef object copy_func = lambda input_x, params: self.func(input_x, params)
        # sympy expressions are IMMUTABLE (self.expr is assigned once in
        # __init__ and never mutated), so SHARE the reference instead of
        # deep-copying it. This removes an O(expr-size) copy per node on every
        # MCMC-step deepcopy and preserves sympy's cached .free_symbols on the
        # shared object. `params` is recomputed from the expr inside __init__
        # (classify_symbols), so the constructor argument is discarded -- there
        # is no point deep-copying it either.
        cdef RealFunctionC copy_rf = RealFunctionC(
                copy_func, self.expr, deepcopy(self.domain, memo2))
                #,deepcopy(self.params, memo2)
        # Phase 2.10 Stage 2: forward the opaque-leaf metadata (theta0 is a
        # fixed reference dict, but copy it anyway so no tree copy can ever
        # alias another's state).
        copy_rf.atom_symbol = self.atom_symbol
        copy_rf.theta0 = None if self.theta0 is None else dict(self.theta0)
        # classify_symbols iterates a sympy SET; re-impose the original
        # deterministic parameter order on the copy (instantiate_atom fixed
        # it on the source; plain functions get their order re-sorted by
        # Tree.get_params anyway, so this is a no-op for them).
        copy_rf.params = list(self.params)
        copy_rf.params_strs = list(self.params_strs)
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
    

    

    

    cdef get_par_string(self):
        return [str(i) for i in self.params]

    def get_expr(self):
        return(self.expr)


    cdef classify_symbols(self):

        cdef list params = []
        cdef list variables = []
        cdef str s_str

        for s in self.symbols:
            s_str = str(s)
            # Flexible domain (plan §0 D4): registry membership takes
            # precedence over the name-shape regexes — 'x0'-style variable
            # names match the parameter pattern, and a variable classified
            # as a parameter would be optimized as a constant. Legacy 'x'
            # lands in `variables` through either route: bit-identical.
            if s_str in _VAR_SET:
                variables.append(s)
            elif re.match(r"^[a-zA-Z]\d+$", s_str):
                params.append(s)
            elif re.match(r"[a-zA-Z]$", s_str):
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
            if "zoo" in self.description:
                return False
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


    cdef object evaluate(self, cnp.ndarray x, dict params=None):
        #print(params)
        if params is None:
            #print('params is NONE')
            params = {}
        # Phase 2.10 Stage 2: opaque atomic leaf. theta0 (set only by
        # instantiate_atom) supplies reference values for any internal
        # parameter missing from `params`, so params-free probes (is_valid,
        # bare evaluate) are well-defined on opaque leaves — the exact
        # analogue of the parameter-leaf constant_value fallback below.
        # A fresh merged dict is built so the caller's dict is never mutated.
        cdef str _nm
        cdef dict _merged
        if self.theta0 is not None:
            for _nm in self.params_strs:
                if _nm not in params:
                    _merged = dict(self.theta0)
                    _merged.update(params)
                    params = _merged
                    break
        #print('params in eval ', params)
        #print('in evaluate: ', self.func(x, params=params))
        if self.is_param():
        # Use params dict if available, otherwise fall back to constant_value
            param_name = self.get_description()  # e.g., "a0"
            if param_name in params:
                return _flat_fill(x, params[param_name])
            return _flat_fill(x, self.constant_value)
        cdef object out = self.func(x, params=params)

        try:
            return _to_float64_array(out, x)
        except:
            return _flat_fill(x, _NAN)
        #return self.func(x, params=params)
    
    def get_eval(self, cnp.ndarray x, dict params=None):
        return self.evaluate(x, params)

    cdef double _log_marginal_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                dict params = None, object x= None):
        if x is None:
            x = self.domain
        if params is None:
            params = {}
        cdef double fitness = self.fitness(y, params, x)
        cdef int N = y.shape[0]
        # Phase 2.13(i): sigma profiled -> counts as a parameter, penalty on
        # len(params)+1 (matches Tree._shortcut_bic's sigma=None branch).
        return -0.5 * N * (_LOG(2.0 * _PI * fitness) + 1.0) - 0.5*(len(params) + 1)*_LOG(N) #-0.5 * N * fitness /s2

    cdef double _log_sigma_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                dict params, cnp.ndarray x,
                                double sigma):
        cdef double mse = self.fitness(y, params, x)
        cdef int N = y.shape[0]
        cdef int k = len(params)
        cdef double var = sigma*sigma

        return (-0.5 * N * mse/var - 0.5 * k * _LOG(N))

    cdef double _log_sigma_array_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                dict params, cnp.ndarray x,
                                cnp.ndarray[cnp.float64_t, ndim=1] sigma):
        
        cdef double mse = self.fitness_sigma(y, params, x, sigma)
        cdef int N = y.shape[0]
        cdef int k = len(params)
        #cdef cnp.ndarray[cnp.float64_t, ndim=1] var = sigma*sigma

        return (-0.5 * N * mse - 0.5 * k * _LOG(N))


    cdef posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                   dict params=None, object fitness=None, object prior=None, 
                   object sigma=1.00, object x = None,  double penalty=0.0):
        if fitness is None:
            #print('FITNESS IS NONE')
            
            if x is None:
                x = self.domain
            fitness = self.posterior_fitness(y, params, x, penalty)
        
        if fitness <0:
            print('invalid fitness ', self.description, penalty)
        if prior is None:
            prior = 0.0 #to be worked on
        #print('fitness is: ', fitness, ' and prior is: ', prior)
        try:
            return -fitness/(2 * sigma**2) - prior  
        except:
            return -np.inf

    # ------------------------------------------------------------------
    # NaN-leak fix (2026-07-15, PLAN §4.2b forensics): the MSE was
    # np.nanmean(resid**2), which silently DROPS NaN predictions — the
    # optimizer then actively drives parameters into NaN territory
    # (e.g. x**8.6e14: NaN on 99.6% of the grid) because that shrinks
    # the mean over the surviving points, and such "NaN monsters" out-
    # scored the exact truth by ~75 nats. Correct semantics: a model
    # with ANY non-finite prediction at an observed point has zero
    # likelihood. Implemented as plain np.sum (NaN/inf PROPAGATE) + one
    # scalar C isfinite + a LARGE FINITE penalty (optimizer-stable: no
    # NaN plateaus, so scipy's bracket() cannot blow up there).
    # PERFORMANCE: on all-finite vectors np.sum is bit-identical to
    # what nanmean computed (same pairwise reduction) and strictly
    # cheaper (no isnan mask / count / Python dispatch). 1e100
    # dominates any is_valid-surviving legitimate mse (<= ~1e24) while
    # -0.5*N*(1e100/sigma^2) stays ~200 orders below double overflow
    # and ~190 orders above the -1e299 hard-reject sentinel.
    # ------------------------------------------------------------------
    cdef posterior_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=None, object x = None, double penalty=0.0):
        if x is None:
            x = self.domain
        cdef double _s = np.sum((y - self.evaluate(x, params))**2)
        cdef double mse
        if _c_isfinite(_s):
            mse = _s / <double>y.shape[0]
        else:
            mse = _MSE_PENALTY
        if mse <0:
            print('invalid mse ', self.description)
        # Fast path: skip regularization when lambda is 0



        return mse + penalty


    cdef fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=None, object x = None):
        #print(x)
        cdef double _s
        if x is None:
            x = self.domain
        try:
            #if self.is_param():
            #print('params in fitness', params, self.evaluate(x,params))
            _s = np.sum((y - self.evaluate(x,  params))**2) #+ lambda_reg * np.sum(np.array(list(params.values()))**2)
            if _c_isfinite(_s):
                return _s / <double>y.shape[0]
            return _MSE_PENALTY
        except:
            return np.inf

    cdef double fitness_sigma(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params, cnp.ndarray x,
                     cnp.ndarray[cnp.float64_t, ndim=1] sigma):

        cdef  cnp.ndarray[cnp.float64_t, ndim=1] var = sigma*sigma
        cdef double _s
        try:
            _s = np.sum(((y-self.evaluate(x, params))**2)/var)
            if _c_isfinite(_s):
                return _s / <double>y.shape[0]
            return _MSE_PENALTY
        except:
            return np.inf
    



    cdef dict _build_gradient_functions(self):
        """
        Compute analytical gradients symbolically using SymPy.
        Returns dict: {param_name: lambdified_gradient_function}
        """
        if len(self.params) == 0:
            return {}
        # Flexible domain: symbolic gradients are a d=1 legacy path (plan
        # §8.2). At d > 1 return no gradient functions — _symbolic_gradient
        # degrades to finite differences (dimension-agnostic: it perturbs
        # theta, not x) and _optimize_with_grad falls back to Powell.
        if len(_VARIABLES) > 1:
            return {}

        cdef dict grad_dict = {}
        cdef object x_sym = _VAR_SYMS[0]
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
                                       cnp.ndarray x,
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

    def objective(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params,
            cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data):
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        #print(x_data, y_data)
        cdef double fit = self.fitness( y_data,  params_dict, x_data)
        #print(self.expr)
        #if flat_params[0] < 2.0 or flat_params[0] > 5.0:  # log only at different param values
        #print(f'  objective called: params={flat_params}, params_dict={params_dict}, fitness={fit}')
        
        return fit

    def objective_sigma(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params,
            cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
            cnp.ndarray[cnp.float64_t, ndim=1] sigma):
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        #print(x_data, y_data)
        cdef double fit = self.fitness_sigma( y_data,  params_dict, x_data, sigma)
        
        #print(self.expr)
        #if flat_params[0] < 2.0 or flat_params[0] > 5.0:  # log only at different param values
        #print(f'  objective called: params={flat_params}, params_dict={params_dict}, fitness={fit}')
        
        return fit
    

    def grad(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data):
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self.gradient( y_data, x_data, params_dict)

    def symb_grad(self, cnp.ndarray[cnp.float64_t, ndim=1] flat_params, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data):
        cdef dict params_dict = dict(zip(self.params_strs, flat_params))
        return self._symbolic_gradient( y_data, x_data, params_dict)
    
    cdef _optimize_with_grad(self, cnp.ndarray x_data,
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

        cdef object result
        try:
            result = minimize(
                fun=self.objective,
                x0=initial_params,
                jac=self.symb_grad if use_gradients else None,
                args=(x_data, y_data),
                method=method_str if use_gradients else 'Powell',
                options={'disp': False}
            )
        except (RuntimeError, ValueError, FloatingPointError,
                OverflowError) as _e:
            # Guarded optimizer failure (2026-07-15): scipy's bracket()
            # raises RuntimeError("Too many iterations.") when a 1-D
            # line-search slice is non-finite (e.g. exp overflow) — not
            # preventable via maxiter options. Keep the warm-start
            # parameters and report +inf, so the NaN-fill scoring path
            # rejects the candidate instead of killing the process.
            # Fires ONLY where the unguarded call would have died.
            result = OptimizeResult(x=initial_params, fun=np.inf,
                                    success=False, status=-1, nit=0,
                                    nfev=0, message="guarded: %r" % (_e,))
        #if use_gradients:
            #print('Optimization with gradients succeeded.')
        #else:
            #print('Optimization without gradients (Powell) succeeded.')
        
        return result
    
    '''
    cdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, 
            cnp.ndarray[cnp.float64_t, ndim=1] y_data,
            dict input_pars, str method_str='Powell'):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] initial_params = np.ascontiguousarray(list(input_pars.values()), dtype=np.float64)
        cdef object result = minimize(
            lambda x: self.objective(x, x_data, y_data),
            initial_params,
            method=method_str,
            options={'disp': False}

        )
        
        return result
    '''
    cdef optimize(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str='Powell'):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] initial_params = np.ascontiguousarray(list(input_pars.values()), dtype=np.float64)
        #cdef double[:] initial_params = np.array(list(input_pars.values()))
        #print(initial_params)
        cdef object result

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    result = minimize(
                            fun=self.objective,
                            x0=initial_params,
                            args=(x_data, y_data),
                            method=method_str,
                            options={'disp': False}
                            )
        except (RuntimeError, ValueError, FloatingPointError,
                OverflowError) as _e:
            # Guarded optimizer failure — see _optimize_with_grad note.
            result = OptimizeResult(x=initial_params, fun=np.inf,
                                    success=False, status=-1, nit=0,
                                    nfev=0, message="guarded: %r" % (_e,))
        #self.params = dict(zip(self.param_str, result.x))
        #print(result)
        return result

    cdef optimize_sigma(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str, cnp.ndarray[cnp.float64_t, ndim=1] sigma):
        cdef cnp.ndarray[cnp.float64_t, ndim=1] initial_params = np.ascontiguousarray(list(input_pars.values()), dtype=np.float64)
        #cdef double[:] initial_params = np.array(list(input_pars.values()))
        #print(initial_params)
        cdef object result

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    result = minimize(
                            fun=self.objective_sigma,
                            x0=initial_params,
                            args=(x_data, y_data, sigma),
                            method=method_str,
                            options={'disp': False}
                            )
        except (RuntimeError, ValueError, FloatingPointError,
                OverflowError) as _e:
            # Guarded optimizer failure — see _optimize_with_grad note.
            result = OptimizeResult(x=initial_params, fun=np.inf,
                                    success=False, status=-1, nit=0,
                                    nfev=0, message="guarded: %r" % (_e,))
        #self.params = dict(zip(self.param_str, result.x))
        #print(result)
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
            #print('Issues: ', self.params, pars, self.description)
            pass

    #cdef?
    cpdef is_param(self):
        """Is function a constant?"""
        if len(self.variables)==0:
            return True
        '''
        cdf cnp.ndarray[cnp.float64_t, ndim=1] y
        
    
        y = self.evaluate(self.domain)
        return np.all(np.isclose(y, y[0]))
        '''
    
    # Use as less as possible: loop on check_single_value is bottleneck 
    def is_defined(self, cnp.ndarray x, dict params = None):
        if params is None:
            params = {}
        try:
            y = self.evaluate(x, params)
            vector_safe_convert = np.vectorize(safe_convert)
            y_safe = vector_safe_convert(y)
            return np.isfinite(y_safe) & np.isreal(y_safe)
        except Exception as e:
            # (n,)-shaped mask regardless of domain layout (plan §1.1).
            # NB x is buffer-typed here, so shape[-1] would compile to raw
            # C pointer indexing (PyArray_DIMS(x)[-1] — an out-of-bounds
            # heap read, found by the adversarial review); index with the
            # explicit non-negative last axis instead.
            return np.full(x.shape[x.ndim - 1], False, dtype=np.bool_)
    
    #needs improvement. useless so far. cdef?.
    cpdef derivative(self,  x,  h=1e-5, dict params = None):
        """Compute derivative"""
        if params is None:
            params = {}

        return (self.evaluate(x + h, params) - self.evaluate(x - h, params)) / (2 * h)

    def __call__(self, cnp.ndarray x):
        """Allows standard calling"""
        return self.evaluate(x)
    # Very important: keep it Python func
    @classmethod
    def from_string(cls, expr_str,  cnp.ndarray x, compute_grad=True):
        """Creates function class from string"""
        # Flexible domain: variables come from the module registry
        # (default ('x',) — sp.Symbol('x') is the identical sympy object,
        # so the d=1 branch below is the historical closure verbatim).
        cdef list var_syms = list(_VAR_SYMS)
        cdef object x_sym = var_syms[0]
        cdef object expr

        expr = sp.sympify(expr_str)
        # Compile ONCE over (*variables, *params). lambdify is a
        # code-generation call (~ms each) and expr.subs is symbolic; doing
        # either on every evaluate() made parameter optimisation thousands
        # of times slower than necessary (Powell calls the objective
        # hundreds of times per candidate, and every leaf was re-lambdified
        # on each call). The compiled callable is captured in the closure
        # and reused, taking numeric parameter values as positional args.
        cdef list param_syms = sorted(
            (s for s in expr.free_symbols if str(s) not in _VAR_SET), key=str)
        cdef list param_names = [str(s) for s in param_syms]
        cdef object compiled = sp.lambdify(var_syms + param_syms, expr, "numpy")
        n_vars = len(var_syms)

        if n_vars == 1:
            def func(cnp.ndarray input_x, dict params = None):
                if params is None:
                    params = {}
                # Fast path: all (if any) free parameters are supplied -> call the
                # pre-compiled function directly. Non-parameter leaves (sin(x),
                # x, ...) have no free params and hit this immediately.
                if not param_names:
                    arg_vals = [input_x]
                elif all(name in params for name in param_names):
                    arg_vals = [input_x]
                    for name in param_names:
                        arg_vals.append(params[name])
                else:
                    # Rare fallback: a needed parameter is missing. Preserve the
                    # original subs+lambdify semantics exactly (returns NaN on any
                    # leftover free symbol via the except below).
                    expr_with_params = expr
                    for param_name, param_value in params.items():
                        expr_with_params = expr_with_params.subs(
                            sp.symbols(param_name), param_value)
                    try:
                        result = sp.lambdify(x_sym, expr_with_params, "numpy")(input_x)
                        return _to_float64_array(result, input_x)
                    except Exception:
                        return np.full_like(input_x, np.nan, dtype=np.float64)

                try:
                    result = compiled(*arg_vals)
                    return _to_float64_array(result, input_x)
                except Exception:
                    return np.full_like(input_x, np.nan, dtype=np.float64)
        else:
            # d > 1: the domain arrives as ONE (d, n) C-contiguous array;
            # the compiled callable receives the d coordinate ROWS as
            # zero-copy views (plan §0 D2). Outputs are always (n,).
            def func(cnp.ndarray input_x, dict params = None):
                if params is None:
                    params = {}
                arg_vals = [input_x[_i] for _i in range(n_vars)]
                if not param_names:
                    pass
                elif all(name in params for name in param_names):
                    for name in param_names:
                        arg_vals.append(params[name])
                else:
                    # Rare fallback: a needed parameter is missing —
                    # subs+lambdify semantics as in the d=1 branch.
                    expr_with_params = expr
                    for param_name, param_value in params.items():
                        expr_with_params = expr_with_params.subs(
                            sp.symbols(param_name), param_value)
                    try:
                        result = sp.lambdify(var_syms, expr_with_params, "numpy")(
                            *[input_x[_i] for _i in range(n_vars)])
                        return _to_float64_array(result, input_x)
                    except Exception:
                        return _flat_fill(input_x, _NAN)

                try:
                    result = compiled(*arg_vals)
                    return _to_float64_array(result, input_x)
                except Exception:
                    return _flat_fill(input_x, _NAN)

        return cls(func, expr, x, compute_grad=compute_grad)

    def get_description(self):
        """Access str(self.expr) directly --> redundancy with str(get_expr(self))"""
        return self.description

    def __add__(self, RealFunctionC other):
        """Addition of two RealFunctions."""
        cdef object new_expr = self.expr + other.expr

        def new_func(cnp.ndarray x, dict params=None):
            return self.evaluate(x, params) + other.evaluate(x, params)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __sub__(self, RealFunctionC other):
        """Subtraction of two RealFunctions."""
        cdef object new_expr = self.expr - other.expr

        def new_func(cnp.ndarray x, dict params=None):
            return self.evaluate(x, params) - other.evaluate(x, params)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __mul__(self, RealFunctionC other):
        """Multiplication of two RealFunctions."""
        cdef object new_expr = self.expr * other.expr

        def new_func(cnp.ndarray x, dict params=None):
            return self.evaluate(x, params) * other.evaluate(x, params)

        return RealFunctionC(new_func, new_expr, self.domain)

    def __truediv__(self, RealFunctionC other):
        """Division of two RealFunctions with protection against division by zero."""
        cdef object new_expr = self.expr / other.expr

        def safe_div(cnp.ndarray x, dict params=None):
            cdef object numerator = self.evaluate(x, params)
            cdef object denominator = other.evaluate(x, params)
            return np.where(denominator == 0, np.inf, numerator / denominator)

        return RealFunctionC(safe_div, new_expr, self.domain)

    def compose(self, RealFunctionC other):
        """Composition of two RealFunctions (f1(f2(x))). d = 1 ONLY.

        Flexible domain (plan §0 D6 / §8.1): with several variables the
        substitution slot is undefined — compose-at-d>1 is a STANDING
        ITEM, not implemented. This is the single innermost gate: every
        'o' dispatch (Node.fetch_fun / update_func, build_tree,
        node_from_skel 'O' rebuild, _expr_to_tree function rebuild) lands
        here, so an 'o' that slips past construction-time validation
        still cannot execute at d > 1."""
        if len(_VARIABLES) > 1:
            raise NotImplementedError(
                "compose ('o') is not defined for multivariate domains "
                "(flexible_domain_plan.md §8.1 — standing item)")
        cdef object x_sym = _VAR_SYMS[0]
        cdef object new_expr = self.expr.subs(x_sym, other.expr)
        if self.is_param():
            new_expr = self.expr * other.expr
        else:
            new_expr = self.expr.subs(x_sym, other.expr)
 
            
        def new_func(cnp.ndarray x, dict params=None):
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

        def new_func(cnp.ndarray x, dict params=None):
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
    


# ---------------------------------------------------------------------------
# Phase 2.10 Stage 2 — opaque atomic primitives (DESIGN_2_10_PRIMITIVES.md §1,
# §5, §9 item 2). A user primitive is an atom of Psi_k: ONE leaf symbol
# carrying k >= 1 internal parameters. The parameter leaf is the k = 1
# constant special case; function leaves are k = 0 — one object class.
# ---------------------------------------------------------------------------

cdef _slot_sort_key(str s):
    """Natural order for placeholder names: ('c', 2) < ('c', 10)."""
    m = re.fullmatch(r'([a-zA-Z]+)(\d+)', s)
    return (m.group(1), int(m.group(2)))


cdef class AtomSpec:
    """A user-registered opaque primitive TEMPLATE.

    Parameters
    ----------
    name : str
        Canonical vocabulary symbol (the Pólya cell key). ONE cell per
        template — instantiations share it regardless of the fresh parameter
        names they receive.
    template : str
        Expression in the variable `x` and k >= 1 placeholder parameters
        matching the codebase's parameter pattern [a-zA-Z]\\d+
        (e.g. "c0*sin(c1*x)"). Every free symbol must be `x` or a
        placeholder; `x` must appear (a template without x is the constant
        map — that is the ordinary parameter leaf, already in the
        vocabulary); k = 0 templates belong in the plain `functions` list.
    """

    def __init__(self, str name, str template):
        if name is None or len(name) == 0:
            raise ValueError("AtomSpec: name must be a non-empty string")
        cdef object expr = sp.sympify(template)
        cdef list slots = []
        cdef bint has_x = False
        cdef str s_str
        for s in expr.free_symbols:
            s_str = str(s)
            # Flexible domain (plan §0 D4): registry membership BEFORE the
            # placeholder regex — 'x0'-style variable names match
            # [a-zA-Z]\d+ and would otherwise become parameter slots.
            # Legacy 'x' takes this branch too (it never matched the
            # regex): bit-identical classification.
            if s_str in _VAR_SET:
                has_x = True
            elif re.fullmatch(r'[a-zA-Z]\d+', s_str):
                slots.append(s_str)
            else:
                raise ValueError(
                    "AtomSpec %r: free symbol %r is neither a registered "
                    "variable %r nor a placeholder parameter "
                    "([a-zA-Z]<digits>)"
                    % (name, s_str, _VARIABLES))
        if not has_x:
            raise ValueError(
                "AtomSpec %r: template must depend on at least one "
                "registered variable %r. A constant template IS the "
                "ordinary parameter leaf (Psi_1), already registered."
                % (name, _VARIABLES))
        if len(slots) == 0:
            raise ValueError(
                "AtomSpec %r: template has no internal parameters (k = 0). "
                "Register it in the plain `functions` vocabulary instead."
                % name)
        slots.sort(key=_slot_sort_key)
        self.name = name
        self.template = template
        self.expr = expr
        self.slots = slots
        self.k = len(slots)

    def __reduce__(self):
        # Phase 2.10 Stage 3b: AtomSpec crosses the MPI wire (worker startup
        # config ships the Forest's atoms). A template is fully determined
        # by (name, template); rebuilding through __init__ re-validates and
        # re-derives expr/slots/k deterministically.
        return (AtomSpec, (self.name, self.template))


cpdef RealFunctionC instantiate_atom(AtomSpec spec,
                    cnp.ndarray domain,
                    list fresh_names):
    """Instantiate `spec` as a concrete opaque-leaf RealFunctionC whose
    internal parameters carry `fresh_names` (positionally matched to
    spec.slots — the caller draws them from the tree's name pool via
    function_treeC.new_par so they never collide with existing tree
    parameters).

    The result is a PLAIN RealFunctionC (same compiled-evaluate path as
    from_string) plus the two opaque markers: atom_symbol = spec.name (the
    stable vocabulary key) and theta0 = {name: 1.0} (reference values for
    params-free validity probes; the sampling-time values live on the GP
    Node, exactly like parameter-leaf values do).

    Cross-module callers: POSITIONAL ARGS ONLY (cpdef + cimport + kwargs is
    the known NameError trap)."""
    cdef int i
    cdef str nm
    if fresh_names is None or len(fresh_names) != spec.k:
        raise ValueError(
            "instantiate_atom(%r): expected %d fresh names, got %r"
            % (spec.name, spec.k, fresh_names))
    if len(set(fresh_names)) != spec.k:
        raise ValueError(
            "instantiate_atom(%r): fresh names must be distinct, got %r"
            % (spec.name, fresh_names))
    for nm in fresh_names:
        if not re.fullmatch(r'[a-zA-Z]\d+', nm):
            raise ValueError(
                "instantiate_atom(%r): %r does not match the parameter "
                "pattern [a-zA-Z]<digits>" % (spec.name, nm))
        if nm in _VAR_SET:
            raise ValueError(
                "instantiate_atom(%r): fresh name %r is a registered "
                "variable — the slot substitution would bind the atom "
                "parameter to the domain" % (spec.name, nm))
    cdef dict sym_map = {}
    for i in range(spec.k):
        sym_map[sp.Symbol(spec.slots[i])] = sp.Symbol(fresh_names[i])
    cdef object new_expr = spec.expr.subs(sym_map, simultaneous=True)
    cdef RealFunctionC rf = RealFunctionC.from_string(
        str(new_expr), domain, compute_grad=False)
    rf.atom_symbol = spec.name
    cdef dict th0 = {}
    for nm in fresh_names:
        th0[nm] = 1.0
    rf.theta0 = th0
    # classify_symbols iterates a sympy set: pin the parameter order to the
    # deterministic fresh_names order (MPI replicas must agree bit-exactly).
    rf.sort_params_list([sp.symbols(nm) for nm in fresh_names])
    return rf


cpdef RealFunctionC rename_leaf_params(RealFunctionC rf, dict name_map):
    """Return a NEW RealFunctionC equal to `rf` with its parameter names
    renamed per name_map (old -> new; names absent from the map are kept).
    Preserves atom_symbol, theta0 (keys remapped) and the deterministic
    parameter order. Used by the label-canonicalisation passes in gp_movesC
    (crossover / compaction) on opaque leaves, where the plain-leaf
    from_string(new_name) rebuild does not apply.

    Cross-module callers: POSITIONAL ARGS ONLY."""
    cdef dict sym_map = {}
    cdef str old, new
    for old in name_map:
        new = name_map[old]
        if old != new:
            sym_map[sp.Symbol(old)] = sp.Symbol(new)
    cdef object new_expr = rf.expr.subs(sym_map, simultaneous=True) if sym_map else rf.expr
    cdef RealFunctionC out = RealFunctionC.from_string(
        str(new_expr), rf.domain, compute_grad=False)
    out.atom_symbol = rf.atom_symbol
    cdef dict th0
    if rf.theta0 is not None:
        th0 = {}
        for old in rf.theta0:
            th0[name_map.get(old, old)] = rf.theta0[old]
        out.theta0 = th0
    out.sort_params_list(
        [sp.symbols(name_map.get(s, s)) for s in rf.params_strs])
    return out


cdef is_parameter(str name):
    """Is name a parameter declaration?"""
    return bool(re.fullmatch(r'[a-zA-Z]\d+', name))

cdef double _wrapper(void *self_ptr, double *x, double w, int ndim) noexcept:
    cdef RealFunctionC realf = <RealFunctionC>self_ptr
    return realf._cfunc(x, w, ndim)
    

cpdef cnp.ndarray _domain_fixer(
                    cnp.ndarray domain):
        """Replace exact zeros with 1e-10 (division guards), in place.

        Accepts the legacy (n,) float64 layout or the flexible (d, n)
        C-contiguous layout (plan §0 D2); shape is preserved — never
        raveled or coerced to 1-D."""
        cdef Py_ssize_t i, N
        cdef double[:] flat
        if domain.ndim == 1:
            flat = domain
        else:
            if not domain.flags['C_CONTIGUOUS']:
                raise ValueError(
                    "_domain_fixer: (d, n) domain must be C-contiguous "
                    "(coordinate rows, points along the last axis)")
            flat = domain.reshape(-1)   # view: in-place writes reach domain
        N = flat.shape[0]
        for i in range(N):
            if flat[i] == 0.0:
                flat[i] = 1e-10
        return domain