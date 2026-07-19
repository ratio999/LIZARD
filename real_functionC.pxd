cimport numpy as cnp
import sympy as sp


# Flexible domain (flexible_domain_plan.md §2.1): module-level variable
# registry. Cross-module callers: POSITIONAL ARGS ONLY (cpdef rule).
cpdef void set_variables(tuple names) except *
cpdef tuple get_variables()
cpdef int n_variables()
cpdef bint is_variable_name(object name)


cdef class AtomSpec:
    # Phase 2.10 Stage 2: a user-registered opaque primitive TEMPLATE
    # (DESIGN_2_10_PRIMITIVES.md par.1): an atom of Psi_k, k >= 1. `name` is
    # the canonical Polya-vocabulary symbol (ONE cell per template, however
    # many instantiations exist); `template` the expression string in x and
    # k placeholder parameters; `slots` the placeholder names in a fixed
    # deterministic (natural) order used positionally by instantiate_atom.
    cdef public str name
    cdef public str template
    cdef public object expr
    cdef public list slots
    cdef public int k


cdef class RealFunctionC:
    cdef str mode
    cdef object func
    cdef object expr, symbols
    cdef bint regular
    cdef object domain
    cdef str description
    cdef list params
    cdef list params_strs
    cdef object prior
    cdef list variables
    cdef object cfit
    cdef object cfit_expr
    cdef dict grad_funcs
    cdef object constant_value
    # Phase 2.10 Stage 2 (opaque atomic leaves): non-None ONLY on functions
    # built by instantiate_atom. atom_symbol = the template's canonical
    # vocabulary name; theta0 = {internal name: reference value} fallback
    # used by evaluate() when a params dict omits the internal names
    # (mirrors the parameter-leaf constant_value convention).
    cdef public str atom_symbol
    cdef public dict theta0
    
    cdef _update_symbols(self)
    cdef evaluate(self, cnp.ndarray x, dict params = ?)
    cdef set_prior(self)
    cdef classify_symbols(self)
    cdef fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=?, 
    object x=?)
    cdef posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
            dict params=?, object fitness=?, object prior=?,
            object sigma=?, object x=?, double penalty=?)
    cdef double _log_marginal_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                dict params = ?, object x= ?)
    cdef posterior_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=?,
            object x=?, double penalty=?)

    cdef dict _build_gradient_functions(self)

    cdef _symbolic_gradient(self,cnp.ndarray[cnp.float64_t, ndim=1] y, 
                  object x=?, dict params=? )
    
    cdef _symbolic_gradient_parallel(self, cnp.ndarray[cnp.float64_t, ndim=1] y, object x=?, 
              dict params=?)
    cdef double fitness_sigma(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params, cnp.ndarray x,
                     cnp.ndarray[cnp.float64_t, ndim=1] sigma)

    cdef double _log_sigma_array_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                dict params, cnp.ndarray x,
                                cnp.ndarray[cnp.float64_t, ndim=1] sigma)
    cdef double _log_sigma_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                dict params, cnp.ndarray x,
                                double sigma)

    cdef double _finite_diff_gradient(self, str param_name, dict params,
                                       cnp.ndarray x,
                                       cnp.ndarray[cnp.float64_t, ndim=1] y,
                                       double h=?)
    cdef _optimize_with_grad(self, cnp.ndarray x_data,
              cnp.ndarray[cnp.float64_t, ndim=1] y_data,
              dict input_pars, str method_str=?)

    cdef optimize(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str=?)
    cdef optimize_sigma(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str, cnp.ndarray[cnp.float64_t, ndim=1] sigma)
        
        
    cdef sort_params_list(self, list pars)
    cdef get_par_string(self)
    cdef gradient(self, cnp.ndarray[cnp.float64_t, ndim=1] y, object x=?, dict params=?) 
    cdef time_to_frequency(self, double[:] x, object data)

    cdef bint is_valid(self, object x=?, double max_abs=?)


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
                    
   

cdef double _wrapper(void *self_ptr, double *x, double w, int ndim) noexcept

cpdef cnp.ndarray _domain_fixer(
                    cnp.ndarray domain)

# Phase 2.10 Stage 2: opaque-leaf factory + rename helper. Both cpdef so
# gp_movesC / function_treeC call them across the cimport boundary --
# POSITIONAL ARGS ONLY at cross-module call sites (kwargs on cpdef across
# cimport raise NameError at runtime; see gp_movesC build_tree note).
cpdef RealFunctionC instantiate_atom(AtomSpec spec,
                    cnp.ndarray domain,
                    list fresh_names)

cpdef RealFunctionC rename_leaf_params(RealFunctionC rf, dict name_map)