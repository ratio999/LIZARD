cimport numpy as cnp
#import sympy as sp
cimport real_functionC

cpdef void set_rng(object py_rng, object np_rng) except *
cpdef tuple get_rng()

cdef class Node:
    cdef real_functionC.RealFunctionC function
    cdef object left
    cdef object right
    cdef str op_symbol
    cdef object domain
    cdef int node_depth
    cdef bint is_valid(self, object x=?, double max_abs=?)
    cdef public cnp.float64_t value    
    cdef reinit_to_par(self, list existent_pars, str user_prefix=?, double value=?)
    cdef _rebuild_ancestor_functions(self, Node node)
    cpdef set_value(self, cnp.float64_t value=?)
    cdef bint bis_param
    # Phase 2.10 Stage 2: internal-parameter values of an opaque atomic leaf
    # (dict name -> float), None on every other node. Lives on the Node so
    # tree deepcopies isolate it, exactly like parameter-leaf .value.
    cdef public object theta
    cpdef is_atomic(self)
    cpdef int leaf_k(self)
    cpdef list leaf_param_names(self)
    cpdef str vocab_symbol(self)
    cdef rename_atomic(self, dict name_map)
    cdef sort_params_by_names(self, list names)
    cdef object fitness
    cdef object posterior
    cdef object prior
    cdef double alpha
    cdef object sigma
    cdef fetch_fun(self, Node left = ?, Node right = ?, str operation = ?)
    cdef optimize(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str=?, bint use_gradients=?)

    cdef optimize_sigma(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, cnp.ndarray[cnp.float64_t, ndim=1] sigma,
        str method_str=?, bint use_gradients=?)

    cdef double _log_marginal_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=?, object x=?)

    cdef evaluate_node_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters)
    
    cdef set_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters)
    
    cdef sort_params(self, list pars)

    cdef set_alpha(self, double newalpha)

    cdef set_prior(self)

    cdef double _log_marginal_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=?, object x=?)
        
    cdef double _log_sigma_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params,
            cnp.ndarray x, double sigma)

    cdef double _log_sigma_array_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params,
            cnp.ndarray x, cnp.ndarray[cnp.float64_t, ndim=1] sigma)

    cpdef is_constant(self)

    cdef set_sigma(self, object newsigma)

    cpdef void set_sigma_recursive(self, object new_sigma)

    cdef set_posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters, double penalty=?)

    cdef evaluate_node_posterior(self,cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters, double penalty=?)
 
    cpdef posterior_tree(self, y, parameters) 
    cpdef fitness_tree(self,  y, parameters)
    cpdef node_count(self)
    cpdef is_param(self)
    cpdef assign_operation(self, operation, bint brute_force=?)
    cpdef get_leaves(self, leaves_list=?)
    cpdef assign_left(self, node, bint brute_force=?)
    cpdef assign_right(self, node, bint brute_force=?)
    cpdef find_parent(self, child)
    cpdef select_random_leaf(self)
    cpdef select_random_node(self, threshold=?)
    cpdef compute_subtree_size(self)
    cpdef get_fun(self)
    cpdef is_leaf(self)
    cpdef is_constant(self)
    cpdef get_left(self)
    cpdef get_right(self)
    cpdef get_op(self)
    cpdef get_depth(self)
    cpdef depth(self)
    cpdef update_func(self, list params_list=?, str user_prefix=?)
    cpdef flag_constants(self, list constants=?)
    cdef replace_constants_with_pars(self, dict name_values=?)

cdef class Tree:
    cdef double T
    cdef Node root
    cdef str current_description
    cdef object domain
    cdef object value
    cdef public double g_tau
    cdef int depth
    cdef list functions
    cdef int Npars
    cdef list leaves
    cdef list params
    cdef dict param_values, name_values
    cdef double penalty
    cdef object fitness_tree
    cdef double alpha
    #cdef public object sigma
    cdef object sigma
    cdef bint bic_loss_is_param_loss
    cdef init_params(self, cnp.ndarray[cnp.float64_t, ndim=1] init_val)
    cpdef get_domain(self)
    cpdef void set_g_tau(self, double tau) except *
    cdef get_param_values(self)
    cpdef optimize(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str=?, bint use_gradients=?)
    cdef _update_parameters(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str=?, bint use_gradients=?)
    cpdef update_parameters(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str=?, bint use_gradients=?)
    
    cdef build_fitness_tree(self, Node node, str prefix=?, bint is_left=?)
    #cpdef build_tree(self, depth, functions, x )
    cpdef get_sigma(self)
    
    cdef replace_constants_with_pars(self)

    cdef compute_posterior_tree(self, y)
    cdef build_posterior_tree(self, Node node, str prefix=?, bint is_left=?)

    cdef compute_superficial_posterior(self, y, double lambda_reg = ?)

    cpdef get_superficial_posterior(self, y, double lambda_reg=?)

    cdef double _log_marginal_bic(self,
                                  cnp.ndarray[cnp.float64_t, ndim=1] y,
                                  bint optimize=?,
                                  bint use_gradients=?,
                                  str method_str=?)

    cpdef double log_marginal_bic(self,
                                  cnp.ndarray[cnp.float64_t, ndim=1] y,
                                  bint optimize=?,
                                  bint use_gradients=?,
                                  str method_str=?)
    cdef double _shortcut_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, double mse)

    cdef double _update_parameters_and_fitness(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
	str method_str=?, bint use_gradients=?)
    cpdef is_defined(self, x=?)

    cpdef compute_fitness_tree(self, y)

    cpdef evaluate(self, object x=?)

    cpdef void set_sigma(self, object new_sigma)

    cpdef get_root(self)
    
    cpdef get_leaves(self)

    cpdef get_params(self, params_list=?)

    #cpdef tree_to_string(self,  node,  prefix=?,  is_left=?)

    cpdef tree_get_description(self)

    cpdef update_tree_description(self)

    cpdef tree_update(self, Node node=?)

    cpdef node_count(self)

    cpdef bint is_valid(self, object x=?, double max_abs=?)

    cdef double _log_likelihood_at(self,
                               cnp.ndarray[cnp.float64_t, ndim=1] y,
                               dict cand_name_values,
                               double sigma)
    
    cdef void _apply_param_values(self, dict new_name_values)

    cdef _root_replacement(self, object replacement=?, int replace_all = ?, dict kwargs= ?, str op = ?,
                            int new_depth=?, bint ensure_valid=?)

cdef _post_order_update(Node node)

# Phase 2.10 Stage 3b: structural (skeleton) serialization for the MPI wire.
cpdef object tree_to_skel(Node node)
cpdef Node node_from_skel(object skel,
                          cnp.ndarray domain,
                          dict atom_specs)

cpdef build_tree(int depth, list functions,
    cnp.ndarray x, str output_type=?,
    int Npars=?, object leaves_count=?, object arr_indices=?,
    dict params_used=?, object value=?, double T=?, double alpha=?, object sigma=?, double g_tau=?,
    list operations=?)

cdef parameter_parser(int Npars)

cdef new_par(list existing_params, str user_choice=?)

cdef _expr_to_tree(object expr, cnp.ndarray domain, list function_list, dict params_used=?)

cdef list _symbol_values(dict name_values)

cdef _parameter_simplifier(object expr, list used_params)
