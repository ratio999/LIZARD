cimport numpy as cnp
#import sympy as sp
cimport real_functionC

cdef class Node:
    cdef real_functionC.RealFunctionC function
    cdef object left
    cdef object right
    cdef str op_symbol
    cdef object domain
    cdef int node_depth
    cdef cnp.float64_t value    
    cdef reinit_to_par(self, list existent_pars, str user_prefix=?) 
    cdef set_value(self, cnp.float64_t value=?)
    cdef bint bis_param
    cdef object fitness
    cdef object posterior
    cdef object prior
    cdef double alpha
    cdef object sigma

    cdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data, 
        dict input_pars, double lambda_reg, str method_str=?)
    
    cdef evaluate_node_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters)
    
    cdef set_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters)
    
    cdef sort_params(self, list pars)

    cdef set_alpha(self, double newalpha)

    cdef set_sigma(self, object newsigma)

    cdef set_posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters)

    cdef evaluate_node_posterior(self,cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters)
 
    cpdef posterior_tree(self, y, parameters) 
    cpdef fitness_tree(self,  y, parameters)
    cpdef node_count(self)
    cpdef is_param(self)
    cpdef assign_operation(self, operation)
    cpdef get_leaves(self, leaves_list=?)
    cpdef assign_left(self, node)
    cpdef assign_right(self, node)
    cpdef find_parent(self, child)
    cpdef select_random_leaf(self)
    cpdef select_random_node(self, threshold=?)
    cpdef compute_subtree_size(self)
    cpdef get_fun(self)
    cpdef is_leaf(self)
    cpdef get_left(self)
    cpdef get_right(self)
    cpdef get_op(self)
    cpdef get_depth(self)
    cpdef depth(self)
    cpdef update_func(self, list params_list=?, str user_prefix=?)

cdef class Tree:
    cdef double T
    cdef Node root
    cdef str current_description
    cdef object domain
    cdef int depth
    cdef list functions
    cdef int Npars
    cdef list leaves
    cdef list params
    cdef dict param_values, name_values
    cdef object fitness_tree
    cdef double alpha
    cdef object sigma
    cdef init_params(self, cnp.ndarray[cnp.float64_t, ndim=1] init_val)
    cpdef get_domain(self)
    cdef get_param_values(self)
    cpdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, 
        cnp.ndarray[cnp.float64_t, ndim=1] y_data, double lambda_reg, 
        str method_str=?)
    cdef build_fitness_tree(self, Node node, str prefix=?, bint is_left=?)
    #cpdef build_tree(self, depth, functions, x )
    cdef compute_posterior_tree(self, y)
    cdef build_posterior_tree(self, Node node, str prefix=?, bint is_left=?)

    cdef compute_superficial_posterior(self, y)

    cpdef get_superficial_posterior(self, y)


    cpdef is_defined(self, x=?)

    cpdef compute_fitness_tree(self, y)

    cpdef evaluate(self, x=?)

    cpdef get_root(self)
    
    cpdef get_leaves(self)

    cpdef get_params(self, params_list=?)

    #cpdef tree_to_string(self,  node,  prefix=?,  is_left=?)

    cpdef tree_get_description(self)

    cpdef update_tree_description(self)

    cpdef tree_update(self, node)

    cpdef node_count(self)

cpdef build_tree(int depth, list functions, 
    cnp.ndarray[cnp.float64_t, ndim=1] x, str output_type=?, 
    int Npars=?, object leaves_count=?, object arr_indices=?,
    dict params_used=?, object value=?, double T=?, double alpha=?, object sigma=?)

cdef parameter_parser(int Npars)

cpdef new_par(list existing_params, str user_choice=?)
