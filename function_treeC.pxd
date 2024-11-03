cimport numpy as cnp
import sympy as sp
cimport real_functionC

cdef class Node:
    cdef real_functionC.RealFunctionC function
    cdef object left
    cdef object right
    cdef str op_symbol
    cdef int node_depth
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
    cpdef update_func(self)

cdef class Tree:
    cdef Node root
    cdef str current_description
    cdef domain
    cdef int depth
    cdef list leaves
    cdef list params

    cpdef get_domain(self)

    #cpdef build_tree(self, depth, functions, x )

    cpdef evaluate(self, x=?)

    cpdef get_root(self)
    
    cpdef get_leaves(self)

    cpdef get_params(self, params_list=?)

    #cpdef tree_to_string(self,  node,  prefix=?,  is_left=?)

    cpdef tree_get_description(self)

    cpdef update_tree_description(self)

    cpdef tree_update(self, node)

cdef build_tree(int depth, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x, str output_type=?)
