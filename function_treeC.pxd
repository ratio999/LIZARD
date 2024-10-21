cimport numpy as cnp
import sympy as sp
cimport real_functionC

cdef class Node:
    cdef real_functionC.RealFunctionC function
    cdef object left
    cdef object right
    cdef str op_symbol

    cpdef get_fun(self)
    cpdef is_leaf(self)
    cpdef get_left(self)
    cpdef get_right(self)
    cpdef get_op(self)

cdef class Tree:
    cdef Node root

    cpdef build_tree(self, depth, functions, x )

    cpdef evaluate(self, x)

    cpdef get_root(self)
    #cpdef tree_to_string(self,  node,  prefix=?,  is_left=?)

    cpdef tree_get_description(self)