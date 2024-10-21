cimport numpy as cnp
import sympy as sp
cimport real_functionC

cdef class Node:
    cdef real_functionC.RealFunctionC function
    cdef object left
    cdef object right
    cdef str op_symbol

    cpdef is_leaf(self)

cdef class Tree:
    cdef Node root

    cpdef build_tree(self, depth, functions, x )

    cpdef evaluate(self, x)

    cpdef get_root(self)
    #cpdef tree_to_string(self,  node,  prefix=?,  is_left=?)

    cpdef tree_get_description(self)