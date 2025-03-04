cimport numpy as cnp
import numpy as np
cimport function_treeC
cimport real_functionC

cdef Polya(function_treeC.Tree previous_tree, list function_list, list operation_list):
    
    
cdef Dirichlet(function_treeC.Tree tree, list function_list, list operation_list):
    cdef double[:] u_k = params(tree, function_list, operation_list) 
