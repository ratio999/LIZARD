import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport real_functionC
cimport gp_movesC
import sympy as sp
#from mpi4py import MPI

cdef class Grove:
    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=1] x, 
        double[:] T, int size, int[:] depth, list funcs,  
                 object node_list=None, int Npars=0, int value=1, 
                 object operations=None):
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
        self.operations = operations
        self.depth = depth
        self.funcs = funcs
        self.domain = x
        self.nodes = node_list
        self.n_pars = Npars
        self.input_value = value
        self.T = np.sort(T)
        self.size = size
        self.root_posteriors = np.empty((self.size,))  
        self.trees = self.grow_grove()
        
        
        
    cpdef set_trees(self, list new_trees):
        self.trees = new_trees 

    cpdef set_single_tree(self, function_treeC.Tree tree, int i):
        self.trees[i] = tree

    cdef grow_grove(self):
        cdef int Ti
        cdef int i
        if np.isinstance(self.T, np.ndarray) and self.T.shape[0]>=self.size:
            return [function_treeC.Tree(self.depth, self.funcs,
                        self.domain, self.nodes[i], self.n_pars, 
                        self.input_value, self.T[i]) 
                    for i in range (0, self.size)]
        elif np.isinstance(self.T, np.ndarray) and self.T.shape[0]<self.size:
            T=np.pad(self.T, (self.size-self.T.shape[0],), 'edge')
            return [function_treeC.Tree(self.depth, self.funcs,
                        self.domain, self.nodes[i], self.n_pars, 
                        self.input_value, self.T[i]) 
                    for i in range (0, self.size)]
        else: 
            return [function_treeC.Tree(self.depth, self.funcs,
                        self.domain, self.nodes[i], self.n_pars, 
                        self.input_value, self.T[i]) 
                    for i in range (0, self.size)]
 
    
    cdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        cdef int i
        for i in range (0, self.size):
            self.trees[i].compute_posterior_tree(y)
            self.root_posteriors[i] = self.trees[i].root.posterior
    '''
    cdef distribute(self, comm):
        if rank==0:
            cdef list all_trees = self.trees 
            comm.scatter(all_trees, root=0)
       

    cdef gather(object local_trees, object comm):
        cdef object all_trees = comm.gather(local_trees, root=0)
        cdef function_treeC.Tree tree
        cdef list sublist
        if comm.Get_rank()==0:
            return [tree for sublist in all_trees for tree in sublist]
    '''