import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport real_functionC
cimport gp_movesC
import sympy as sp
from mpi4py import MPI

cdef class Forest:
    def __init__(self, float[:] T, int size, int[:] depth, list funcs, 
                 cnp.ndarray[cnp.float64_t, ndim=1] x, object nodes_list=None,
                 int Npars=0, int value=1, object operations=None):
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
        self.operations = operations
        self.depth = depth
        self.funcs = funcs
        self.domain = x
        self.nodes = nodes_list
        self.n_pars = Npars
        self.input_value = value
        self.T = np.sort(T)
        self.size = size
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.trees = self.grow_forest()
        self.posteriors = np.empty((self.size,)) 
        
        
    cpdef set_trees(self, list new_trees):
        self.trees = new_trees 

    cpdef set_single_tree(self, function_treeC.Tree tree, int i):
        self.trees[i] = tree

    cdef grow_forest(self):
        cdef int Ti
        cdef int i
        if np.isinstance(self.T, np.ndarray) and self.T.shape>=self.size:
            return [function_treeC.Tree(self.depth, self.funcs,
                        self.domain, self.nodes[i], self.n_pars, 
                        self.input_value, self.T[i]) for i in range (0, self.size)]
        elif np.isinstance(T, np.ndarray) and T.shape<self.size:
            T=np.pad(T, (size-T.shape,), 'edge')
            return [function_treeC.Tree(self.depth, self.funcs,
                        self.domain, self.nodes[i], self.n_pars, 
                        self.input_value, self.T[i]) for i in range (0, self.size)]
        else: 
            return [function_treeC.Tree(self.depth, self.funcs,
                        self.domain, self.nodes[i], self.n_pars, 
                        self.input_value, self.T[i]) for i in range (0, self.size)]
 
    
    cdef compute_posterior_forest(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        cdef int i
        for i in range (0, self.size):
            self.trees[i].compute_posterior_tree(y)
            self.posteriors[i] = self.trees[i].root.posterior
    
    cdef distribute(self, comm):
        cdef list all_trees = self.get_trees() if rank==0 else None
        comm.scatter(all_trees, root=0)
       

    cdef gather(object local_trees, object comm)):
        cdef object all_trees = comm.gather(local_trees, root=0)
        cdef function_treeC.Tree tree
        cdef list sublist
        if comm.Get_rank()==0:
            return [tree for sublist in all_trees for tree in sublist]
    
    cdef calculate_posteriors(self, float[:] data):
        for i in range (0, self.size):
            self.posteriors[i] = calculate_posterior(self.trees[i], data)

    cdef parallel_evolve(self, int generations):
        self.perform_mutations()
        if crossover:
            self.crossover_between_processes()
        self.gather()
