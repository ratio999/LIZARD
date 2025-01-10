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
                 int Npars=0, int value=1):
        self.depth = depth
        self.funcs = funcs
        self.domain = x
        self.nodes = nodes_list
        self.n_pars = Npars
        self.input_value = value
        self.T = T
        self.size = size
        self.rank = MPI.COMM_WORLD.Get_rank()
        self.trees = self.grow_forest()

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
 
    cdef genstep(self, dict T_moves=None):
        if T_moves is None:
            T_moves = self.random_move(self.T)
        cdef function_treeC.Tree tree
        for tree in self.trees:
            tree.T_moves[tree.T]
        
    cdef distribute(self, comm):
        cdef list all_trees = self.get_trees() if rank==0 else None
        comm.scatter(all_trees, root=0)
       

    cdef gather(object local_trees, object comm)):
        cdef object all_trees = comm.gather(local_trees, root=0)
        cdef function_treeC.Tree tree
        cdef list sublist
        if comm.Get_rank()==0:
            return [tree for sublist in all_trees for tree in sublist]

    cdef parallel_evolve(self, int generations):
        self.perform_mutations()
        if crossover:
            self.crossover_between_processes()
        self.gather()
