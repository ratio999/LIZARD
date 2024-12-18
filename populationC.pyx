import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport real_functionC
cimport gp_movesC
import sympy as sp
from mpi4py import MPI

cdef class Population:
    def __init__(self, int[:] T):
        cdef int Ti
        self.trees = [function_treeC.Tree(Ti) for Ti in T]
    
    cdef genstep(self, dict T_moves=None):
        if T_moves is None:
            T_moves = random_move(self.T)
        cdef function_treeC.Tree tree
        for tree in self.trees:
            tree.T_moves[tree.T]
        
    cdef distribute(self, comm):
        cdef int rank = comm.Get_rank()
        cdef in size = comm.Get_size()
        cdef list all_trees = self.get_trees() if rank==0 else None
        cdef object local_trees = comm.scatter(all_trees, root=0)
        return local_trees

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
