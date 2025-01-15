# sampler.pyx
from libc.stdlib cimport rand, srand
cimport numpy as cnp
cimport function_treeC
cimport gp_movesC as cgp
cimport groveC
import numpy as np
import copy
cimport function_treeC
from mpi4py import MPI

cdef class MCMCWalker:

    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=1] x, 
                 cnp.ndarray[cnp.float64_t, ndim=1] data, double[:] T, int size,
                 int[:] depth, list funcs, object node_list = None, int Npars=0,
                 int value = 1, object operations = None, parallel=False):
        self.grove = groveC.Grove(x, T, size, depth, funcs, node_list, 
                           Npars, value, operations) 
        self.data = data
        self.grove.compute_posterior_grove(self.data) 
        self.parallel_bint = parallel
    
    cdef evolve(self, int i):
        cdef function_treeC.Tree current_tree = copy.copy(self.grove.trees[i])
        cdef function_treeC.Tree old_tree = copy.copy(self.grove.trees[i])
 
        cdef  function_treeC.Tree other_tree  
        srand(i)
        cdef int move_type = rand() % 5  
            
        if move_type == 0:
            cgp.subtree_mutation(current_tree, self.grove.funcs)
        elif move_type == 1:
            other_tree = copy.copy(self.grove.trees[i+1])
            cgp.subtree_crossover(current_tree, other_tree)
        elif move_type == 2:
            cgp.point_mutation(current_tree, self.grove.funcs, self.grove.operations)
        elif move_type == 3:
            cgp.hoist_mutation(current_tree)
        else:
            cgp.add_node(current_tree, self.grove.funcs, self.grove.operations)
        current_tree.compute_posterior_tree(self.data)
        cdef double old_posterior = self.grove.trees[i].root.posterior 
        cdef double proposed_posterior = current_tree.root.posterior 
        cdef double acceptance_ratio = np.exp((proposed_posterior - old_posterior) / self.grove.T[i])
        if acceptance_ratio > rand():
            return current_tree
        else:
            return old_tree


    cpdef parallel_evolve(self):
        if self.parallel_bint:
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            comm_size = comm.Get_size()
            if self.grove.size != comm_size:
                chunk_size = self.grove.size // comm_size
        
                start = rank * chunk_size
                end = self.grove.size if rank == comm_size - 1 else start + chunk_size
                sub_grove = self.grove.trees[start:end]
                method = getattr(self, "evolve")
                evolved_subgrove = [method(j) for j in range(0, len(sub_grove))]
           
            elif self.grove.size == comm_size:
                method = getattr(self, "evolve")
                evolved_subgrove=method(rank)
            
            evolved_grove = comm.gather(evolved_subgrove, root = 0)
            if rank==0:
                self.grove.set_trees(evolved_grove)
                self.grove.compute_posteriors(self.data)
        else:
            evolved_grove = [None]*self.grove.size
            for i in range(0, self.grove.size):
                evolved_grove[i] = self.evolve(i)
            self.grove.set_trees(evolved_grove)
            self.grove.compute_posteriors(self.data)
    '''
    cdef propose_new_trees(self):
        """
        Propose a new tree obtained by applying some GP move to the current tree.
        """
        cdef int move_type
        cdef function_treeC.Tree current_tree
        cdef list return_list = copy.copy(self.forest.trees)
        for i in range (0, self.forest.size):
            current_tree = copy.copy(self.forest.trees[i])
            move_type = rand_r() % 5  
            
            if move_type == 0:
                cgp.subtree_mutation(current_tree, self.forest.funcs)
            elif move_type == 1:
                other_tree = copy.copy(self.forest.trees[i+1])
                cgp.subtree_crossover(current_tree, other_tree)
            elif move_type == 2:
                cgp.point_mutation(current_tree, self.forest.funcs, self.forest.operations)
            elif move_type == 3:
                cgp.hoist_mutation(current_tree)
            else:
                cgp.add_node(current_tree, self.forest.funcs, self.forest.operations)

            proposed_posterior = calculate_posterior(current_tree, self.data)
            acceptance_ratio = np.exp((proposed_posterior - self.current_posterior) / self.temperature)
            if acceptance_ratio > rand():
                return_list[i] = current_tree
            
                
        return return_list

    cdef step(self):
        """
        New MCMC step
        """
        #cdef object proposed_posterior
        #cdef cnp.float64_t acceptance_ratio
        self.current_trees = self.propose_new_trees()
    '''        
    cdef _run(self, n_steps):
        """
        Run MCMC sampling
        """
        for i in range(n_steps):
            self.parallel_evolve()
    def run(self, n_steps):
        self._run(n_steps)

    cpdef get_best_tree(self):
        """
        Returns gest tree found so far.
        """
        return np.min(self.grove.root_posteriors)
