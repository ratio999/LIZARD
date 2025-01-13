# sampler.pyx
from libc.stdlib cimport rand, srand, rand_r
cimport forestC
cimport numpy as cnp
cimport function_treeC
from fitnessC cimport calculate_likelihood
from posteriorC cimport calculate_posterior
cimport gp_movesC as cgp
import numpy as np
import copy

cdef class MCMCSampler:

    def __init__(self, forestC.Forest forest, cnp.ndarray[cnp.float64_t, ndim=1] data,
        cnp.ndarray[cnp.float64_t, ndim=1] temperatures=None, parallel = False):
        self.forest = forest 
        #self.current_trees = forest.trees 
        self.data = data
        self.current_posterior = np.zeros
        self.parallel_bint = parallel
        #MPI INSTANCE SHOULD BE CALLED HERE(?)
        for i in range (0, forest.size):
            self.current_posterior[i] = calculate_posterior(forest.trees[i],
                                                            data)
           
    cdef evolve(self, int i):
        cdef function_treeC.Tree current_tree = copy.copy(self.forest.trees[i])

        cdef  function_treeC.Tree other_tree  
        cdef int move_type = rand_r() % 5  
            
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

        cdef float proposed_posterior = calculate_posterior(current_tree, self.data)
        cdef float acceptance_ratio = np.exp((proposed_posterior - self.current_posterior) / self.temperature)
        if acceptance_ratio > rand():
            return current_tree
    
    cpdef parallel_evolve(self):
        comm = MPI.COMM_world
        rank = comm.Get_rank()
        comm_size = comm.Get_size()
        chunk_size = self.forest.size // comm_size

        start = rank * chunk_size
        end = self.forest.size if rank == comm_size - 1 else start + chunk_size
        grove = self.forest.trees[start:end]
        method = getattr(self, evolve)
        evolved_grove = [method(j) for j in range(0, len(self.forest.trees))]
        evolved_forest = comm.gather(evolved grove, root = 0)
        if rank==0:
            self.forest.set_trees(evolved_forest)
            self.forest.calculate_posteriors(self.data)
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
    def run(self, nsteps):
        self._run(n_steps)

    cpdef get_best_tree(self):
        """
        Returns gest tree found so far.
        """
        return np.min(self.forest.posteriors)
