# sampler.pyx
from libc.stdlib cimport rand, srand, rand_r
cimport numpy as cnp
cimport function_treeC
from fitnessC cimport calculate_likelihood
from posteriorC cimport calculate_posterior
cimport gp_movesC as cgp
import numpy as np
import copy

cdef class MCMCSampler:

    def __init__(self, Forest forest, cnp.ndarray[cnp.float64_t, ndim=1] data,
        cnp.ndarray[cnp.float64_t, ndim=1] temperatures=None):
        self.forest = forest 
        #self.current_trees = forest.trees 
        self.data = data
        self.current_posterior = np.zeros
        for i in range (0, forest.size):
            self.current_posterior[i] = calculate_posterior(forest.trees[i],
                                                            data)
            

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
                cgp.point_mutation(current_tree, self.forest.funcs, self.operations)
            elif move_type == 3:
                cgp.hoist_mutation(current_tree)
            else:
                cgp.add_node(current_tree, self.forest.funcs, self.operations)

            proposed_posterior = calculate_posterior(current_tree, self.data)
            acceptance_ratio = np.exp((proposed_posterior - self.current_posterior) / self.temperature)
            if acceptance_ratio > rand():
                return_list[i] = current_tree
            
                
        return return_list

    cdef step(self):
        """
        New MCMC step
        """
        cdef dict proposed_trees
        #cdef object proposed_posterior
        #cdef cnp.float64_t acceptance_ratio
        self.current_trees = self.propose_new_trees()
        
    cpdef run(self, n_steps):
        """
        Run MCMC sampling
        """
        for i in range(n_steps):
            self.step()

    cpdef get_best_tree(self):
        """
        Returns gest tree found so far.
        """
        return self.current_tree
