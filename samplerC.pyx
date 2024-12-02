# sampler.pyx
from libc.stdlib cimport rand, srand
cimport numpy as cnp
cimport function_treeC
from fitnessC cimport calculate_likelihood
from posteriorC cimport calculate_posterior
cimport gp_movesC as cgp
import numpy as np
import copy

cdef class MCMCSampler:

    def __init__(self, dict initial_trees, list functions, list operations, 
            cnp.ndarray[cnp.float64_t, ndim=1] temperatures=None):
        self.functions = functions
        self.operations = operations
        self.current_trees = initial_trees
        if temperatures is None or len(temperatures)<=0:
            temperatures = np.ndarray([1])
        self.temperatures = temperatures
        for T in self.temperatures:
            self.current_posterior = calculate_posterior(initial_trees[T])

    cdef propose_new_trees(self):
        """
        Propose a new tree obtained by applying some GP move to the current tree.
        """
        cdef int move_type
        cdef function_treeC.Tree current_tree
        cdef dict return_dict = copy.copy(self.current_trees)
        for i in range (0, len(self.temperatures)):
            current_tree = copy.copy(self.current_trees[self.temperatures[i]])
            move_type = rand() % 5  
            
            if move_type == 0:
                cgp.subtree_mutation(current_tree, self.functions)
            elif move_type == 1:
                other_tree = copy.copy(self.current_trees[self.temperatures[i+1]])
                cgp.subtree_crossover(current_tree, other_tree)
            elif move_type == 2:
                cgp.point_mutation(current_tree, self.functions, self.operations)
            elif move_type == 3:
                cgp.hoist_mutation(current_tree)
            else:
                cgp.add_node(current_tree, self.functions, self.operations)

            #return_dict[self.temperature[i]] = current_tree
            proposed_posterior = calculate_posterior(current_tree)
            acceptance_ratio = np.exp((proposed_posterior - self.current_posterior) / self.temperature)
            if acceptance_ratio > rand():
                return_dict[self.temperatures[i]] = current_tree
            
                
        return return_dict

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
