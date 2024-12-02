# sampler.pyx
from libc.stdlib cimport rand, srand
cimport numpy as cnp
cimport function_treeC
from fitness cimport calculate_likelihood
from posterior cimport calculate_posterior
import numpy as np
cdef class MCMCSampler:

    def __init__(self, dict initial_trees, double temperature=1.0):
        self.current_trees = initial_trees
        self.current_posterior = calculate_posterior(initial_tree)
        self.temperature = temperature

    cdef propose_new_tree(self, list functions, list operations):
        """
        Propose a new tree obtained by applying some GP move to the current tree.
        """
        cdef int move_type
        cdef function_treeC.Tree current_tree
        cdef dict return_dict = {}
        for i in range (0, len(self.temperatures)):
            current_tree = self.current_trees[self.temperatures[i]]
            move_type = rand() % 5  # 0: Mutazione, 1: Crossover, 2: Variazione parametri
            
            if move_type == 0:
                cgp.subtree_mutation(current_tree, functions)
            elif move_type == 1:
                other_tree = self.current_trees[self.temperatures[i+1]]
                cgp.subtree_crossover(current_tree, other_tree)
            elif move_type == 2:
                cgp.point_mutation(current_tree, functions, operations)
            elif move_type == 3:
                cgp.hoist_mutation(current_tree)
            else:
                cgp.add_node(current_tree, functions, operations)

            return_dict[self.temperature[i]] = current_tree
            proposed_posterior = calculate_posterior(current_tree)
            acceptance_ratio = np.exp((proposed_posterior - self.current_posterior) / self.temperature)
        return return_dict

    cdef step(self):
        """
        New MCMC step
        """
        cdef function_treeC.Tree proposed_tree
        cdef object proposed_posterior
        cdef cnp.float64_t acceptance_ratio
        proposed_tree = propose_new_tree(self.current_trees)
        proposed_posterior = calculate_posterior(proposed_trees)
        
        acceptance_ratio = np.exp((proposed_posterior - self.current_posterior) / self.temperature)
        
        if acceptance_ratio > rand():
            self.current_tree = proposed_tree
            self.current_posterior = proposed_posterior

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
