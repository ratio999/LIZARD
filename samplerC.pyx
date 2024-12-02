# sampler.pyx
from libc.stdlib cimport rand, srand
cimport numpy as cnp
cimport function_treeC
from fitness cimport calculate_likelihood
from proposal cimport propose_new_tree
from posterior cimport calculate_posterior
import numpy as np
cdef class MCMCSampler:

    def __init__(self, function_treeC.Tree initial_tree, double temperature=1.0):
        self.current_tree = initial_tree
        self.current_posterior = calculate_posterior(initial_tree)
        self.temperature = temperature

    cdef step(self):
        """
        New MCMC step
        """
        cdef function_treeC.Tree proposed_tree
        cdef object proposed_posterior
        cdef cnp.float64_t acceptance_ratio
        proposed_tree = propose_new_tree(self.current_tree)
        proposed_posterior = calculate_posterior(proposed_tree)
        
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
