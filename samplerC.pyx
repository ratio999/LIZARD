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
import random

cdef class MCMCWalker:

    def __init__(self, cnp.ndarray[cnp.float64_t, ndim=1] x, 
                 cnp.ndarray[cnp.float64_t, ndim=1] data, double[:] T, int size,
                 int[:] depth, list funcs, object node_list = None, int Npars=0,
                 int value = 1, object operations = None, parallel=False, int max_iter = 1000,
                 bint verbose=False, dict gp_probabilities = None):
        self.grove = groveC.Grove(x, T, size, depth, funcs, node_list, 
                           Npars, value, operations) 
        self.data = data
        self.grove.compute_posterior_grove(self.data) 
        self.parallel_bint = parallel
        self.max_iter = max_iter
        self.verbose = verbose
        self.funcs=funcs
        self.operations = operations
        self.standard_probabilities ={'root_replacement': 0.05, 
                                     'mutation': 0.2, 
                                     'crossover':0.2,
                                     'point':0.25, 
                                     'hoist': 0.15,
                                     'addition': 0.15}
  
        if gp_probabilities is None:
            self.gp_probabilities = self.standard_probabilities
        else:
            self.gp_probabilities = gp_probabilities
        
        cdef float opsum = 0.0
        cdef float norm = sum(self.gp_probabilities.items())
        
        if norm != 1.0:
            print('Unnormalized probabilities provided. Normalizing...')
            for item in gp_probabilities.items():
                item /= norm
        
        cdef int i = 0
        
        for i, key in enumerate(self.gp_probabilities.keys()):
            self.op_range[opsum] = key 
            opsum += self.gp_probabilities[key]
            #self._lambda_operations[key] = _cgp_ops_list[i]

    cdef evolve(self, int i):
        
        cdef function_treeC.Tree current_tree = copy.deepcopy(self.grove.trees[i])
        cdef function_treeC.Tree old_tree = copy.deepcopy(self.grove.trees[i])
        
        np.random.seed(i)
        
        cdef float move_type = np.random.uniform((0,1))
        
        lesses = [v for v in op_range.keys() if v <= move_type]
        
        cdef float move = max(lesses)
        cdef str selected_move = self.op_range[move] 
        
        cgp._operation_call(selected_move, current_tree, self.funcs, self.operations)

        cdef double old_posterior = self.grove.trees[i].root.posterior 
        cdef double proposed_posterior = current_tree.root.posterior 
        cdef double acceptance_ratio = np.exp((proposed_posterior - old_posterior) / self.grove.T[i])
        
        np.random.seed(i*1000 + int(MPI.Wtime()*1000))

        if acceptance_ratio > np.random.rand():
            return current_tree
        else:
            return old_tree
    
    cdef move(object move_type, function_treeC.Tree current_tree):
        if move_type == 'root_replacement' 
            cgp.reset_tree(current_tree) #to define reset tree in gp_moves
        if move_type == 'mutation'
            cgp.subtree_mutation(current_tree, self.grove.funcs)
        elif move_type == crossover:
            random.seed(i**2 + 1000)
            if random.choice([False, True]):
                other_tree = copy.deepcopy(self.grove.trees[i+1]) if i+1 < self.grove.size else copy.deepcopy(self.grove.trees[i-1])
            else:
                other_tree = copy.deepcopy(self.grove.trees[i-1]) if i != 0 else copy.deepcopy(self.grove.trees[i+1])
            cgp.subtree_crossover(current_tree, other_tree)
        elif move_type == 'point':
            cgp.point_mutation(current_tree, self.grove.funcs, self.grove.operations)
        elif move_type == 'hoist':
            cgp.hoist_mutation(current_tree)
        elif move_type = 'addition' 
            cgp.add_node(current_tree, self.grove.funcs, self.grove.operations)
        

    cdef print_best_tree(self):
        cdef int j = np.argsort(self.grove.posteriors)
        print(self.grove.trees[j])

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
                self.grove.set_trees([tree for sublist in evolved_grove for tree in sublist])
                self.grove.compute_posteriors(self.data)
        else:
            evolved_grove = [None]*self.grove.size
            for i in range(0, self.grove.size):
                evolved_grove[i] = self.evolve(i)
            self.grove.set_trees(evolved_grove)
            self.grove.compute_posteriors(self.data)
        if self.verbose:
            self.print_best_tree()
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
        for i in range(min(n_steps, self.max_iter)):
            self.parallel_evolve()

    def run(self, n_steps):
        self._run(n_steps)

    cpdef get_best_tree(self):
        """
        Returns gest tree found so far.
        """
        return np.min(self.grove.root_posteriors)



def example_run():
    size = 100
    x = np.linspace(0,10,100)
    data = np.sin(x) + 0.1*np.random.randn(size)
    T=np.full(size, 1.0)
    depth = np.full(size, 3, dtype=np.int32)
    funcs = ["sin(x)", "cos(x)", "exp(x)", "log(x)", "x"]
    Npars = 2
    walker = MCMCWalker(x, data, T, size, depth, funcs, Npars=Npars, max_iter=50, verbose=True)
    walker.run(20)
    best_tree = walker.get_best_tree()
