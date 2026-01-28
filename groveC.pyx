# cython: language_level=3
import random
import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport gp_movesC
cimport real_functionC
import itertools
from tqdm import tqdm
import time
import sys
import os
from copy import deepcopy
from libc.math cimport exp, fmax
from mpi4py import MPI
#from mpi4py cimport MPI as cMPI
#from mpi4py.libmpi cimport *

cdef inline double _safe_exp(double x) nogil:
    # mild guard for very negative x
    if x < -700.0:
        return 0.0
    elif x > 700.0:
        return 1.0e304  # not used directly as prob, but keep bounded
    return exp(x)

cdef inline double _accept_prob(double dlogpost, double T) nogil:
    # T <= 0 means greedy acceptance
    if T <= 0.0:
        return 1.0 if dlogpost >= 0.0 else 0.0
    return fmax(0.0, 1.0 if dlogpost >= 0.0 else _safe_exp(dlogpost / T))

cdef class Grove:
    """
    A set of Trees evolving at a single temperature T (Metropolis-Hastings).
    Higher T broader proposals; lower T conservative proposals.
    """
    
    def __init__(self,
                 cnp.ndarray[cnp.float64_t, ndim=1] x,
                 cnp.ndarray[cnp.float64_t, ndim=1] y,
                 double T,  # accepts scalar or array; use T_arr[0]
                 int size,
                 int depth,
                 list funcs,
                 object node_list=None,
                 int Npars=0,
                 object init_value=1.0,
                 list operations=None,
                 double base_temp=1.0,
                 double lambda_reg=0.00,
                 bint use_gradients=True,
                 bint burn_in=False,
                 int use_mpi=0,
                 bint use_spawn=False):

        # Memory view - safer alternative to raw pointers
        cdef cnp.float64_t[::1] x_view = x
        # Store y as instance attribute for use in optimization and posterior computation
        self.data = np.asarray(y, dtype=np.float64)
        self.lambda_reg = lambda_reg  # default; can be changed in optimize()
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
        self.operations = operations
        self.funcs = funcs
        self.domain = real_functionC._domain_fixer(x)
        self.use_gradients = use_gradients
        self.method_str = 'L-BFGS-B' if self.use_gradients else 'Powell'
        self.depth = depth
        self.size = size
        self.use_mpi = use_mpi >= 2
        self.n_procs = use_mpi
        self.use_spawn = use_spawn
        self.T = float(T)
        self.base_temp = base_temp if base_temp > 0.0 else 1.0
        #self.trees = self._init_trees(node_list, Npars, init_value)
        if self.size > 0:
            self.root_posteriors = np.full((self.size,), np.nan, dtype=np.float64)
            self.burn_in = burn_in
            self.size=0            
        else:
            self.root_posteriors=None
            self.burn_in = True

        self.acceptance_count = 0.0
        self.proposal_count = 0.0
        self.cache = {}
        if self.burn_in:
            # Run two-stage burn-in procedure to initialize trees
            # Stage 1: Generate all depth-1 combinations (primitive op primitive)
            depth1_candidates = self._get_depth1_candidates(y if 'y' in locals() else self.data, tree_cache=self.cache, optimize_params=True)
            
            # Stage 2: Combine top depth-1 trees to create depth-2 trees
            if len(depth1_candidates) > 0 and not self.use_mpi:
                self._burn_in_stage2(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    self.cache,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True
                )
            elif len(depth1_candidates) > 0 and self.use_mpi and not self.use_spawn:
                self._burn_in_stage2_parallel(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    self.cache,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True
                )
            
            elif len(depth1_candidates) > 0 and self.use_mpi and self.use_spawn:
                self._burn_in_stage2_parallel_spawn(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    self.cache,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True,
                    n_workers = self.n_procs
                )
            
            # Initialize trees list if not already done
            if not hasattr(self, 'trees') or self.trees is None:
                self.trees = [None] * self.size
                self.root_posteriors = np.full((self.size,), np.nan, dtype=np.float64)
        else:
            self.trees = self._init_trees(node_list, Npars, init_value)
        
    cdef _init_trees(self, list node_list, int Npars, object init_value):
        cdef list trees = [None] * self.size
        cdef int i
        for i in range(self.size):
            if node_list is not None and len(node_list)>i:
                    trees[i] = function_treeC.Tree(self.depth, self.funcs, self.domain,
                                               node_list[i], Npars, init_value, self.T)
            else:
                trees[i] = function_treeC.Tree(self.depth, self.funcs, self.domain,
                                               Npars=Npars, value=init_value, T=self.T)
        return trees

    cpdef set_trees(self, list new_trees):
        self.trees = new_trees

    cpdef set_single_tree(self, function_treeC.Tree tree, int i):
        self.trees[i] = tree

    cpdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=False):
        """
        Update/compute root posteriors. If full, compute entire tree; else superficial (root only).
        """
        cdef int i
        if full:
            for i in range(self.size):
                self.trees[i].compute_posterior_tree(y) #controlla i metdi in functiontree
                self.root_posteriors[i] = self.trees[i].root.posterior
        else:
            for i in range(self.size):
                #print('FITNESS', self.trees[i])
                self.root_posteriors[i] = self.trees[i].get_superficial_posterior(y)

    cpdef best_index(self):
        return int(np.nanargmax(self.root_posteriors))

    cpdef best_tree(self):
        return self.trees[self.best_index()]

    cdef _choose_move(self) :
        """
        Select a move and parameters based on temperature.
        Returns (name, kwargs_dict).
        """
        # normalize temperature to [0, +inf): higher → more disruptive moves
        cdef double tau = self.T / self.base_temp
        # probabilities
        # small/local moves favored when tau << 1
        p_point = 0.35 * (1.0 / (1.0 + tau)) + 0.05
        p_hoist = 0.20 * (1.0 / (1.0 + tau)) + 0.05
        p_add   = 0.20 * (1.0 / (1.0 + tau)) + 0.05
        # bigger moves favored when tau >> 1
        p_subtree = 0.15 * (tau / (1.0 + tau)) + 0.10
        p_cross   = 0.10 * (tau / (1.0 + tau)) + 0.05
        probs = np.array([p_point, p_hoist, p_add, p_subtree, p_cross], dtype=np.float64)
        probs = probs / probs.sum()
        idx = int(np.random.choice(np.arange(5), p=probs))
        random_depth = (tau > 0.7)  # deeper/remoter edits at higher T
        if idx == 0:
            return ("point_mutation", {"operations": self.operations})
        elif idx == 1:
            return ("hoist_mutation", {})
        elif idx == 2:
            return ("add_node", {"operations": self.operations, "random_depth": random_depth})
        elif idx == 3:
            return ("subtree_mutation", {"random_depth": random_depth})
        else:
            return ("subtree_crossover", {})  # needs donor tree

    cdef _propose(self, int i, int max_attempts=2) :
        """
        Build a proposal from tree i using a move drawn from the temperature-dependent kernel.
        Returns a deepcopy candidate (original tree remains unchanged).
        """
        cdef function_treeC.Tree cand 
        cdef str move_name
        cdef dict kwargs
        cdef int attempts = 0
               
        while attempts < max_attempts:
            cand = deepcopy(self.trees[i])
            move_name, kwargs = self._choose_move()
            print(move_name)
            if move_name == "point_mutation":
                gp_movesC.point_mutation(cand, self.funcs, kwargs.get("operations", None))
            elif move_name == "hoist_mutation":
                gp_movesC.hoist_mutation(cand)
            elif move_name == "add_node":
                gp_movesC.add_node(cand, self.funcs,
                                kwargs.get("operations", None),
                                kwargs.get("random_depth", False))
            elif move_name == "subtree_mutation":
                #print(kwargs.get('random_depth', False))
                gp_movesC.subtree_mutation(cand, self.funcs,
                                        kwargs.get("random_depth", False))
            elif move_name == "subtree_crossover":
                # pick a donor tree (not i), deep-copied to keep original intact
                if self.size > 1:
                    donor_idx = i
                    while donor_idx == i:
                        donor_idx = random.randrange(self.size)
                    donor = deepcopy(self.trees[donor_idx])
                    print('DONOR:')
                    print(donor)
                    gp_movesC.subtree_crossover(cand, donor)
                else:
                    # fallback to a local move if only one tree
                    gp_movesC.point_mutation(cand, self.funcs, self.operations)
            else:
                # default safety: point mutation
                gp_movesC.point_mutation(cand, self.funcs, self.operations)
            
            if cand.is_valid(self.domain):
                print('VALID:')
                print(cand)
                
                return cand
            attempts += 1
            #print(attempts)
        return deepcopy(self.trees[i])

    cdef _step(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        One MH step on a random tree in this grove.
        Returns True if accepted.
        """
        if self.size == 0:
            return False
        cdef int i = 0 #random.randrange(self.size)

        # ensure current posterior is available
        if np.isnan(self.root_posteriors[i]):
            self.root_posteriors[i] = self.trees[i].get_superficial_posterior(y)

        cdef double old_post = float(self.root_posteriors[i])

        cdef function_treeC.Tree cand = self._propose(i)
        #print(cand.name_values)
        
        if len(cand.param_values)>0:
            cand.update_parameters(self.domain, y, self.lambda_reg,
                 self.method_str, self.use_gradients)
        
        cdef double new_post = float(cand.get_superficial_posterior(y))
        cdef double dlogpost = new_post - old_post

        cdef double acc_p = _accept_prob(dlogpost, self.T)
        self.proposal_count += 1.0
        if random.random() < acc_p:
            print('ACCEPTED')
            self.trees[i] = cand
            self.root_posteriors[i] = new_post
            self.acceptance_count += 1.0
            return True
        return False

    cpdef run(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps):
        """
        Run n_steps MH updates within this grove.
        """
        cdef int t
        for t in range(n_steps):
            self._step(y)

    cpdef acceptance_rate(self):
        if self.proposal_count == 0.0:
            return 0.0
        return self.acceptance_count / self.proposal_count

    
    cpdef burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=True, bint optimize_params=True):
        self._burn_in_combinations(y, replace_all=replace_all, 
                                optimize_params=optimize_params)

    cdef _burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=True, bint optimize_params=True):
        """
        Systematic burn-in: generate all combinations of (func1 op func2) 
        for all operations and all pairs of functions (including parameters).
        
        Parameters
        ----------
        y : array
            Target data for posterior evaluation.
        replace_all : bool
            If True, replace all trees. If False, replace only if better.
        optimize_params : bool
            If True, optimize parameters for each candidate.
        """
      
        cdef list candidates = []
        cdef list all_primitives = []
        cdef str func_str, left_str, right_str, op
        cdef function_treeC.Tree tree
        cdef double post
        cdef int i, idx
        
        # Build full primitive set: functions + one parameter
        all_primitives = self.funcs.copy()
        all_primitives.append("a0")  # add a parameter primitive
        
        print(f"Burn-in: exploring {len(all_primitives)} squared times {len(self.operations)} = "
              f"{len(all_primitives)**2 * len(self.operations)} combinations")
        
        # Generate all (left, op, right) combinations
        for left_str, right_str in itertools.product(all_primitives, repeat=2):
            for op in self.operations:
                '''if op == 'o':  # composition: only makes sense when right is a function of x
                    # Skip if right is a parameter
                    if left_str == "a0" or right_str == "a0":
                        continue'''
                
                
                tree = function_treeC.Tree(
                    depth=1, 
                    functions=self.funcs, 
                    x=self.domain, 
                    Npars=0,  # will add params manually
                    value=1.0, 
                    T=self.T
                )
                
                # Build left child
                if left_str == "a0":
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.left.set_value(1.0)
                else:
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(left_str, self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Build right child
                if right_str == "a0":
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a1", self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.right.set_value(1.0)
                else:
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(right_str, self.domain, compute_grad = False),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Assign operation and update
                tree.root.assign_operation(op)
                tree.tree_update(tree.root)
                
                # Validate
                if not tree.is_valid(self.domain):
                    continue
                print(tree)
                # Optimize parameters if any exist
                if optimize_params and len(tree.params) > 0:
                    tree.update_parameters(self.domain, y, self.lambda_reg, 
                        self.method_str, self.use_gradients)
                    
                
                # Compute posterior
                post = tree.get_superficial_posterior(y)
                if np.isfinite(post):
                    candidates.append((tree, post))
            
        
        if len(candidates) == 0:
            print("Warning: No valid burn-in candidates generated")
            return
        
        # Sort by posterior (descending = better)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Burn-in: {len(candidates)} valid candidates generated")
        print(f"  Best 3 posteriors: {[c[1] for c in candidates[:3]]}")
        print(f"  Best 3 expressions: {[c[0].tree_get_description() for c in candidates[:3]]}")
        
        # Replace grove trees
        if replace_all:
            # Take top `size` candidates, cycling if fewer
            for i in range(self.size):
                idx = i % len(candidates)
                self.trees[i] = deepcopy(candidates[idx][0])
                self.root_posteriors[i] = candidates[idx][1]
        else:
            # Replace only if new candidate is better
            for i in range(min(self.size, len(candidates))):
                if np.isnan(self.root_posteriors[i]) or candidates[i][1] > self.root_posteriors[i]:
                    self.trees[i] = deepcopy(candidates[i][0])
                    self.root_posteriors[i] = candidates[i][1]
        
        print(f"Burn-in complete. Grove best posterior: {self.root_posteriors[self.best_index()]:.4f}")

    cpdef burn_in_depth2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                         bint replace_all=True, bint optimize_params=True):
        """
        Stage 2 burn-in: combine top depth-1 trees with primitives and each other.
        Call this AFTER burn_in_combinations to build depth-2 expressions.
        
        Parameters
        ----------
        y : target data
        replace_all : if True, replace all grove trees with best depth-2 candidates
        optimize_params : if True, optimize parameters for each candidate
        
        """
        cdef dict tree_cache = {}
        # First, run stage 1 to get depth-1 candidates
        depth1_candidates = self._get_depth1_candidates(y, tree_cache, optimize_params)
        
        if len(depth1_candidates) == 0:
            print("Warning: No depth-1 candidates; cannot build depth-2")
            return
        print(tree_cache)
        # Now combine them
        self._burn_in_stage2(y, depth1_candidates, tree_cache, replace_all, optimize_params)
    
    cdef _get_depth1_candidates(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict tree_cache,
                                      bint optimize_params=True):
        """
        Build all depth-1 trees (primitive op primitive) and return sorted list.
        Reuses logic from _burn_in_combinations but returns candidates instead of replacing.
        """
        cdef list candidates = []
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        cdef str left_str, right_str, op
        cdef function_treeC.Tree tree, primitive_tree
        cdef function_treeC.Node root_node
        cdef double post
        print(f"Stage 1: building {len(all_primitives)**2 * len(self.operations)} depth-1 trees")
        # Initialize candidates with primitive functions (depth-0 trees)
        for func_str in all_primitives:
            if func_str == "a0":
            
                root_node= function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                    )
                root_node.set_value(1.0)
            
            else:
                root_node = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(func_str, self.domain, compute_grad = False),
                    )
            primitive_tree = function_treeC.Tree(
                depth=root_node.node_depth, functions=self.funcs, x=self.domain, node=root_node,
                Npars=len(root_node.function.params), value=1.0, T=self.T
            )
            
            
            #primitive_tree.tree_update(primitive_tree.root)
            primitive_expression = primitive_tree.tree_get_description()
            
            if primitive_tree.is_valid(self.domain) and primitive_expression not in tree_cache:
                
                if optimize_params and len(primitive_tree.params) > 0:
                    primitive_tree.update_parameters(self.domain, y, self.lambda_reg, 
                    self.method_str, self.use_gradients)
                
                post = primitive_tree.get_superficial_posterior(y)
                
                tree_cache[primitive_expression] = post
                if np.isfinite(post):
                    candidates.append((primitive_tree, post, 0))  # depth 0
                
        cdef function_treeC.Node root_left, root_right, root_tree
        for left_str, right_str in itertools.product(all_primitives, repeat=2):
            if left_str=='a0' and right_str=='a0':
                continue
            for op in self.operations:
                
                if left_str == "a0":
                    root_left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                    )
                    root_left.set_value(1.0)
                else:
                    root_left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(left_str, self.domain, compute_grad = False),
                    )
                
                # Build right child
                if right_str == "a0":
                    root_right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain, compute_grad = False),
                    )
                    root_right.set_value(1.0)
                else:
                    root_right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(right_str, self.domain, compute_grad = False),
                    )
                print(op, root_left.function.get_description(), root_right.function.get_description())
                root_tree = function_treeC.Node(
                    function=None,
                    left=root_left,
                    right=root_right,
                    symbol=op
                )
                print('root built', root_tree.function.get_description())                
                tree = function_treeC.Tree(
                    depth=root_tree.depth(), functions=self.funcs, x=self.domain, node=root_tree,
                    Npars=len(root_tree.function.params), T=self.T
                )
                if len(root_tree.function.params) > 1:
                    print('# parameters > 1')
                
                #tree.tree_update(tree.root)
                expression = tree.tree_get_description()
                
                if expression in tree_cache:
                    continue
                if not tree.is_valid(self.domain):
                    continue
                print(tree)
                if optimize_params and len(tree.params) > 0:
                    print('computing params for ', expression)
                    tree.update_parameters(self.domain, y, self.lambda_reg,
                        self.method_str, self.use_gradients)
                
                post = tree.get_superficial_posterior(y)
                tree_cache[expression] = post
                if np.isfinite(post):
                    candidates.append((tree, post, 1)) # depth 1
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 1: {len(candidates)} valid depth-1 candidates")
        if len(candidates) > 0:
            print(f"  Top-3: {[(c[0].tree_get_description(), f'{c[1]:.3f}') for c in candidates[:3]]}")
        
        return candidates

    cdef _burn_in_stage2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                          list up2depth1_candidates,
                          dict tree_cache,
                          bint replace_all=True,
                          bint optimize_params=True
                          ):
        """
        Combine top depth-1 trees with primitives and with each other to build depth-2 trees.
        """
        cdef list candidates = []
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        # Filter for only depth-1 candidates (third element == 1)
        
        depth1_candidates = [c for c in up2depth1_candidates if c[2] == 1]
        

        cdef int n_top = len(depth1_candidates)
        cdef int total_combinations = n_top * (n_top + len(all_primitives)) * len(self.operations)
        start_time = time.time()
        pbar = tqdm(total=total_combinations, 
                    desc="Stage 2",
                    unit=" combos",
                    position=0,
                    leave=True,
                    dynamic_ncols=True,
                    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        print(f"Stage 2: combining {n_top} depth-1 trees with {len(all_primitives)} primitives + themselves")
        print(f"  Estimated combinations: ~{n_top * (len(all_primitives) + n_top) * len(self.operations)}")
        print(f"  Estimated RAM occupancy of the grove: ~{n_top * (len(all_primitives) + n_top) * len(self.operations) * 160 / 1000000} MegaBytes")
        
        cdef str prim_str, op
        cdef function_treeC.Tree tree, d1_tree, d1_tree_2
        cdef double post
        cdef int i, j, idx
        cdef function_treeC.Node prim_node
        
        # 1) depth-1 op primitive (both directions)
        
        for i in range(n_top):
            for prim_str in all_primitives:
                prim_node = function_treeC.Node(function=real_functionC.RealFunctionC.from_string(prim_str, self.domain, compute_grad = False))
                for op in self.operations:
                    # left = depth-1, right = primitive
                    d1_tree = depth1_candidates[i][0] 
                    tree = self._combine_depth1(d1_tree.root, prim_node, op)
                    expression = tree.tree_get_description()
                    
                    if tree is not None and tree.is_valid(self.domain) and expression not in tree_cache:
            
                        if optimize_params and len(tree.params) > 0:
                            tree.update_parameters(self.domain, y, self.lambda_reg, 
                                self.method_str, self.use_gradients)
                        post = tree.get_superficial_posterior(y)
                        tree_cache[expression] = post
                        if np.isfinite(post):
                            candidates.append((tree, post))
                    
                    # left = primitive, right = depth-1
                    tree = self._combine_depth1(prim_node, d1_tree.root, op)
                    expression = tree.tree_get_description()
                    
                    if tree is not None and tree.is_valid(self.domain) and expression not in tree_cache:
                        if optimize_params and len(tree.params) > 0:
                            tree.update_parameters(self.domain, y, self.lambda_reg,
                                self.method_str, self.use_gradients)
                        post = tree.get_superficial_posterior(y)
                        tree_cache[expression] = post
                        if np.isfinite(post):
                            candidates.append((tree, post)) # depth 2
                    pbar.update(1)
                    pbar.set_postfix({"iter/s": f"{(pbar.n / (time.time() - start_time)):.1f}"})
        # 2) depth-1 op depth-1 (all pairs)
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    d1_tree = depth1_candidates[i][0]
                    d1_tree_2 = depth1_candidates[j][0]
                    tree = self._combine_depth1(d1_tree.root, d1_tree_2.root, op)
                    expression = tree.tree_get_description()
                    
                    if tree is not None and tree.is_valid(self.domain) and expression not in tree_cache:
                        if optimize_params and len(tree.params) > 0:
                            tree.update_parameters(self.domain, y, self.lambda_reg,
                                self.method_str, self.use_gradients)
                        post = tree.get_superficial_posterior(y)
                        tree_cache[expression] = post
                        if np.isfinite(post):
                            candidates.append((tree, post)) # depth 2
                    pbar.update(1)
                    pbar.set_postfix({"iter/s": f"{(pbar.n / (time.time() - start_time)):.1f}"})
        pbar.close()
        if len(candidates) == 0:
            print("Warning: No valid depth-2 candidates")
            return
        '''
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 2: {len(candidates)} valid depth-2 candidates")
        print(f"  Top-3: {[(c[0].tree_get_description(), f'{c[1]:.3f}') for c in candidates[:3]]}")
        '''
        # Replace grove trees
        # Append depth-1 candidates to depth-2 candidates
        for i in range(len(up2depth1_candidates)):
            candidates.append(up2depth1_candidates[i])
        
        # Re-sort the combined list (maybe delete?)
        candidates.sort(key=lambda x: x[1], reverse=True)
            
        if replace_all:
            if self.trees is None:
                self.trees = [None] * (len(candidates))
                self.root_posteriors = np.full((len(candidates),),
                     np.nan, dtype=np.float64)
                self.size = len(candidates)
                for i in range(self.size):
                    self.trees[i] = candidates[i][0]
                    self.root_posteriors[i] = candidates[i][1]
            else:
                for i in range(self.size):
                    idx = i % len(candidates)
                    self.trees[i] = candidates[idx][0]
                    self.root_posteriors[i] = candidates[idx][1]
        else:
            for i in range(min(self.size, len(candidates))):
                if np.isnan(self.root_posteriors[i]) or candidates[i][1] > self.root_posteriors[i]:
                    self.trees[i] = candidates[i][0]
                    self.root_posteriors[i] = candidates[i][1]
        
        print(f"Burn-in depth-2 complete. Best: {self.root_posteriors[self.best_index()]:.4f}")
    

    cdef  _combine_depth1(self, function_treeC.Node left, function_treeC.Node right, object op):
        """
        Combine depth-1 tree with a primitive (or vice versa).
        
        """
        
        # Left side
        cdef function_treeC.Node left_node = deepcopy(left)
        cdef function_treeC.Node right_node = deepcopy(right)
        cdef function_treeC.Node root_node = function_treeC.Node(function=None,
                                            left=left_node, right=right_node, symbol=op )
        cdef function_treeC.Tree tree = function_treeC.Tree(depth=root_node.node_depth, functions=self.funcs,
                                    x=self.domain, node = root_node,
                                    Npars=len(root_node.function.params), value=1.0, T=self.T)
        return tree
    
    cpdef _burn_in_stage2_parallel(self, cnp.ndarray[cnp.float64_t, ndim=1] y, list up2depth1_candidates,
                            dict tree_cache, bint replace_all=True, bint optimize_params=True):
        """
        MPI-parallel version of _burn_in_stage2.
        """
        cdef object comm = MPI.COMM_WORLD
        cdef rank = int(comm.Get_rank())
        cdef n_procs = int(comm.Get_size())
        print('number of processes: ', n_procs)
        # Filter depth-1 candidates
        cdef list depth1_candidates = [c for c in up2depth1_candidates if c[2] == 1]
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        
        cdef int n_top = len(depth1_candidates)
        
        # ============================================================
        # BUILD ALL WORK ITEMS (list of tasks)
        # ============================================================
        cdef list work_items = []
        
        # Type 1: depth-1 op primitive (both directions)
        for i in range(n_top):
            for prim_str in all_primitives:
                for op in self.operations:
                    work_items.append(('d1_prim', i, prim_str, op))
        
        # Type 2: depth-1 op depth-1
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    work_items.append(('d1_d1', i, j, op))
        
        cdef int total_work = len(work_items)
        
        # ============================================================
        # DIVIDE WORK AMONG RANKS
        # ============================================================
        # Each rank gets a slice of work_items
        cdef int chunk_size = (total_work + n_procs - 1) // n_procs
        cdef int start_idx = rank * chunk_size
        cdef int end_idx = min(start_idx + chunk_size, total_work)
        cdef object my_work = work_items[start_idx:end_idx]
        
        if rank == 0:
            print(f"Stage 2: {n_top} depth-1 trees, {total_work} total combinations")
            print(f"  Distributed across {n_procs} MPI processes")
        
        # ============================================================
        # LOCAL PROGRESS BAR (only rank 0 shows it)
        # ============================================================
        cdef list local_candidates = []
        cdef dict local_cache = {}
        
        # Only rank 0 shows progress bar
        if rank == 0:
            pbar = tqdm(total=total_work, desc="Stage 2 (MPI)", unit=" combos")
        cdef function_treeC.Node prim_node
        cdef function_treeC.Tree tree, d1_tree, d1_tree_2
        #cdef int local_count, all_counts

        for idx, item in enumerate(my_work):
            if item[0] == 'd1_prim':
                _, i, prim_str, op = item
                prim_node = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(
                        prim_str, self.domain, compute_grad=False))
                d1_tree = depth1_candidates[i][0]
                
                # Left = depth-1, Right = primitive
                tree = self._combine_depth1(d1_tree.root, prim_node, op)
                self._process_candidate(tree, y, optimize_params, local_cache, local_candidates)
                
                # Right = depth-1, Left = primitive
                tree = self._combine_depth1(prim_node, d1_tree.root, op)
                self._process_candidate(tree, y, optimize_params, local_cache, local_candidates)
                
            elif item[0] == 'd1_d1':
                _, i, j, op = item
                d1_tree = depth1_candidates[i][0]
                d1_tree_2 = depth1_candidates[j][0]
                tree = self._combine_depth1(d1_tree.root, d1_tree_2.root, op)
                self._process_candidate(tree, y, optimize_params, local_cache, local_candidates)
            
            # ============================================================
            # UPDATE PROGRESS BAR (aggregate from all ranks)
            # ============================================================
            # Every N iterations, sync progress counts
            
            if (idx + 1) % 100 == 0 or idx == len(my_work) - 1:
                # Gather counts from all ranks to rank 0
                local_count = idx + 1
                all_counts = comm.gather(local_count, root=0)
                
                if rank == 0:
                    total_done = sum(all_counts)
                    pbar.n = total_done
                    pbar.refresh()
        
        if rank == 0:
            pbar.close()
        
        # ============================================================
        # GATHER ALL CANDIDATES TO RANK 0
        # ============================================================
        #cdef object all_candidates_list = comm.gather(local_candidates, root=0)
        cdef list all_cache_list = comm.gather(local_cache, root=0)
        cdef dict all_cache_dict = comm.gather(local_cache, root=0)[0]
        
        if rank==0:
            print(len(all_cache_dict))
            tree_cache.update(all_cache_dict)
        
        #cdef list all_candidates_list = [function_treeC._expr_to_tree(ex, self.domain, self.funcs) for ex in all_cache_list]
        if rank == 0:
            all_cache_list_of_tuples = [(k, v) for k, v in all_cache_dict.items()]
            #all_cache_list_of_tuples.sort(key=_bis2_parallel_helper, reverse=True)
            
            # Flatten the list of lists
            candidates = []
            for cache_tuple in all_cache_list_of_tuples:
                expr_str, post = cache_tuple[0], cache_tuple[1]
                
                tree = function_treeC._expr_to_tree(
                    real_functionC.RealFunctionC.from_string(
                        expr_str, self.domain, compute_grad=False).get_expr(), 
                    self.domain, self.funcs)
                candidates.append((tree, post))
            # Merge caches
            ''' 
            for cache_dict in all_cache_list:
            tree_cache.update(cache_dict)
            
            # Add depth-1 candidates
            for c in up2depth1_candidates:
                candidates.append(c)
            '''
            
            # Sort and assign to grove
            candidates.sort(key=_bis2_parallel_helper, reverse=True)
            
            # ... rest of the assignment logic (same as original)
            if replace_all:
                self.trees = [c[0] for c in candidates]
                self.root_posteriors = np.array([c[1] for c in candidates])
            
            print(f"Burn-in complete. Best: {self.root_posteriors[0]:.4f}")
            print(self.trees[0])
            print(self.trees[0].get_param_dicts()[1])
        
        # Broadcast final trees to all ranks if needed
        #self.trees = comm.bcast(self.trees if rank == 0 else None, root=0)
        #self.root_posteriors = comm.bcast(self.root_posteriors if rank == 0 else None, root=0)

       
    cdef _burn_in_stage2_parallel_spawn(self, 
                                    cnp.ndarray[cnp.float64_t, ndim=1] y,
                                    list up2depth1_candidates,
                                    dict tree_cache,
                                    bint replace_all=True,
                                    bint optimize_params=True,
                                    int n_workers=4):
        """
        Spawn MPI workers for parallel burn-in, then collect results.
        
        This function runs in a SINGLE process. It spawns workers,
        distributes work, collects results, and workers exit.
        No need to run with mpirun/mpiexec.
        
        Parameters
        ----------
        n_workers : int
            Number of worker processes to spawn (default: 4)
        """
        import subprocess
        from mpi4py import MPI
        from operator import itemgetter
        
        # =========================================================
        # PREPARE DATA
        # =========================================================
        cdef list depth1_candidates = [c for c in up2depth1_candidates if c[2] == 1]
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        cdef int n_top = len(depth1_candidates)
        
        if n_top == 0:
            print("No depth-1 candidates to combine")
            return
        
        # Serialize depth-1 trees for workers
        cdef list  depth1_data = []
        cdef function_treeC.Tree tree
        cdef object post
        cdef int depth, i
        cdef object expr
        cdef dict names_params
        for tree, post, depth in depth1_candidates:
            expr = tree.root.function.expr
            names_params = tree.name_values if tree.name_values else {}
            depth1_data.append((expr, names_params, post))
        
        # Build ALL work items
        cdef list work_items = []
        cdef str prim_str, op
        for i in range(n_top):
            for prim_str in all_primitives:
                for op in self.operations:
                    work_items.append(('d1_prim', i, prim_str, op))
        
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    work_items.append(('d1_d1', i, j, op))
        
        cdef int total_work = len(work_items)
        print(f"Stage 2: {n_top} depth-1 trees × {len(all_primitives)} primitives × {len(self.operations)} ops")
        print(f"  Total combinations: {total_work}")
        print(f"  Spawning {n_workers} worker processes...")
        
        # =========================================================
        # SPAWN WORKERS
        # =========================================================
        # Path to worker script (same directory as this module)
        #cdef object worker_script = os.path.join(os.path.dirname(__file__), 'mpi_worker.py')
        
        # Spawn worker processes
        # MPI.COMM_SELF is the communicator for just this process
        intercomm = MPI.COMM_SELF.Spawn(
            sys.executable,  # Python interpreter
            args=["-c", "from mpi_worker import worker_main; worker_main()"],
            maxprocs=n_workers
        )
        
        # =========================================================
        # BROADCAST CONFIG TO WORKERS
        # =========================================================
        config = {
            'domain': self.domain,
            'funcs': self.funcs,
            'operations': self.operations,
            'y': np.asarray(y),
            'lambda_reg': self.lambda_reg,
            'method_str': self.method_str,
            'use_gradients': self.use_gradients,
            'depth1_data': depth1_data,
            'all_primitives': all_primitives,
            'T': self.T
        }
        
        intercomm.bcast(config, root=MPI.ROOT)
        
        # =========================================================
        # SCATTER WORK ITEMS TO WORKERS
        # =========================================================
        # Divide work items into n_workers chunks
        chunk_size = (total_work + n_workers - 1) // n_workers
        work_chunks = []
        for w in range(n_workers):
            start = w * chunk_size
            end = min(start + chunk_size, total_work)
            work_chunks.append(work_items[start:end])
        
        # Scatter chunks to workers
        intercomm.scatter(work_chunks, root=MPI.ROOT)
        
        print(f"  Work distributed. Waiting for workers...")
        
        cdef object pbar=tqdm(total=total_work, desc="Stage 2 (Spawned)", unit=" combos", 
                dynamic_ncols=True)
        
        cdef dict progress_received={w: 0 for w in range(n_workers)}
        cdef object last_update_time = time.time()
        cdef double timeout = 5.0  # seconds - if no updates for 5s, assume workers done

        while True:
            got_update = False
            
            # Non-blocking check for incoming messages
            for source in range(n_workers):
                status = MPI.Status()
                if intercomm.Iprobe(source=source, tag=99, status=status):
                    # Progress update from worker
                    count = intercomm.recv(source=source, tag=99)
                    delta = count - progress_received[source]
                    progress_received[source] = count
                    pbar.update(delta)
                    pbar.refresh()
                    sys.stdout.flush()
                    last_update_time = time.time()
                    got_update = True
            
            # Check for timeout - if no updates for X seconds, workers likely done
            elapsed_since_update = time.time() - last_update_time
            if elapsed_since_update > timeout and sum(progress_received.values()) > 0:
                # No updates for a while and we've received some progress = done
                break
            
            time.sleep(0.05)  # Short sleep to avoid busy-waiting
        
        pbar.close()
        sys.stdout.flush()  # Final flush after closing
        # =========================================================
        # GATHER RESULTS FROM WORKERS
        # =========================================================
        # Gather returns a list of lists (one per worker)
        all_results = intercomm.gather(None, root=MPI.ROOT)
        
        # =========================================================
        # DISCONNECT WORKERS
        # =========================================================
        intercomm.Disconnect()
        print(f"  Workers finished and disconnected.")
        
        # =========================================================
        # PROCESS RESULTS
        # =========================================================
        candidates = []
        
        # Flatten results from all workers
        for worker_results in all_results:
            for expr, params, post in worker_results:
                # Rebuild tree from expression
                tree = function_treeC._expr_to_tree(expr, self.domain, self.funcs)
                if tree is not None:
                    if params:
                        tree.params = list(params)
                    candidates.append((tree, post))
                    tree_cache[expr] = post
        
        print(f"  Received {len(candidates)} valid candidates from workers")
        
        # Add depth-1 candidates
        for c in up2depth1_candidates:
            candidates.append((c[0], c[1]))
        
        # Sort by posterior (best first)
        candidates.sort(key=itemgetter(1), reverse=True)
        
        # =========================================================
        # ASSIGN TO GROVE
        # =========================================================
        if replace_all:
            self.size = len(candidates)
            self.trees = [c[0] for c in candidates]
            self.root_posteriors = np.array([c[1] for c in candidates], dtype=np.float64)
        else:
            for i in range(min(self.size, len(candidates))):
                if np.isnan(self.root_posteriors[i]) or candidates[i][1] > self.root_posteriors[i]:
                    self.trees[i] = candidates[i][0]
                    self.root_posteriors[i] = candidates[i][1]
        
        print(f"Burn-in complete. Best posterior: {self.root_posteriors[0]:.4f}")
        print(f"  Best expression: {self.trees[0].tree_get_description()}")



    cdef _process_candidate(self, function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                        bint optimize_params, dict local_cache, list local_candidates):
        """Helper to process a single candidate tree."""
        if tree is None:
            return
        cdef str expression = tree.tree_get_description()
        if expression in local_cache:
            return
        if not tree.is_valid(self.domain):
            return
        
        if optimize_params and len(tree.params) > 0:
            tree.update_parameters(self.domain, y, self.lambda_reg,
                                self.method_str, self.use_gradients)
        
        cdef double post = tree.get_superficial_posterior(y)
        local_cache[expression] = post
        
        if np.isfinite(post):
            local_candidates.append((tree, post))

cdef class Forest:
    """
    A collection of Groves at different temperatures (parallel tempering scaffold).
    Provides within-grove MH and optional adjacent-grove swaps.
    """

    def __init__(self,
                 cnp.ndarray[cnp.float64_t, ndim=1] x,
                 cnp.ndarray[cnp.float64_t, ndim=1] temperatures,
                 int size_per_grove,
                 int depth,
                 list funcs,
                 object node_lists=None,       # optional list of node_list per grove
                 int Npars=0,
                 object init_value=1.0,
                 list operations=None,
                 double base_temp=1.0,
                 int n_procs=0):
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.groves = []
        self.n_procs = n_procs
        self.use_mpi = self.n_procs >= 2
        cdef int G = self.temperatures.shape[0]
        cdef int g
        for g in range(G):
            nodes_g = None
            if node_lists is not None:
                nodes_g = node_lists[g]
            self.groves.append(
                Grove(x, self.temperatures[g], size_per_grove, depth, funcs,
                      node_list=nodes_g, Npars=Npars, init_value=init_value,
                      operations=operations, base_temp=base_temp)
            )

    cpdef run_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps_per_grove):
        """
        Run MH in each grove independently.
        """
        for gr in self.groves:
            gr.run(y, n_steps_per_grove)

    cpdef compute_posteriors_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=False):
        cdef Grove gr
        for gr in self.groves:
            gr.compute_posteriors(y, full=full)

    cdef bint _attempt_swap_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        Attempt a parallel-tempering swap between neighboring groves g_left and g_left+1.
        Chooses the same index i in both (assumes same size).
        """
        cdef Grove A = self.groves[g_left]
        cdef Grove B = self.groves[g_left + 1]
        cdef int sizeA = A.size
        cdef int sizeB = B.size
        if sizeA == 0 or sizeB == 0:
            return False
        cdef int i = random.randrange(min(sizeA, sizeB))
        # ensure posteriors
        if np.isnan(A.root_posteriors[i]):
            A.root_posteriors[i] = A.trees[i].get_superficial_posterior(y)
        if np.isnan(B.root_posteriors[i]):
            B.root_posteriors[i] = B.trees[i].get_superficial_posterior(y)
        cdef double postA = float(A.root_posteriors[i])
        cdef double postB = float(B.root_posteriors[i])
        cdef double Ta = A.T
        cdef double Tb = B.T
        if Ta <= 0.0 and Tb <= 0.0:
            # both greedy: only swap if improves both (rare); default reject
            return False
        # Parallel tempering swap acceptance:
        # acc = exp((1/Ta - 1/Tb) * (postB - postA))
        cdef double delta = (1.0 / (Ta if Ta > 0.0 else 1e-12) -
                             1.0 / (Tb if Tb > 0.0 else 1e-12)) * (postB - postA)
        cdef double acc_p = _safe_exp(delta)
        if random.random() < min(1.0, acc_p):
            # swap trees and their posteriors
            tmp_tree = A.trees[i]; A.trees[i] = B.trees[i]; B.trees[i] = tmp_tree
            tmp_post = A.root_posteriors[i]; A.root_posteriors[i] = B.root_posteriors[i]; B.root_posteriors[i] = tmp_post
            return True
        return False

    cpdef int attempt_swaps(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        Try swaps between all adjacent pairs once (left-to-right).
        Returns number of accepted swaps.
        """
        cdef int G = len(self.groves)
        cdef int acc = 0
        cdef int g
        if G <= 1:
            return 0
        for g in range(G - 1):
            if self._attempt_swap_pair(g, y):
                acc += 1
        return acc

    cpdef function_treeC.Tree best_tree(self):
        """
        Best (highest posterior) tree across all groves.
        """
        cdef double best_val = -1e300
        cdef function_treeC.Tree best = None
        cdef Grove gr
        cdef int idx
        for gr in self.groves:
            if gr.root_posteriors is None or np.any(np.isnan(gr.root_posteriors)):
                # ensure computed
                # cannot access y here; caller should have called compute_posteriors_all
                pass
            idx = gr.best_index()
            if gr.root_posteriors[idx] > best_val:
                best_val = gr.root_posteriors[idx]
                best = gr.trees[idx]
        return best


cdef _bis2_parallel_helper(object x):
        return x[1]