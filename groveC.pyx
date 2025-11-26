# cython: language_level=3
import random
import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport gp_movesC
cimport real_functionC
import itertools
from copy import deepcopy
from libc.math cimport exp, fmax

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
                 bint burn_in=False):

        # Memory view - safer alternative to raw pointers
        cdef cnp.float64_t[::1] x_view = x
        # Store y as instance attribute for use in optimization and posterior computation
        self.data = np.asarray(y, dtype=np.float64)
        self.lambda_reg = lambda_reg  # default; can be changed in optimize()
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
        self.operations = operations
        self.funcs = funcs
        self.domain = x
        self.depth = depth
        self.size = size
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
        if self.burn_in:
            # Run two-stage burn-in procedure to initialize trees
            # Stage 1: Generate all depth-1 combinations (primitive op primitive)
            depth1_candidates = self._get_depth1_candidates(y if 'y' in locals() else self.data, optimize_params=True)
            
            # Stage 2: Combine top depth-1 trees to create depth-2 trees
            if len(depth1_candidates) > 0:
                self._burn_in_stage2(
                    y if 'y' in locals() else self.data,
                    depth1_candidates,
                    replace_all=True,  # Will replace after trees are initialized
                    optimize_params=True,
                    max_depth1_trees=10
                )
            
            # Initialize trees list if not already done
            if not hasattr(self, 'trees') or self.trees is None:
                self.trees = [None] * self.size
                self.root_posteriors = np.full((self.size,), np.nan, dtype=np.float64)
        else:
            self.trees = self._init_trees(node_list, Npars, init_value)
        
    cdef list _init_trees(self, list node_list, int Npars, object init_value):
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

    cpdef int best_index(self):
        return int(np.nanargmax(self.root_posteriors))

    cpdef function_treeC.Tree best_tree(self):
        return self.trees[self.best_index()]

    cdef tuple _choose_move(self) :
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

    cdef function_treeC.Tree _propose(self, int i, int max_attempts=2) :
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

    cdef bint _step(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
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
            cand.update_parameters(self.domain, y, self.lambda_reg)
        
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

    cpdef double acceptance_rate(self):
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
                    domain=self.domain, 
                    Npars=0,  # will add params manually
                    value=1.0, 
                    T=self.T
                )
                
                # Build left child
                if left_str == "a0":
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.left.set_value(1.0)
                else:
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(left_str, self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Build right child
                if right_str == "a0":
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a1", self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.right.set_value(1.0)
                else:
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(right_str, self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Assign operation and update
                tree.root.assign_operation(op)
                tree.tree_update(tree.root)
                
                # Validate
                if not tree.is_valid(self.domain):
                    continue
                
                # Optimize parameters if any exist
                if optimize_params and len(tree.params) > 0:
                    tree.update_parameters(self.domain, y, self.lambda_reg)
                    
                
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
                         bint replace_all=True, bint optimize_params=True,
                         int max_depth1_trees=10):
        """
        Stage 2 burn-in: combine top depth-1 trees with primitives and each other.
        Call this AFTER burn_in_combinations to build depth-2 expressions.
        
        Parameters
        ----------
        y : target data
        replace_all : if True, replace all grove trees with best depth-2 candidates
        optimize_params : if True, optimize parameters for each candidate
        max_depth1_trees : how many top depth-1 trees to combine (avoid explosion)
        """
        # First, run stage 1 to get depth-1 candidates
        depth1_candidates = self._get_depth1_candidates(y, optimize_params)
        
        if len(depth1_candidates) == 0:
            print("Warning: No depth-1 candidates; cannot build depth-2")
            return
        
        # Now combine them
        self._burn_in_stage2(y, depth1_candidates, replace_all, optimize_params, max_depth1_trees)
    
    cdef list _get_depth1_candidates(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                      bint optimize_params=True):
        """
        Build all depth-1 trees (primitive op primitive) and return sorted list.
        Reuses logic from _burn_in_combinations but returns candidates instead of replacing.
        """
        cdef list candidates = []
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        
        print(f"Stage 1: building {len(all_primitives)**2 * len(self.operations)} depth-1 trees")
        # Initialize candidates with primitive functions (depth-0 trees)
        for func_str in all_primitives:
            tree = function_treeC.Tree(
                depth=0, functions=self.funcs, domain=self.domain,
                Npars=0, value=1.0, T=self.T
            )
            if func_str == "a0":
                tree.root = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string("a0", self.domain),
                    alpha=tree.alpha, sigma=tree.sigma
                )
                tree.root.set_value(1.0)
            else:
                tree.root = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(func_str, self.domain),
                    alpha=tree.alpha, sigma=tree.sigma
                )
            tree.tree_update(tree.root)
            if tree.is_valid(self.domain):
                if optimize_params and len(tree.params) > 0:
                    tree.update_parameters(self.domain, y, self.lambda_reg)
                post = tree.get_superficial_posterior(y)
                if np.isfinite(post):
                    candidates.append((tree, post))

        cdef str left_str, right_str, op
        cdef function_treeC.Tree tree
        cdef double post
        
        for left_str, right_str in itertools.product(all_primitives, repeat=2):
            if left_str=='a0' and right_str=='a0':
                continue
            for op in self.operations:
                tree = function_treeC.Tree(
                    depth=1, functions=self.funcs, x=self.domain,
                    Npars=0, value=1.0, T=self.T
                )
                
                # Build left child
                if left_str == "a0":
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a0", self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.left.set_value(1.0)
                else:
                    tree.root.left = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(left_str, self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                # Build right child
                if right_str == "a0":
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string("a1", self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                    tree.root.right.set_value(1.0)
                else:
                    tree.root.right = function_treeC.Node(
                        function=real_functionC.RealFunctionC.from_string(right_str, self.domain),
                        alpha=tree.alpha, sigma=tree.sigma
                    )
                
                tree.root.assign_operation(op)
                tree.tree_update(tree.root)
                
                if not tree.is_valid(self.domain):
                    continue
                
                if optimize_params and len(tree.params) > 0:
                    tree.update_parameters(self.domain, y, self.lambda_reg)
                
                post = tree.get_superficial_posterior(y)
                if np.isfinite(post):
                    candidates.append((tree, post))
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 1: {len(candidates)} valid depth-1 candidates")
        if len(candidates) > 0:
            print(f"  Top-3: {[(c[0].tree_get_description(), f'{c[1]:.3f}') for c in candidates[:3]]}")
        
        return candidates

    cdef _burn_in_stage2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                          list depth1_candidates,
                          bint replace_all=True,
                          bint optimize_params=True,
                          int max_depth1_trees=10):
        """
        Combine top depth-1 trees with primitives and with each other to build depth-2 trees.
        """
        cdef list candidates = []
        cdef list all_primitives = self.funcs.copy()
        all_primitives.append("a0")
        
        # Keep top N to avoid combinatorial explosion
        cdef int n_top = min(max_depth1_trees, len(depth1_candidates))
        cdef list top_depth1 = [c[0] for c in depth1_candidates[:n_top]]
        
        print(f"Stage 2: combining {n_top} depth-1 trees with {len(all_primitives)} primitives + themselves")
        print(f"  Estimated combinations: ~{n_top * (len(all_primitives) + n_top) * len(self.operations)}")
        
        cdef str prim_str, op
        cdef function_treeC.Tree tree, d1_tree
        cdef double post
        cdef int i, j, idx
        
        # 1) depth-1 op primitive (both directions)
        '''
        for i in range(n_top):
            for prim_str in all_primitives:
                for op in self.operations:
                    # left = depth-1, right = primitive
                    tree = self._combine_depth1_prim(top_depth1[i], prim_str, op, left_is_tree=True)
                    if tree is not None and tree.is_valid(self.domain):
                        if optimize_params and len(tree.params) > 0:
                            tree.update_parameters(self.domain, y, self.lambda_reg)
                        post = tree.get_superficial_posterior(y)
                        if np.isfinite(post):
                            candidates.append((tree, post))
                    
                    # left = primitive, right = depth-1
                    tree = self._combine_depth1_prim(prim_str, top_depth1[i], op, left_is_tree=False)
                    if tree is not None and tree.is_valid(self.domain):
                        if optimize_params and len(tree.params) > 0:
                            tree.update_parameters(self.domain, y, self.lambda_reg)
                        post = tree.get_superficial_posterior(y)
                        if np.isfinite(post):
                            candidates.append((tree, post))
        '''
        # 2) depth-1 op depth-1 (all pairs)
        for i in range(n_top):
            for j in range(n_top):
                for op in self.operations:
                    tree = self._combine_two_depth1(top_depth1[i], top_depth1[j], op)
                    if tree is not None and tree.is_valid(self.domain):
                        if optimize_params and len(tree.params) > 0:
                            tree.update_parameters(self.domain, y, self.lambda_reg)
                        post = tree.get_superficial_posterior(y)
                        if np.isfinite(post):
                            candidates.append((tree, post))
        
        if len(candidates) == 0:
            print("Warning: No valid depth-2 candidates")
            return
        
        candidates.sort(key=lambda x: x[1], reverse=True)
        print(f"Stage 2: {len(candidates)} valid depth-2 candidates")
        print(f"  Top-3: {[(c[0].tree_get_description(), f'{c[1]:.3f}') for c in candidates[:3]]}")
        
        # Replace grove trees
        if replace_all:
            if self.trees is None:
                self.trees = [None] * len(candidates)
                self.root_posteriors = np.full((len(candidates),), np.nan, dtype=np.float64)
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
    

    cdef function_treeC.Tree _combine_depth1_prim(self, object left, object right, object op,
                                                   bint left_is_tree=False):
        """
        Combine depth-1 tree with a primitive (or vice versa).
        left/right can be Tree or str (primitive name).
        """
        
        cdef function_treeC.Tree tree = function_treeC.Tree(depth=2, functions=self.funcs, domain=self.domain,
                                    Npars=0, value=1.0, T=self.T)
        cdef str next_par

        # Left side
        if left_is_tree:
            tree.root.left = deepcopy(left.root)
        else:
            # left is primitive string
            if left == "a0":
                tree.root.left = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string("a0", self.domain),
                    alpha=tree.alpha, sigma=tree.sigma
                )
                tree.root.left.set_value(1.0)
            else:
                tree.root.left = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(<str>left, self.domain),
                    alpha=tree.alpha, sigma=tree.sigma
                )
        
        # Right side
        if not left_is_tree:  # right is tree
            tree.root.right = deepcopy(right.root)
        else:  # right is primitive string
            if right == "a0":
                # Use next available parameter name
                next_par = f"a{len([p for p in tree.get_params()])}" #THIS HOULD BE OK BUT DOBULE CHECK
                tree.root.right = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(next_par, self.domain),
                    alpha=tree.alpha, sigma=tree.sigma
                )
                tree.root.right.set_value(1.0)
            else:
                tree.root.right = function_treeC.Node(
                    function=real_functionC.RealFunctionC.from_string(<str>right, self.domain),
                    alpha=tree.alpha, sigma=tree.sigma
                )
        
        tree.root.assign_operation(op)
        tree.tree_update(tree.root)
        return tree
        

    cdef function_treeC.Tree _combine_two_depth1(self, function_treeC.Tree left_tree,
                                                  function_treeC.Tree right_tree, object op):
        """
        Combine two depth-1 trees: left_tree op right_tree.
        """
        cdef function_treeC.Tree tree
        tree = function_treeC.Tree(depth=2, functions=self.funcs, domain=self.domain,
                                    Npars=0, value=1.0, T=self.T)
        tree.root.left = deepcopy(left_tree.root)
        tree.root.right = deepcopy(right_tree.root)
        tree.root.assign_operation(op)
        tree.tree_update(tree.root)
        return tree
        



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
                 double base_temp=1.0):
        self.temperatures = np.asarray(temperatures, dtype=np.float64)
        self.groves = []
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
