# cython: language_level=3
import random
import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport gp_movesC

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
                 double T,  # accepts scalar or array; use T_arr[0]
                 int size,
                 int depth,
                 list funcs,
                 object node_list=None,
                 int Npars=0,
                 object init_value=1.0,
                 list operations=None,
                 double base_temp=1.0):
        if operations is None:
            operations = ["+", "-", "*", "/", "**", "o"]
        self.operations = operations
        self.funcs = funcs
        self.domain = np.asarray(x, dtype=np.float64)
        self.depth = depth
        self.size = size
        self.T = float(T)
        self.base_temp = base_temp if base_temp > 0.0 else 1.0
        self.trees = self._init_trees(node_list, Npars, init_value)
        self.root_posteriors = np.full((self.size,), np.nan, dtype=np.float64)
        self.acceptance_count = 0.0
        self.proposal_count = 0.0

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
        probs = np.array([p_point, p_hoist, p_add, p_subtree], dtype=np.float64)
        probs = probs / probs.sum()
        idx = int(np.random.choice(np.arange(4), p=probs))
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
                    gp_movesC.subtree_crossover(cand, donor)
                else:
                    # fallback to a local move if only one tree
                    gp_movesC.point_mutation(cand, self.funcs, self.operations)
            else:
                # default safety: point mutation
                gp_movesC.point_mutation(cand, self.funcs, self.operations)
            
            if cand.is_valid(self.domain):
                print('is valid')
                return cand
            attempts += 1
            #print(attempts)
        print('is_not valid')
        return deepcopy(self.trees[i])

    cdef bint _step(self, cnp.ndarray[cnp.float64_t, ndim=1] y):
        """
        One MH step on a random tree in this grove.
        Returns True if accepted.
        """
        if self.size == 0:
            return False
        cdef int i = random.randrange(self.size)

        # ensure current posterior is available
        if np.isnan(self.root_posteriors[i]):
            self.root_posteriors[i] = self.trees[i].get_superficial_posterior(y)

        cdef double old_post = float(self.root_posteriors[i])

        cdef function_treeC.Tree cand = self._propose(i)
        cdef double new_post = float(cand.get_superficial_posterior(y))
        cdef double dlogpost = new_post - old_post

        cdef double acc_p = _accept_prob(dlogpost, self.T)
        self.proposal_count += 1.0
        if random.random() < acc_p:
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