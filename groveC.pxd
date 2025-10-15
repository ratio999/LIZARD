import numpy as np
cimport numpy as cnp
cimport function_treeC

cdef class Grove:
    cdef list operations
    cdef int size
    cdef int depth
    cdef list funcs
    cdef object x
    cdef list nodes
    cdef int n_pars
    cdef double T
    cdef object root_posteriors
    cdef public list trees
    cdef double lambda_reg
    cdef double base_temp
    cdef object domain
    cdef double acceptance_count
    cdef double proposal_count

    cdef list _init_trees(self, list node_list, int Npars,
                            object init_value)

    cpdef set_trees(self, list new_trees)

    cpdef set_single_tree(self, function_treeC.Tree tree, int i)

    #cdef grow_grove(self)

    cpdef int best_index(self)

    cpdef function_treeC.Tree best_tree(self)

    cdef tuple _choose_move(self)
    cpdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=*)
    
    cdef function_treeC.Tree _propose(self, int i, int max_attempts=?)

    cdef bint _step(elf, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef run(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps)

    cpdef burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=?, bint optimize_params=?)

    cdef _burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=?, bint optimize_params=?)

    cpdef burn_in_depth2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                         bint replace_all=?, bint optimize_params=?,
                         int max_depth1_trees=?)

    cdef list _get_depth1_candidates(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                      bint optimize_params=?)    

    cdef _burn_in_stage2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                          list depth1_candidates,
                          bint replace_all=?,
                          bint optimize_params=?,
                          int max_depth1_trees=?)

    cdef function_treeC.Tree _combine_depth1_prim(self, object left, object right, object op,
                                                   bint left_is_tree=?)

    cdef function_treeC.Tree _combine_two_depth1(self, function_treeC.Tree left_tree,
                                                  function_treeC.Tree right_tree, object op)

    cpdef double acceptance_rate(self)

cdef class Forest:
    cdef list groves
    cdef object temperatures
    cpdef run_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps_per_grove)
    cpdef compute_posteriors_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=*)
    cdef bint _attempt_swap_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef int attempt_swaps(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef function_treeC.Tree best_tree(self)