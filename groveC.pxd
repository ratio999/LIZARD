import numpy as np
cimport numpy as cnp
cimport function_treeC

cdef class Grove:
    cdef list operations
    cdef cnp.float64_t[::1] data
    cdef int size
    cdef int depth
    cdef list funcs
    cdef object x
    cdef list nodes
    cdef int n_pars
    cdef double T
    cdef public object root_posteriors
    cdef public list trees
    cdef double lambda_reg
    cdef double base_temp
    cdef object domain
    cdef double acceptance_count
    cdef double proposal_count
    cdef bint burn_in
    cdef bint use_mpi
    cdef int n_procs
    cdef bint use_spawn
    cdef bint use_gradients
    cdef object method_str
    cdef dict cache
    # Optional Pólya tree prior (None => behaves as before).
    cdef public object polya
    cdef public dict move_proposals
    cdef public dict move_accepts
    cdef public dict _am_states
    # Noise scale for raw-likelihood scoring inside parameter_refresh.
    # Placeholder until Phase 2.7(a) wires sigma=None -> profiled sigma.
    cdef public double sigma
    # PR mixture weight (Phase 2.1.PR D3, default 0.02).
    cdef public double pr_prob

    cdef  _init_trees(self, list node_list, int Npars,
                            object init_value)

    cpdef set_trees(self, list new_trees)

    cpdef set_single_tree(self, function_treeC.Tree tree, int i)

    #cdef grow_grove(self)

    cpdef  best_index(self)

    cpdef  best_tree(self)

    cdef  _choose_move(self)
    cpdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=*)
    
    cdef  tuple _propose(self, int i, int max_attempts=?)

    cdef  legacy_step(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cdef _step(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef run(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps)
    cpdef dict move_stats(self)
    cdef _step_parameter_refresh(self, cnp.ndarray[cnp.float64_t, ndim=1] y)

    cpdef burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=?, bint optimize_params=?)

    cdef _burn_in_combinations(self, cnp.ndarray[cnp.float64_t, ndim=1] y, 
                                bint replace_all=?, bint optimize_params=?)

    cpdef burn_in_depth2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                         bint replace_all=?, bint optimize_params=?)

    cdef  _get_depth1_candidates(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict tree_cache,
                                      bint optimize_params=?)    

    cdef _burn_in_stage2(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                          list up2depth1_candidates,
                          dict tree_cache,
                          bint replace_all=?,
                          bint optimize_params=?)
    cpdef _burn_in_stage2_parallel(self, cnp.ndarray[cnp.float64_t, ndim=1] y, list up2depth1_candidates,
                            dict tree_cache, bint replace_all=?, bint optimize_params=?)
    
    cdef _burn_in_stage2_parallel_spawn(self, 
                                    cnp.ndarray[cnp.float64_t, ndim=1] y,
                                    list up2depth1_candidates,
                                    dict tree_cache,
                                    bint replace_all=?,
                                    bint optimize_params=?,
                                    int n_workers=?)

    cdef _process_candidate(self, function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                    bint optimize_params, dict local_cache, list local_candidates)
    
    cdef  _combine_depth1(self, function_treeC.Node left, function_treeC.Node right, object op)
    cdef  _combine_depth1_trees(self, function_treeC.Tree left, function_treeC.Tree right, object op)

    cpdef  acceptance_rate(self)

cdef class Forest:
    cdef list groves
    cdef object temperatures
    cdef int n_procs
    cdef bint use_mpi
    cpdef run_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps_per_grove)
    cpdef compute_posteriors_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=*)
    cdef bint _attempt_swap_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef int attempt_swaps(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef function_treeC.Tree best_tree(self)

cdef _bis2_parallel_helper(object x)
