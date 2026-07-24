import numpy as np
cimport numpy as cnp
cimport function_treeC
cimport polya_treeC

cdef class Grove:
    cdef list operations
    # Phase 2.10 Stage 3a: opaque-primitive templates (list of
    # real_functionC.AtomSpec, or None). Passed UNIFORMLY to every
    # atom-aware kernel move in _propose — mixed wiring would break the
    # birth/death density pairing exactly like an operations mismatch.
    cdef public list atoms
    cdef object data
    cdef public int size
    cdef int depth
    cdef public list funcs
    cdef object x
    cdef list nodes
    cdef int n_pars
    cdef double T
    cdef public object sigma
    cdef public double g_tau
    cdef public object root_posteriors
    cdef public list trees
    cdef double lambda_reg
    cdef object _py_rng
    cdef object _np_rng
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
    cdef public polya_treeC.PolyaTree polya
    cdef public dict move_proposals
    cdef public dict move_accepts
    cdef public dict _am_states
    # Noise scale for raw-likelihood scoring inside parameter_refresh.
    # Placeholder until Phase 2.7(a) wires sigma=None -> profiled sigma.
    #cdef public double sigma
    # PR mixture weight (Phase 2.1.PR D3, default 0.02).
    cdef public double pr_prob
    # Phase 2.7(f): per-topology BIC cache.
    # Maps tree_get_description() -> (log_evidence, dict(name_values)).
    # Invalidated whenever the y reference changes between _step calls.
    cdef public dict _bic_cache
    cdef public object _cache_y
    # Phase 4.4 (structural-constraint prior tilt, 2026-07-19): opt-in
    # user callable tree -> log-prior increment (e.g. -lambda*C(tree)
    # for an asymptotic-behaviour constraint). A PRIOR term: enters the
    # acceptance sums next to the Polya log-prior, never mse/BIC/
    # root_posteriors. None (default) = bit-identical legacy.
    cdef public object structural_prior
    cdef public dict _sp_cache          # description -> tilt (y-independent)
    cdef public bint structural_prior_cache
    # v2 (2026-07-22): the declarative ('module', 'builder', kwargs)
    # spec the callable was resolved from, or None. Picklable — it is
    # what run_mpi_spawn ships so workers rebuild the tilt locally.
    cdef public object structural_prior_spec
    # Phase 2.8: optional adaptive move-weight calibration (OFF by default).
    # _move_logits holds the 6 logical-group logits; _adapt_t counts
    # adaptive updates for the diminishing step size gamma_t.
    cdef public bint adapt_moves
    cdef public double adapt_gamma0
    cdef public double adapt_t0
    cdef public double adapt_floor
    cdef public double adapt_logit_bound
    cdef public double adapt_target
    cdef public cnp.ndarray _move_logits
    cdef public long _adapt_t
    # Phase 2.12 (MPI sampling): defer Polya reinforcement for replay-sync.
    cdef public bint polya_defer
    cdef public list _deferred_paths

    cdef  _init_trees(self, list node_list, int Npars,
                            object init_value)
    cpdef void set_sigmas(self, object new_sigma)

    cpdef set_trees(self, list new_trees)

    cpdef set_single_tree(self, function_treeC.Tree tree, int i)

    #cdef grow_grove(self)

    cpdef  best_index(self)

    cpdef  best_tree(self)

    cdef  _choose_move(self)
    cdef cnp.ndarray _compute_move_probs(self)
    cpdef cnp.ndarray get_move_probs(self)
    cdef cnp.ndarray _adapted_move_probs(self)
    cdef int _move_group(self, str move_name)
    cdef void _adapt_move(self, str move_name, bint accepted)
    cpdef compute_posteriors(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=*)
    
    cdef  tuple _propose(self, int i, int max_attempts=?)

    cdef  legacy_step(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cdef _step(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    # Phase 2.13(b1): joint accept/install for the within-grove crossover pair.
    cdef bint _step_pair_crossover(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                                   int i, int j,
                                   function_treeC.Tree candA,
                                   function_treeC.Tree candB,
                                   double lqf, double lqr)
    cdef double _score_post(self, function_treeC.Tree cand,
                            cnp.ndarray[cnp.float64_t, ndim=1] y,
                            bint optimize=*)
    # Phase 4.4: structural-prior tilt evaluator (0.0 when feature off).
    cdef double _sp(self, function_treeC.Tree tree)
    cpdef run(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps)
    cpdef dict move_stats(self)
    cpdef list drain_deferred_paths(self)
    cdef _step_parameter_refresh(self, cnp.ndarray[cnp.float64_t, ndim=1] y)

    # Phase 2.10 (final): burn-in token list + AtomSpec registry.
    cdef tuple _burn_in_primitives(self)

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
    cdef public list groves
    cdef public object temperatures
    cdef int n_procs
    cdef bint use_mpi
    # Phase 2.10 Stage 3a: shared opaque-primitive templates, forwarded to
    # every Grove; their symbols are pre-registered in the shared Polya at
    # construction (registration invariance, THEORY 5.3).
    cdef public list atoms
    cdef public polya_treeC.PolyaTree polya
    cdef object _py_rng
    cdef object _np_rng
    # Phase 2.12 (MPI sampling): Grove ctor args for spawned workers.
    cdef dict _ctor
    # Phase 4.1 (PT ladder tuning): per-consecutive-pair monitors (pure
    # observability; read via ladder_stats()).
    cdef public cnp.ndarray _swap_attempts
    cdef public cnp.ndarray _swap_accepts
    cdef public cnp.ndarray _xover_attempts
    cdef public cnp.ndarray _xover_accepts
    cpdef dict ladder_stats(self)
    cpdef run_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, int n_steps_per_grove)
    cpdef compute_posteriors_all(self, cnp.ndarray[cnp.float64_t, ndim=1] y, bint full=*)
    cdef bint _attempt_swap_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef int attempt_swaps(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cdef bint _attempt_crossover_pair(self, int g_left, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef int attempt_crossovers(self, cnp.ndarray[cnp.float64_t, ndim=1] y)
    cpdef dict run_mpi_spawn(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                             int n_rounds, int n_steps_per_round,
                             object seed=*,
                             object max_tree_depth=*,
                             object subtree_rebuild_max_height=*)
    cpdef function_treeC.Tree best_tree(self)

# Phase 2.10 (final): burn-in primitive-token resolver (mpi_worker uses it
# for its 'd1_prim' work items). POSITIONAL calls only.
cpdef function_treeC.Node _burn_in_primitive_node(str prim_str, object domain,
                                                  dict atom_specs,
                                                  str prefix, int name_start)

# Phase 2.10 Stage 3b: cpdef (mpi_worker delegates here; tests drive them).
# Cross-module callers: POSITIONAL ARGS ONLY.
cpdef tuple _tree_to_wire(function_treeC.Tree tree, double post,
                          bint force_skel=*)
cpdef function_treeC.Tree _tree_from_wire(object expr, dict name_values,
                                          object domain, list funcs,
                                          object sigma, double g_tau,
                                          object atom_specs=?)

cdef _bis2_parallel_helper(object x)
