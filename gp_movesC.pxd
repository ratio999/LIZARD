cimport numpy as cnp
cimport real_functionC
cimport function_treeC



# Per-Grove RNG handles.
cpdef void set_rng(object py_rng, object np_rng) except *
cpdef tuple get_rng()

cpdef void set_max_tree_depth(int d)

cpdef int get_max_tree_depth()

cpdef void set_subtree_rebuild_max_height(int h)

cpdef int get_subtree_rebuild_max_height()

cdef list _flatten_nodes(function_treeC.Node node, list out)
cpdef list flatten_nodes(function_treeC.Node node, list out=?)
cdef function_treeC.Node _select_uniform_node(function_treeC.Tree tree)
cdef bint _is_perfect_binary(function_treeC.Node node, int h)
cdef list _collect_parameter_leaves(function_treeC.Node node, list out)
cdef bint _contains_atomic(function_treeC.Node node)
cdef function_treeC.Node _make_atom_leaf(object spec_obj, object domain,
                                         list name_pool, double g_tau)
cdef int _count_selectable(function_treeC.Node node)
cdef tuple _subtree_mutation(function_treeC.Tree tree, list functions,
                             list operations)
cpdef tuple subtree_mutation(function_treeC.Tree tree , list functions=?,
                             list operations=?)
cpdef legacy_subtree_mutation(function_treeC.Tree tree, functions, random_depth=?)
cpdef legacy_subtree_crossover(function_treeC.Tree parent1, function_treeC.Tree parent2)
cdef tuple _subtree_crossover(function_treeC.Tree parent1, function_treeC.Tree parent2)
cpdef tuple subtree_crossover(function_treeC.Tree parent1, function_treeC.Tree parent2)
cdef tuple _point_mutation(function_treeC.Tree tree, list functions, list operations,
                           list atoms=?)
cpdef tuple point_mutation(function_treeC.Tree tree, list functions=?, list operations=?, function_treeC.Node node_to_mutate=?,
                           list atoms=?)
cpdef legacy_point_mutation(function_treeC.Tree tree, functions,operations=?, function_treeC.Node node_to_mutate=?)
cpdef legacy_hoist_mutation(function_treeC.Tree tree, function_treeC.Node node_to_lift=?)
cdef tuple _hoist_mutation(function_treeC.Tree tree, list functions, list operations,
                           list atoms=?)
cpdef tuple hoist_mutation(function_treeC.Tree tree, list functions=?, list operations=?,
                           list atoms=?)
cdef tuple _leaf_hoist(function_treeC.Tree tree, list functions, list operations,
                       list atoms=?)
cpdef tuple leaf_hoist(function_treeC.Tree tree, list functions=?, list operations=?,
                       list atoms=?)
cdef tuple _internal_hoist(function_treeC.Tree tree, list functions, list operations,
                           list atoms=?)
cpdef tuple internal_hoist(function_treeC.Tree tree, list functions=?, list operations=?,
                           list atoms=?)
#cpdef size_fair_mutation(tree, functions, x)
#cpdef replace_with_subtree(tree, functions, x)
#cpdef add_node(function_treeC.Tree tree, functions, operations)
cpdef legacy_add_node(function_treeC.Tree tree, functions, operations=?, random_depth=?)
cdef tuple _sample_truncated_geometric(double rho, int d_max)
cdef double _truncated_geometric_log_pmf(int d, double rho, int d_max)
cdef int _count_nodes(function_treeC.Node node)
cdef double _eval_build_density(function_treeC.Node S, int d,
                                int M, int O, double p_par, double g_tau)
cdef function_treeC.Node _build_subtree_with_params(
        int d, list functions, list operations, object domain,
        double p_par, list existing_param_names, double g_tau,
        list atoms=?)
cdef tuple _add_node(function_treeC.Tree tree, list functions, list operations,
                     list atoms=?)
cpdef tuple add_node(function_treeC.Tree tree, list functions=?, list operations=?,
                     random_depth=?, list atoms=?)
cdef tuple _split(function_treeC.Tree tree, list functions, list operations,
                  list atoms=?)
cpdef tuple split(function_treeC.Tree tree, list functions=?, list operations=?,
                  list atoms=?)
cdef list _collect_ets(function_treeC.Node node, list out)
cdef tuple _draw_leaf_et(list functions, object domain, list name_pool, double g_tau,
                         list atoms=?)
cdef tuple _build_random_et(int arity, list functions, list operations,
                            object domain, list name_pool, double g_tau,
                            list atoms=?)
cdef tuple _etr_mutation(function_treeC.Tree tree, list functions, list operations,
                         list atoms=?)
cpdef tuple etr_mutation(function_treeC.Tree tree, list functions=?, list operations=?,
                         list atoms=?)
#cpdef select_random_node(node)
#cpdef find_parent(root, child)
#cpdef compute_subtree_size(node)
#cpdef generate_subtree_with_size(size, functions, x)
cpdef root_replacement(function_treeC.Tree tree)
#cpdef prune_node(function_treeC.Tree tree, function_treeC.Node node_to_prune=?)
cpdef rename_prefix(function_treeC.Tree tree, str new_prefix=?)
cdef refresh_params(function_treeC.Tree tree)
cdef _compact_param_labels(function_treeC.Tree tree, str prefix=?)
cpdef rename_prefix_downwards(function_treeC.Tree tree, str new_prefix = ?)

cdef refresh_params_downwards(function_treeC.Tree tree)


cdef class AMState:
    cdef public tuple param_names
    cdef public int K
    cdef public cnp.ndarray mu
    cdef public cnp.ndarray C
    cdef public double log_lambda
    cdef public int n_visits
    cdef public int n_accepts
    cdef public double alpha_star
    cdef public double kappa
    cdef public double rel
    cdef public double eps
    cdef public double jitter

    cdef double _gamma(self)
    cdef cnp.ndarray _propose(self, cnp.ndarray theta)
    cpdef cnp.ndarray propose(self, cnp.ndarray theta)
    cdef void _update(self, cnp.ndarray theta_after, bint accepted, double alpha_obs)
    cpdef void update(self, cnp.ndarray theta_after, bint accepted, double alpha_obs)
    
cpdef list pr_coord_names(function_treeC.Tree tree)

'''sigma is object-typed here: a scalar keeps the historical path, while
a per-point float64 array routes through
Tree._log_likelihood_at_sigma_arr. When this was a `double` an array
raised TypeError at the coercion.'''
cdef tuple _parameter_refresh(function_treeC.Tree tree,
                              cnp.ndarray[cnp.float64_t, ndim=1] y,
                              object sigma,
                              AMState am_state,
                              double T)

cpdef tuple parameter_refresh(function_treeC.Tree tree,
                              cnp.ndarray[cnp.float64_t, ndim=1] y,
                              object sigma,
                              AMState am_state,
                              double T)
# Transparent-atom helpers.
cdef bint _contains_flex(function_treeC.Node node)
cdef _rename_flex_internals(function_treeC.Node fnode, dict amap)
cdef tuple _flex_slot_mutation(function_treeC.Tree tree, list functions,
                               list operations)
cpdef tuple flex_slot_mutation(function_treeC.Tree tree, list functions=?,
                               list operations=?)
