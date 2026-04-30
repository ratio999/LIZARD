cimport numpy as cnp
cimport real_functionC
cimport function_treeC

cdef list _flatten_nodes(function_treeC.Node node, list out)
cdef function_treeC.Node _select_uniform_node(function_treeC.Tree tree)
cdef bint _is_perfect_binary(function_treeC.Node node, int h)
cdef list _collect_parameter_leaves(function_treeC.Node node, list out)
cdef int _count_selectable(function_treeC.Node node)
cdef tuple _subtree_mutation(function_treeC.Tree tree, list functions)
cpdef tuple subtree_mutation(function_treeC.Tree tree , list functions)
cpdef legacy_subtree_mutation(function_treeC.Tree tree, functions, random_depth=?)
cpdef subtree_crossover(function_treeC.Tree parent1, function_treeC.Tree parent2)
cpdef point_mutation(function_treeC.Tree tree, functions,operations=?, function_treeC.Node node_to_mutate=?)
cpdef hoist_mutation(function_treeC.Tree tree, function_treeC.Node node_to_lift=?)
#cpdef size_fair_mutation(tree, functions, x)
#cpdef replace_with_subtree(tree, functions, x)
#cpdef add_node(function_treeC.Tree tree, functions, operations)
cpdef add_node(function_treeC.Tree tree, functions, operations=?, random_depth=?)
#cpdef select_random_node(node)
#cpdef find_parent(root, child)
#cpdef compute_subtree_size(node)
#cpdef generate_subtree_with_size(size, functions, x)
cpdef root_replacement(function_treeC.Tree tree)
cpdef prune_node(function_treeC.Tree tree, function_treeC.Node node_to_prune=?)
cpdef rename_prefix(function_treeC.Tree tree, str new_prefix=?)
cdef refresh_params(function_treeC.Tree tree)

cpdef rename_prefix_downwards(function_treeC.Tree tree, str new_prefix = ?)

cdef refresh_params_downwards(function_treeC.Tree tree)