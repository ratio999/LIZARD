cimport numpy as cnp
cimport real_functionC
cimport function_treeC

cpdef subtree_mutation(function_treeC.Tree tree, functions, random_depth=?)
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
cdef rename_prefix(function_treeC.Tree tree, str new_prefix=?)
cdef refresh_params(function_treeC.Tree tree)