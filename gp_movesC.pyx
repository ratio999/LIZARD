import random
cimport cython
from libc.stdlib cimport malloc, free
cimport function_treeC
cimport real_functionC
cimport numpy as cnp

@cython.boundscheck(False)  # Per migliorare le prestazioni
@cython.wraparound(False)
def subtree_mutation(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Sostituisce un sottoalbero con un nuovo sottoalbero generato casualmente."""
    cdef function_treeC.Node node_to_replace = select_random_node(tree.root)
    cdef int max_depth = random.randint(1, 3)
    node_to_replace.left = tree.build_tree(max_depth, functions, x).left
    node_to_replace.right = tree.build_tree(max_depth, functions, x).right

@cython.boundscheck(False)
@cython.wraparound(False)
def subtree_crossover(function_treeC.Tree parent1, function_treeC.Tree parent2):
    """Esegue il crossover tra due alberi sostituendo un sottoalbero di parent1 con un sottoalbero di parent2."""
    cdef function_treeC.Node node1 = select_random_node(parent1.root)
    cdef function_treeC.Node node2 = select_random_node(parent2.root)
    node1.left = node2.left
    node1.right = node2.right

@cython.boundscheck(False)
@cython.wraparound(False)
def point_mutation(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Modifica un singolo nodo cambiando la funzione o l'operazione."""
    cdef function_treeC.Node node_to_mutate = select_random_node(tree.root)
    cdef str new_func_str = random.choice(functions)
    node_to_mutate.function = real_functionC.RealFunctionC.from_string(new_func_str, x)

@cython.boundscheck(False)
@cython.wraparound(False)
def hoist_mutation(function_treeC.Tree tree):
    """Sostituisce un sottoalbero con uno dei suoi sottoalberi."""
    cdef function_treeC.Node subtree = select_random_node(tree.root)
    if not subtree.is_leaf():
        parent_node = find_parent(tree.root, subtree)
        if random.choice([True, False]):
            parent_node.left = subtree.left
        else:
            parent_node.right = subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def size_fair_mutation(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Effettua una mutazione preservando la dimensione dell'albero, sostituendo una parte con un'altra di dimensioni simili."""
    cdef function_treeC.Node node_to_replace = select_random_node(tree.root)
    cdef int subtree_size = compute_subtree_size(node_to_replace)
    new_subtree = generate_subtree_with_size(subtree_size, functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_with_subtree(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Sostituisce un nodo con un sottoalbero casuale."""
    cdef function_treeC.Node node_to_replace = select_random_node(tree.root)
    new_subtree = tree.build_tree(random.randint(1, 3), functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def add_node(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Aggiunge un nodo a un albero esistente."""
    cdef function_treeC.Node node_to_expand = select_random_node(tree.root)
    new_func_str = random.choice(functions)
    new_node = function_treeC.Node(real_functionC.RealFunctionC.from_string(new_func_str, x))
    if random.choice([True, False]):
        node_to_expand.left = new_node
    else:
        node_to_expand.right = new_node

# Funzioni di utilità per operare sui nodi
@cython.boundscheck(False)
@cython.wraparound(False)
def select_random_node(function_treeC.Node node):
    """Seleziona un nodo casuale in un albero."""
    if node.is_leaf():
        return node
    if random.random() < 0.5:
        return select_random_node(node.left)
    else:
        return select_random_node(node.right)

@cython.boundscheck(False)
@cython.wraparound(False)
def find_parent(function_treeC.Node root, function_treeC.Node child):
    """Trova il nodo genitore di un dato nodo."""
    if root is None or root.is_leaf():
        return None
    if root.left == child or root.right == child:
        return root
    if root.left:
        parent = find_parent(root.left, child)
        if parent:
            return parent
    if root.right:
        return find_parent(root.right, child)
    return None

@cython.boundscheck(False)
@cython.wraparound(False)
def compute_subtree_size(function_treeC.Node node):
    """Calcola la dimensione di un sottoalbero."""
    if node is None:
        return 0
    return 1 + compute_subtree_size(node.left) + compute_subtree_size(node.right)

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_subtree_with_size(int size, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Genera un nuovo sottoalbero di una dimensione specifica."""
    depth = size // 2  # Calcola una profondità approssimativa
    tree = function_treeC.Tree(depth, functions, x)
    return tree.root
