import random
cimport cython
from libc.stdlib cimport malloc, free
cimport function_treeC
cimport real_functionC
cimport numpy as cnp


cdef assign_operation(function_treeC.Node left, function_treeC.Node right, str op_symbol):
    cdef real_functionC.RealFunctionC func 
    if op_symbol == '+':
        func = left.function + right.function    
    elif op_symbol == '-':
        func = left.function - right.function
    elif op_symbol == '*':
        func = left.function * right.function
    elif op_symbol == '/':
        func = left.function / right.function
    elif op_symbol == '**':
        func = left.function ** right.function  # Operazione di potenza
    elif op_symbol == '∘':
        func = left.function.compose(right.function)
    #print(func)
    return function_treeC.Node(function=func, left=left, right=right, symbol=op_symbol)

@cython.boundscheck(False)  # Per migliorare le prestazioni
@cython.wraparound(False)
def subtree_mutation( function_treeC.Tree tree,list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Sostituisce un sottoalbero con un nuovo sottoalbero generato casualmente."""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef int max_depth = node_to_replace.depth()
    cdef function_treeC.Node parent_node = tree.root.find_parent(node_to_replace)
    cdef str parent_op = parent_node.op_symbol
    
    cdef function_treeC.Node new_subtree = tree.build_tree(max_depth, functions, x)
    #print('op: ', node_to_replace.function.get_description(), node_to_replace.depth(), node_to_replace.is_leaf())

    if parent_node is None:
        # Se non c'è genitore, significa che stai sostituendo la radice
        tree.root = new_subtree
    elif parent_node.left == node_to_replace:
        parent_node.left = new_subtree
        
    elif parent_node.right == node_to_replace:
        parent_node.right = new_subtree
        
    #print('op_new: ', new_subtree.function.get_description(),new_subtree.depth(), new_subtree.is_leaf())
    
    tree.tree_update(parent_node)
    #print('nuova descr: ', tree.tree_get_description())

@cython.boundscheck(False)
@cython.wraparound(False)
def subtree_crossover( function_treeC.Tree parent1, function_treeC.Tree parent2):
    """Esegue il crossover tra due alberi sostituendo un sottoalbero di parent1 con un sottoalbero di parent2."""
    cdef function_treeC.Node node1 = parent1.root.select_random_node()
    cdef function_treeC.Node node2 = parent2.root.select_random_node()
    cdef function_treeC.Node parent_node = parent1.root.find_parent(node1)
    #print("Prima del crossover:", node1.function.get_description(), node2.function.get_description())

    # Sostituisci il sottoalbero in parent1 con node2
    if parent_node is None:
        # Se node1 è la radice, sostituisci direttamente la radice
        parent1.root = node2
    elif parent_node.left == node1:
        parent_node.left = node2
    elif parent_node.right == node1:
        parent_node.right = node2

    # Aggiorna la struttura dell'albero
    parent1.tree_update(parent_node)

    
    

@cython.boundscheck(False)
@cython.wraparound(False)
def point_mutation(function_treeC.Tree tree, list functions, list operations, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Modifica un singolo nodo cambiando la funzione o l'operazione."""
    cdef function_treeC.Node node_to_mutate = tree.root.select_random_node()
    cdef str new_func_str
    cdef function_treeC.Node parent = tree.root.find_parent(node_to_mutate)
    if node_to_mutate.is_leaf():
        new_func_str= random.choice(functions)
        node_to_mutate.function = real_functionC.RealFunctionC.from_string(new_func_str, x)
    else:
        node_to_mutate.op_symbol = random.choice(operations)
        node_to_mutate.update_func()

    if parent is None:
        # Se node1 è la radice, sostituisci direttamente la radice
        tree.root = node_to_mutate
    elif parent.left == node_to_mutate:
        parent.left = node_to_mutate
    elif parent.right == node_to_mutate:
        parent.right = node_to_mutate

    # Aggiorna la struttura dell'albero
    tree.tree_update(parent)
    
@cython.boundscheck(False)
@cython.wraparound(False)
def hoist_mutation(function_treeC.Tree tree):
    """Sostituisce un sottoalbero con uno dei suoi sottoalberi."""
    cdef function_treeC.Node subtree = tree.root.select_random_node()
    if not subtree.is_leaf():
        parent_node = tree.root.find_parent(subtree)
        if random.choice([True, False]):
            parent_node.left = subtree.left
        else:
            parent_node.right = subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def size_fair_mutation(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Effettua una mutazione preservando la dimensione dell'albero, sostituendo una parte con un'altra di dimensioni simili."""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef int subtree_size = node_to_replace.compute_subtree_size()
    cdef function_treeC.Node new_subtree = generate_subtree_with_size(subtree_size, functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_with_subtree(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Sostituisce un nodo con un sottoalbero casuale."""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef function_treeC.Node new_subtree = tree.build_tree(random.randint(1, 3), functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def add_node(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Aggiunge un nodo a un albero esistente."""
    cdef function_treeC.Node node_to_expand = tree.root.select_random_node()
    new_func_str = random.choice(functions)
    cdef function_treeC.Node new_node = function_treeC.Node(real_functionC.RealFunctionC.from_string(new_func_str, x))
    if node_to_expand.is_leaf():
        if random.choice([True, False]):
            node_to_expand.left = new_node
        else:
            node_to_expand.right = new_node
    else:
        # Se il nodo ha già dei figli, scegli uno dei figli da sostituire o aggiungere
        if node_to_expand.left is None:
            node_to_expand.left = new_node
        elif node_to_expand.right is None:
            node_to_expand.right = new_node
        else:
            # Sostituisci casualmente uno dei due figli
            if random.choice([True, False]):
                node_to_expand.left = new_node
            else:
                node_to_expand.right = new_node
    
# Funzioni di utilità per operare sui nodi



def generate_subtree_with_size(int size, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Genera un nuovo sottoalbero di una dimensione specifica."""
    depth = size // 2  # Calcola una profondità approssimativa
    cdef function_treeC.Tree tree = function_treeC.Tree(depth, functions, x)
    return tree.root

#Se fai cdef su tutto forse puoi tornare a cpdef nel file .pxd
#add_node causa segmentation fault