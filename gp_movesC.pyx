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
    cdef function_treeC.Node node_to_replace = select_random_node(tree.root)
    cdef int max_depth = node_to_replace.depth()
    cdef function_treeC.Node parent_node = find_parent(tree.root, node_to_replace)
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
    
    tree_update(tree, parent_node)
    #print('nuova descr: ', tree.tree_get_description())

cdef tree_update(function_treeC.Tree tree, function_treeC.Node node):
    current_node = node
    while current_node is not None:
        current_node.update_func()
        current_node = find_parent(tree.root, current_node)

    tree.update_tree_description()

@cython.boundscheck(False)
@cython.wraparound(False)
def subtree_crossover( function_treeC.Tree parent1, function_treeC.Tree parent2):
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
    cdef function_treeC.Node new_subtree = generate_subtree_with_size(subtree_size, functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_with_subtree(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Sostituisce un nodo con un sottoalbero casuale."""
    cdef function_treeC.Node node_to_replace = select_random_node(tree.root)
    cdef function_treeC.Node new_subtree = tree.build_tree(random.randint(1, 3), functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right

@cython.boundscheck(False)
@cython.wraparound(False)
def add_node(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Aggiunge un nodo a un albero esistente."""
    cdef function_treeC.Node node_to_expand = select_random_node(tree.root)
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
@cython.boundscheck(False)
@cython.wraparound(False)
cdef select_random_node(function_treeC.Node node, float threshold=0.1):
    """
    Seleziona un nodo casuale in un albero, con probabilità influenzata dalla sua fitness.
    Nodi con fitness bassa hanno una maggiore probabilità di essere selezionati.
    """
    #cdef float fitness = compute_fitness(node)
    cdef float fitness = 0.8
    # Più è bassa la fitness, più è alta la probabilità di selezionare il nodo
    selection_prob = max(1.0 - fitness, threshold)
    
    if random.random() < selection_prob:
        return node
    
    if node.is_leaf():
        return node
    
    # Se il nodo non viene selezionato, passa casualmente ai figli
    if random.random() < 0.5:  # 50% di probabilità di andare verso il nodo sinistro
        return select_random_node(node.left, threshold)
    else:  # 50% di probabilità di andare verso il nodo destro
        return select_random_node(node.right, threshold)

@cython.boundscheck(False)
@cython.wraparound(False)
cdef find_parent(function_treeC.Node root, function_treeC.Node child):
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
cdef compute_subtree_size(function_treeC.Node node):
    """Calcola la dimensione di un sottoalbero."""
    if node is None:
        return 0
    return 1 + compute_subtree_size(node.left) + compute_subtree_size(node.right)

@cython.boundscheck(False)
@cython.wraparound(False)
def generate_subtree_with_size(int size, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Genera un nuovo sottoalbero di una dimensione specifica."""
    depth = size // 2  # Calcola una profondità approssimativa
    cdef function_treeC.Tree tree = function_treeC.Tree(depth, functions, x)
    return tree.root

#Se fai cdef su tutto forse puoi tornare a cpdef nel file .pxd
#add_node causa segmentation fault