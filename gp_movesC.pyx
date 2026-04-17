import random
import numpy as np
cimport cython
from libc.stdlib cimport malloc, free
cimport function_treeC
#from function_treeC import build_tree
cimport real_functionC
cimport numpy as cnp

@cython.boundscheck(True) 
cpdef subtree_mutation( function_treeC.Tree tree, functions, random_depth=False):
    """Substitutes a subtree with a random-generated new subtree"""
    if tree.root.is_leaf():
        #print('actually addnode')
        add_node(tree, functions, random_depth=random_depth)
        return
    parent_node = None
    #print('starting while')
    while parent_node is None:
        node_to_replace = tree.root.select_random_node()
        parent_node = tree.root.find_parent(node_to_replace)
    cdef str parent_op = parent_node.get_op()
    if not random_depth:
        max_depth = node_to_replace.depth()
    else:
        #max_depth = max(1, int(random.expovariate(1/2)))
        max_depth = random.choice(list(np.arange(1, node_to_replace.depth()+2)))
    cdef function_treeC.Node new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)

    if parent_node is None:
        tree.root = new_subtree
    elif parent_node.get_left() == node_to_replace:
        parent_node.assign_left(new_subtree)
        
    elif parent_node.get_right() == node_to_replace:
        parent_node.assign_right(new_subtree)
        
    #print('op_new: ', new_subtree.function.get_description(),new_subtree.depth(), new_subtree.is_leaf())
    
    tree.tree_update(parent_node)
    #print('nuova descr: ', tree.tree_get_description())

@cython.boundscheck(True)
cpdef subtree_crossover( function_treeC.Tree parent1, function_treeC.Tree parent2):
    """Crossover between two subtrees from different trees"""
    #cdef function_treeC.Node node1 = parent1.root.select_random_node()
    #cdef function_treeC.Node node2 = parent2.root.select_random_node()
    #cdef function_treeC.Node parent_node = parent1.root.find_parent(node1)
    #print("Prima del crossover:", node1.function.get_description(), node2.function.get_description())
    cdef function_treeC.Tree  old2 = parent2
    rename_prefix(old2)
    
    cdef function_treeC.Node node1 = parent1.root.select_random_node()
    cdef function_treeC.Node node2 = old2.root.select_random_node()
    cdef function_treeC.Node parent_node = parent1.root.find_parent(node1)
    
    if parent_node is None:
        parent1.root = node2
    elif parent_node.left == node1:
        parent_node.left = node2
    elif parent_node.right == node1:
        parent_node.right = node2
    parent1.tree_update(parent_node)
    
    refresh_params(parent1)

cpdef rename_prefix(function_treeC.Tree tree, str new_prefix = 'b'):
    '''Renames parameters of a tree using a new prefix'''
    cdef function_treeC.Node node
    
    
    for node in tree.params:
        old_rep = node.function.get_description()
        param_suffix = old_rep[1:]
        node.function = real_functionC.RealFunctionC.from_string(f"{new_prefix}{param_suffix}", tree.domain, compute_grad = False)
    tree.tree_update(tree.root)
    
cdef refresh_params(function_treeC.Tree tree):
    '''Conforms all parameters to one prefix'''
    
    cdef list params = [tree.params[j].get_fun().get_description() for j in range (0, len(tree.params))]
    cdef dict rename_map = {}
    cdef int max_index = -1
    for param in params:
        if param[0] == 'a' and param[1:].isdigit():  
            max_index = max(max_index, int(param[1:]))
    cdef int counter = max_index + 1    
    for param in params:
        if param[0] =='b' :
            rename_map[param] = f"a{counter}"
            counter += 1
    
    cdef function_treeC.Node node
    for node in tree.params:
        param_name = node.function.get_description()
        if param_name in rename_map:
            parent_node = tree.root.find_parent(node)
            old_node = node
            node.function = real_functionC.RealFunctionC.from_string(rename_map[param_name], tree.domain, compute_grad = False)
            node.update_func()
            if parent_node is None:
    
                tree.root = node
            elif parent_node.get_left() == old_node:
                parent_node.assign_left(node)
        
            elif parent_node.get_right() == old_node:
                parent_node.assign_right(node)
            
            tree.tree_update(parent_node)


@cython.boundscheck(True)
cpdef point_mutation(function_treeC.Tree tree, functions, operations=None, function_treeC.Node node_to_mutate=None):
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    
    if node_to_mutate is None:
        node_to_mutate = tree.root.select_random_node() 
    
    cdef str new_func_str
    cdef function_treeC.Node parent = tree.root.find_parent(node_to_mutate)
    if node_to_mutate.is_leaf():
        new_func_str= random.choice(functions)
        node_to_mutate.function = real_functionC.RealFunctionC.from_string(new_func_str, tree.domain, compute_grad = False)
    else:
        node_to_mutate.op_symbol = random.choice(operations)
        node_to_mutate.update_func()
    '''
    if parent is None:
      
        tree.root = node_to_mutate
    elif parent.left == node_to_mutate:
        parent.left = node_to_mutate
    elif parent.right == node_to_mutate:
        parent.right = node_to_mutate
    '''
    # Aggiorna la struttura dell'albero
    if parent is None:
        tree.tree_update(tree.root)
    else:
        tree.tree_update(parent)
    
@cython.boundscheck(True)
@cython.wraparound(True)
cpdef hoist_mutation(function_treeC.Tree tree, function_treeC.Node node_to_lift = None):
    """Replaces subtree with one of its subtrees"""
 
    cdef function_treeC.Node subtree, parent_node, granpa
   
    if not tree.root.is_leaf():
        check=False
        if node_to_lift is None:
            while not check:
                subtree = tree.root.select_random_node()
                parent_node = tree.root.find_parent(subtree)
            
                check = parent_node is not None
        else:
            subtree = node_to_lift
            parent_node = tree.root.find_parent(subtree)
            if parent_node is None:
                pass
        #parent_node = subtree
        if not subtree.is_leaf():
            parent_node.assign_left(subtree.left)
            parent_node.assign_right(subtree.right)
            parent_node.assign_operation(subtree.get_op())
            tree.tree_update(parent_node)

        else:
            granpa = tree.root.find_parent(parent_node)
            if granpa is not None:

                if granpa.left == parent_node:
                    granpa.assign_left(subtree)
                elif granpa.right == parent_node:
                    granpa.assign_right(subtree)
                tree.tree_update(granpa) 
            else:

                tree.root.assign_left(None)
                tree.root.assign_right(None)
                tree.root.assign_operation(None)
                tree.root = function_treeC.Node(subtree.get_fun())
                tree.tree_update(tree.root)
            #parent_node.assign_left(None)
            #parent_node.assign_right(None)
            #parent_node.assign_operation(None)
            #print(subtree.get_fun().get_description())
            #parent_node = function_treeC.Node(subtree.get_fun())
            #print(parent_node.get_fun().get_description()) 
            #granpa = tree.root.find_parent(parent_node)
            #if granpa is not None:
            #    if granpa.get_left() == parent_node
              
        '''
        if not subtree.is_leaf():
            #print('subtree node: ', subtree.get_fun().get_description(), 'left:', subtree.left.get_fun().get_description(),
            # 'right: ', subtree.right.get_fun().get_description())
            choice = random.choice([True, False])
            if choice:
                try:
                    parent_node.assign_left(subtree.get_left())
                except: 
                    parent_node.assign_right(subtree.get_right())
            else:
                try:
                    parent_node.assign_right(subtree.get_right())
                except:
                    parent_node.assign_left(subtree.get_left())
            
            tree.tree_update(parent_node)
        '''
        '''
        elif subtree.is_leaf():
            parent_node.assign_left(None)
            parent_node.assign_right(None)
            parent_node.assign_operation(None)
            parent_node = subtree
        '''
           
            #print('new parent node: ', parent_node.get_fun().get_description(), 'left: ', parent_node.left.get_fun().get_description(),
            # 'right: ', parent_node.right.get_fun().get_description())
    
        
#@cython.boundscheck(False)
#@cython.wraparound(False)
'''
def size_fair_mutation(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Effettua una mutazione preservando la dimensione dell'albero, sostituendo una parte con un'altra di dimensioni simili."""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef int subtree_size = node_to_replace.compute_subtree_size()
    cdef function_treeC.Node new_subtree = generate_subtree_with_size(subtree_size, functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right
    tree.tree_update(node_to_replace)
'''

@cython.boundscheck(False)
@cython.wraparound(False)
def replace_with_subtree(function_treeC.Tree tree, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Replaces a subtree with a random generated subtree GONNA BE DELETED PROBABLY"""
    cdef function_treeC.Node node_to_replace = tree.root.select_random_node()
    cdef function_treeC.Node new_subtree = function_treeC.build_tree(random.randint(1, 3), functions, x)
    node_to_replace.left = new_subtree.left
    node_to_replace.right = new_subtree.right
'''
@cython.boundscheck(True)
cpdef add_node(function_treeC.Tree tree, functions, operations):
    cdef function_treeC.Node node_to_expand = tree.root.select_random_leaf()
    cdef function_treeC.Node parent = tree.root.find_parent(node_to_expand)
    cdef str new_func_str = random.choice(functions)
    cdef function_treeC.Node cache = node_to_expand
    x = tree.domain
    op = random.choice(operations)
    cdef function_treeC.Node new_node = function_treeC.build_tree(random.choice(list(np.arange(1, tree.depth))), functions, x)
    if random.choice([True, False]):
        node_to_expand.assign_right(node_to_expand)
        node_to_expand.assign_left(new_node)
    else:
        node_to_expand.assign_left(node_to_expand)
        node_to_expand.assign_right(new_node)
    node_to_expand.assign_operation(op)
    if parent is None:
        tree.root = node_to_expand
    elif parent.get_left() == node_to_expand:
        parent.assign_left(node_to_expand)
    elif parent.get_right() ==node_to_expand:
        parent.assign_right(node_to_expand)
    
    #print('check5')
    #print(node_to_expand.get_fun().get_description())
    #print(parent.get_fun().get_description())
    parent.update_func()
    #print(parent.get_fun().get_description())
    #print('check6')
    #print(node_to_expand.is_leaf())
    #print(tree.root.find_parent(node_to_expand).update_func())
    tree.tree_update(parent)

'''

@cython.boundscheck(True)  # Per migliorare le prestazioni
cpdef add_node( function_treeC.Tree tree, functions, operations=None, random_depth=False):
    """Adds a node from a random location in an existing tree"""
    if operations is None:
        operations = ["+", "-", "*", "/", "**", "o"]
    cdef str op = random.choice(operations)
    if not tree.root.is_leaf():
        parent_node = None
        while parent_node is None:
            node_to_replace = tree.root.select_random_leaf()
            parent_node = tree.root.find_parent(node_to_replace)
        parent_op = parent_node.get_op()
        if not random_depth:
            #max_depth = tree.root.depth()
            max_depth = 2
        else:
            max_depth = random.choice(list(np.arange(1, tree.root.depth()+1)))
        new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)
        if random.choice([True, False]):
            node_to_replace.assign_left(node_to_replace)
            node_to_replace.assign_right(new_subtree)
        else:
            node_to_replace.assign_right(node_to_replace)
            node_to_replace.assign_left(new_subtree)

        node_to_replace.assign_operation(op)
       

        if parent_node is None:
            tree.root = new_subtree
        elif parent_node.get_left() == node_to_replace:
            parent_node.assign_left(new_subtree)
        
        elif parent_node.get_right() == node_to_replace:
            parent_node.assign_right(new_subtree)
 
        node_to_replace.update_func()  
    #print('op_new: ', new_subtree.function.get_description(),new_subtree.depth(), new_subtree.is_leaf())
    
        tree.tree_update(parent_node)
    #print('nuova descr: ', tree.tree_get_description())
    else:
        parent_node = function_treeC.Node(function=tree.root.get_fun())
        max_depth = random.choice([1,2])
        new_subtree = function_treeC.build_tree(max_depth, functions, tree.domain)
        if random.choice([True, False]):
            tree.root.assign_right(parent_node)
            tree.root.assign_left(new_subtree)
        else:
            tree.root.assign_left(parent_node)
            tree.root.assign_right(new_subtree)
       
        tree.root.assign_operation(op)
        #print(tree.root.get_fun().get_description(), parent_node.is_leaf())
        
        #print(tree.root.get_fun().get_description(), tree.root.is_leaf(), parent_node.is_leaf())
        #print(parent_node, tree.root)
        tree.tree_update(parent_node)
        
def generate_subtree_with_size(int size, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x):
    """Generates a random subtree with chosen size"""
    depth = size // 2  #  profondità approssimativa
    cdef function_treeC.Tree tree = function_treeC.Tree(depth, functions, x)
    return tree.root


cpdef root_replacement(function_treeC.Tree tree):
    cdef int depth = tree.depth
    cdef object x = tree.domain
    cdef list funcs = tree.funcs
    cdef function_treeC.Tree new_tree = function_treeC.Tree(depth, funcs, x)
    tree = new_tree
    tree.tree_update(tree.root) #RIGUARDA QUESTO METODO

cpdef prune_node(function_treeC.Tree tree, function_treeC.Node node_to_prune = None):
    if node_to_prune is None:
       hoist_mutation(tree)
       pass

    cdef function_treeC.Node parent = tree.root.find_parent(node_to_prune)  
    if parent is not None:
        if parent.right == node_to_prune:
            hoist_mutation(tree, parent.left)
        else:
            hoist_mutation(tree, parent.right)
    else:
        pass
        #point_mutation(tree, node_to_prune)
