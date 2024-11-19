from real_functionC import RealFunctionC
import cython
import time
import numpy as np
from function_treeC import Tree
from gp_movesC import (subtree_mutation, subtree_crossover, point_mutation,
                       hoist_mutation, add_node, replace_with_subtree, add_node)

def main():
    # Definire le funzioni disponibili
    functions = ["sin(x)", "cos(x)", "exp(x)", "x**2", "x**3"]
    operations = ["+", "-", "*", "/"]
    # Crea due alberi di funzioni
    x = np.linspace(0, 10, 100).astype(np.float64)

    tree1 = Tree(depth=3, functions=functions, x=x)
    print(tree1.tree_get_description())
    tree2 = Tree(depth=3, functions=functions, x=x)
    
    print("Albero 1:")
    print(tree1)
    print("Albero 2:")
    print(tree2)

 
    print(tree1.get_root().get_fun().get_description())
    
    print(tree1.get_root().get_fun().is_defined(x), tree1.get_root().get_fun().evaluate(x))
    # Eseguire le mosse di genetic programming
    subtree_mutation(tree1, functions, random_depth=True)
    print("Dopo la mutazione del sottoalbero:")
    print(tree1)
    print(tree1.get_root().get_fun().get_description())
    
    subtree_crossover(tree1, tree2)
    print("Dopo il crossover:")
    print(tree1)
    print(tree1.tree_get_description())
    
    point_mutation(tree1, functions, operations, x)
    print("Dopo la mutazione puntuale:")
    print(tree1)
    print(tree1.tree_get_description())
    
    hoist_mutation(tree1)
    print("Dopo la mutazione di sollevamento:")
    print(tree1)
    print(tree1.tree_get_description())
    '''
    size_fair_mutation(tree1, functions, x)
    print("Dopo la mutazione di dimensione equa:")
    print(tree1)
    print(tree1.tree_get_description())
    '''
    
    add_node(tree1, functions, operations)
    print("Dopo l'aggiunta di un nodo:")
    print(tree1)
    print(tree1.tree_get_description())
    '''
    replace_with_subtree(tree1, functions, x)
    print("Dopo la sostituzione con un sottoalbero:")
    print(tree1)
    print(tree1.tree_get_description())
    '''
if __name__ == "__main__":
    main()
