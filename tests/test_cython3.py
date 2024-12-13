from real_functionC import RealFunctionC
from fitnessC import calculate_likelihood
from posteriorC import calculate_posterior
import cython
import time
import numpy as np
from function_treeC import Tree
from gp_movesC import (subtree_mutation, subtree_crossover, point_mutation,
                       hoist_mutation, add_node, replace_with_subtree, add_node)
import matplotlib.pyplot as plt

def main():
    # Definire le funzioni disponibili
    functions = ["sin(x)", "cos(x)", "exp(x)", "x**2", "x**3"]
    operations = ["+", "-", "*", "/"]
    # Crea due alberi di funzioni
    x = np.linspace(0, 10, 100).astype(np.float64)
    y = np.sin(x)

    tree1 = Tree(depth=3, functions=functions, x=x, Npars = 6, value=np.array([3, 4, 5, 6, 7, 8]))
    print(tree1.tree_get_description())
    tree1.compute_fitness_tree(y)
    tree1.print_fitness_tree()
    tree2 = Tree(depth=3, functions=functions, x=x, Npars = 6, value=np.array([545, 546, 547, 548]))
    print(tree1.node_count(), calculate_likelihood(tree1, y), calculate_posterior(tree1, y))
    print("Albero 1:")
    print(tree1)
    print("Albero 2:")
    print(tree2)
    print(tree1.get_root().get_fun().get_params())
    parameters = tree1.get_param_dicts()[1]
    #print(tree1.get_root().get_fun().get_objective_L2(y, parameters), tree1.get_root().get_fun().get_gradient_L2(parameters) )





    for i in tree1.get_params():
        print(i.get_fun().get_description())
    print(tree1.get_param_dicts()[1], tree2.get_param_dicts()[1])
    print(tree1.get_root().get_fun().get_description())
    #try:
    #    plt.plot(x, tree1.evaluate(x))
    #    plt.show()
    #except:
    #    pass
    print(tree1.is_defined(x), tree1.evaluate(x))
    # Eseguire le mosse di genetic programming
    subtree_mutation(tree1, functions, random_depth=True)
    print("Dopo la mutazione del sottoalbero:")
    print(tree1)
    print(tree1.get_root().get_fun().get_description())
    print(tree1.get_param_dicts()[1], tree2.get_param_dicts()[1])
   
    subtree_crossover(tree1, tree2)
    print("Dopo il crossover:")
    print(tree1)
    print(tree1.tree_get_description())
    print(tree1.get_root().get_fun().get_params())
    
    for i in tree1.get_params():
        print(i.get_fun().get_description())
    print(tree1.get_param_dicts()[1])
 
    point_mutation(tree1, functions, operations)
    print("Dopo la mutazione puntuale:")
    print(tree1)
    print(tree1.tree_get_description())
    print(tree1.get_root().get_fun().get_params())
    for i in tree1.get_params():
        print(i.get_fun().get_description())
    #print(tree1.get_params(),'\n\n\n' , tree2.get_params())
    print(tree1.get_param_dicts()[1])
    hoist_mutation(tree1)
    print("Dopo la mutazione di sollevamento:")
    print(tree1)
    print(tree1.tree_get_description())
    print(tree1.get_root().get_fun().get_params())
    
    for i in tree1.get_params():
        print(i.get_fun().get_description())
    print(tree1.get_param_dicts()[1])
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
    print(tree1.get_root().get_fun().get_params())
    
    for i in tree1.get_params():
        print(i.get_fun().get_description())
    print(tree1.get_param_dicts()[1])
    '''
    replace_with_subtree(tree1, functions, x)
    print("Dopo la sostituzione con un sottoalbero:")
    print(tree1)
    print(tree1.tree_get_description())
    '''
if __name__ == "__main__":
    main()
