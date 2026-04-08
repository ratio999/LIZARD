import tqdm
from real_functionC import RealFunctionC
#from fitnessC import calculate_likelihood
#from posteriorC import calculate_posterior
import sympy as sp
import cython
import time
import numpy as np
from function_treeC import Tree, expr_to_tree
from gp_movesC import (subtree_mutation, subtree_crossover, point_mutation,
                       hoist_mutation, add_node, replace_with_subtree, add_node)
from groveC import Grove
import matplotlib.pyplot as plt

'''def grove_smoke_test():
    # Domain and target
    x = np.linspace(0, 10, 200).astype(np.float64)
    y = 2.0 + np.sin(x)

    # Primitive set
    functions = ["sin(x)", "cos(x)", "exp(x)", "x**2", "x**3"]
    operations = ["+", "-", "*", "/", "o", "**"]

    # Grove config
    T = 1.0
    size = 2
    depth = 1
    grove = Grove(x, T, size, depth, functions, Npars=1, init_value=1.0, operations=operations, base_temp=1.0)
    for tree in grove.trees:
        print(tree)
        print(tree.get_param_dicts())
    # Initialize posteriors, run a few steps
    grove.compute_posteriors(y, full=False)
    grove.run(y, n_steps=10)

    # Report
    print(f"Grove acceptance rate: {grove.acceptance_rate():.3f}")
    best = grove.best_tree()
    print("Best tree (structure):")
    print(best)
    print("Best posterior:", best.get_superficial_posterior(y))
'''

def grove_smoke_test():
    # Domain and target
    x = np.linspace(0.001, 10, 200).astype(np.float64)
    y = 2.0 + np.sin(x)
    
    # Primitive set
    functions = ["sin(x)", "cos(x)", "x"]
    operations = ["+", "-", "*", "/"]

    # Grove config
    T = 1.0
    size = 6
    depth = 2
    grove = Grove(x,y, T, size, depth, functions, Npars=4, init_value=1.0, 
                  operations=operations, base_temp=1.0, use_gradients=False, 
                  burn_in=True, use_mpi=4, use_spawn=True, lambda_reg=0.01)
    
    print("=" * 60)
    print("Initial random trees:")
    for i, tree in enumerate(grove.trees[-10:]):
        print(f"  Tree {i}: {tree.tree_get_description()}")
    print("=" * 60)
    
    # Two-stage burn-in (does both depth-1 and depth-2 internally)
    #print("\n--- Two-Stage Burn-in ---")
    #grove.burn_in_depth2(y, replace_all=True, optimize_params=True )
    
    print(f"\nAfter burn-in, grove contains some depth-2 trees")
    for i, tree in enumerate(grove.trees)[:10]:
        print(f"  Tree {i}: {tree.tree_get_description()} | posterior: {grove.root_posteriors[i]:.4f}")
    
    # Now run MCMC on the initialized grove
    print("\n--- Running MCMC evolution ---")
    grove.run(y, n_steps=50)
    
    # Report
    print("\n" + "=" * 60)
    print(f"Final acceptance rate: {grove.acceptance_rate():.3f}")
    best = grove.best_tree()
    print("Best tree after MCMC:")
    print(f"  Expression: {best.tree_get_description()}")
    print(f"  Posterior: {best.get_superficial_posterior(y):.4f}")
    print(f"  Target: 2.0 + sin(x)")
    print("=" * 60)

def main():
    # Definire le funzioni disponibili
    functions = ["sin(x)", "cos(x)", "exp(x)", "x**2", "x**3"]
    operations = ["+", "-", "*", "/", "o", "**"]
    # Crea due alberi di funzioni
    x = np.linspace(0, 10, 100).astype(np.float64)
    y = np.sin(x)
    
    tree1 = Tree(depth=1, functions=functions, x=x, Npars = 24, value=np.array([3, 4, 5, 6, 7, 8]))
    
    print(tree1.tree_get_description())
    tree1.compute_fitness_tree(y)
    tree1.print_fitness_tree()
    tree2 = Tree(depth=5, functions=functions, x=x, Npars =24, value=np.array([545, 546, 547, 548]))
    n=0
    for param in tree2.get_params():
        param.set_value(n)
        print(param.get_fun().get_description())
        n+=1
    tree2.tree_update()
    for param in tree2.get_params():
        print(param.value)
    #print(tree1.node_count(), calculate_likelihood(tree1, y), calculate_posterior(tree1, y))
    print("Tree 1:")
    print(tree1)
    print(tree1.get_param_dicts())
    print("Tree 2:")
    print(tree2)
    #print(tree1.get_root().get_fun().get_params())
    parameters = tree1.get_param_dicts()[1]
    #print(tree1.get_root().get_fun().get_objective_L2(y, parameters), tree1.get_root().get_fun().get_gradient_L2(parameters) )





    #for i in tree1.get_params():
    #    print(i.get_fun().get_description())
    #print(tree1.get_param_dicts()[1], tree2.get_param_dicts()[1])
    #print(tree1.get_root().get_fun().get_description())
    #try:
    #    plt.plot(x, tree1.evaluate(x))
    #    plt.show()
    #except:
    #    pass
    print(tree1.is_defined(x), tree1.evaluate(x))
    # Eseguire le mosse di genetic programming
    
    subtree_mutation(tree1, functions, random_depth=True)
    
    #print(tree1.optimize(x, y).x)
    print("Subtree mutation:")
    print(tree1)
    print(tree1.get_root().get_fun().get_description())
    print(tree1.is_valid())
    #print(tree1.get_param_dicts()[1], tree2.get_param_dicts()[1])
    print('other tree for crossover')
    print(tree2)
    print('prima', tree1.get_param_dicts()[1], tree2.get_param_dicts()[1])
    subtree_crossover(tree1, tree2)
    print("Crossover:")
    print(tree1)
    print(tree1.tree_get_description())
    print('dopo', tree1.get_param_dicts()[1])
    #for i in tree1.get_params():
    #    print(i.get_fun().get_description())
    #print(tree1.get_param_dicts()[1])
 
    point_mutation(tree1, functions)
    print("Point mutation:")
    print(tree1)
    print(tree1.tree_get_description())
    #print(tree1.get_root().get_fun().get_params())
    #for i in tree1.get_params():
    #    print(i.get_fun().get_description())
    #print(tree1.get_params(),'\n\n\n' , tree2.get_params())
    #print(tree1.get_param_dicts()[1])
    hoist_mutation(tree1)
    print("Hoist mutation:")
    print(tree1)
    print(tree1.tree_get_description())
    #print(tree1.get_root().get_fun().get_params())
    
    #for i in tree1.get_params():
    #    print(i.get_fun().get_description())
    #print(tree1.get_param_dicts()[1])
    '''
    size_fair_mutation(tree1, functions, x)
    print("Dopo la mutazione di dimensione equa:")
    print(tree1)
    print(tree1.tree_get_description())
    '''
    
    add_node(tree1, functions, operations)
    print("Node addition:")
    print(tree1)
    print(tree1.tree_get_description())
    #print(tree1.get_root().get_fun().get_params())
    
    #for i in tree1.get_params():
    #    print(i.get_fun().get_description())
    #print(tree1.get_param_dicts()[1])
    '''
    replace_with_subtree(tree1, functions, x)
    print("Dopo la sostituzione con un sottoalbero:")
    print(tree1)
    print(tree1.tree_get_description())
    '''

    domain = np.linspace(-1,1, 100)
    x, a1, a0 = sp.Symbol('x'), sp.Symbol('a1'), sp.Symbol('a0')
    expr = sp.exp(sp.sin(x+2)) ** a1**x**(2 * a0)/sp.cos(x)
    tree = expr_to_tree(expr, domain,functions)
    print(str(expr), tree.tree_get_description())
    print('AFTER EXPR2TREE', tree, tree.get_param_dicts())
    grove_smoke_test()



if __name__ == "__main__":
    main()
