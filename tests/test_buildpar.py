import numpy as np
from real_functionC import RealFunctionC  # Assicurati di importare il modulo corretto
from function_treeC import build_tree, Node  # Importa il modulo Cython che contiene build_tree

# Funzione di esempio per testare
def test_build_tree():
    # Esempio di funzioni disponibili per costruire l'albero
    functions = ['sin(x)', 'cos(x)', 'exp(x)', 'log(x)', 'sqrt(x)']

    # Creazione di un array x (dominio su cui valutare le funzioni)
    x = np.linspace(0, 10, 100, dtype=np.float32)

    # Numero di parametri da inserire nelle foglie (es. 3 parametri)
    Npars = 3
    depth = 3  # Profondità dell'albero

    # Chiamata alla funzione build_tree per creare l'albero
    tree = build_tree(depth=depth, functions=functions, x=x, output_type='tree',  Npars=Npars, param_count=0)

    # Stampa l'albero o il nodo risultante
    print("Albero o nodo creato:")
    print(tree)

# Esegui il test
if __name__ == "__main__":
    test_build_tree()
