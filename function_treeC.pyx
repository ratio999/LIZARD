import numpy as np
cimport numpy as cnp
import sympy as sp
cimport sympy as csp
cimport real_functionC # cimport per accedere a RealFunctionC a livello di Cython
#from real_functionC import RealFunctionC  # Importa l'API ad alto livello di RealFunctionC se necessario
import random

cdef class Node:
    
    def __init__(self, real_functionC.RealFunctionC function=None, object left=None, object right=None, str symbol=''):
        self.function = function
        self.left = left
        self.right = right
        self.op_symbol = symbol
        self.node_depth = self.depth()

    cpdef is_leaf(self):
        return self.left is None and self.right is None
    
    cpdef update_func(self):
        if self.is_leaf():
            return 
        else:
            if self.op_symbol == '+':
                self.function = self.left.get_fun() + self.right.get_fun()
                
            elif self.op_symbol == '-':
                self.function = self.left.get_fun() - self.right.get_fun()
                
            elif self.op_symbol == '*':
                self.function = self.left.get_fun() * self.right.get_fun()
                
            elif self.op_symbol == '/':
                self.function = self.left.get_fun() / self.right.get_fun()
                
            elif self.op_symbol == '**':
                self.function = self.left.get_fun() ** self.right.get_fun()  # Operazione di potenza
                
            elif self.op_symbol == '∘':
                self.function = self.left.get_fun().compose(self.right.get_fun())
                
             
    
    cpdef get_left(self):
        return self.left
    
    cpdef get_right(self):
        return self.right
    
    cpdef get_op(self):
        return self.op_symbol
    
    cpdef get_fun(self):
        return self.function

    cpdef get_depth(self):
        return self.node_depth

    cpdef depth(self):
        if self.is_leaf():
            return 0
        else:
            left_depth = self.left.depth() if self.left is not None else 0
            right_depth = self.right.depth() if self.right is not None else 0
            return 1 + max(left_depth, right_depth)

cdef class Tree:
    

    def __init__(self, int depth, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x, Node node=None):
        if node==None:
            self.root = self.build_tree(depth, functions, x)
        else:
            self.root = node
        self.current_description = self.root.function.get_description()
    cpdef get_root(self):
        return self.root
        
    cpdef build_tree(self, depth, functions, x ):
        cdef str func_str
        if depth == 0:
            func_str = random.choice(functions)
            return Node(function=real_functionC.RealFunctionC.from_string(func_str, x))  # Usa il tipo Cython

        # Ricorsione per costruire l'albero
        cdef Node left = self.build_tree(depth - 1, functions, x)
        cdef Node right = self.build_tree(depth - 1, functions, x)

        # Seleziona casualmente un'operazione tra quelle disponibili
        cdef str operation = random.choice(['add', 'sub', 'mul', 'div', 'pow', 'compose'])
        cdef real_functionC.RealFunctionC func  # Usa il tipo Cython
        cdef str op_symbol

        if operation == 'add':
            func = left.function + right.function
            op_symbol = '+'
        elif operation == 'sub':
            func = left.function - right.function
            op_symbol = '-'
        elif operation == 'mul':
            func = left.function * right.function
            op_symbol = '*'
        elif operation == 'div':
            func = left.function / right.function
            op_symbol = '/'
        elif operation == 'pow':
            func = left.function ** right.function  # Operazione di potenza
            op_symbol = '**'
        elif operation == 'compose':
            func = left.function.compose(right.function)
            op_symbol = '∘'  # Composizione di funzioni

        # Crea il nodo con la funzione e i simboli dell'operazione
        
        return Node(function=func, left=left, right=right, symbol=op_symbol)

    cpdef evaluate(self,  x):
        # Valuta la funzione nell'albero radicato
        return self.root.function.evaluate(x)

    def __str__(self):
        # Stampa la rappresentazione dell'albero
        
        return self.tree_to_string(self.root)

    def tree_to_string(self,  Node node, str prefix='', bint is_left=True):
        # Rappresentazione visiva dell'albero
        
        if node.is_leaf():
            return f"{prefix}{'└── ' if is_left else '├── '}{str(node.function.get_description())}\n"
        
        result = f"{prefix}{'└── ' if is_left else '├── '}{node.op_symbol}\n"
        prefix += '    ' if is_left else '│   '

        result += self.tree_to_string(node.left, prefix, True)
        result += self.tree_to_string(node.right, prefix, False)

        return result
        
    

    cpdef tree_get_description(self):
        return self.current_description

    cpdef update_tree_description(self):
        """Ricalcola la descrizione dell'intero albero."""
        # Chiamare il metodo per ottenere la descrizione dalla radice
        
        #self.update_root()
        
        
        self.current_description = self.root.function.get_description()


        
        