import numpy as np
cimport numpy as cnp
import sympy as sp
cimport sympy as csp
cimport real_functionC # cimport per accedere a RealFunctionC a livello di Cython
#from real_functionC import RealFunctionC  # Importa l'API ad alto livello di RealFunctionC se necessario
import random

cdef class Node:
    
    def __init__(self, real_functionC.RealFunctionC function=None, object left=None, object right=None, str symbol=None):
        self.function = function
        self.left = left
        self.right = right
        self.op_symbol = symbol
        self.node_depth = self.depth()
        self.domain = self.function.domain

    def get_domain(self):
        return self.domain

    cpdef is_param(self):
        return self.function.is_param()

    cpdef assign_operation(self, operation):
        if not self.is_param():
            self.op_symbol = operation
            self.update_func()

    cpdef assign_left(self, node):
        if not self.is_param():
            self.left = node
    
    cpdef assign_right(self, node):
        if not self.is_param():
            self.right = node

    cpdef find_parent(self, child):
        """Trova il nodo genitore di un dato nodo."""
        if self is None or self.is_leaf():
            return None
        if self.left == child or self.right == child:
            return self
        if self.left:
            parent = self.left.find_parent(child)
            if parent:
                return parent
        if self.right:
            return self.right.find_parent(child)
        return None
    
    cpdef select_random_leaf(self):
        if self.is_leaf():
            return self
        else:
            if random.choice([True, False]):
                return self.left.select_random_leaf()
            else:
                return self.right.select_random_leaf()

    cpdef select_random_node(self, threshold=0.1):
        """
        Seleziona un nodo casuale in un albero, con probabilità influenzata dalla sua fitness.
        Nodi con fitness bassa hanno una maggiore probabilità di essere selezionati.
        """
        #cdef float fitness = compute_fitness(node)
        cdef float fitness = 0.8
        # Più è bassa la fitness, più è alta la probabilità di selezionare il nodo
        cdef float selection_prob = max(1.0 - fitness, threshold)
        
        if random.random() < selection_prob:
            return self
        
        if self.is_leaf():
            return self
        
        # Se il nodo non viene selezionato, passa casualmente ai figli
        if random.random() < 0.5:  # 50% di probabilità di andare verso il nodo sinistro
            return self.left.select_random_node(threshold)
        else:  # 50% di probabilità di andare verso il nodo destro
            return self.right.select_random_node(threshold)

    cpdef compute_subtree_size(self):
        """Calcola la dimensione di un sottoalbero."""
        if self is None:
            return 0
        left_size = self.left.compute_subtree_size() if self.left is not None else 0
        right_size = self.right.compute_subtree_size() if self.right is not None else 0
        return 1 + left_size + right_size

    cpdef is_leaf(self):
        return self.left is None and self.right is None
    
    cpdef update_func(self):
        if self.is_param():
            self.op_symbol=None
            self.right = None
            self.left = None
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
                
            elif self.op_symbol == '∘':   # Composition symbol
                self.function = self.left.get_fun().compose(self.right.get_fun())
                
    cpdef get_leaves(self, leaves_list = []):
        if self.is_leaf():
            leaves_list.append(self)
        else:
            self.left.get_leaves(leaves_list = leaves_list)
            self.right.get_leaves(leaves_list = leaves_list)
        
        return leaves_list

         
    
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
        if self.is_leaf() or self.is_param():
            return 0
        else:
            left_depth = self.left.depth() if self.left is not None else 0
            right_depth = self.right.depth() if self.right is not None else 0
            return 1 + max(left_depth, right_depth)
'''
cdef class ParameterNode(Node):
    cdef double value  # Valore del parametro

    def __init__(self, double value):
        super().__init__()  # Inizializza la classe padre
        self.value = value  # Imposta il valore del parametro

    def __repr__(self):
        return f"ParameterNode(value={self.value})"

    def mutate(self, double mutation_rate):
        """Mutazione del valore del parametro."""
        import random
        if random.random() < mutation_rate:
            # Varia il valore del parametro con un piccolo scostamento
            self.value += random.uniform(-, 0.1)
'''
cdef class Tree:
    

    def __init__(self, int depth, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x, Node node=None):
        self.domain = x
        self.depth = depth
        if node==None:
            self.root = build_tree(depth, functions, self.domain)
        else:
            self.root = node
        self.current_description = self.root.function.get_description()
        self.leaves = self.get_leaves()
        self.params = self.get_params()

    cpdef get_root(self):
        return self.root
    
    cpdef get_leaves(self):
         return self.root.get_leaves()

    cpdef get_params(self, params_list = []):
        for leaf in self.leaves:
            if leaf.is_param():
                params_list.append(leaf)
        return params_list

    cpdef get_domain(self):
        return self.domain

    cpdef evaluate(self,  x=None):
        if x is None:
        # Valuta la funzione nell'albero radicato
            return self.root.function.evaluate(self.domain)
        else:
            
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

    cpdef tree_update(self, node):
        current_node = node
        while current_node is not None:
            current_node.update_func()
            current_node = self.root.find_parent(current_node)
            
        self.update_tree_description()

cdef build_tree(int depth, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x, str output_type=None):
        cdef str func_str
        #cdef cnp.ndarray[cnp.float32_t, ndim=1] x = self.domain
        if depth == 0:
            func_str = random.choice(functions)
 
            #print(funct.get_domain()) 
            return Node(real_functionC.RealFunctionC.from_string(func_str, x))  # Usa il tipo Cython

        # Ricorsione per costruire l'albero
        cdef Node left = build_tree(depth - 1, functions, x)
        cdef Node right = build_tree(depth - 1, functions, x)

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
        if output_type is None or output_type=='node' or output_type=='Node':
            
 
            return Node(function=func, left=left, right=right, symbol=op_symbol)
        else:
            return Tree(depth, functions, x, 
                    node =Node(function=func, left=left, right=right, symbol=op_symbol))

    

        
        
