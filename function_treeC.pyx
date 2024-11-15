import numpy as np
cimport numpy as cnp
import sympy as sp
cimport sympy as csp
cimport real_functionC 

import random
import re

cdef class Node:
    '''Node of a Tree: either a leaf, or an operation on two other nodes'''    
    def __init__(self, real_functionC.RealFunctionC function=None, object left=None, object right=None, str symbol=None):
        self.function = function
        self.left = left
        self.right = right
        self.op_symbol = symbol
        self.node_depth = self.depth()
        self.domain = self.function.domain

    def get_domain(self):
        '''Obtain domain from Python.'''
        return self.domain
    # def?
    cpdef is_param(self):
        '''Is Node a constant function?'''
        if self.is_leaf() and len(self.function.variables)==0:
            return True
        elif self.is_leaf() and len(self.function.variables)>0:
            return False
        else:
            return self.function.is_param()
   
    # DANGEROUS!!! Use cases need be minimized!
    cdef reinit_to_par(self, set existing_pars, str user_prefix=None):
        '''Re-initializes a node to a constant.'''
        cdef object new_param = new_par(existing_pars, user_prefix) 
        self.function = real_functionC.RealFunctionC.from_string(str(new_param), self.domain)
        self.left, self.right, self.op_symbol, self.node_depth = None, None, None, 0
    #Useless not supported so far    
    cpdef assign_operation(self, operation):
        if not self.is_param():
            self.op_symbol = operation
            self.update_func()
    #Useless not supported so far
    cpdef assign_left(self, node):
        if not self.is_param():
            self.left = node
    #Useless not supported so far
    cpdef assign_right(self, node):
        if not self.is_param():
            self.right = node
    #cdef?
    cpdef find_parent(self, child):
        """Find the node whose left or right branch is child node"""
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
    #cdef?
    cpdef select_random_leaf(self):
        """Select a random leaf (used by genetic operations)"""
        if self.is_leaf():
            return self
        else:
            if random.choice([True, False]):
                return self.left.select_random_leaf()
            else:
                return self.right.select_random_leaf()
    #cdef?
    cpdef select_random_node(self, threshold=0.1):
        """Select a random node with probability influenced by node fitness (NOT IMPLEMENTED YET)"""
        #cdef float fitness = compute_fitness(node)
        cdef float fitness = 0.8

        cdef float selection_prob = max(1.0 - fitness, threshold)
        
        if random.random() < selection_prob:
            return self
        
        if self.is_leaf():
            return self
        
        
        if random.random() < 0.5:
            return self.left.select_random_node(threshold)
        else: 
            return self.right.select_random_node(threshold)

    cpdef compute_subtree_size(self):
        """A figure of merit for Tree or subTree dimension"""
        if self is None:
            return 0
        left_size = self.left.compute_subtree_size() if self.left is not None else 0
        right_size = self.right.compute_subtree_size() if self.right is not None else 0
        return 1 + left_size + right_size

    cpdef is_leaf(self):
        """Am I a leaf?"""
        return self.left is None and self.right is None
    #cdef?
    cpdef update_func(self, set params_list=set(), str user_prefix=None):
        """Update function after attribute changes"""
        if self.is_param():
            self.reinit_to_par(params_list, user_prefix)
           
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
        """ Find leaves in the subtree rooted by self"""
        if self.is_leaf():
            leaves_list.append(self)
        else:
            self.left.get_leaves(leaves_list = leaves_list)
            self.right.get_leaves(leaves_list = leaves_list)
        
        return leaves_list

         
    #def?
    cpdef get_left(self):
        """Access self.left from Python"""
        return self.left
    #def?
    cpdef get_right(self):
        """Access self.right from Python"""
        return self.right
    #def?
    cpdef get_op(self):
        """Access self.op_symbol from Python"""
        return self.op_symbol
    #def?
    cpdef get_fun(self):
        """Acces self.function from Python"""
        return self.function
    #def?
    cpdef get_depth(self):
        """Access self.depth from Python"""
        return self.node_depth
    #cdef?
    cpdef depth(self):
        """Compute how deep self is"""
        if self.is_leaf():
            return 0
        else:
            left_depth = self.left.depth() if self.left is not None else 0
            right_depth = self.right.depth() if self.right is not None else 0
            return 1 + max(left_depth, right_depth)

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
    
    #def?
    cpdef get_root(self):
        """Access Tree's root from Python"""
        return self.root
    #def?
    cpdef get_leaves(self):
        """Access Tree's leaves from Python"""
        return self.root.get_leaves()
    
    cpdef get_params(self, params_list = []):
        """How many constants in the Tree?"""
        for leaf in self.leaves:
            if leaf.is_param():
                params_list.append(leaf)
        return params_list
    #def?
    cpdef get_domain(self):
        """Access root's function domain from Python"""
        return self.domain
    #cdef?
    cpdef evaluate(self,  x=None):
        """Evaluate root on its domain or on a specified one"""
        if x is None:

            return self.root.function.evaluate(self.domain)
        else:
            
            return self.root.function.evaluate(x)

    def __str__(self):
        """print the Tree fancy scheme. REVERSE POLISH"""
        
        return self.tree_to_string(self.root)

    def tree_to_string(self,  Node node, str prefix='', bint is_left=True):
        """Creation of fancy scheme. REVERSE POLISH"""
        
        if node.is_leaf():
            return f"{prefix}{'└── ' if is_left else '├── '}{str(node.function.get_description())}\n"
        
        result = f"{prefix}{'└── ' if is_left else '├── '}{node.op_symbol}\n"
        prefix += '    ' if is_left else '│   '

        result += self.tree_to_string(node.left, prefix, True)
        result += self.tree_to_string(node.right, prefix, False)

        return result
        
    
    #def?
    cpdef tree_get_description(self):
        """access root function description from Python""" 
        return self.current_description
    #cdef?
    cpdef update_tree_description(self):
        """Update the whole tree nodes' functions' descriptions"""

        self.current_description = self.root.function.get_description()
    #cdef?
    cpdef tree_update(self, node):
        """Update the whole tree nodes' functions"""
        current_node = node
        while current_node is not None:
            current_node.update_func()
            current_node = self.root.find_parent(current_node)
            
        self.update_tree_description()

#cdef --> is cpdef for testing purposes only
cpdef build_tree(int depth, list functions, cnp.ndarray[cnp.float32_t, ndim=1] x, 
            str output_type=None, int Npars=0, int param_count=0, 
            bint first_call = True, list params_used=None):
        '''Builds a Tree from scratch'''
        cdef str func_str
        cdef list par_list = parameter_parser(Npars)
        cdef int max_leaves = int(2**depth)
        cdef str param_name

        
        if first_call:
            if Npars > max_leaves:
                print('More parameters than leaves --> constant function')
                Npars = max_leaves

            if params_used is None:
                params_used = []

            
        if depth==0:
            if param_count < Npars and (param_count%3) < 2:
                for p in par_list:
                    if p not in params_used:
                        param_name = p
                        
                        params_used.append(param_name) 
                        param_count += 1
                        break
                else:
                    func_str = random.choice(functions)
                    return Node(real_functionC.RealFunctionC.from_string(func_str, x)) 



                return Node(real_functionC.RealFunctionC.from_string(param_name, x))

            func_str = random.choice(functions)
            return Node(real_functionC.RealFunctionC.from_string(func_str, x))

        #Recursion needed
        cdef Node left = build_tree(depth - 1, functions, x,
                'node', Npars, param_count, False, params_used)
        cdef Node right = build_tree(depth - 1, functions, x, 
                'node', Npars, param_count, False, params_used)
       
       
        cdef str operation = random.choice(['add', 'sub', 'mul', 'div', 'pow', 'compose'])
        cdef real_functionC.RealFunctionC func  
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
            func = left.function ** right.function 
            op_symbol = '**'
        elif operation == 'compose':
            func = left.function.compose(right.function)
            op_symbol = '∘' 
        

        if output_type is None or output_type=='node' or output_type=='Node':
            
 
            return Node(function=func, left=left, right=right, symbol=op_symbol)
        else:
            return Tree(depth, functions, x, 
                    node =Node(function=func, left=left, right=right, symbol=op_symbol))

cdef parameter_parser(int Npars):
    """Create parameter names list(str())"""
    cdef list parameters = []
    cdef str prefix = 'a'
    for i in range(Npars):
        parameters.append(f"{prefix}{i}")

    return parameters

#Naming parameters requires fix!    
cpdef new_par(set existing_params, str user_choice=None):
    '''Add a new parameter named with a specific logic'''
    if len(existing_params) > 0:
        param_pattern = re.compile(r"([a-zA-Z]+)(\d+)")
        param_names = [param_pattern.match(param).groups()[0] for param in existing_params if param_pattern.match(param)]
        unique_prefixes = set(param_names)
        if user_choice not in unique_prefixes or user_choice==None:
            user_choice = random.choice(list(unique_prefixes))
       


        indices = sorted([int(param_pattern.match(param).groups()[1]) for param in existing_params if param.startswith(user_choice)])
        next_index = 0
        for index in indices:
            if index == next_index:
                next_index += 1
            else:
                break

       
        existing_params.add(f"{user_choice}{next_index}")
        return f"{user_choice}{next_index}" 
    else:
        if user_choice is None:
            existing_params.add('a0')
            return 'a0'
        else:
            existing_params.add(f"{user_choice}{0}")
            return f"{user_choice}{0}" 
