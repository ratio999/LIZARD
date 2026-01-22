import numpy as np
cimport numpy as cnp
import sympy as sp
cimport sympy as csp
cimport real_functionC 
from libc.stdlib cimport malloc, free
from libc.stdio cimport snprintf, printf
#rom libc.string cimport snpritf
from cython.parallel cimport prange
from copy import copy, deepcopy
import random
import re

cdef class Node:
    '''Node of a Tree: either a leaf, or an operation on two other nodes'''    
    def __init__(self, real_functionC.RealFunctionC function=None, object left=None, object right=None, str symbol=None, double alpha = 0.01, object sigma = 1.00):
        self.function = function if function is not None else self.fetch_fun()
        self.left = left
        self.right = right
        self.op_symbol = symbol
        self.node_depth = self.depth()
        self.domain = self.function.domain
        self.bis_param = self.is_param()
        if self.bis_param:
            self.value = 1
        else:
            self.value = np.nan
        self.fitness = None
        self.posterior = None
        self.prior = None
        self.alpha = alpha #keep alpha as it could weight an externally computed prior
        self.sigma = sigma
        
    def __copy__(self):
        return Node(copy(self.function), copy(self.left), copy(self.right),
            self.op_symbol, alpha=self.alpha, sigma=self.sigma)

    def __deepcopy__(self, memo1):
        cdef Node node_copy = Node(deepcopy(self.function, memo1),
                deepcopy(self.left, memo1), deepcopy(self.right, memo1),
                self.op_symbol, alpha=self.alpha, sigma=self.sigma
                )
        memo1[id(self)] = node_copy
        return node_copy

    def get_root(self):
        print(self.root)
    def get_domain(self):
        '''Obtain domain from Python.'''
        return self.domain
    # def?
    cpdef is_param(self):
        '''Is Node a constant function?'''
        if self.is_leaf() and len(self.function.variables)==0:
            return True
        #elif self.is_leaf() and len(self.function.variables)>0:
        
        return False
    
    cdef set_alpha(self, double newalpha):
        self.alpha = newalpha

    cdef set_sigma(self, object newsigma):
        self.sigma = newsigma
    
    cdef set_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters):
        self.fitness = self.evaluate_node_fitness(y, parameters)
    
    cdef set_posterior(self,cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters):
        self.posterior = self.evaluate_node_posterior(y, parameters)
    
 
    cpdef set_value(self, cnp.float64_t value = 1):
        '''Forcedly sets parameter to value'''
        if self.bis_param:

            self.value = value 
        
    cdef sort_params(self, list pars):
        cdef Node i
        cdef list pars_str = [sp.symbols(i.function.get_description()) for i in pars]
        self.function.sort_params_list(pars_str)

    cdef fetch_fun(self, Node left = None, Node right = None, str operation = None):
        '''Fetches function if none is provided'''
        
        cdef real_functionC.RealFunctionC auxiliary_fun  
        if left is None:
            left = self.left
        if right is None:
            right = self.right     
        if operation is None:
            operation = self.op_symbol
        
        if left is None and right is None and operation is None:
            return None
        
        if operation == '+':
            auxiliary_fun = left.function + right.function
        elif operation == '-':
            auxiliary_fun = left.function - right.function
        elif operation == '*':
            auxiliary_fun = left.function * right.function
        elif operation == '/':
            auxiliary_fun = left.function / right.function
        elif operation == '**':
            auxiliary_fun = left.function ** right.function  # Operazione di potenza
        elif operation == 'o':   # Composition symbol
            auxiliary_fun = left.function.compose(right.function)
        print('fetched function: ', auxiliary_fun.get_description())
        return auxiliary_fun

    cpdef fitness_tree(self,  y, parameters):
        self.set_fitness(y, parameters)
        if self.left:
            self.left.fitness_tree(y, parameters)
        if self.right:
            self.right.fitness_tree(y, parameters)
    cpdef posterior_tree(self,  y, parameters):
        self.set_posterior(y, parameters)
        if self.left:
            self.left.posterior_tree(y, parameters)
        if self.right:
            self.right.posterior_tree(y, parameters)
         
    cdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data, 
        dict input_pars, double lambda_reg, str method_str='L-BFGS-B', bint use_gradients=False):
        if use_gradients:
            return self.function._optimize_with_grad(x_data, y_data, input_pars, lambda_reg, method_str)
        else:
            return self.function.optimize(x_data, y_data, input_pars, lambda_reg, method_str='Powell')
        
        
    cdef set_prior(self):
        self.prior = self.function.prior
   
    # DANGEROUS!!! Use cases need be minimized!
    cdef reinit_to_par(self, list existing_pars, str user_prefix=None):
        '''Re-initializes a node to a constant.'''
        cdef object new_param = new_par(existing_pars, user_prefix) 
        self.function = real_functionC.RealFunctionC.from_string(str(new_param), self.domain)
        print('REINITIALIZATION OF NODE ', self, ' TO PARAMETER!')
        self.left, self.right, self.op_symbol, self.node_depth, self.value = None, None, None, 0, self.set_value()
    
    cdef evaluate_node_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters):
        return self.function.fitness(y, parameters)

    cdef evaluate_node_posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters):
        
        return self.function.posterior(y, parameters, self.fitness, self.prior, 
                                       self.sigma)

    cpdef assign_operation(self, operation):
        '''Assigns an operation between self.left and self.right'''
        if not self.bis_param:
            self.op_symbol = operation
            self.update_func()
   
    cpdef assign_left(self, node):
        '''Assigns left node'''
        if not self.bis_param:
            self.left = node
  
    cpdef assign_right(self, node):
        '''Assigns right node'''
        if not self.bis_param:
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
        """Is self a leaf?"""
        return self.left is None and self.right is None
    #cdef?
    cpdef update_func(self, list params_list=[], str user_prefix=None):
        """Update function after attribute changes"""
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
                
            elif self.op_symbol == 'o':   # Composition symbol
                self.function = self.left.get_fun().compose(self.right.get_fun())

    cdef bint is_valid(self,object x=None, double max_abs=1e12):
        print('function: ', self.function.get_description())
        return self.function.is_valid(x, max_abs)
                
    cpdef get_leaves(self, leaves_list = None):
        """ Find leaves in the subtree rooted by self"""
        if leaves_list is None:
            leaves_list = []
        if self.is_leaf():
            leaves_list.append(self)
            #print(self.get_fun().get_description())
        else:
            
            self.left.get_leaves(leaves_list = leaves_list)
            self.right.get_leaves(leaves_list = leaves_list)
    
        return leaves_list

    cpdef node_count(self):
        
        cdef int count = 1  
        if self.left:
            count += self.left.node_count()
        if self.right:
            count += self.right.node_count()
        return count
        
    
    
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
     

    def __init__(self, int depth, list functions, 
            cnp.ndarray[cnp.float64_t, ndim=1] x, 
            Node node=None, int Npars = 0, object value=1, 
            double T=0, double alpha=0.01, object sigma = 1.0):
        '''Tree class initialization'''
        self.T = T
        self.domain = x
        self.depth = depth
        self.functions = functions
        self.Npars=Npars
        self.alpha = alpha
        self.sigma = sigma
        self.value = value   
        if node==None:
            self.root = build_tree(depth, functions, self.domain, Npars=Npars, value=value, alpha=self.alpha, sigma=self.sigma)
        else:
            self.root = node
            self.depth = node.depth()
        self.current_description = self.root.function.get_description()
        self.leaves = self.get_leaves()
        self.params = self.get_params()
        self.param_values, self.name_values = self.get_param_values()
        self.fitness_tree = None
    def __copy__(self):
        return Tree(self.depth, self.functions, self.domain, self.root,
            self.Npars, self.value, self.T, self.alpha, self.sigma)

    def __deepcopy__(self, memo):
        cdef Tree copy_tree = Tree(
                self.depth,
                deepcopy(self.functions, memo),
                deepcopy(self.domain, memo),
                deepcopy(self.root, memo),
                self.Npars, self.value, self.T, self.alpha, self.sigma
                )
        memo[id(self)] = copy_tree
        return copy_tree
    
    
    
    cdef get_param_values(self):
        '''Builds parameters' dictionaries: one with memory locations and values 
        and one with strings and values'''

        cdef dict params_values = {}
        cdef Node par
        cdef dict names_values = {}
        for par in self.params:
            #print(par.value)
            params_values[par] = par.value
            names_values[par.function.get_description()] = par.value
        return params_values, names_values

    def get_param_dicts(self):
        return self.param_values, self.name_values

        #self.param_values = self.init_params(init_val = 1)
    cdef init_params(self, cnp.ndarray[cnp.float64_t, ndim=1] init_val):
        '''
        CANDIDATE FOR DELETION
        cdef Node par
        for par in self.params:
            par.set_value(init_val)
        '''
        pass

    #def?
    cpdef get_root(self):
        """Access Tree's root from Python"""
        return self.root
    #def?
    cpdef get_leaves(self):
        """Access Tree's leaves from Python"""
        return self.root.get_leaves()
     
    cpdef get_params(self, params_list = None):
        """How many constants in the Tree?"""
        if params_list is None:
            params_list = []
        
        for leaf in self.leaves:
            #print(leaf.get_fun().get_description())
            if leaf.is_param():
                
               
                params_list.append(leaf)
        self.root.sort_params(params_list)        
        return params_list
    #def?

    cpdef node_count(self):
        return self.root.node_count()


    cpdef get_domain(self):
        """Access root's function domain from Python"""
        return self.domain
    #cdef?
    cpdef is_defined(self, x=None):
        '''Applies real_functionC's is_defined on self.root'''
        if x is None:

            return self.root.function.is_defined(self.domain, self.name_values)
        else:
            
            return self.root.function.is_defined(x, self.name_values)


    cpdef evaluate(self,  object x=None):
        """Evaluate root on its domain or on a specified one"""
        if x is None:

            return self.root.function.evaluate(self.domain, self.name_values)
        else:
            
            return self.root.function.evaluate(x, self.name_values)

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
        """Access root function description from Python""" 
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
        self.leaves = self.get_leaves()
        #print('before', self.name_values)
        self.params = self.get_params()
        
        self.param_values, self.name_values = self.get_param_values()
        #print('after', self.name_values)
        #self.update_tree_description()
        #for j in range (0, len(self.get_params())):
        #    print(self.get_params()[j].get_fun().get_description())

    cpdef compute_fitness_tree(self, y):
        self.root.fitness_tree(y, self.name_values)     
        
    cdef build_fitness_tree(self, Node node, str prefix='', bint is_left=True):
        if node.is_leaf():
            return f"{prefix}{'└── ' if is_left else '├── '}{str(node.fitness)}\n"
        
        result = f"{prefix}{'└── ' if is_left else '├── '}{node.fitness}\n"
        prefix += '    ' if is_left else '│   '

        result += self.build_fitness_tree(node.left, prefix, True)
        result += self.build_fitness_tree(node.right, prefix, False)

        return result  

    def print_fitness_tree(self):
        print(self.build_fitness_tree(self.root))

    cdef compute_posterior_tree(self, y):
        self.root.posterior_tree(y, self.name_values)     
        
    cdef build_posterior_tree(self, Node node, str prefix='', bint is_left=True):
        if node.is_leaf():
            return f"{prefix}{'└── ' if is_left else '├── '}{str(node.posterior)}\n"
        
        result = f"{prefix}{'└── ' if is_left else '├── '}{node.posterior}\n"
        prefix += '    ' if is_left else '│   '

        result += self.build_posterior_tree(node.left, prefix, True)
        result += self.build_posterior_tree(node.right, prefix, False)

        return result  

    def print_posterior_tree(self):
        print(self.build_posterior_tree(self.root))

    cdef compute_superficial_posterior(self, y):
        self.root.set_posterior(y, self.name_values)

    cpdef get_superficial_posterior(self, y):
        if self.root.posterior is None:
            self.compute_superficial_posterior(y)
        print('computed posterior', self.root.posterior)
        return self.root.posterior
            
    cpdef optimize(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, 
        cnp.ndarray[cnp.float64_t, ndim=1] y_data, double lambda_reg, 
        str method_str='L-BFGS-B', bint use_gradients=False):
        cdef dict input_pars = self.name_values
        return self.root.optimize(x_data, y_data, input_pars, lambda_reg, method_str, use_gradients)
    
    cdef _update_parameters(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, 
        cnp.ndarray[cnp.float64_t, ndim=1] y_data, double lambda_reg, 
        str method_str='L-BFGS-B', bint use_gradients=False):
        '''Finds optimal parameters and sets them in the tree'''
        
        cdef dict input_pars = self.name_values
        cdef object result
        cdef dict optimized_params
        cdef Node par
        
        result = self.root.optimize(x_data, y_data, input_pars, lambda_reg, method_str, use_gradients)
        optimized_params = dict(zip(input_pars.keys(), result.x))
        
        for par in self.params:
            par.set_value(optimized_params[par.function.get_description()])
        
        self.param_values, self.name_values = self.get_param_values()
        self.tree_update(self.root)
        print('Updated parameters to: ', self.name_values) #comment out in production
        
    cpdef update_parameters(self, cnp.ndarray[cnp.float64_t, ndim=1] x_data, 
        cnp.ndarray[cnp.float64_t, ndim=1] y_data, double lambda_reg, 
        str method_str='L-BFGS-B', bint use_gradients=False):
        self._update_parameters(x_data, y_data, lambda_reg, method_str, use_gradients)
    cpdef bint is_valid(self, object x=None, double max_abs=1e12):
        """
        Strict validity: evaluation must succeed, be finite, and not explode.
        """
        return self.root.is_valid(x, max_abs)
        


    cdef _root_replacement(self, object replacement=None, int replace_all = 0, dict kwargs= None,
     str op = None, int new_depth=1, bint ensure_valid=True):
        cdef Node old_root = deepcopy(self.root)
        if replace_all==0:
            if isinstance(replacement, sp.core.expr.Expr):
                tmp_tree=_expr_to_tree(replacement, self.domain, self.functions)
                self.root = tmp_tree.root
            elif isinstance(replacement, Node):
                self.root=replacement
            elif isinstance(replacement, Tree):
                self.root = replacement.root
            else:    
                self.root = build_tree(max(0, min(new_depth, self.depth)), self.functions, self.domain,
                            Npars=self.Npars, value=self.value,
                            alpha=self.alpha, sigma=self.sigma) #riguardare per distribuire depth e Npars (specialmente Npars)
        elif replace_all==1:
            if kwargs is not None:
                self.root.left = Node(**kwargs)
            else:
                self.root.left = build_tree(max(0, new_depth), self.functions, 
                        self.domain, Npars=random.choice((0,1)))
            if op is not None:
                self.root.assign_operation(op)
        elif replace_all==2:
            if kwargs is not None:
                self.root.right = Node(**kwargs)
            else:
                self.root.right = build_tree(max(0, new_depth), self.functions, 
                        self.domain, Npars=random.choice((0,1)))
            if op is not None:    
                self.root.assign_operation(op)
        self.tree_update(self.root)
        if ensure_valid and not self.is_valid(self.domain):
            # revert if invalid
            self.root = old_root
            self.tree_update(self.root)
            return False
        return True
 

#cdef --> is cpdef for testing purposes only
cpdef build_tree(int depth, list functions, cnp.ndarray[cnp.float64_t, ndim=1] x, 
        str output_type=None, int Npars=0, object leaves_count = None, object arr_indices = None, 
             dict params_used=None, object value = 1.0, double T=0, double alpha=0.01, object sigma=1.00):
        '''Builds a Tree from scratch'''
        if isinstance(value, (float, int)):
            assign_value = np.full((Npars,), value, dtype=np.float64)
        elif isinstance(value, np.ndarray):
            if value.shape[0] > Npars:
                assign_value = value[:Npars].astype(np.float64)
            elif value.shape[0] < Npars:
                assign_value = np.pad(value, (0,Npars - value.shape[0]), mode='constant', constant_values=(1.0)).astype(np.float64)
            elif value.shape[0]==Npars:
                assign_value = value.astype(np.float64)
        if leaves_count is None:
            leaves_count = [-1]
        cdef str func_str
        cdef list par_list = parameter_parser(Npars)
        cdef int max_leaves = int(2**depth)
        cdef str param_name
        cdef bint first_call = False
        cdef Node return_node
        if arr_indices is None:
            raw_indices = np.linspace(0, max_leaves - 1, Npars, dtype = float)
            noise = np.random.uniform(-0.5, 0.5)
            arr_indices = np.floor(raw_indices + noise).astype(int)
            arr_indices = np.sort(np.where(arr_indices<0, arr_indices + max_leaves, arr_indices))
            arr_indices = np.unique(arr_indices)
  
            first_call = True

            if Npars > max_leaves:
                print('More parameters than leaves --> constant function')
                Npars = max_leaves

            if params_used is None:
                params_used = {}


        if depth==0:
            leaves_count[0] += 1

            if leaves_count in arr_indices:
                if len(params_used) < Npars:
                    param_name = par_list[len(params_used)]
                    return_node = Node(real_functionC.RealFunctionC.from_string(param_name, x, compute_grad = False),
                            alpha=alpha, sigma=sigma)
 

                  
                    return_node.set_value(value=assign_value[len(params_used)])
                    params_used[param_name] = return_node                   
                    return return_node 
                else:
                    print('ParamsError')
            func_str = random.choice(functions)
            return_node =Node(real_functionC.RealFunctionC.from_string(func_str, x, compute_grad = False),
                              alpha=alpha, sigma=sigma) 
            return return_node 
        '''
                for p in par_list:
                    if p not in params_used:
                        param_name = p
 
                        params_used.append(param_name) 
                        break
                else:
                    func_str = random.choice(functions)
                    return Node(real_functionC.RealFunctionC.from_string(func_str, x)) 



                return Node(real_functionC.RealFunctionC.from_string(param_name, x))

            func_str = random.choice(functions)
            return Node(real_functionC.RealFunctionC.from_string(func_str, x))
        '''
        #Recursion needed
        cdef Node left = build_tree(depth - 1, functions, x,
                'node', Npars, leaves_count,  arr_indices, params_used, value=value,
                alpha = alpha, sigma = sigma)
        cdef Node right = build_tree(depth - 1, functions, x, 
                'node', Npars, leaves_count, arr_indices, params_used, value=value,
                alpha = alpha, sigma = sigma)
       
       
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
            op_symbol = 'o' 
        if output_type is None or output_type=='node' or output_type=='Node':
            
            return_node = Node(function=func, left=left, right=right, symbol=op_symbol, alpha=alpha, sigma=sigma) 
            return return_node 
        else:
            return Tree(depth, functions, x, 
                    node =Node(function=func, left=left, right=right, symbol=op_symbol), T=T, alpha = alpha, sigma = sigma)

cdef parameter_parser(int Npars):
    """Create parameter names list(str())"""
    cdef list parameters = [None]*Npars
    cdef str prefix = 'a'
    cdef int i
    cdef char **buffer = <char **>malloc(Npars * sizeof(char *))
    try:
        for i in prange(Npars, nogil=True):
            buffer[i] = <char *>malloc(16)
            if buffer[i]==NULL:
                printf("ohno NULL BUFFER IN PARAMETER PARSER")
            snprintf(buffer[i], 16, b"a%d", i)

        for i in range(Npars):
            parameters[i] = buffer[i].decode('utf-8')
    
    finally:
        for i in prange(Npars, nogil=True):
            if buffer[i] != NULL:
                free(buffer[i])
        free(buffer)
    '''
    for i in range(Npars):
        parameters.append(f"{prefix}{i}")
    '''
    return parameters

#Naming parameters requires fix!    
cdef new_par(list existing_params, str user_choice=None):
    '''Add a new parameter named with a specific logic'''
    existing_params = [item for sublist in existing_params for item in (sublist if isinstance(sublist, list) else [sublist])]
    if len(existing_params) > 0:
        param_pattern = re.compile(r"([a-zA-Z]+)(\d+)")
        param_names = [param_pattern.match(param).groups()[0] for param in existing_params if param_pattern.match(param)]
        
        #unique_prefixes = set(param_names)
        if user_choice not in param_names or user_choice is None:
            user_choice = random.choice(param_names)
       


        indices = [int(param_pattern.match(param).groups()[1]) for param in existing_params if param.startswith(user_choice)]
        new_index = max(indices) + 1 if indices else 0

        return f"{user_choice}{new_index}"
    
    user_choice = user_choice if user_choice else 'a'
    return f"{user_choice}0"

cdef _expr_to_tree(object expr, cnp.ndarray[cnp.float64_t, ndim=1] domain, list function_list, dict params_used = None):
    """
    Sympy expr to Tree

    Parameters:
        expr (sympy.Expr): expression to translate
        domain: function domain

    Returns:
        Tree
    """
    cdef Tree left_tree, right_tree, arg_tree
    cdef Node root_node
    cdef real_functionC.RealFunctionC func, root_func, composed_func
    
    if params_used is None:
        params_used = {}
    print(params_used.values()) 
    if expr.is_Number:
        param_name = new_par(list(params_used.values()))
        print(param_name)
        if str(expr) not in params_used: 
            params_used[str(expr)] = [param_name]
        else:
            params_used[str(expr)].append(param_name)
 
        root_node = Node(real_functionC.RealFunctionC.from_string(param_name, domain))
        root_node.set_value(np.float64(expr))
        return Tree(0, function_list, domain, root_node)

    if isinstance(expr, sp.Symbol):
        symbol_str = str(expr)
        if symbol_str == 'x':
            func = real_functionC.RealFunctionC.from_string(str(expr), domain)
            root_node = Node(function = func)
        else:
            if symbol_str not in params_used:
                param_name = new_par(list(params_used.values()))
                params_used[symbol_str] = param_name
            else:
                param_name = params_used[symbol_str]
            root_node = Node(real_functionC.RealFunctionC.from_string(param_name, domain))

        return Tree(root_node.node_depth, function_list, root_node.domain, root_node)

    if isinstance(expr, sp.Add):
        left_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
        right_tree = _expr_to_tree(expr.args[1], domain, function_list, params_used)
        root_func = left_tree.root.get_fun() + right_tree.root.get_fun()
        root_node = Node(root_func, left_tree.root, right_tree.root, "+")
        return Tree(root_node.node_depth, function_list, root_node.domain, root_node)

    if isinstance(expr, sp.Mul):
        left_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
        right_tree = _expr_to_tree(expr.args[1], domain, function_list, params_used)
        root_func = left_tree.root.get_fun() * right_tree.root.get_fun()
        root_node = Node(root_func, left_tree.root, right_tree.root, "*")
        return Tree(root_node.node_depth, function_list, root_node.domain, root_node)

    if isinstance(expr, sp.Pow):
        left_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
        right_tree = _expr_to_tree(expr.args[1], domain, function_list, params_used)
        root_func = left_tree.root.get_fun() ** right_tree.root.get_fun()
        root_node = Node(root_func, left_tree.root, right_tree.root, "**")
        return Tree(root_node.node_depth, function_list, root_node.domain, root_node)

    if isinstance(expr, sp.Function):
        func_name = ''.join((expr.func.__name__, '(x)'))
        print('NAME: ', func_name)
        if func_name in function_list:
            root_function = real_functionC.RealFunctionC.from_string(func_name, domain) 
            if len(expr.args)==1 and expr.args[0] in expr.free_symbols:
                root_node = Node(root_function)
                return Tree(0, function_list, domain, root_node)
            arg_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
            composed_function = root_function.compose(arg_tree.root.get_fun())
            #make subnode for composed_func
             
            print('last stage: ', composed_function, root_function, arg_tree)
            root_node = Node(composed_function, Node(root_function), arg_tree.root, "o")
            return Tree(root_node.node_depth, function_list, root_node.domain, root_node)
        else:
            raise NotImplementedError(f"Conversion {func_name} not supported") 

    raise NotImplementedError(f" Expression{type(expr)} not supported.")

def expr_to_tree(expr, domain, function_list):
    return _expr_to_tree(expr, domain, function_list)

