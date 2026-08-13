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
import warnings

warnings.filterwarnings('ignore', message=r'invalid value encountered.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'divide by zero encountered.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'overflow encountered.*', category=RuntimeWarning)
warnings.filterwarnings('ignore', message=r'underflow encountered.*', category=RuntimeWarning)

cdef object _LOG = np.log
cdef double _PI = np.pi
'''
 module-level kill switch for
the BIC complexity penalty in Tree._shortcut_bic. Default True =
bit-identical legacy behaviour; toggled ONLY by the ablation harness.
'''
cdef bint _BIC_PENALTY_ON = True

cpdef void set_bic_penalty(bint on):
    """Ablation harness only: enable/disable the BIC complexity penalty
    (default enabled). Affects the EMC scoring path (_shortcut_bic);
    the burn-in ranking is deliberately unaffected, so all ablation
    arms share identical seeding."""
    global _BIC_PENALTY_ON
    _BIC_PENALTY_ON = on

cpdef bint get_bic_penalty():
    """ 
    read the ablation switch — the joint
    reduction's penalty reconstruction (groveC._md_dataset_penalty) must
    mirror _shortcut_bic's branch or min/softmin decomposes L_j wrongly
    under ablation.
    """
    return _BIC_PENALTY_ON
'''
module-scope flag consumed by
_expr_to_tree's numeric-literal branch. Set ONLY by the public
expr_to_tree wrapper (try/finally restored), so the ~15 recursive call
sites stay untouched — same single-threaded module-global convention as
 the RNG handles. False = bit-identical legacy (literals -> free params).
'''
cdef bint _PIN_NUMBERS = False
cdef double _NAN = np.nan

# Per-process RNG handles used by every randomized routine in this module
# (build_tree, select_random_leaf/node, _root_replacement). Default to the
# stdlib `random` module and the numpy legacy module so unseeded callers
# behave as before. Grove installs its own (random.Random, default_rng)
# pair via set_rng() to make tree construction reproducible per the
# Grove's seed -- without this hook, build_tree would silently pull from
# Python's global RNG and break determinism even when gp_movesC is
# correctly seeded.
cdef object _PY_RNG = random
cdef object _NP_RNG = np.random

cpdef void set_rng(object py_rng, object np_rng) except *:
    global _PY_RNG, _NP_RNG
    _PY_RNG = py_rng if py_rng is not None else random
    _NP_RNG = np_rng if np_rng is not None else np.random

cpdef tuple get_rng():
    return (_PY_RNG, _NP_RNG)

cdef class Node:
    '''Node of a Tree: either a leaf, or an operation on two other nodes'''    
    def __init__(self, real_functionC.RealFunctionC function=None, object left=None, object right=None,
            str symbol=None, double alpha = 0.01, object sigma = 1.00, object value = None,
            object theta = None):
        self.left = left
        self.right = right
        self.op_symbol = symbol
        self.function = function if function is not None else self.fetch_fun()
        self.node_depth = self.depth()
        if self.function is not None:
            self.domain = self.function.domain
            self.bis_param = self.is_param()
        else:
            self.domain = None
            self.bis_param = False
        if self.bis_param and value==None:
            self.value = 1
        elif self.bis_param and value is not None:
            self.value = value
        else:
            self.value = np.nan
        '''
        internal-parameter values of an opaque atomic
        leaf live on the Node (deepcopy-safe, like parameter-leaf .value
        above). Auto-init from the function's reference theta0 so
        hand-built Nodes are immediately consistent; move code overwrites
        with its warm-start draws.
        '''
        if theta is not None:
            self.theta = dict(theta)
        elif (self.function is not None
                and self.function.atom_symbol is not None):
            self.theta = dict(self.function.theta0) \
                if self.function.theta0 is not None else {}
        else:
            self.theta = None
        self.fitness = None
        self.posterior = None
        self.prior = None
        self.alpha = alpha #keep alpha as it could weight an externally computed prior
        self.sigma = sigma

    def __copy__(self):
        cdef Node shallow = Node(copy(self.function), copy(self.left), copy(self.right),
            self.op_symbol, alpha=self.alpha, sigma=self.sigma,
            theta=(None if self.theta is None else dict(self.theta)))
        
        shallow.flex_spec = self.flex_spec
        shallow.flex_names = (None if self.flex_names is None
                              else list(self.flex_names))
        return shallow

    def __deepcopy__(self, memo1):
        cdef Node node_copy = Node(deepcopy(self.function, memo1),
                deepcopy(self.left, memo1), deepcopy(self.right, memo1),
                self.op_symbol, alpha=self.alpha, sigma=self.sigma,
                value=self.value,
                theta=(None if self.theta is None else dict(self.theta))
                )
        '''
        forward the scaffold identity. The copied function's
        closure still references the ORIGINAL slot function object --
        values are identical at copy time (closures are state-free wrt
        theta), and any structural slot change on the copy goes through
        update_func, which rebuilds the composition over the COPIED
        slot, so stale closures cannot leak.
        '''
        node_copy.flex_spec = self.flex_spec
        node_copy.flex_names = (None if self.flex_names is None
                                else list(self.flex_names))
        memo1[id(self)] = node_copy
        return node_copy

    cdef double _log_marginal_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params=None, object x=None):
        return self.function._log_marginal_bic(y, params, x)

    
    cdef double _log_sigma_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params,
            cnp.ndarray x, double sigma):
            return self.function._log_sigma_bic(y, params, x, sigma)

    cdef double _log_sigma_array_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict params,
            cnp.ndarray x, cnp.ndarray[cnp.float64_t, ndim=1] sigma):
            return self.function._log_sigma_array_bic(y, params, x, sigma)

    def get_root(self):
        print(self.root)
    def get_domain(self):
        '''Obtain domain from Python.'''
        return self.domain
    def get_description(self):
        return self.function.get_description()    
    # def?
    cpdef is_param(self):
        '''Is Node a constant function?'''
        '''A pinned numeric literal (expr_to_tree pin_numbers=True) is a
        constant map that must NOT count as a parameter. The flag reads
        False on every ordinary function.'''
        if self.is_leaf() and len(self.function.variables)==0 \
                and not self.function.pinned_const:
            return True
        #elif self.is_leaf() and len(self.function.variables)>0:

        return False
    
    cpdef is_constant(self):
        '''A flex scaffold is NEVER foldable as a constant, even when
        its slot is momentarily variable-free: folding would evaluate
        np.float64(Psi(...)) on an undefined Function head, crashing,
        and would destroy the immortal scaffold.'''
        return ((len(self.function.variables)==0) and not self.is_leaf()
                and self.flex_spec is None)

    # ------------------------------------------------------------------
    '''The atomic-leaf taxonomy (DESIGN §1, §9 item 2). One object
    class covers all three kinds of leaf, distinguished only by how many
    internal parameters they carry:

      function leaf   : k = 0                  (is_param False, is_atomic False)
      parameter leaf  : k = 1, expr == symbol  (is_param True,  is_atomic False)
      opaque leaf     : k = k_psi >= 1, from an AtomSpec template
                                               (is_param False, is_atomic True)
    '''
    # ------------------------------------------------------------------
    cpdef is_atomic(self):
        '''Is this leaf an opaque atomic primitive (instantiated from an
        AtomSpec template)? Plain parameter leaves and function leaves
        return False.'''
        return (self.is_leaf() and self.function is not None
                and self.function.atom_symbol is not None)

    cpdef bint is_flex(self):
        '''Is this node a TRANSPARENT (flexible) atom: a frozen scaffold
        (real_functionC.FlexAtomSpec) whose mutable slot subtree lives
        in self.left, with self.right None? It is neither a leaf nor
        opaque-atomic. The outer move kernel treats it as an
        indivisible, immortal unit; only gp_movesC.flex_slot_mutation
        edits the slot. False on every ordinary node.'''
        return self.flex_spec is not None

    cpdef int leaf_k(self):
        '''Continuous parameters this LEAF contributes to the tree's
        effective dimension (the k_psi of DESIGN §5): 0 for a function
        leaf, 1 for a plain parameter leaf, k_psi for an opaque leaf.
        0 on internal nodes (their functions merely inherit leaf
        parameters; counting there would double-count).'''
        if not self.is_leaf() or self.function is None:
            return 0
        if self.function.atom_symbol is not None:
            return len(self.function.params_strs)
        if self.bis_param:
            return 1
        return 0

    cpdef list leaf_param_names(self):
        '''Names of the parameters carried by this leaf, in the leaf's
        deterministic order: [] for a function leaf, [description] for a
        plain parameter leaf, the instantiated internal names for an
        opaque leaf.'''
        if not self.is_leaf() or self.function is None:
            return []
        if self.function.atom_symbol is not None:
            return list(self.function.params_strs)
        if self.bis_param:
            return [self.function.get_description()]
        return []

    cpdef str vocab_symbol(self):
        '''Canonical Pólya-vocabulary key of this leaf: the atom template's
        registered symbol for opaque leaves (ONE cell per template — the
        instantiated description varies with the fresh parameter names and
        must never be registered), the plain description otherwise.'''
        if self.function is not None and self.function.atom_symbol is not None:
            return self.function.atom_symbol
        return self.function.get_description()

    cdef rename_atomic(self, dict name_map):
        '''Rename this opaque leaf's internal parameters (function expr AND
        theta keys) per name_map (old -> new). No-op on non-atomic nodes.'''
        cdef dict new_theta
        cdef str key
        if not self.is_atomic():
            return
        self.function = real_functionC.rename_leaf_params(self.function, name_map)
        new_theta = {}
        if self.theta is not None:
            for key in self.theta:
                new_theta[name_map.get(key, key)] = self.theta[key]
        self.theta = new_theta

    cdef sort_params_by_names(self, list names):
        '''Impose `names` (str list) as this node's function's parameter
        order. Generalizes sort_params to plain name lists so opaque-leaf
        internal names — which have no dedicated parameter Node — can be
        appended after the plain parameter leaves' names.'''
        cdef list syms = [sp.symbols(n) for n in names]
        self.function.sort_params_list(syms)

    cdef set_alpha(self, double newalpha):
        self.alpha = newalpha

    cdef set_sigma(self, object newsigma):
        self.sigma = newsigma

    cpdef void set_sigma_recursive(self, object new_sigma):
        self.set_sigma(new_sigma)
        if not self.is_leaf():
            self.left.set_sigma_recursive(new_sigma)
            # Flex scaffolds are unary (right is None). For an ordinary
            # node this guard is a no-op: non-leaf means both children.
            if self.right is not None:
                self.right.set_sigma_recursive(new_sigma)
    
   
    
    cdef set_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters):
        self.fitness = self.evaluate_node_fitness(y, parameters)
    
    cdef set_posterior(self,cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters, double penalty=0.0):
        self.posterior = self.evaluate_node_posterior(y, parameters, penalty)
    
 
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
        #print(operation, left.function.get_description(), right.function.get_description())
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
        #print('fetched function: ', auxiliary_fun.get_description())
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
         
    cdef optimize(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, str method_str='L-BFGS-B', bint use_gradients=False):
        if use_gradients:
            return self.function._optimize_with_grad(x_data, y_data, input_pars, method_str)
        else:
            return self.function.optimize(x_data, y_data, input_pars, method_str='Powell')

    cdef optimize_sigma(self, cnp.ndarray x_data, cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        dict input_pars, cnp.ndarray[cnp.float64_t, ndim=1] sigma, str method_str='L-BFGS-B', bint use_gradients=False):
        return self.function.optimize_sigma(x_data, y_data, input_pars, method_str = 'Powell', sigma=sigma)
        
    cdef set_prior(self):
        self.prior = self.function.prior
   
    # DANGEROUS!!! Use cases need be minimized!
    cdef reinit_to_par(self, list existing_pars, str user_prefix=None, double value=1.0):
        '''Re-initializes a node to a constant.'''
        cdef object new_param = new_par(existing_pars, user_prefix) 
        self.function = real_functionC.RealFunctionC.from_string(str(new_param), self.domain)
        #print('REINITIALIZATION OF NODE ', self, ' TO PARAMETER!')
        self.left, self.right, self.op_symbol, self.node_depth, self.value = None, None, None, 0, value
    
    cdef _rebuild_ancestor_functions(self, Node node):
        cdef Node parent = self.find_parent(node)
        while parent is not None:
            parent.function = parent.fetch_fun()  # Rebuild from children
            parent.function._update_symbols()     # Refresh params/variables
            parent = self.find_parent(parent)

    cdef evaluate_node_fitness(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters):
        return self.function.fitness(y, parameters)

    cdef evaluate_node_posterior(self, cnp.ndarray[cnp.float64_t, ndim=1] y, dict parameters, double penalty=0.0):
        
        return self.function.posterior(y, parameters, self.fitness, self.prior, 
                                       self.sigma, None, penalty)

    cpdef assign_operation(self, operation, bint brute_force=False):
        '''Assigns an operation between self.left and self.right'''
        if not self.bis_param or brute_force:
            self.op_symbol = operation
            self.update_func()
        else:
            raise ValueError("Parameters have no children nor operations.")
   
    cpdef assign_left(self, node, bint brute_force=False):
        '''Assigns left node'''
        if not self.bis_param or brute_force:
            self.left = node
        else:
            raise ValueError("Parameters have no children nor operations.")
   
  
    cpdef assign_right(self, node, bint brute_force=False):
        '''Assigns right node'''
        if not self.bis_param or brute_force:
            self.right = node
        else:
            raise ValueError("Parameters have no children nor operations.")
   
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
            '''A unary flex scaffold descends into its slot WITHOUT
            consuming RNG. The check sits before the draw so ordinary
            trees keep their historical RNG stream bit for bit.'''
            if self.right is None:
                return self.left.select_random_leaf()
            if _PY_RNG.choice([True, False]):
                return self.left.select_random_leaf()
            else:
                return self.right.select_random_leaf()
    #cdef?
    cpdef select_random_node(self, threshold=0.1):
        """Select a random node with probability influenced by node fitness (NOT IMPLEMENTED YET)"""
        #cdef float fitness = compute_fitness(node)
        cdef float fitness = 0.8

        cdef float selection_prob = max(1.0 - fitness, threshold)
        
        if _PY_RNG.random() < selection_prob:
            return self
        
        if self.is_leaf():
            return self

        '''Same as in select_random_leaf: a unary flex scaffold descends
        left without consuming RNG, and the guard precedes the draw so
        the ordinary stream is unchanged.'''
        if self.right is None:
            return self.left.select_random_node(threshold)
        if _PY_RNG.random() < 0.5:
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
        if self.flex_spec is not None:
            '''Rebuild the numeric composition over the CURRENT slot
            child. Structural slot changes always replace function
            objects, so this is the single point that keeps the
            scaffold's closure and expr in sync. The op-symbol chain
            below would leave the function stale, since op_symbol is
            None here.'''
            self.function = _flex_function(self.flex_spec, self.flex_names,
                                           self.left)
            return
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
            '''A flex scaffold is unary (right is None), and its SLOT
            leaves — the ones under left — ARE tree leaves, so slot
            parameters flow into params/name_values unchanged. The
            None-guard is a no-op for an ordinary node, where non-leaf
            means both children are present.'''
            if self.right is not None:
                self.right.get_leaves(leaves_list = leaves_list)

        return leaves_list

    cpdef node_count(self):
        
        cdef int count = 1  
        if self.left:
            count += self.left.node_count()
        if self.right:
            count += self.right.node_count()
        return count
        
    cpdef flag_constants(self, list constants=None):
        if constants is None:
            constants = []
        
        #cdef list my_symbols

        if self.is_constant():
           
            constants.append((self, self.function.symbols))
        else:
            
            if self.left is not None:
                self.left.flag_constants(constants)
            
            if self.right is not None:    
                self.right.flag_constants(constants)
        
        return constants

    cdef replace_constants_with_pars(self, dict name_values=None):
        """Fold every constant subtree into a fresh parameter leaf seeded
        with the constant's numeric value at the current `name_values`.

        Returns
        -------
        bint
            True  -> all constants folded successfully.
            False -> at least one constant evaluated to a complex number,
                     NaN, or could otherwise not be converted to float64.
                     The caller is expected to discard the tree (the
                     parameters have drifted into a pathological basin
                     where the symbolic expression is no longer real).
                     The tree may be left partially modified: do NOT
                     continue using it.
        """
        cdef tuple constant
        cdef double value
        cdef Node constant_node
        cdef object raw_value
        cdef list symbols_values = _symbol_values(name_values) if name_values else None
        cdef list existing_params = list(name_values.keys())
        cdef set constant_pars

        for constant in self.flag_constants():
            
            constant_node = constant[0]
            constant_pars = constant[1]

            for sy in constant_pars:
                sy_str = str(sy)
                if sy_str in existing_params:  # May need str() depending on types
                    existing_params.remove(sy_str)
            
            if symbols_values:
                
                subs_dict = [(sy, val) for sy, val in symbols_values if sy in constant_pars]
                # Guard: sympy may return a complex (e.g. (-x)**0.5 with x>0
                # after substitution) or a non-finite number. np.float64() of
                # a complex raises TypeError; that has historically killed
                # the worker mid-burn-in. Signal failure to the caller so
                # the offending tree gets discarded.
                raw_value = constant_node.function.expr.subs(subs_dict)
                try:
                    value = np.float64(raw_value)
                except (TypeError, ValueError):
                    return False
                if not np.isfinite(value):
                    return False
            else:
                
                value = 1.0

            constant_node.reinit_to_par(existing_params, user_prefix=None, value=value)
            self._rebuild_ancestor_functions(constant_node)
        return True



    
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
            cnp.ndarray x,
            Node node=None, int Npars = 0, object value=1,
            double T=0, double alpha=0.01, object sigma = 1.0,
            double g_tau=1.0, list operations=None):
        '''Tree class initialization.

        `operations` (op symbols) is forwarded to build_tree when the
        root is built from scratch, so a kernel with a restricted op
        vocabulary starts INSIDE its own support. An out-of-vocabulary
        op in the initial tree could never be legally removed, since
        every reverse density for it is zero. Default None gives the
        historical 6-op set.'''
        self.T = T
        self.domain = real_functionC._domain_fixer(x)
        self.depth = depth
        self.functions = functions
        self.Npars=Npars
        self.alpha = alpha
        self.sigma = sigma
        self.value = value
        self.g_tau = g_tau
        if node==None:
            self.root = build_tree(depth, functions, self.domain, Npars=Npars,
                        value=value, alpha=self.alpha, sigma=self.sigma,  g_tau = self.g_tau,
                        operations=operations)
        else:
            self.root = node
            self.depth = node.depth()
        self.current_description = self.root.function.get_description()
        self.leaves = self.get_leaves()
        self.params = self.get_params()
        self.param_values, self.name_values, self.penalty = self.get_param_values()
        self.fitness_tree = None
    def __copy__(self):
        # Forward g_tau: it used to be dropped here, so copies silently
        # re-defaulted to 1.0. Positional slot 10, after sigma.
        return Tree(self.depth, self.functions, self.domain, self.root,
            self.Npars, self.value, self.T, self.alpha, self.sigma,
            self.g_tau)

    def __deepcopy__(self, memo):
        '''Forward g_tau. Grove._propose deepcopies before every move,
        so without this the birth warm-start draw N(0, g_tau^2) used the
        wrong scale: 1.0 instead of the configured std(y).'''
        cdef Tree copy_tree = Tree(
                self.depth,
                deepcopy(self.functions, memo),
                deepcopy(self.domain, memo),
                deepcopy(self.root, memo),
                self.Npars, self.value, self.T, self.alpha, self.sigma,
                self.g_tau
                )
        #copy_tree.tree_update(copy_tree.root)  # Ensure descriptions and params are updated
        #copy_tree.name_values = deepcopy(self.name_values, memo)
        memo[id(self)] = copy_tree
        return copy_tree
    
    cpdef void set_g_tau(self, double tau) except *:
        
        if tau <= 0.0:
            raise ValueError("set_g_tau: tau must be > 0")
        self.g_tau = tau
    
    cdef get_param_values(self):
        '''Builds parameters' dictionaries: one with memory locations and values 
        and one with strings and values'''

        cdef dict params_values = {}
        cdef Node par
        cdef Node leaf
        cdef str nm
        cdef dict names_values = {}
        cdef double par_value = 0.0
        cdef double penalty = 0.0
        for par in self.params:
            #print(par.value)
            par_value = par.value
            params_values[par] = par_value
            names_values[par.function.get_description()] = par_value
            penalty += 1 #np.abs(par_value)
        '''Merge the opaque leaves' internal parameters AFTER the plain
        ones, matching the order get_params imposed on the root
        function. Each internal parameter counts 1 toward the penalty,
        exactly like a plain parameter leaf, and that is what makes
        _shortcut_bic's k = len(name_values) the DESIGN §5 effective
        dimension k(T) + sum_j k_psi_j with no further changes.'''
        for leaf in self.leaves:
            if leaf.is_atomic():
                for nm in leaf.leaf_param_names():
                    names_values[nm] = leaf.theta[nm]
                    penalty += 1
        '''Flex-scaffold internal parameters, in pre-order flex-node
        order, after the plain and opaque names — mirroring get_params.
        Each counts 1 toward the penalty, so _shortcut_bic's k stays the
        true effective dimension. On a tree without scaffolds the walk
        is empty and costs nothing.'''
        cdef Node fx_node
        cdef str fx_nm
        for fx_node in _collect_flex_nodes(self.root, []):
            for fx_nm in fx_node.flex_names:
                names_values[fx_nm] = fx_node.theta[fx_nm]
                penalty += 1
        return params_values, names_values, penalty

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
    

    cpdef void set_sigma(self, object new_sigma):
        self.sigma = new_sigma
        self.root.set_sigma_recursive(new_sigma)
        

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
        cdef list order_names = []
        cdef Node leaf
        for leaf in self.leaves:
            #print(leaf.get_fun().get_description())
            if leaf.is_param():


                params_list.append(leaf)
                order_names.append(leaf.function.get_description())
        '''Opaque atomic leaves carry internal parameters with no
        dedicated parameter Node. Append their names AFTER the plain
        ones, in leaf (pre-order) order — the identical order in which
        get_param_values inserts them into name_values — so the root
        function's parameter order, name_values and the optimizer's flat
        vector stay aligned. With no atomic leaves this reduces exactly
        to the historical sort_params(params_list) call.'''
        for leaf in self.leaves:
            if leaf.is_atomic():
                order_names.extend(leaf.leaf_param_names())
        # Flex internal names continue that same order, mirroring
        # get_param_values so root-function order, name_values and the
        # optimizer's flat vector stay aligned.
        cdef Node fx_node
        for fx_node in _collect_flex_nodes(self.root, []):
            order_names.extend(fx_node.flex_names)
        self.root.sort_params_by_names(order_names)
        return params_list
    #def?

    cpdef node_count(self):
        return self.root.node_count()

    cpdef get_sigma(self):
        return self.sigma

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

        if node.is_flex():
            # Unary scaffold: print the head plus internal names, then
            # the slot subtree (right is None).
            result = f"{prefix}{'└── ' if is_left else '├── '}{node.flex_spec.name}[{', '.join(node.flex_names)}]\n"
            prefix += '    ' if is_left else '│   '
            result += self.tree_to_string(node.left, prefix, True)
            return result

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
    cpdef tree_update(self, Node node = None):
        """Update the whole tree nodes' functions"""
        if node is None:
            node = self.root
        current_node = node
        while current_node is not None:
            current_node.update_func()
            current_node = self.root.find_parent(current_node)
            
        self.update_tree_description()
        self.leaves = self.get_leaves()
        #print('before', self.name_values)
        self.params = self.get_params()
        
        self.param_values, self.name_values, self.penalty = self.get_param_values()
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
    

    cdef replace_constants_with_pars(self):
        self.root.replace_constants_with_pars(self.name_values)
        self.tree_update(self.root)

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

    cdef compute_superficial_posterior(self, y, double lambda_reg = 0.0):
        self.root.set_posterior(y, self.name_values, self.penalty * lambda_reg)

    cpdef get_superficial_posterior(self, y, double lambda_reg = 0.0):
        if self.root.posterior is None:
            self.compute_superficial_posterior(y, lambda_reg)
        
        return self.root.posterior
            
    cpdef optimize(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str='L-BFGS-B', bint use_gradients=False):
        cdef dict input_pars = self.name_values
        return self.root.optimize(x_data, y_data, input_pars, method_str, use_gradients)
    
    cpdef double log_marginal_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
                           bint optimize=True,
                           bint use_gradients=False,
                           str method_str='L-BFGS-B'):
        return self._log_marginal_bic(y, optimize, use_gradients, method_str)
    cdef double _shortcut_bic(self, cnp.ndarray[cnp.float64_t, ndim=1] y,
			double mse):
        cdef int N = y.shape[0]
        cdef int k = len(self.name_values)
        '''Opt-in kill switch for the complexity penalty, giving
        pure-likelihood scoring. Default ON is the ordinary behaviour.
        One C bool branch, toggled only by the ablation harness via
        set_bic_penalty().'''
        if not _BIC_PENALTY_ON:
            if self.sigma is None:
                return -0.5 * N * (_LOG(2.0 * _PI * mse) + 1.0)
            return -0.5 * N * mse
        if self.sigma is None:
            '''sigma is PROFILED here (sigma_hat^2 = SSR/N), so it
            counts as an estimated parameter and the penalty is on k+1.
            This is a constant +0.5*log(N) shift across all models, so
            MH ratios are unchanged, but reported evidences and
            fixed-versus-profiled comparisons become honest.'''
            return -0.5 * N * (_LOG(2.0 * _PI * mse) + 1.0) - 0.5 * (k + 1) * _LOG(N)
        else:

            return -0.5 * N * mse - 0.5 *k * _LOG(N)
    cdef double _log_marginal_bic(self,
                                  cnp.ndarray[cnp.float64_t, ndim=1] y,
                                  bint optimize=True,
                                  bint use_gradients=False,
                                  str method_str='L-BFGS-B'):
        cdef double mse
        self.bic_loss_is_param_loss = True #placeholder to remove when using custom defined loss functions
        '''Gate on name_values, not self.params: an opaque-leaf tree can
        carry continuous parameters (k_psi >= 1) with ZERO plain
        parameter Nodes, and its theta must still be optimized. For
        atom-free trees the two gates are equivalent, since every
        parameter Node inserts a name.'''
        if self.bic_loss_is_param_loss and optimize and len(self.name_values)>0:
            mse = self._update_parameters_and_fitness(self.domain, y,
                                        method_str, use_gradients)
            '''Careful: this gate does NOT re-validate at theta_hat.
            is_valid takes no params and evaluates at the
            construction-time reference values — the CBC forensics
            proved it, PLAN §4.2b. The actual theta_hat pathology gate
            is the fitness() semantics themselves: in real_functionC any
            non-finite prediction yields _MSE_PENALTY, hence an
            effectively -inf posterior. This call is kept for its
            original structural checks (zoo, reference evaluation,
            magnitude cap) and for trajectory continuity.'''
            if not self.is_valid(self.domain):
                return -np.inf
            if self.sigma is not None and (not hasattr(self.sigma, '__len__')):
                mse /= (<double>self.sigma*<double>self.sigma)
            return self._shortcut_bic(y, mse)

        if optimize and len(self.name_values) > 0:

            self._update_parameters(self.domain, y,
                                    method_str, use_gradients)
            # Same theta_hat caveat as above, on the slow path.
            if not self.is_valid(self.domain):
                return -np.inf
        if self.sigma is None:
            return self.root._log_marginal_bic(y, self.name_values, self.domain)
        elif hasattr(self.sigma, '__len__'):
                '''Coerce at the typed boundary — a no-op view when the
                caller already holds a contiguous float64 vector — so a
                list or float32 sigma cannot raise TypeError deep inside
                the scorer. Behaviour for ndarray callers is
                unchanged.'''
                return self.root._log_sigma_array_bic(
                    y, self.name_values, self.domain,
                    np.ascontiguousarray(self.sigma, dtype=np.float64).ravel())
        else:
            return self.root._log_sigma_bic(y, self.name_values, self.domain, <double>(self.sigma))

    cdef _update_parameters(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str='L-BFGS-B', bint use_gradients=False):
        '''Finds optimal parameters and sets them in the tree'''

        cdef dict input_pars = self.name_values
        cdef object result
        cdef dict optimized_params
        cdef Node par
        cdef Node leaf
        cdef str nm
        #print(f'EXPR: {self.tree_get_description()}')
        #print(f'input_pars id: {id(input_pars)}, value: {input_pars}')
        #print(f'self.name_values id: {id(self.name_values)}')
        if not hasattr(self.sigma, '__len__'):
            result = self.root.optimize(x_data, y_data, input_pars, method_str, use_gradients)
        else:
            '''Argument order matters here. This call used to pass
            (input_pars, method_str, use_gradients, self.sigma) into
            Node.optimize_sigma(..., input_pars, SIGMA, method_str,
            use_gradients), so the str method_str landed in the typed
            sigma slot and raised TypeError — which made the
            per-point-sigma optimize path unreachable. Arguments now
            follow the declaration order, and sigma is coerced to a
            contiguous float64 vector at this boundary (a no-op for
            ndarray callers).'''
            result = self.root.optimize_sigma(
                x_data, y_data, input_pars,
                np.ascontiguousarray(self.sigma, dtype=np.float64).ravel(),
                method_str, use_gradients)
        optimized_params = dict(zip(input_pars.keys(), result.x))
        #print(self.tree_get_description(), result)
        for par in self.params:
            par.set_value(optimized_params[par.function.get_description()])
        # Write the optimizer's values back into the opaque leaves;
        # their names are keys of name_values, hence of optimized_params.
        for leaf in self.leaves:
            if leaf.is_atomic():
                for nm in leaf.leaf_param_names():
                    if nm in optimized_params:
                        leaf.theta[nm] = optimized_params[nm]
        # Same write-back for flex-scaffold internal theta.
        cdef Node fx_node
        for fx_node in _collect_flex_nodes(self.root, []):
            for nm in fx_node.flex_names:
                if nm in optimized_params:
                    fx_node.theta[nm] = optimized_params[nm]

        self.param_values, self.name_values, self.penalty = self.get_param_values()

        self.tree_update(self.root)
        #print('Updated parameters to: ', self.name_values) #comment out in production
       
    cdef double _update_parameters_and_fitness(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str='L-BFGS-B', bint use_gradients=False):
        '''Finds optimal parameters and sets them in the tree'''
        
        cdef dict input_pars = self.name_values
        cdef object result
        cdef dict optimized_params
        cdef Node par
        cdef Node leaf
        cdef str nm
        #print(f'EXPR: {self.tree_get_description()}')
        #print(f'input_pars id: {id(input_pars)}, value: {input_pars}')
        #print(f'self.name_values id: {id(self.name_values)}')
        if not hasattr(self.sigma, '__len__'):
            result = self.root.optimize(x_data, y_data, input_pars, method_str, use_gradients)
        else:
            # Same argument-order care as in _update_parameters above:
            # shifting them by one slot puts method_str in the typed
            # sigma buffer and raises TypeError.
            result = self.root.optimize_sigma(
                x_data, y_data, input_pars,
                np.ascontiguousarray(self.sigma, dtype=np.float64).ravel(),
                method_str, use_gradients)
        optimized_params = dict(zip(input_pars.keys(), result.x))
        #print(self.tree_get_description(), result)
        cdef double mse=result.fun
        for par in self.params:
            par.set_value(optimized_params[par.function.get_description()])
        # Same opaque-leaf write-back as _update_parameters.
        for leaf in self.leaves:
            if leaf.is_atomic():
                for nm in leaf.leaf_param_names():
                    if nm in optimized_params:
                        leaf.theta[nm] = optimized_params[nm]
        # And the same for flex-scaffold internal theta.
        cdef Node fx_node
        for fx_node in _collect_flex_nodes(self.root, []):
            for nm in fx_node.flex_names:
                if nm in optimized_params:
                    fx_node.theta[nm] = optimized_params[nm]

        self.param_values, self.name_values, self.penalty = self.get_param_values()
        self.tree_update(self.root)
        return mse

    cpdef update_parameters(self, cnp.ndarray x_data,
        cnp.ndarray[cnp.float64_t, ndim=1] y_data,
        str method_str='L-BFGS-B', bint use_gradients=False):
        self._update_parameters(x_data, y_data, method_str, use_gradients)

    # ------------------------------------------------------------------
    '''Explicit-argument scoring, for joint evidence over several
    datasets. _log_marginal_bic above is left untouched; this mirror
    substitutes the PASSED x at every site where that method reads
    self.domain: the optimizer x, both is_valid gates, and the three
    no-optimize scorer calls. n still comes from y.shape inside
    _shortcut_bic, and sigma routing still comes from self.sigma, which
    score_dataset below swaps around the call — one O(1) attribute
    assignment, never the node-recursive setter.'''
    # ------------------------------------------------------------------
    cdef double _log_marginal_bic_x(self,
                                    cnp.ndarray x,
                                    cnp.ndarray[cnp.float64_t, ndim=1] y,
                                    bint optimize=True,
                                    bint use_gradients=False,
                                    str method_str='Powell'):
        cdef double mse
        self.bic_loss_is_param_loss = True
        if self.bic_loss_is_param_loss and optimize and len(self.name_values) > 0:
            mse = self._update_parameters_and_fitness(x, y,
                                        method_str, use_gradients)
            # Same contract as the legacy path: is_valid evaluates at the
            # reference values ON THIS DATASET'S x (strictest-wins comes
            # from the caller looping every dataset); the theta_hat
            # pathology gate is the fitness() NaN-penalty semantics.
            if not self.is_valid(x):
                return -np.inf
            if self.sigma is not None and (not hasattr(self.sigma, '__len__')):
                mse /= (<double>self.sigma * <double>self.sigma)
            return self._shortcut_bic(y, mse)

        if optimize and len(self.name_values) > 0:
            self._update_parameters(x, y, method_str, use_gradients)
            if not self.is_valid(x):
                return -np.inf
        if self.sigma is None:
            return self.root._log_marginal_bic(y, self.name_values, x)
        elif hasattr(self.sigma, '__len__'):
            return self.root._log_sigma_array_bic(
                y, self.name_values, x,
                np.ascontiguousarray(self.sigma, dtype=np.float64).ravel())
        else:
            return self.root._log_sigma_bic(y, self.name_values, x,
                                            <double>(self.sigma))

    cpdef tuple score_dataset(self,
                              cnp.ndarray x,
                              cnp.ndarray[cnp.float64_t, ndim=1] y,
                              object sigma,
                              bint optimize=True,
                              bint use_gradients=False,
                              str method_str='Powell'):
        """Score THIS tree against ONE explicit dataset (x, y, sigma),
        independent of the construction-time domain and of the tree's
        configured sigma.

        Fits theta on the passed data (when optimize is set and the tree
        has parameters), leaves the fitted theta installed in the tree,
        and returns (post, dict(name_values) snapshot). The caller owns
        the snapshot, which is that dataset's theta_j. self.sigma is
        saved and restored around the call, so the tree's configured
        sigma routing is untouched on exit even if something raises.

        Cross-module callers: POSITIONAL ARGS ONLY."""
        cdef object saved_sigma = self.sigma
        cdef double post
        try:
            self.sigma = sigma
            post = self._log_marginal_bic_x(x, y, optimize,
                                            use_gradients, method_str)
        finally:
            self.sigma = saved_sigma
        return (post, dict(self.name_values))
    cpdef bint is_valid(self, object x=None, double max_abs=1e12):
        """
        Strict validity: evaluation must succeed, be finite, and not explode.
        """
        return self.root.is_valid(x, max_abs)

    cdef double _log_likelihood_at(self,
                               cnp.ndarray[cnp.float64_t, ndim=1] y,
                               dict cand_name_values,
                               double sigma):
        """
        Raw Gaussian log-likelihood at a *candidate* parameter assignment, WITHOUT
        mutating any tree state. Used by parameter_refresh: the PR move
        scores on un-profiled likelihood, so structural-move scoring
        (BIC, Polya prior) is intentionally bypassed here.

        Returns -inf on any non-finite evaluation, shape mismatch, or sigma<=0.
        """
        if sigma <= 0.0:
            return -np.inf
        cdef object raw
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_pred
        try:
            raw = self.root.function.evaluate(self.domain, cand_name_values)
            y_pred = np.asarray(raw, dtype=np.float64).ravel()
        except Exception:
            return -np.inf
        if y_pred.shape[0] != y.shape[0]:
            return -np.inf
        cdef cnp.ndarray[cnp.float64_t, ndim=1] resid = y - y_pred
        '''Plain sum, NOT nansum. With nansum an all-NaN residual gave
        ssr = 0, which is finite, and slipped straight past the guard
        below. Now any non-finite prediction propagates to NaN and is
        rejected: zero likelihood, consistent with the fitness()
        semantics in real_functionC.'''
        cdef double ssr = float(np.sum(resid * resid))
        if not np.isfinite(ssr):
            return -np.inf
        cdef int n = y.shape[0]
        cdef double s2 = sigma * sigma
        return -0.5 * n * np.log(2.0 * np.pi * s2) - 0.5 * ssr / s2

    cdef double _log_likelihood_at_sigma_arr(self,
                               cnp.ndarray[cnp.float64_t, ndim=1] y,
                               dict cand_name_values,
                               cnp.ndarray[cnp.float64_t, ndim=1] sigma):
        """
        Per-point-sigma companion of _log_likelihood_at; the scalar
        method above is untouched.
        Raw heteroscedastic Gaussian log-likelihood at a *candidate*
        parameter assignment, WITHOUT mutating any tree state. Used by
        parameter_refresh when the run's sigma is a per-point array:

            logL = -0.5 * sum_i log(2 pi sigma_i^2)
                   -0.5 * sum_i (y_i - f_i)^2 / sigma_i^2

        Returns -inf on any non-finite evaluation, shape mismatch, or any
        sigma_i <= 0 / non-finite (mirror of the scalar sigma<=0 guard).
        Plain np.sum, never nansum: any non-finite prediction propagates
        to NaN and is rejected (zero-likelihood semantics, PLAN 4.2b).
        """
        if sigma.shape[0] != y.shape[0]:
            return -np.inf
        if (not np.all(np.isfinite(sigma))) or np.any(sigma <= 0.0):
            return -np.inf
        cdef object raw
        cdef cnp.ndarray[cnp.float64_t, ndim=1] y_pred
        try:
            raw = self.root.function.evaluate(self.domain, cand_name_values)
            y_pred = np.asarray(raw, dtype=np.float64).ravel()
        except Exception:
            return -np.inf
        if y_pred.shape[0] != y.shape[0]:
            return -np.inf
        cdef cnp.ndarray[cnp.float64_t, ndim=1] resid = y - y_pred
        cdef double wssr = float(np.sum((resid * resid) / (sigma * sigma)))
        if not np.isfinite(wssr):
            return -np.inf
        cdef double logdet = float(np.sum(np.log(2.0 * np.pi * sigma * sigma)))
        return -0.5 * logdet - 0.5 * wssr


    cdef void _apply_param_values(self, dict new_name_values):
        """
        Install a parameter assignment INTO the tree (mutates state).
        Used by parameter_refresh on accept. Loops Tree.params (Node objects),
        setting each parameter Node's value via set_value, then calls
        tree_update(self.root) so name_values, param_values and penalty stay
        in sync. Names not present in new_name_values are left untouched.
        """
        cdef Node par
        cdef Node leaf
        cdef str name
        for par in self.params:
            name = par.function.get_description()
            if name in new_name_values:
                par.set_value(float(new_name_values[name]))
        # Opaque leaves' internal parameters are ordinary coordinates of
        # the refresh vector (DESIGN §5), so install them too.
        for leaf in self.leaves:
            if leaf.is_atomic():
                for name in leaf.leaf_param_names():
                    if name in new_name_values:
                        leaf.theta[name] = float(new_name_values[name])
        # Flex-scaffold internal theta are refresh coordinates as well.
        cdef Node fx_node
        for fx_node in _collect_flex_nodes(self.root, []):
            for name in fx_node.flex_names:
                if name in new_name_values:
                    fx_node.theta[name] = float(new_name_values[name])
        self.tree_update(self.root)

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
                            alpha=self.alpha, sigma=self.sigma, g_tau = self.g_tau) #riguardare per distribuire depth e Npars (specialmente Npars)
        elif replace_all==1:
            if kwargs is not None:
                self.root.left = Node(**kwargs)
            else:
                self.root.left = build_tree(max(0, new_depth), self.functions, 
                        self.domain, Npars=_PY_RNG.choice((0,1)), g_tau = self.g_tau)
            if op is not None:
                self.root.assign_operation(op)
        elif replace_all==2:
            if kwargs is not None:
                self.root.right = Node(**kwargs)
            else:
                self.root.right = build_tree(max(0, new_depth), self.functions, 
                        self.domain, Npars=_PY_RNG.choice((0,1)), g_tau = self.g_tau)
            if op is not None:    
                self.root.assign_operation(op)
        self.tree_update(self.root)
        if ensure_valid and not self.is_valid(self.domain):
            # revert if invalid
            self.root = old_root
            self.tree_update(self.root)
            return False
        return True

cdef _post_order_update(Node node):
    if node is None or node.is_leaf():
        return
    if node.left is not None:
        _post_order_update(node.left)
    if node.right is not None:
        _post_order_update(node.right)
    node.update_func()


# ---------------------------------------------------------------------------
'''Structural (skeleton) tree serialization.

The MPI wire cannot ship a raw sympy expression for an atom-bearing
tree: _expr_to_tree rebuilds from the EXPRESSION structure, and an
opaque leaf's contribution (a0*sin(a1*x), say) is indistinguishable from
an ordinary Mul subtree, so atomicity, the Polya vocabulary key and the
description round-trip would all break.

The skeleton is a picklable recursive tuple preserving the exact tree
shape and the opaque leaves' identity:

    ('L', description)                      plain leaf (function/parameter)
    ('A', atom_symbol, (name1, name2, ...)) opaque atomic leaf
    ('O', op_symbol, left_skel, right_skel) internal node

Parameter VALUES travel separately in name_values, exactly as on the
older wire. Both functions are cpdef so groveC and mpi_worker can call
them across the cimport boundary — POSITIONAL ARGS ONLY there.'''
# ---------------------------------------------------------------------------

cpdef object tree_to_skel(Node node):
    """Serialize the (sub)tree rooted at `node` to a skeleton tuple.

    Transparent atoms add one more tag:
        ('F', head_name, (theta_names...), slot_skel)
    Exact-shape and sympy-free, mirroring the 'A' contract: values travel
    in name_values, and the caller installs them and runs tree_update."""
    if node.is_leaf():
        if node.is_atomic():
            return ('A', node.function.atom_symbol,
                    tuple(node.leaf_param_names()))
        return ('L', node.function.get_description())
    if node.flex_spec is not None:
        return ('F', node.flex_spec.name, tuple(node.flex_names),
                tree_to_skel(node.left))
    return ('O', node.op_symbol,
            tree_to_skel(node.left), tree_to_skel(node.right))


cpdef Node node_from_skel(object skel,
                          cnp.ndarray domain,
                          dict atom_specs):
    """Rebuild a Node tree from a skeleton tuple.

    `atom_specs` maps atom_symbol -> real_functionC.AtomSpec; a skeleton
    that references a template absent from the map raises KeyError (the
    wire consumers turn that into their reject-the-exchange convention).
    Values are NOT installed here — the caller applies name_values
    afterwards (plain params via set_value, opaque theta via the leaf
    dicts) and runs tree_update."""
    cdef str tag = <str> skel[0]
    cdef Node left, right
    cdef real_functionC.RealFunctionC rf
    if tag == 'L':
        return Node(real_functionC.RealFunctionC.from_string(
            <str> skel[1], domain, compute_grad=False))
    if tag == 'A':
        if atom_specs is None or skel[1] not in atom_specs:
            raise KeyError(
                "node_from_skel: unknown atom template %r — pass the "
                "atom_specs dict covering every template on the wire"
                % (skel[1],))
        rf = real_functionC.instantiate_atom(
            <real_functionC.AtomSpec> atom_specs[skel[1]], domain,
            list(skel[2]))
        return Node(rf)
    if tag == 'F':
        # Rebuild the slot first, then the scaffold over it.
        # Registry shares the atom_specs dict (FlexAtomSpec entries);
        # a missing template keeps the KeyError->reject convention.
        if atom_specs is None or skel[1] not in atom_specs:
            raise KeyError(
                "node_from_skel: unknown flex template %r — pass the "
                "atom_specs dict covering every template on the wire"
                % (skel[1],))
        left = node_from_skel(skel[3], domain, atom_specs)
        return make_flex_node(atom_specs[skel[1]], left, list(skel[2]), None)
    if tag != 'O':
        raise ValueError("node_from_skel: bad skeleton tag %r" % (tag,))
    left = node_from_skel(skel[2], domain, atom_specs)
    right = node_from_skel(skel[3], domain, atom_specs)
    return Node(None, left, right, <str> skel[1])

# ---------------------------------------------------------------------------
'''Transparent-atom machinery.

A flex node is a UNARY scaffold: a frozen shape (FlexAtomSpec) over ONE
mutable slot subtree in node.left, with node.right None. Its identity is
the UNEXPANDED sympy Function head name(theta..., slot_expr), which is
injective with respect to transparent expressions, and no sympy
substitution ever happens — evaluation is numeric composition of the
spec's compiled template over the slot's output.

Scaffolds enter by seeding only and are immortal in the outer kernel;
slots evolve both ways through gp_movesC.flex_slot_mutation.
Cross-module callers: POSITIONAL ARGS ONLY.'''
# ---------------------------------------------------------------------------

cdef real_functionC.RealFunctionC _flex_function(object spec_obj, list names,
                                                 Node slot):
    """Compose spec's template numerically over the CURRENT slot function.

    The returned RealFunctionC's expr keeps the unexpanded head; its
    closure captures the slot's function OBJECT — every structural slot
    change replaces function objects and re-runs update_func, so stale
    closures cannot leak. Missing internal names in `params` fall back to
    the 1.0 reference (theta0 convention: params-free probes like
    is_valid stay well-defined). Any evaluation failure or complex output
    yields an all-NaN vector -> zero-likelihood semantics upstream."""
    cdef object spec = spec_obj
    cdef object slot_rf = slot.function
    cdef object compiled = (<real_functionC.FlexAtomSpec> spec_obj).compiled
    cdef list nm_list = list(names)
    cdef object expr = sp.Function(
        (<real_functionC.FlexAtomSpec> spec_obj).name)(
        *([sp.Symbol(nm) for nm in nm_list] + [slot_rf.get_expr()]))

    def func(cnp.ndarray input_x, dict params=None):
        if params is None:
            params = {}
        s_out = slot_rf.get_eval(input_x, params)
        args = [s_out]
        for nm in nm_list:
            args.append(params.get(nm, 1.0))
        try:
            arr = np.asarray(compiled(*args), dtype=np.float64)
            if arr.ndim == 0:
                return np.full(input_x.shape[input_x.ndim - 1],
                               float(arr), dtype=np.float64)
            return np.ascontiguousarray(arr, dtype=np.float64).ravel()
        except Exception:
            return np.full(input_x.shape[input_x.ndim - 1], _NAN,
                           dtype=np.float64)

    return real_functionC.RealFunctionC(func, expr, slot_rf.get_domain(),
                                        compute_grad=False)


cdef list _collect_flex_nodes(Node node, list out):
    """Pre-order collection of flex-scaffold nodes. Does NOT descend into
    slots (nesting is forbidden at construction; the walk stays O(outer
    nodes)). Empty on every legacy tree."""
    if node is None:
        return out
    if node.flex_spec is not None:
        out.append(node)
        return out
    if not node.is_leaf():
        _collect_flex_nodes(node.left, out)
        _collect_flex_nodes(node.right, out)
    return out


cpdef list collect_flex_nodes(Node node, list out=None):
    """Python/cross-module wrapper around _collect_flex_nodes."""
    if out is None:
        out = []
    return _collect_flex_nodes(node, out)


cpdef bint tree_has_flex(Node node):
    """Cheap presence test used by wire predicates and PR routing."""
    return len(_collect_flex_nodes(node, [])) > 0


cpdef Node make_flex_node(object spec_obj, Node slot, list fresh_names,
                          object theta=None):
    """Instantiate a transparent atom: scaffold `spec_obj`
    (real_functionC.FlexAtomSpec) over the given `slot` subtree, internal
    parameters named `fresh_names` (positionally matched to spec.slots,
    len == spec.k, distinct, parameter-pattern, never variable names).
    `theta` optionally pins internal values (default 1.0 reference).

    This is the SEEDING entry point: flex atoms only enter trees here
    and via the 'F' wire tag, and the move kernel never births or
    destroys them. The slot must be a legal flex-free subtree with a
    live function. Cross-module callers: POSITIONAL ARGS ONLY.

    Seeds must start IN SUPPORT. The scaffold contributes 1 + depth(slot)
    to the OUTER tree's depth, and the kernel's bounded support is
    depth <= gp_movesC.get_max_tree_depth(). Nothing validates that here,
    since gp_movesC is not importable from this module without a cycle,
    so it is the caller's responsibility — the same convention as
    node_list seeding."""
    cdef object spec = spec_obj
    cdef str nm
    cdef dict th = {}
    cdef Node out
    if not isinstance(spec_obj, real_functionC.FlexAtomSpec):
        raise TypeError(
            "make_flex_node: spec must be a real_functionC.FlexAtomSpec, "
            "got %r" % type(spec_obj))
    if slot is None or slot.function is None:
        raise ValueError("make_flex_node: slot must be a live Node")
    if len(_collect_flex_nodes(slot, [])) > 0:
        raise ValueError(
            "make_flex_node: nested flex scaffolds are not supported (v1)")
    if (fresh_names is None
            or len(fresh_names) != (<real_functionC.FlexAtomSpec> spec_obj).k):
        raise ValueError(
            "make_flex_node(%r): expected %d fresh names, got %r"
            % ((<real_functionC.FlexAtomSpec> spec_obj).name,
               (<real_functionC.FlexAtomSpec> spec_obj).k, fresh_names))
    if len(set(fresh_names)) != len(fresh_names):
        raise ValueError(
            "make_flex_node: fresh names must be distinct, got %r"
            % (fresh_names,))
    for nm in fresh_names:
        if not re.fullmatch(r'[a-zA-Z]\d+', nm) \
                or real_functionC.is_variable_name(nm):
            raise ValueError(
                "make_flex_node: %r is not a legal internal parameter "
                "name" % (nm,))
        th[nm] = 1.0 if theta is None else float(theta.get(nm, 1.0)) \
            if isinstance(theta, dict) else 1.0
    out = Node(_flex_function(spec_obj, list(fresh_names), slot),
               slot, None, None)
    out.flex_spec = spec_obj
    out.flex_names = list(fresh_names)
    out.theta = th
    return out


#cdef --> is cpdef for testing purposes only
cpdef build_tree(int depth, list functions, cnp.ndarray x,
        str output_type=None, int Npars=0, object leaves_count = None, object arr_indices = None,
             dict params_used=None, object value = 1.0, double T=0, double alpha=0.01, object sigma=1.00,
             double g_tau = 1.0, list operations=None):
        '''Builds a Tree from scratch.

        Internal ops are drawn uniformly from `operations` (the symbols
        '+', '-', '*', '/', '**', 'o'). Default None keeps the historical
        6-op set IN THE SAME ORDER as the old internal
        ['add','sub','mul','div','pow','compose'] list, so default-ops
        RNG streams — and every determinism test — stay bit-identical.
        Callers with a restricted kernel vocabulary (safe-ops fixtures,
        for instance) MUST pass their own list, otherwise the proposal
        build-densities computed with O = len(operations) do not match
        the true sampling law.'''
        if operations is None:
            operations = ['+', '-', '*', '/', '**', 'o']
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
            noise = _NP_RNG.uniform(-0.5, 0.5)
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
            func_str = _PY_RNG.choice(functions)
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
                alpha = alpha, sigma = sigma, g_tau = g_tau, operations = operations)
        cdef Node right = build_tree(depth - 1, functions, x,
                'node', Npars, leaves_count, arr_indices, params_used, value=value,
                alpha = alpha, sigma = sigma, g_tau = g_tau, operations = operations)


        # Draw the op SYMBOL uniformly from the caller's vocabulary; the
        # default list above preserves the historical stream.
        cdef str op_symbol = _PY_RNG.choice(operations)
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
            func = left.function ** right.function
        elif op_symbol == 'o':
            func = left.function.compose(right.function)
        else:
            raise ValueError(
                "build_tree: unknown operation symbol %r (supported: "
                "'+', '-', '*', '/', '**', 'o')" % op_symbol)
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

cdef bint _prefix_mints_variable(object prefix):
    """True iff prefix+digits could equal a REGISTERED variable name.

    Flexible-domain guard (plan §0 D4): with registry ('x0','x1') the
    prefix 'x' is forbidden for parameter minting; with the legacy
    registry ('x',) NO prefix is forbidden ('x' alone never matches
    prefix+digits), so legacy naming is bit-identical. (`object`-typed:
    this module compiles under language_level 2, where unicode locals
    do not implicitly coerce to `str` parameters.)"""
    cdef object v
    for v in real_functionC.get_variables():
        if re.fullmatch(re.escape(prefix) + r"\d+", v):
            return True
    return False


#Naming parameters requires fix!
cdef new_par(list existing_params, str user_choice=None):
    '''Add a new parameter named with a specific logic'''
    existing_params = [item for sublist in existing_params for item in (sublist if isinstance(sublist, list) else [sublist])]
    if len(existing_params) > 0:
        param_pattern = re.compile(r"([a-zA-Z]+)(\d+)")
        param_names = [param_pattern.match(param).groups()[0] for param in existing_params if param_pattern.match(param)]
        # Flexible-domain guard: never derive a prefix that could mint a
        # registered variable name (no-op filter under the legacy
        # registry — same list, same single RNG draw below).
        param_names = [p for p in param_names if not _prefix_mints_variable(p)]
        if not param_names:
            param_names = ['a']

        #unique_prefixes = set(param_names)
        if user_choice is not None and _prefix_mints_variable(user_choice):
            user_choice = None
        if user_choice not in param_names or user_choice is None:
            user_choice = _PY_RNG.choice(param_names)



        indices = [int(param_pattern.match(param).groups()[1]) for param in existing_params if param.startswith(user_choice)]
        new_index = max(indices) + 1 if indices else 0
        # Last-resort skip: a parameter name must never equal a variable.
        while real_functionC.is_variable_name(f"{user_choice}{new_index}"):
            new_index += 1

        return f"{user_choice}{new_index}"

    user_choice = user_choice if user_choice else 'a'
    if _prefix_mints_variable(user_choice):
        user_choice = 'a'
    new_index = 0
    while real_functionC.is_variable_name(f"{user_choice}{new_index}"):
        new_index += 1
    return f"{user_choice}{new_index}"

cdef _expr_to_tree(object expr, cnp.ndarray domain, list function_list, dict params_used = None ):
    """
    Sympy expr to Tree

    Parameters:
        expr (sympy.Expr): expression to translate
        domain: function domain

    Returns:
        Tree
    """

    cdef Tree left_tree, right_tree, arg_tree, result_tree, inner_tree, neg_one_tree
    cdef Node root_node
    cdef real_functionC.RealFunctionC func, root_func, composed_func

    # Flexible domain (plan §0 D8): the raw-sympy-expr rebuild is a
    # d=1-only path — function applications rebuild as COMPOSITIONS (see
    # the sp.Function branch below), which are undefined at d > 1.
    # Multivariate trees always ride the exact-shape skeleton wire
    # (tree_to_skel / node_from_skel) instead.
    if real_functionC.n_variables() > 1:
        raise NotImplementedError(
            "_expr_to_tree: raw-expression rebuild is d=1-only; "
            "multivariate trees use the skeleton wire "
            "(flexible_domain_plan.md §0 D8)")

    if params_used is None:
        params_used = {}
    #print(params_used.values()) 
    
    if expr.is_Number:
        if _PIN_NUMBERS:
            '''The literal becomes a CONSTANT leaf, never a fitted
            parameter: it stays out of get_params/name_values, so k and
            the optimizer vector are untouched. The pinned_const flag is
            what keeps Node.is_param() False for it, because in this
            kernel's ontology an unmarked constant map IS the parameter
            leaf.

            These trees are meant for SCORING and reference use. Constant
            leaves sit OUTSIDE the move vocabulary, so the usual
            seeds-in-support convention applies if such a tree is ever
            evolved, and burn-in constant folding would refold them —
            though burn-in never sees them.'''
            _pc = real_functionC.RealFunctionC.from_string(
                str(expr), domain)
            (<real_functionC.RealFunctionC> _pc).pinned_const = True
            # Zero-variable functions evaluate as params.get(desc,
            # constant_value), and constant_value defaults to 1.0, so
            # the literal's value has to live there: nothing else feeds
            # a non-parameter constant at eval time.
            (<real_functionC.RealFunctionC> _pc).constant_value = \
                float(expr)
            root_node = Node(_pc)
            return Tree(0, function_list, domain, root_node)
        param_name = new_par(list(params_used.values()))
        #print(param_name)
        if str(expr) not in params_used:
            params_used[str(expr)] = [param_name]
        else:
            params_used[str(expr)].append(param_name)

        root_node = Node(real_functionC.RealFunctionC.from_string(param_name, domain))
        root_node.set_value(np.float64(expr))
        return Tree(0, function_list, domain, root_node)

    if isinstance(expr, sp.Symbol):
        symbol_str = str(expr)
        # Flexible domain (plan §0 D4): registry membership, not the
        # literal 'x' — a registered variable rebuilt as a parameter
        # would silently lose the expression's x-dependence.
        if real_functionC.is_variable_name(symbol_str):
            func = real_functionC.RealFunctionC.from_string(symbol_str, domain)
            root_node = Node(function = func)
        else:
            if symbol_str not in params_used:
                param_name = new_par(list(params_used.values()))
                params_used[symbol_str] = param_name
            else:
                param_name = params_used[symbol_str]
            root_node = Node(real_functionC.RealFunctionC.from_string(param_name, domain))

        return Tree(root_node.node_depth, function_list, root_node.domain, root_node)

    '''if isinstance(expr, sp.Add):
        left_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
        right_tree = _expr_to_tree(expr.args[1], domain, function_list, params_used)
        root_func = left_tree.root.get_fun() + right_tree.root.get_fun()
        root_node = Node(root_func, left_tree.root, right_tree.root, "+")
        return Tree(root_node.node_depth, function_list, root_node.domain, root_node)
    '''
    if isinstance(expr, sp.Add):
        if len(expr.args) >= 2:
            positives = []
            negatives = []
            for arg in expr.args:
                arg_is_negative = False
                if isinstance(arg, sp.Mul) and len(arg.args) > 0:
                    '''if arg.args[0] == -1:  # Check if the first argument is -1
                        arg_is_negative = True'''
                    if arg.args[0].is_Number and arg.args[0] < 0:
                        arg_is_negative = True
                elif arg.is_Number and arg < 0:
                    arg_is_negative = True
                if arg_is_negative:
                    negatives.append(-arg)
                else:
                    positives.append(arg)

            if not positives:
                # All terms are negative: build -(|a1| + |a2| + ...) as (-1) * sum
                pos_expr = sp.Add(*negatives)
                inner_tree = _expr_to_tree(pos_expr, domain, function_list, params_used)
                neg_one_tree = _expr_to_tree(sp.Integer(-1), domain, function_list, params_used)
                
                root_node = Node(neg_one_tree.root.get_fun() * inner_tree.root.get_fun(),
                        neg_one_tree.root, inner_tree.root, '*')
                return Tree(root_node.node_depth, function_list, domain, root_node)

            result_tree = _expr_to_tree(positives[0], domain, function_list, params_used)
            for arg in positives[1:]:
                arg_tree = _expr_to_tree(arg, domain, function_list, params_used)
                root_node = Node(result_tree.root.get_fun() + arg_tree.root.get_fun(),
                        result_tree.root, arg_tree.root, '+')
                result_tree = Tree(root_node.node_depth, function_list, domain,
                    root_node)

            for positive_arg in negatives:
                arg_tree = _expr_to_tree(positive_arg, domain, function_list, params_used)
                
                root_node = Node(result_tree.root.get_fun() - arg_tree.root.get_fun(),
                        result_tree.root, arg_tree.root, '-')

                result_tree = Tree(root_node.node_depth, function_list, domain,
                    root_node)
            return result_tree
        else:
            print("something's off in addition")
            '''    
                if arg_is_negative:
                    # Handle the negative argument as a subtraction
                    positive_arg = -arg
                    arg_tree = _expr_to_tree(positive_arg, domain, function_list, params_used)
                    if 'result_tree' not in locals():
                        result_tree = arg_tree
                    else:
                        result_tree = Tree(result_tree.depth, function_list, domain,
                            Node(result_tree.root.get_fun() - arg_tree.root.get_fun(), 
                                result_tree.root, arg_tree.root, '-'))
                else:
                    # Handle the positive argument as an addition
                    arg_tree = _expr_to_tree(arg, domain, function_list, params_used)
                    if 'result_tree' not in locals():
                        result_tree = arg_tree
                    else:
                        result_tree = Tree(result_tree.depth, function_list, domain,
                            Node(result_tree.root.get_fun() + arg_tree.root.get_fun(), 
                                result_tree.root, arg_tree.root, '+'))
            return result_tree
            
        else:
            print("something's off in addition")'''

    if isinstance(expr, sp.Mul):    
        numer, denom = expr.as_numer_denom()
        if denom != 1:
            left_tree = _expr_to_tree(numer, domain, function_list, params_used)
            right_tree = _expr_to_tree(denom, domain, function_list, params_used)
            op_symbol = '/'
            root_func = left_tree.root.get_fun() / right_tree.root.get_fun() 
            root_node = Node(root_func, left_tree.root, right_tree.root, op_symbol)
            result_tree = Tree(root_node.node_depth, function_list, domain,
                root_node)
        else:  
            # Handle multiple factors by recursively building the tree
            factors = list(expr.args)
            result_tree = _expr_to_tree(factors[0], domain, function_list, params_used)
            for factor in factors[1:]:
                arg_tree = _expr_to_tree(factor, domain, function_list, params_used)
                result_tree = Tree(result_tree.depth, function_list, domain,
                    Node(result_tree.root.get_fun() * arg_tree.root.get_fun(), 
                        result_tree.root, arg_tree.root, '*'))
        return result_tree
    
    if isinstance(expr, sp.Pow):
        result_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
        for arg in expr.args[1:]:
            arg_tree = _expr_to_tree(arg, domain, function_list, params_used)
            root_node =Node(result_tree.root.get_fun() ** (arg_tree.root.get_fun()), 
                    result_tree.root, arg_tree.root, '**')
            result_tree = Tree(root_node.node_depth, function_list, domain,
                root_node)
        
        return result_tree

    if isinstance(expr, sp.Function):
        # Registry-aware primitive key ('sin(x)' legacy; 'sin(f)' for a
        # custom d=1 variable name). d>1 never reaches here (entry guard).
        func_name = ''.join(
            (expr.func.__name__, '(', real_functionC.get_variables()[0], ')'))
        #print('NAME: ', func_name)
        if func_name in function_list:
            root_function = real_functionC.RealFunctionC.from_string(func_name, domain) 
            if len(expr.args)==1 and expr.args[0] in expr.free_symbols:
                root_node = Node(root_function)
                return Tree(0, function_list, domain, root_node)
            arg_tree = _expr_to_tree(expr.args[0], domain, function_list, params_used)
            composed_function = root_function.compose(arg_tree.root.get_fun())
            #make subnode for composed_func
             
            #print('last stage: ', composed_function, root_function, arg_tree)
            root_node = Node(composed_function, Node(root_function), arg_tree.root, "o")
            return Tree(root_node.node_depth, function_list, root_node.domain, root_node)
        else:
            raise NotImplementedError(f"Conversion {func_name} not supported") 

    raise NotImplementedError(f" Expression{type(expr)} not supported.")

def expr_to_tree(expr, domain, function_list, pin_numbers=False):
    """Convert a sympy expression to a Tree.

    pin_numbers=True renders numeric literals as CONSTANT leaves instead
    of free parameters; the default False is the ordinary behaviour. The
    flag is module-scoped so the recursion can see it, and it is always
    restored."""
    global _PIN_NUMBERS
    # Set the flag on BOTH branches and always restore False, so that a
    # nested ordinary call made during a pin_numbers conversion cannot
    # inherit the raised flag.
    _PIN_NUMBERS = bool(pin_numbers)
    try:
        return _expr_to_tree(expr, domain, function_list)
    finally:
        _PIN_NUMBERS = False


cdef list _symbol_values(dict names_values):
    cdef list return_list = []
    for key in names_values:
        return_list.append((sp.symbols(key), names_values[key]))
    return return_list


cdef _parameter_simplifier(object expr, list used_params):
    cdef str expr_str
    expr_str = new_par(used_params)
    expr = sp.sympify(expr_str)
