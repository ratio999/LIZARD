# cython: language_level=3
# mpi_worker.py
"""
MPI worker script for parallel burn-in.
Spawned by the main process, receives work, computes, sends results back.
"""
from mpi4py import MPI
import numpy as np
cimport numpy as cnp
import sys
import os
from copy import deepcopy
# Add the module path
from sympy import sympify, symbols
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import after path setup
cimport function_treeC
cimport real_functionC
cimport gp_movesC
cimport polya_treeC
cimport groveC
cdef function_treeC.Tree _helper(object expr, cnp.ndarray[cnp.float64_t, ndim=1] domain, 
            list funcs, dict names_params):
    
    cdef function_treeC.Tree tree
    cdef function_treeC.Node param

    tree = function_treeC._expr_to_tree(expr, domain, funcs)
    #tree.
    
    if tree is not None and names_params and len(names_params) > 0:
        for param in tree.params:
            if not param.function.expr.is_Number:
                param.set_value(names_params[param.get_description()])
    return tree



cdef void _worker_main():
    """
    Worker entry point. Communicates with parent (spawner) process.
    """
    # Get the intercommunicator to the parent
    cdef object parent_comm = MPI.Comm.Get_parent()
    
    if parent_comm == MPI.COMM_NULL:
        print("Error: This script must be spawned by MPI, not run directly.")
        sys.exit(1)
    
    # Get my rank among workers
    cdef object worker_comm = MPI.COMM_WORLD
    cdef int rank = worker_comm.Get_rank()
    cdef int n_workers = worker_comm.Get_size()
    
    # =========================================================
    # RECEIVE SHARED DATA FROM PARENT
    # =========================================================
    # Parent broadcasts the configuration to all workers
    cdef dict config = parent_comm.bcast(None, root=0)
    
    cdef cnp.ndarray[cnp.float64_t, ndim=1] domain = config['domain']
    cdef list funcs = config['funcs']
    cdef list operations = config['operations']
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y = config['y']
    cdef double lambda_reg = config['lambda_reg']
    cdef str method_str = config['method_str']
    cdef bint use_gradients = config['use_gradients']
    cdef list depth1_data = config['depth1_data']  # List of (expression, params, posterior)
    cdef list all_primitives = config['all_primitives']
    cdef double T = config['T']
    cdef dict local_cache = config['tree_cache']
    # Phase 2.10 (final): opaque templates — resolve '@atom:' tokens in
    # all_primitives and rebuild 'skel3b' depth-1 payloads. None/absent for
    # atom-free burn-ins (legacy configs included).
    cdef list atoms = config.get('atoms')
    cdef dict atom_specs = None
    if atoms:
        atom_specs = {}
        for _spec_obj in atoms:
            atom_specs[_spec_obj.name] = _spec_obj

    # =========================================================
    # RECEIVE MY WORK ITEMS
    # =========================================================
    # Parent scatters work items to each worker
    cdef object my_work = parent_comm.scatter(None, root=0)

    # =========================================================
    # REBUILD DEPTH-1 TREES FROM SERIALIZED DATA
    # =========================================================
    cdef list depth1_trees = []
    cdef int i=0
    cdef object expr
    cdef dict names_params
    cdef object post
    cdef function_treeC.Tree tree, d1_tree, d1_tree_2

    for expr, names_params, post in depth1_data:
        #print(i, expr, names_params)
        # Phase 2.10 (final): atom-bearing depth-1 trees arrive as
        # ('skel3b', skeleton, description) payloads — rebuild through the
        # Stage 3b wire (sigma/g_tau 1.0/1.0: the same Tree defaults the
        # legacy _helper path produces). A failed rebuild leaves None in
        # the slot, which the work loop below already skips.
        if isinstance(expr, tuple) and len(expr) == 3 and expr[0] == 'skel3b':
            tree = groveC._tree_from_wire(expr, names_params, domain, funcs,
                                          1.0, 1.0, atom_specs)
            depth1_trees.append((tree, post))
            if tree is None:
                print(f'[Worker {rank}] Warning: skeleton depth-1 rebuild failed for {expr[2]!r}.')
            elif tree.tree_get_description() not in local_cache:
                local_cache[tree.tree_get_description()] = post
            continue
        tree = _helper(expr, domain, funcs, names_params)
        depth1_trees.append((tree, post))
        if tree.tree_get_description() not in local_cache:
            print(f'[Worker {rank}] Cache miss for depth-1 tree: {tree.tree_get_description()}')
            local_cache[tree.tree_get_description()] = post
        if tree.tree_get_description() != str(expr):
            print(f'[Worker {rank}] Warning: Reconstructed tree description "{tree.tree_get_description()}" does not match original expression string "{expr}".')
    # =========================================================
    # PROCESS MY WORK ITEMS
    # =========================================================
    cdef list local_results = []  # (expression, params_list, posterior)
    
    cdef object item
    cdef object item_type
    cdef function_treeC.Node prim_node
    cdef str prim_str, op
    cdef tuple result
    cdef int idx

    for idx, item in enumerate(my_work):
        # NOTE: every exception inside this loop MUST be swallowed. If a
        # worker dies here, the parent's collective `gather` below never
        # completes and the whole burn-in freezes. `_evaluate_tree` already
        # returns None for "expected" failures (invalid domain, stale
        # parameters that fold to complex, non-finite posterior); this
        # try/except is the last-resort net for anything else.
        try:
            item_type = item[0]
            
            
            if item_type == 'd1_prim':
                _, i, prim_str, op = item

                # Create primitive node. Phase 2.10 (final): via the token
                # resolver; the 'c' name band because _combine_nodes below
                # renames nothing by default, and 'c' collides with neither
                # the all-'a' depth-1 names nor the 'b' rename buffer.
                prim_node = groveC._burn_in_primitive_node(
                    prim_str, domain, atom_specs, 'c', 1)

                d1_tree = depth1_trees[i][0]
                if d1_tree is None:
                    continue
                
                # Direction 1: depth-1 op primitive
                tree = _combine_nodes(d1_tree.root, prim_node, op, domain, funcs, T)
                result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
                expr_string = tree.tree_get_description()
                if result is not None:
                    local_results.append(result)
                    local_cache[expr_string] = result[2]  # Cache the posterior 
                # Direction 2: primitive op depth-1
                tree = _combine_nodes(prim_node, d1_tree.root, op, domain, funcs, T)
                result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
                expr_string = tree.tree_get_description()
                if result is not None:
                    local_results.append(result)
                    local_cache[expr_string] = result[2]  # Cache the posterior        
            elif item_type == 'd1_d1':
                _, i, j, op = item
                
                d1_tree = depth1_trees[i][0]
                d1_tree_2 = depth1_trees[j][0]
                
                if d1_tree is None or d1_tree_2 is None:
                    continue
                
                tree = _combine_nodes(d1_tree.root, d1_tree_2.root, op, domain, funcs, T)
                result = _evaluate_tree(tree, y, domain, lambda_reg, method_str, use_gradients, T)
                expr_string = tree.tree_get_description()
                if result is not None:
                    local_results.append(result)
                    local_cache[expr_string] = result[2]  # Cache the posterior
            if (idx +1) % 50 == 0:
                parent_comm.isend(idx+1, dest=0, tag=99)
        except Exception as e:
            # Discard this work item. Print once for diagnostics but do
            # NOT re-raise: keeping the worker alive is more important
            # than perfect coverage of the candidate space.
            print(f'[Worker {rank}] skipped item {idx} ({type(e).__name__}: {e})')
            sys.stdout.flush()
            continue
        #print(idx, rank)
        #sys.stdout.flush()
    # =========================================================
    # SEND RESULTS BACK TO PARENT
    # =========================================================
    worker_comm.Barrier()
    print(f'[Worker {rank}] About to gather {len(local_results)} results...')
    sys.stdout.flush()
    
    try:
        parent_comm.gather(local_results, root=0)
        print(f'[Worker {rank}] Gather completed successfully!')
    except Exception as e:
        print(f'[Worker {rank}] Gather failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.flush()
    
    # =========================================================
    # DISCONNECT AND EXIT
    # =========================================================
    #worker_comm.Disconnect()
    parent_comm.Disconnect()
    print(f'[Worker {rank}] disconnected.')

cdef function_treeC.Tree _combine_nodes(function_treeC.Node left_node, function_treeC.Node right_node, object op,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, list funcs, double T):
    """Combine two nodes with an operation into a new tree."""


    cdef function_treeC.Node left_copy = deepcopy(left_node)
    cdef function_treeC.Node right_copy = deepcopy(right_node)
    #cdef function_treeC.Tree preliminary_tree = function_treeC.Tree(
	#depth = right_copy.node_depth, functions = funcs,
	#x = domain, node = right_copy)

    #gp_movesC.rename_prefix(preliminary_tree)

    # Phase 2.10 (final): this combiner historically renames NOTHING (the
    # rename_prefix above was disabled), which lets same-named PLAIN
    # parameters couple across sides — a tolerated legacy quirk kept
    # bit-identical here. Opaque internals must NOT couple (two atoms
    # sharing theta names would tie their fitted values), so when the
    # right side carries atomic leaves it is renamed wholesale into the
    # 'b' namespace (plain + atomic; refresh_params below re-canonicalises
    # the plains to 'a'). Atom-free behaviour is unchanged.
    cdef function_treeC.Tree _pre
    if gp_movesC._contains_atomic(right_copy):
        _pre = function_treeC.Tree(
            depth=right_copy.node_depth, functions=funcs,
            x=domain, node=right_copy)
        gp_movesC.rename_prefix(_pre)
        right_copy = _pre.root

    cdef function_treeC.Node root_node = function_treeC.Node(
        function=None,
        left=left_copy,
        right=right_copy,
        symbol=op
    )
    
    cdef function_treeC.Tree tree = function_treeC.Tree(
        depth=root_node.node_depth,
        functions=funcs,
        x=domain,
        node=root_node,
        Npars=len(root_node.function.params),
        value=1.0,
        T=T
    )
    gp_movesC.refresh_params(tree)
    #del preliminary_tree
    return tree


cdef tuple _evaluate_tree(function_treeC.Tree tree, cnp.ndarray[cnp.float64_t, ndim=1] y,
                    cnp.ndarray[cnp.float64_t, ndim=1] domain, double lambda_reg, 
                    str method_str, bint use_gradients, double T):
    """Evaluate a tree and return serializable result or None."""
    if tree is None:
        return None
    
    #cdef object expression = tree.root.function.expr
    
    if not tree.is_valid(domain):
        return None
    
    # Optimize parameters if needed. Phase 2.10 (final): gate on
    # name_values, not tree.params — opaque internals are continuous
    # parameters with no plain parameter Node, and skipping the fit would
    # score the reference theta as if it were the MLE (same generalization
    # as the Stage 3a optim_need gate; identical truthiness for atom-free
    # trees).
    if len(tree.name_values) > 0:
        tree.update_parameters(domain, y, method_str, use_gradients)
    
    # Fold constants -> fresh parameter leaves. If any constant subexpression
    # evaluates to complex / non-finite at the optimised theta (e.g. (-a0)**0.5
    # with a0 > 0), replace_constants_with_pars returns False; the tree has
    # drifted into a pathological basin and must be discarded.
    cdef bint ok = tree.replace_constants_with_pars()
    if not ok:
        return None
    cdef double post = tree.root._log_marginal_bic(y)

    if not np.isfinite(post):
        return None
    # Phase 2.10 (final): atom-bearing candidates must return as skeleton
    # wires — sympify(description) below would decompose the opaque leaves
    # into ordinary subtrees on the master's rebuild. _tree_to_wire yields
    # the same (payload, name_values, post) 3-tuple shape.
    if gp_movesC._contains_atomic(tree.root):
        return groveC._tree_to_wire(tree, post)
    cdef object expression = tree.tree_get_description()
    try:

        # Let sympify auto-detect and create symbols
        expression = sympify(expression)
    except:
        expression = tree.root.function.expr
        pass
    # Return serializable tuple
    return (expression, tree.name_values if tree.name_values else {}, post)

def worker_main():
    _worker_main()


# ===========================================================================
# Phase 2.12: MPI-parallel EMC sampling worker (spawned by
# groveC.Forest.run_mpi_spawn; one worker per temperature rung).
#
# The worker builds and PERSISTENTLY OWNS one Grove for the whole run:
#   * mutation steps run locally (grove.run) with polya_defer=True so the
#     accepted Polya paths are collected, gathered by the master each round
#     and replayed identically on every replica (deterministic replay);
#   * exchange / crossover decisions happen on the master; this worker only
#     serves 'get_tree' (serialize a slot) and 'set_tree' (install a slot)
#     commands, using the same (sympy expr, name_values, post) wire format
#     as the burn-in spawn path.
# The command loop is bcast-driven: EVERY worker receives every command and
# only the addressed rank acts on the point-to-point parts, so no
# interleaving can deadlock. `worker_main` (burn-in) is untouched.
# ===========================================================================
# Phase 2.10 Stage 3b: the wire has ONE implementation — groveC's cpdef
# pair (dual payload: legacy sympy expr for atom-free trees, 'skel3b'
# structural skeleton for atom-bearing ones). These wrappers only keep the
# historical local names; calls are POSITIONAL (cpdef across cimport).
cdef tuple _sampler_tree_to_wire(function_treeC.Tree tree, double post):
    return groveC._tree_to_wire(tree, post)


cdef function_treeC.Tree _sampler_tree_from_wire(
        object expr, dict name_values,
        cnp.ndarray[cnp.float64_t, ndim=1] domain, list funcs,
        object sigma, double g_tau, object atom_specs=None):
    return groveC._tree_from_wire(expr, name_values, domain, funcs,
                                  sigma, g_tau, atom_specs)


cdef void _sampler_main():
    cdef object parent_comm = MPI.Comm.Get_parent()
    if parent_comm == MPI.COMM_NULL:
        print("Error: sampler_main must be spawned by MPI, not run directly.")
        sys.exit(1)
    cdef int rank = MPI.COMM_WORLD.Get_rank()

    # ---- startup config ---------------------------------------------------
    cdef dict config = parent_comm.bcast(None, root=0)
    cdef dict ctor = config['ctor']
    cdef cnp.ndarray[cnp.float64_t, ndim=1] y = \
        np.ascontiguousarray(config['y'], dtype=np.float64)
    cdef cnp.ndarray[cnp.float64_t, ndim=1] temps = \
        np.ascontiguousarray(config['temperatures'], dtype=np.float64)
    cdef double my_T = float(temps[rank])
    cdef object my_seed = config['worker_seeds'][rank]
    cdef cnp.ndarray[cnp.float64_t, ndim=1] x = \
        np.ascontiguousarray(ctor['x'], dtype=np.float64)

    # Optional caps (module-level in THIS process's gp_movesC instance).
    if config.get('max_tree_depth') is not None:
        gp_movesC.set_max_tree_depth(int(config['max_tree_depth']))
    if config.get('subtree_rebuild_max_height') is not None:
        gp_movesC.set_subtree_rebuild_max_height(
            int(config['subtree_rebuild_max_height']))

    # ---- local Polya replica (exact copy of the master's) ------------------
    cdef polya_treeC.PolyaTree polya = None
    cdef dict pstate = config.get('polya_state')
    if pstate is not None:
        polya = polya_treeC.PolyaTree(
            list(pstate['operations']), list(pstate['primitives']),
            alpha_0=float(pstate['alpha_0']))
        polya.set_state(pstate)

    # ---- Phase 2.10 Stage 3b: opaque-primitive templates --------------------
    # Shipped in the config (AtomSpec pickles via __reduce__). They go to
    # (a) the Grove, so this worker's kernel moves propose/retire opaque
    # leaves with the same M+1+A densities as the master's, and (b) every
    # wire rebuild, so 'skel3b' payloads reconstruct atoms as units. The
    # atom SYMBOLS are already in polya_state's vocabulary, so the
    # ('kind', idx) path mapping is identical on every replica.
    cdef list atoms = config.get('atoms')
    cdef dict atom_specs = None
    if atoms:
        atom_specs = {}
        for _spec_obj in atoms:
            atom_specs[_spec_obj.name] = _spec_obj

    # ---- build MY grove -----------------------------------------------------
    cdef groveC.Grove grove = groveC.Grove(
        x, y, my_T, int(ctor['size']), int(ctor['depth']),
        list(ctor['funcs']), sigma=ctor['sigma'],
        Npars=int(ctor['Npars']), init_value=ctor['init_value'],
        operations=list(ctor['operations']),
        base_temp=float(ctor['base_temp']),
        lambda_reg=float(ctor['lambda_reg']),
        use_gradients=bool(ctor['use_gradients']),
        burn_in=False, use_mpi=0,
        polya=polya, seed=my_seed, g_tau=ctor['g_tau'],
        adapt_moves=bool(ctor['adapt_moves']),
        atoms=atoms)
    grove.polya_defer = True

    # Install the master's CURRENT trees (start exactly where it stands).
    cdef list wires = config['init_trees'][rank]
    cdef list new_trees = []
    cdef function_treeC.Tree t
    cdef int s
    for s in range(len(wires)):
        t = _sampler_tree_from_wire(wires[s][0], wires[s][1],
                                    grove.domain, list(ctor['funcs']),
                                    grove.sigma, grove.g_tau, atom_specs)
        if t is None:
            # Fall back to the locally built tree for this slot (the master
            # will simply rescore it through the round replies).
            t = grove.trees[s]
        new_trees.append(t)
        grove.root_posteriors[s] = wires[s][2]
    grove.set_trees(new_trees)

    # ---- command loop (COLLECTIVE gather/scatter, same shape as the
    # burn-in spawn path: bcast a command, gather results, scatter updates).
    # The collectives (bcast / gather / scatter) run OUTSIDE any try so master
    # and workers stay in lockstep; only the local computation is guarded. ---
    cdef object cmd, incoming
    cdef str op
    cdef list paths, new_wires, all_paths, final_wires
    cdef function_treeC.Tree nt
    cdef int slot
    while True:
        cmd = parent_comm.bcast(None, root=0)
        op = <str> cmd[0]
        if op == 'run':
            try:
                grove.run(y, int(cmd[1]))
            except Exception as e:
                print(f"[Sampler {rank}] run failed: {type(e).__name__}: {e}")
                sys.stdout.flush()
            paths = grove.drain_deferred_paths()
            wires = [
                _sampler_tree_to_wire(
                    <function_treeC.Tree> grove.trees[slot],
                    float(grove.root_posteriors[slot]))
                for slot in range(grove.size)]
            # Send this grove's (paths, trees); receive its updated trees +
            # the global path list to replay. Collective, always in lockstep.
            parent_comm.gather((paths, wires), root=0)
            incoming = parent_comm.scatter(None, root=0)
            new_wires = <list> incoming[0]
            all_paths = <list> incoming[1]
            for slot in range(min(grove.size, len(new_wires))):
                nt = _sampler_tree_from_wire(
                    new_wires[slot][0], new_wires[slot][1], grove.domain,
                    list(ctor['funcs']), grove.sigma, grove.g_tau,
                    atom_specs)
                if nt is not None:
                    grove.set_single_tree(nt, slot)
                    grove.root_posteriors[slot] = float(new_wires[slot][2])
            if polya is not None:
                polya.replay_paths(all_paths)
        elif op == 'stop':
            final_wires = [
                _sampler_tree_to_wire(
                    <function_treeC.Tree> grove.trees[slot],
                    float(grove.root_posteriors[slot]))
                for slot in range(grove.size)]
            parent_comm.gather(final_wires, root=0)
            break

    parent_comm.Disconnect()
    print(f"[Sampler {rank}] disconnected.")


def sampler_main():
    _sampler_main()


if __name__ == "__main__":
    worker_main()
