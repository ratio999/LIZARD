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
                       hoist_mutation, add_node, replace_with_subtree, add_node,
                       AMState, parameter_refresh, flatten_nodes)
from groveC import Grove
import matplotlib.pyplot as plt
from gp_movesC import split
from polya_treeC import PolyaTree
# ---------------------------------------------------------------------------
# Phase 2.1.K smoke tests: add_node / hoist_mutation / split
# ---------------------------------------------------------------------------
def _count_nodes_py(node):
    """Python-side recursive node count (mirrors cdef _count_nodes)."""
    if node is None:
        return 0
    if node.is_leaf():
        return 1
    return 1 + _count_nodes_py(node.get_left()) + _count_nodes_py(node.get_right())


def _count_internal_py(node):
    if node is None or node.is_leaf():
        return 0
    return 1 + _count_internal_py(node.get_left()) + _count_internal_py(node.get_right())

def crossover_smoke_test():
    print("[X] subtree_crossover symmetric swap")
    x_sym, a0 = sp.Symbol('x'), sp.Symbol('a0')
    domain = np.linspace(0.001, 10, 50).astype(np.float64)
    funs = ["sin(x)", "cos(x)", "x"]
    for _ in range(50):
        ta = expr_to_tree(sp.sin(x_sym) + a0, domain, funs)
        tb = expr_to_tree(sp.cos(x_sym) * a0, domain, funs)
        n_a, n_b = _count_nodes_py(ta.get_root()), _count_nodes_py(tb.get_root())
        ni_a, ni_b = _count_internal_py(ta.get_root()), _count_internal_py(tb.get_root())
        lqf, lqr = subtree_crossover(ta, tb)
        n_ap, n_bp = _count_nodes_py(ta.get_root()), _count_nodes_py(tb.get_root())
        ni_ap, ni_bp = _count_internal_py(ta.get_root()), _count_internal_py(tb.get_root())
        # Conservation of total node count: every node either stays put or
        # crosses to the other tree. Therefore n_a + n_b == n_a' + n_b'.
        assert n_a + n_b == n_ap + n_bp, \
            f"node total not conserved: {n_a}+{n_b} -> {n_ap}+{n_bp}"
        # Option-1 (internal-only) selection densities.
        expected_lqf = -np.log(ni_a) - np.log(ni_b)
        expected_lqr = -np.log(ni_ap) - np.log(ni_bp)
        assert abs(lqf - expected_lqf) < 1e-10, \
            f"lqf {lqf} != {expected_lqf} (ni_a={ni_a}, ni_b={ni_b})"
        assert abs(lqr - expected_lqr) < 1e-10, \
            f"lqr {lqr} != {expected_lqr} (ni_ap={ni_ap}, ni_bp={ni_bp})"
        # No 'b'-prefixed params should remain after refresh.
        for tree in (ta, tb):
            for p in tree.get_params():
                name = p.get_fun().get_description()
                assert not name.startswith('b'), f"unrefreshed label {name}"
    print("    crossover OK (50 random pairs, conservation + densities + labels)")
    # Crossover post-condition: parameter labels must be contiguous a0..a{K-1}.
    for tree in (ta, tb):
        names = sorted(p.get_fun().get_description() for p in tree.get_params())
        expected = [f"a{i}" for i in range(len(names))]
        assert sorted(names) == expected, \
            f"non-contiguous labels after crossover: {names}"

    # All swap targets must be internal nodes (sanity for option-1 restriction):
    # we test by counting. After many crossovers leaves(T_a)+leaves(T_b) is
    # conserved (only internal subtrees were swapped, and a swap maps internal
    # subtrees to internal subtrees, so the leaf populations of each tree shuffle
    # with no creation/destruction of leaves).
    n_leaves_before = (
        sum(1 for n in flatten_nodes(ta.get_root()) if n.is_leaf()) +
        sum(1 for n in flatten_nodes(tb.get_root()) if n.is_leaf())
    )
    subtree_crossover(ta, tb)
    n_leaves_after = (
        sum(1 for n in flatten_nodes(ta.get_root()) if n.is_leaf()) +
        sum(1 for n in flatten_nodes(tb.get_root()) if n.is_leaf())
    )
    assert n_leaves_before == n_leaves_after
def k_changing_smoke_test():
    """Diagnostics for Phase 2.1.K K-changing moves.

    Checks (in order):
      1) Single-call sanity for each of add_node / hoist / split:
         finite densities, tree shape changes correctly, no crash.
      2) add_node x N: |N| always grows; lqf, lqr both finite; log delta
         summary; subtree depth distribution sampled from truncated geom.
      3) split   x N: same checks, restricted to trees with >=1 internal
         node (we use a slightly deeper seed tree).
      4) hoist   x N: |N| always shrinks; report off-support fraction (must
         be << 1 because every leaf-hoist of a perfect-binary sibling is
         on-support); on-support: mean log_q_rev - log_q_fwd.
      5) Detailed-balance spot check: lqr returned by add_node must equal
         -log(|N(T')|-1), since the paired leaf-hoist's selection density
         on T' is exactly that.
      6) Symmetric spot check: lqr returned by split must equal
         -log(|N(T')|-1) for the internal-hoist selection.
      7) Pair sanity for hoist (leaf branch): independently recompute the
         add_node density on T' for the lifted leaf and compare to the
         lqr returned by hoist_mutation.
    """
    print("=" * 60)
    print("Phase 2.1.K: add_node / hoist_mutation / split smoke test")
    print("=" * 60)
    import random as _r
    np.random.seed(1)
    _r.seed(1)

    domain = np.linspace(0.001, 10.0, 50).astype(np.float64)
    functions = ["sin(x)", "cos(x)", "x"]
    operations = ["+", "-", "*", "/"]
    x_sym = sp.Symbol('x')
    a0 = sp.Symbol('a0')

    def make_tree_simple():
        # sin(x) + a0  -> 3 nodes (1 internal, 2 leaves: 1 fn + 1 par)
        return expr_to_tree(sp.sin(x_sym) + a0, domain, functions)

    def make_tree_deeper():
        # sin(x) * (cos(x) + a0)  -> 5 nodes (2 internal, 3 leaves)
        return expr_to_tree(sp.sin(x_sym) * (sp.cos(x_sym) + a0),
                            domain, functions)

    # --- 1. single-call invocation ----------------------------------------
    print("\n[1] Single-call invocation")
    for name, fn, mk in [
        ("add_node",       lambda t: add_node(t, functions, operations),       make_tree_simple),
        ("split",          lambda t: split(t, functions, operations),          make_tree_deeper),
        ("hoist_mutation", lambda t: hoist_mutation(t, functions, operations), make_tree_deeper),
    ]:
        t = mk()
        n_before = _count_nodes_py(t.get_root())
        lqf, lqr = fn(t)
        n_after = _count_nodes_py(t.get_root())
        delta = "+inf" if not np.isfinite(lqr) else f"{lqr - lqf:+.3f}"
        print(f"  {name:16s}: |N| {n_before:>2d}->{n_after:<2d}  "
              f"lqf={lqf:+8.3f}  lqr={lqr:+8.3f}  delta={delta}")
        assert np.isfinite(lqf), f"{name}: lqf must be finite"

    # --- 2. add_node x N --------------------------------------------------
    N = 300
    deltas_a, sizes_a = [], []
    for _ in range(N):
        t = make_tree_simple()
        n_before = _count_nodes_py(t.get_root())
        #print(t)
        lqf, lqr = add_node(t, functions, operations)
        #print(t)
        n_after = _count_nodes_py(t.get_root())
        assert n_after > n_before, "add_node failed to grow tree"
        assert np.isfinite(lqf) and np.isfinite(lqr), \
            "add_node densities must both be finite (on-support by construction)"
        deltas_a.append(lqr - lqf)
        sizes_a.append(n_after - n_before - 1)   # = nodes added by S
    sizes_a = np.array(sizes_a)
    print(f"\n[2] add_node x{N}:")
    print(f"    mean lqr-lqf      = {np.mean(deltas_a):+.3f}  "
          f"std={np.std(deltas_a):.3f}")
    print(f"    |S| (nodes added) min/median/max = "
          f"{sizes_a.min()}/{int(np.median(sizes_a))}/{sizes_a.max()}")
    # truncated-geom on {1,..,4} with rho=0.5 puts mass mostly on d=1 (|S|=1).
    # Larger d are exponentially rarer.

    # --- 3. split x N -----------------------------------------------------
    deltas_s, sizes_s = [], []
    for _ in range(N):
        t = make_tree_deeper()
        n_before = _count_nodes_py(t.get_root())
        lqf, lqr = split(t, functions, operations)
        n_after = _count_nodes_py(t.get_root())
        assert n_after > n_before, "split failed to grow tree"
        assert np.isfinite(lqf) and np.isfinite(lqr), \
            "split densities must both be finite"
        deltas_s.append(lqr - lqf)
        sizes_s.append(n_after - n_before - 1)
    sizes_s = np.array(sizes_s)
    print(f"\n[3] split x{N}:")
    print(f"    mean lqr-lqf      = {np.mean(deltas_s):+.3f}  "
          f"std={np.std(deltas_s):.3f}")
    print(f"    |S| min/median/max = "
          f"{sizes_s.min()}/{int(np.median(sizes_s))}/{sizes_s.max()}")

    # --- 4. hoist x N -----------------------------------------------------
    n_off = 0
    deltas_h, branch_leaf, branch_int = [], 0, 0
    for _ in range(N):
        # Use a richer seed tree so hoist has both leaf and internal targets.
        t = make_tree_deeper()
        n_before = _count_nodes_py(t.get_root())
        lqf, lqr = hoist_mutation(t, functions, operations)
        n_after = _count_nodes_py(t.get_root())
        assert n_after < n_before, f"hoist failed to shrink tree: {n_before}->{n_after}"
        assert np.isfinite(lqf), "hoist lqf must be finite"
        if not np.isfinite(lqr):
            n_off += 1
        else:
            deltas_h.append(lqr - lqf)
    print(f"\n[4] hoist_mutation x{N}:")
    print(f"    off-support (lqr=-inf): {n_off}/{N} = {n_off/N:.1%}")
    if deltas_h:
        print(f"    on-support mean lqr-lqf = {np.mean(deltas_h):+.3f}  "
              f"std={np.std(deltas_h):.3f}")

    # --- 5. add_node detailed-balance spot check --------------------------
    print("\n[5] add_node q_rev == -log(|N(T')|-1)  (paired leaf-hoist selection)")
    n_pass = 0
    for _ in range(100):
        t = make_tree_simple()
        lqf, lqr = add_node(t, functions, operations)
        n_Tp = _count_nodes_py(t.get_root())
        expected = -np.log(n_Tp - 1)
        if abs(lqr - expected) < 1e-10:
            n_pass += 1
    print(f"    {n_pass}/100 match")
    assert n_pass == 100

    # --- 6. split detailed-balance spot check -----------------------------
    print("\n[6] split    q_rev == -log(|N(T')|-1)  (paired internal-hoist selection)")
    n_pass = 0
    for _ in range(100):
        t = make_tree_deeper()
        lqf, lqr = split(t, functions, operations)
        n_Tp = _count_nodes_py(t.get_root())
        expected = -np.log(n_Tp - 1)
        if abs(lqr - expected) < 1e-10:
            n_pass += 1
    print(f"    {n_pass}/100 match")
    assert n_pass == 100

    # --- 7. hoist forward density spot check ------------------------------
    # hoist's lqf must be -log(|N(T)|-1), the uniform pick over non-root nodes.
    print("\n[7] hoist     q_fwd == -log(|N(T)|-1)  (uniform pick of v)")
    n_pass = 0
    for _ in range(100):
        t = make_tree_deeper()
        n_T = _count_nodes_py(t.get_root())
        lqf, _ = hoist_mutation(t, functions, operations)
        expected = -np.log(n_T - 1)
        if abs(lqf - expected) < 1e-10:
            n_pass += 1
    print(f"    {n_pass}/100 match")
    assert n_pass == 100

    print("\n" + "=" * 60)
    print("Phase 2.1.K smoke test: ALL CHECKS PASSED")
    print("=" * 60)
    print()
# ---------------------------------------------------------------------------
# Phase 2.1.PR smoke tests: AMState + parameter_refresh
# ---------------------------------------------------------------------------
def am_state_smoke_test():
    """Diagnostics for Adaptive Metropolis state and parameter_refresh.

    Checks (in order):
      1) constructor + attribute exposure (cdef public reachability)
      2) K=1 path: cold-start scalar shape, warm-start after K+1 visits
      3) K=3 path: cov stays SPD after RM updates; symmetry preserved
      4) lambda adaptation: drive RM with synthetic alpha_obs, expect
         log_lambda to move toward target
      5) end-to-end parameter_refresh on a fixed-topology tree
         (sin(x) + a0) with truth a0=2.0; expect convergence + accept rate
         hovering near alpha_star=0.44 after warm-up.
    """
    print("=" * 60)
    print("AMState / parameter_refresh smoke test")
    print("=" * 60)
    rng = np.random.default_rng(0)
    np.random.seed(0)

    # --- 1. constructor + attribute checks --------------------------------
    s1 = AMState(["a0"])
    s3 = AMState(["a0", "a1", "a2"])
    assert s1.K == 1 and s3.K == 3, "K mismatch"
    assert s1.alpha_star == 0.44, f"K=1 alpha* should be 0.44, got {s1.alpha_star}"
    assert s3.alpha_star == 0.234, f"K=3 alpha* should be 0.234, got {s3.alpha_star}"
    assert s1.mu.shape == (1,) and s3.C.shape == (3, 3), "shape init wrong"
    assert s1.log_lambda == 0.0 and s1.n_visits == 0
    print(f"[1] OK  param_names={s1.param_names}  K=1 alpha*={s1.alpha_star}")
    print(f"        K=3  C.shape={s3.C.shape}  mu={s3.mu}")

    # --- 2. K=1 propose: cold-start then warm -----------------------------
    theta = np.array([3.0])
    cold_props = np.array([s1.propose(theta)[0] for _ in range(500)])
    cold_sd = cold_props.std()
    expected_cold_sd = max(s1.eps, abs(theta[0]) * s1.rel)  # = 0.3
    print(f"[2] K=1 cold-start: empirical sd={cold_sd:.3f}  expected~{expected_cold_sd:.3f}")
    assert 0.5 * expected_cold_sd < cold_sd < 1.5 * expected_cold_sd, \
        "cold-start sd off by >50%"

    # Force warm regime: pretend we visited K+1 times, then check that
    # propose() doesn't crash and uses Cholesky path.
    s1.n_visits = 5
    warm_sample = s1.propose(theta)
    assert warm_sample.shape == (1,), f"warm K=1 shape {warm_sample.shape} != (1,)"
    print(f"        warm propose returns shape {warm_sample.shape}: OK")

    # --- 3. K=3 update: covariance stays symmetric & PSD ------------------
    s3b = AMState(["a0", "a1", "a2"])
    for t in range(200):
        theta_t = rng.normal(size=3) * np.array([1.0, 2.0, 0.5])
        s3b.update(theta_t, accepted=True, alpha_obs=0.234)
    sym_err = float(np.max(np.abs(s3b.C - s3b.C.T)))
    eigs = np.linalg.eigvalsh(s3b.C)
    print(f"[3] K=3 after 200 updates:")
    print(f"        symmetry err = {sym_err:.2e}")
    print(f"        eigenvalues  = {eigs}")
    print(f"        empirical mu = {s3b.mu}")
    assert sym_err < 1e-12, "C lost symmetry"
    assert eigs.min() >= -1e-10, "C lost positive-semidefiniteness"

    # --- 4. lambda adaptation: feed alpha_obs above/below target ----------
    s_hi = AMState(["a0", "a1", "a2"])  # alpha* = 0.234
    s_lo = AMState(["a0", "a1", "a2"])
    for t in range(2000):
        # high acceptance -> log_lambda should INCREASE (broader proposals)
        s_hi.update(np.zeros(3), accepted=True, alpha_obs=0.9)
        # low acceptance -> log_lambda should DECREASE
        s_lo.update(np.zeros(3), accepted=False, alpha_obs=0.05)
    print(f"[4] lambda adaptation (target alpha*=0.234, 2000 RM steps):")
    print(f"        high-accept feed (alpha_obs=0.9):  log_lambda -> {s_hi.log_lambda:+.3f}")
    print(f"        low-accept  feed (alpha_obs=0.05): log_lambda -> {s_lo.log_lambda:+.3f}")
    assert s_hi.log_lambda > 0.5, "log_lambda should grow when alpha_obs >> target"
    assert s_lo.log_lambda < -0.5, "log_lambda should shrink when alpha_obs << target"

    # --- 5. parameter_refresh end-to-end on a fixed-topology tree ---------
    # Build  sin(x) + a0  with a0 initialized far from truth (= 2.0).
    x = np.linspace(0.001, 10.0, 200).astype(np.float64)
    y_true = np.sin(x) + 2.0
    domain = x.copy()
    a0_sym = sp.symbols("a0")
    expr = sp.sin(sp.symbols("x")) + a0_sym
    tree = expr_to_tree(expr, domain, ["sin(x)", "cos(x)", "x"])
    # Force a0 far from truth.
    for p in tree.get_params():
        p.set_value(-5.0)
    tree.tree_update(tree.get_root())
    # Cache the parameter Node directly: tree.tree_update() rebuilds the
    # name_values dict on every accept, so any dict reference captured here
    # would go stale. The Node object itself is mutated in place by
    # set_value(), so reading p.value each iteration is the safe path.
    pnode = tree.get_params()[0]
    init_val = pnode.value
    print(f"[5] end-to-end parameter_refresh on  sin(x) + a0  (truth a0=2.0)")
    print(f"        initial a0 = {init_val:.3f}")

    names = [p.get_fun().get_description() for p in tree.get_params()]
    am = AMState(names)
    sigma = 0.1
    T = 1.0

    accepts = 0
    n_iter = 1500
    trace = np.empty(n_iter, dtype=np.float64)
    for t in range(n_iter):
        accepted, _, _ = parameter_refresh(tree, y_true, sigma, am, T)
        if accepted:
            accepts += 1
        trace[t] = pnode.value

    final_a0 = trace[-1]
    post_warmup_mean = float(np.mean(trace[500:]))
    post_warmup_sd = float(np.std(trace[500:]))
    accept_rate = accepts / n_iter
    print(f"        final a0 = {final_a0:.4f}   "
          f"post-warmup mean = {post_warmup_mean:.4f} +/- {post_warmup_sd:.4f}")
    print(f"        accept rate = {accept_rate:.3f}   "
          f"(target alpha* = {am.alpha_star})")
    print(f"        AM final: log_lambda={am.log_lambda:+.3f}  "
          f"n_visits={am.n_visits}  C[0,0]={am.C[0, 0]:.3e}")
    # Loose convergence check: posterior mean should be within ~3 sigma
    # of the truth given sigma=0.1 and n=200 (analytic SE ~ sigma/sqrt(n) ~ 0.007).
    assert abs(post_warmup_mean - 2.0) < 0.1, \
        f"PR did not converge: post-warmup mean {post_warmup_mean} vs truth 2.0"
    # Acceptance should be in a reasonable window around target.
    assert 0.15 < accept_rate < 0.75, \
        f"accept rate {accept_rate} outside sane window"
    print("=" * 60)
    print("AMState smoke test: ALL CHECKS PASSED")
    print("=" * 60)
    print()

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
    polya = PolyaTree(operations, functions, alpha_0=1.0,
                      auto_register=True, strict_ops=True)

    grove = Grove(x,y, T, size, depth, functions, Npars=4, init_value=1.0, 
                  operations=operations, base_temp=1.0, use_gradients=False, 
                  burn_in=True, use_mpi=4, use_spawn=True, lambda_reg=0.01, polya=polya)
    
    print("=" * 60)
    print("Initial random trees:")
    for i, tree in enumerate(grove.trees[-10:]):
        print(f"  Tree {i}: {tree.tree_get_description()}")
    print("=" * 60)
    
    # Two-stage burn-in (does both depth-1 and depth-2 internally)
    #print("\n--- Two-Stage Burn-in ---")
    #grove.burn_in_depth2(y, replace_all=True, optimize_params=True )
    
    print(f"\nAfter burn-in, grove contains some depth-2 trees")
    for i, tree in enumerate(grove.trees[:10]):
        print(f"  Tree {i}: {tree.tree_get_description()} | posterior: {grove.root_posteriors[i]:.4f}")
    
    # Now run MCMC on the initialized grove
    print("\n--- Running MCMC evolution ---")
    t0 = time.time()
    #grove.run(y, n_steps=500)
    grove.run(y, n_steps=50)
    t1 = time.time()
    print(f"MCMC run completed in {t1 - t0:.2f} seconds")

    # After grove.run(...):
    stats = grove.move_stats()
    print("Per-move stats:")
    for name, s in stats.items():
        print(f"  {name:20s} proposed={s['proposed']:5d} "
            f"accepted={s['accepted']:5d}  rate={s['rate']:.3f}")
    print(f"AM states cached: {len(grove._am_states)} topologies")
    # Report
    print("\n" + "=" * 60)
    print(f"Final acceptance rate: {grove.acceptance_rate():.3f}")
    best = grove.best_tree()
    print("Best tree after MCMC:")
    print(f"  Expression: {best.tree_get_description()}")
    print(f"  Posterior: {best.get_superficial_posterior(y):.4f}")
    print(f"  Target: 2.0 + sin(x)")
    print("=" * 60)

    for tree in grove.trees:
        polya.assert_covers(tree.get_root())
    print(f"polya vocab after run: n_ops={polya.n_ops} n_prims={polya.n_prims}")
    print(f"  primitives: {polya.primitives}")

def main():
    # Definire le funzioni disponibili
    functions = ["sin(x)", "cos(x)", "exp(x)", "x**2", "x**3"]
    operations = ["+", "-", "*", "/", "o", "**"]
    # Crea due alberi di funzioni
    x = np.linspace(0, 10, 100).astype(np.float64)
    y = np.sin(x)
    
    tree1 = Tree(depth=3, functions=functions, x=x, Npars = 2, value=np.array([3, 4, 5, 6, 7, 8]))
    
    print(tree1.tree_get_description())
    tree1.compute_fitness_tree(y)
    tree1.print_fitness_tree()
    tree2 = Tree(depth=4, functions=functions, x=x, Npars =4, value=np.array([545, 546, 547, 548]))
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
    
    subtree_mutation(tree1, functions)
    
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

    
    print("\nSplit move:")
    lq_f, lq_r = split(tree1, functions, operations)
    print(f"  shape: {tree1.tree_get_description()}")
    print(f"  log_q_fwd={lq_f:.3f}  log_q_rev={lq_r:.3f}  delta={lq_r - lq_f:+.3f}")
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
    crossover_smoke_test()
    am_state_smoke_test()
    k_changing_smoke_test()
    grove_smoke_test()
    



if __name__ == "__main__":
    main()
