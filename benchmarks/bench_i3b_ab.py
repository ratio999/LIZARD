"""bench_i3b_ab.py -- speed A/B for the I3-B kernel change (Grove burn-in
budget). Uses ONLY API common to the pre-change and post-change kernel
(no burn_in_top_k / burn_in_cap keywords), so the same script runs
against both builds. Three workloads, each repeated and reported as
the minimum wall (the fairest single number under OS noise):

  1. d = 1 ladder EMC throughput (6 rungs x 4 slots, P1 round shape)
     on a Feynman-style target: proposals per second.
  2. d = 1 serial EXHAUSTIVE burn-in wall (the code path the budget
     change touches; default arguments = historical path).
  3. d = 3 ladder EMC throughput on 1e4 pooled rows.

Run: python bench_i3b_ab.py --label old|new   (prints one JSON line)
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SRC = os.path.join(os.path.dirname(_HERE), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import numpy as np

import real_functionC
import exp_common
from groveC import Grove
from function_treeC import node_from_skel

OPS4 = ["+", "-", "*", "/"]
LADDER = [1.0 / b for b in (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)]


def emc_throughput(X, y, sigma, funcs, skels, rounds, seed):
    roots = [node_from_skel(s, X, None) for s in skels]
    forest = exp_common.seeded_forest(X, y, LADDER, 4, funcs, OPS4, sigma,
                                      seed, roots)
    t0 = time.perf_counter()
    for _ in range(rounds):
        forest.run_all(y, 6)
        for _ in range(4):
            forest.attempt_swaps(y)
        forest.attempt_crossovers(y)
    wall = time.perf_counter() - t0
    n_prop = sum(sum(g.move_proposals.values()) for g in forest.groves)
    return wall, n_prop


def part1(reps):
    real_functionC.set_variables(("x",))
    exp_common.set_caps()
    rng = np.random.default_rng(1)
    x = np.linspace(0.5, 3.5, 200)
    y = (np.sin(x) + 0.5 * x + rng.normal(0.0, 0.05, x.size)).astype(
        np.float64)
    funcs = ["sin(x)", "cos(x)", "x"]
    skels = [('O', '+', ('O', '*', ('L', 'a0'), ('L', 'x')), ('L', 'sin(x)')),
             ('O', '*', ('L', 'a0'), ('L', 'cos(x)')),
             ('O', '+', ('L', 'sin(x)'), ('L', 'a0')),
             ('O', '/', ('L', 'x'), ('O', '+', ('L', 'a0'), ('L', 'cos(x)')))]
    best = None
    for r in range(reps):
        wall, n_prop = emc_throughput(x, y, 0.05, funcs, skels, 40, 11)
        if best is None or wall < best[0]:
            best = (wall, n_prop)
    return dict(d1_emc_wall_s=round(best[0], 2), d1_emc_proposals=best[1],
                d1_emc_prop_per_s=round(best[1] / best[0], 1))


def part2(reps):
    real_functionC.set_variables(("x",))
    rng = np.random.default_rng(2)
    x = np.linspace(0.5, 3.5, 60)
    y = (np.sin(x) + 0.5 * x + rng.normal(0.0, 0.05, x.size)).astype(
        np.float64)
    funcs = ["sin(x)", "cos(x)", "x"]
    best = None
    for r in range(reps):
        t0 = time.perf_counter()
        g = Grove(x, y, 1.0, 8, 2, list(funcs), sigma=0.05,
                  operations=OPS4, use_gradients=False, burn_in=True,
                  use_mpi=0, seed=5)
        wall = time.perf_counter() - t0
        n = len(g.trees)
        if best is None or wall < best[0]:
            best = (wall, n)
    return dict(d1_burnin_wall_s=round(best[0], 2), d1_burnin_candidates=best[1])


def part3(reps):
    import i3b_pool
    sims = i3b_pool.synthetic_sims(n_sims=100, m=100, seed=3)
    pool = i3b_pool.build_pooled(sims)
    pool, n_tr = i3b_pool.split_train_first(pool, 0)
    X, y, sigma = pool["X"], pool["y"], pool["sigma"]
    real_functionC.set_variables(("x0", "x1", "x2"))
    exp_common.set_caps()
    funcs = ["x0", "x1", "x2", "exp(x0)", "exp(x1)", "exp(x2)", "tanh(x0)",
             "tanh(x1)", "tanh(x2)", "log(x1)", "log(x2)"]
    skels = [('O', '+', ('O', '*', ('L', 'a0'), ('L', 'x0')), ('L', 'a1')),
             ('O', '*', ('O', '*', ('L', 'a0'), ('L', 'x1')), ('L', 'x2')),
             ('O', '+', ('O', '*', ('L', 'a0'), ('L', 'exp(x1)')),
              ('O', '*', ('L', 'a1'), ('L', 'x0'))),
             ('O', '/', ('L', 'x0'), ('O', '+', ('L', 'a0'), ('L', 'tanh(x2)')))]
    best = None
    for r in range(reps):
        wall, n_prop = emc_throughput(X, y, sigma, funcs, skels, 6, 3)
        if best is None or wall < best[0]:
            best = (wall, n_prop)
    real_functionC.set_variables(("x",))
    return dict(d3_n=int(y.shape[0]), d3_emc_wall_s=round(best[0], 2),
                d3_emc_proposals=best[1],
                d3_emc_prop_per_s=round(best[1] / best[0], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    rec = dict(label=args.label)
    rec.update(part1(args.reps))
    rec.update(part2(args.reps))
    rec.update(part3(max(2, args.reps - 1)))
    print("AB " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
