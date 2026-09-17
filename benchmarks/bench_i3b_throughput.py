"""bench_i3b_throughput.py -- I3-B preparation, D4: proposal throughput
and memory of the ladder EMC at d = 3 versus the pooled size n.

Builds the synthetic pooled dataset (i3b_pool.synthetic_sims), thins it
to n rows, seeds a Forest with the I3-B ladder shape (6 rungs x 4
slots) from hand-built depth-2 skeletons, and runs R rounds of the P1
protocol (6 steps per slot, 4 swaps, 1 crossover per round). Reports
proposals per second (within-grove MH proposals summed over rungs),
the wall per round, the single-evaluate and single-BIC-scoring cost,
and the peak resident set size. One JSON line per n.

Run (WSL, mcmc_koopman, from gpu_ms):
    python bench_i3b_throughput.py --n 10000 --rounds 6
"""
import argparse
import json
import os
import resource
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np

import real_functionC
import exp_common
import i3b_pool
from function_treeC import node_from_skel, Tree
from real_functionC import RealFunctionC

OPS4 = ["+", "-", "*", "/"]
FUNCS = ["x0", "x1", "x2", "exp(x0)", "exp(x1)", "exp(x2)", "tanh(x0)",
         "tanh(x1)", "tanh(x2)", "log(x1)", "log(x2)"]
LADDER = [1.0 / b for b in (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)]
SEEDS = [
    ('O', '+', ('O', '*', ('L', 'a0'), ('L', 'x0')), ('L', 'a1')),
    ('O', '*', ('O', '*', ('L', 'a0'), ('L', 'x1')), ('L', 'x2')),
    ('O', '+', ('O', '*', ('L', 'a0'), ('L', 'exp(x1)')),
     ('O', '*', ('L', 'a1'), ('L', 'x0'))),
    ('O', '/', ('L', 'x0'), ('O', '+', ('L', 'a0'), ('L', 'tanh(x2)'))),
]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    # pooled synthetic data thinned to ~n rows
    n_sims = 100
    m = max(20, int(np.ceil(args.n / n_sims)))
    sims = i3b_pool.synthetic_sims(n_sims=n_sims, m=m, seed=args.seed)
    pool = i3b_pool.build_pooled(sims)
    pool = i3b_pool.thin_pooled(pool, n_t=m, seed=args.seed)
    pool, n_tr = i3b_pool.split_train_first(pool, held_fold=0)
    X = pool["X"]
    y = pool["y"]
    sigma = pool["sigma"]
    n = y.shape[0]
    rss0 = rss_mb()

    real_functionC.set_variables(i3b_pool.VARIABLES)
    exp_common.set_caps()

    # single-leaf evaluate and single BIC scoring cost at this n
    rf = RealFunctionC.from_string("a0*exp(x1)*x0 + a1*tanh(x2)", X)
    t0 = time.perf_counter()
    for _ in range(200):
        rf.get_eval(X, {"a0": 1.1, "a1": 0.3})
    t_eval = (time.perf_counter() - t0) / 200.0
    root = node_from_skel(SEEDS[2], X, None)
    tree = Tree(0, list(FUNCS), X, root, 0, 1.0, 1.0, 0.01, sigma)
    t0 = time.perf_counter()
    tree.log_marginal_bic(y, True)
    t_bic = time.perf_counter() - t0

    roots = [node_from_skel(s, X, None) for s in SEEDS]
    t0 = time.perf_counter()
    forest = exp_common.seeded_forest(X, y, LADDER, 4, FUNCS, OPS4,
                                      sigma, args.seed, roots)
    t_seed = time.perf_counter() - t0
    rss1 = rss_mb()

    t0 = time.perf_counter()
    for _ in range(args.rounds):
        forest.run_all(y, 6)
        for _ in range(4):
            forest.attempt_swaps(y)
        forest.attempt_crossovers(y)
    wall = time.perf_counter() - t0
    n_prop = sum(sum(g.move_proposals.values()) for g in forest.groves)
    n_acc = sum(sum(g.move_accepts.values()) for g in forest.groves)
    rss2 = rss_mb()
    rec = dict(n=int(n), n_train=int(n_tr), rounds=args.rounds,
               proposals=int(n_prop), accepts=int(n_acc),
               wall_s=round(wall, 2),
               proposals_per_s=round(n_prop / wall, 2),
               s_per_round=round(wall / args.rounds, 2),
               eval_ms=round(1e3 * t_eval, 3), bic_score_s=round(t_bic, 3),
               seed_forest_s=round(t_seed, 2),
               rss_mb_data=round(rss0, 1), rss_mb_forest=round(rss1, 1),
               rss_mb_peak=round(rss2, 1),
               best=forest.best_tree().tree_get_description()[:70])
    print("BENCH " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
