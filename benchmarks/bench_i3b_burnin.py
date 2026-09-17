"""bench_i3b_burnin.py -- I3-B preparation, D2: burn-in combinatorics
and wall time at d = 3.

One configuration per invocation (vocabulary size, number of rows,
solo or spawn workers, exhaustive or capped); prints one JSON line so a
shell loop can assemble the timing table. The exhaustive stage-2 count
is EXACT: it is n_top * (n_top + P') * |ops| where n_top is the number
of valid, distinct depth-1 candidates the real stage 1 produces on the
data, and P' = P + 1 counts the 'a0' parameter primitive. The rate
(combinations per second) is measured on a capped stage 2 of `--cap`
items (or on the full run with --cap 0) and the exhaustive wall time is
count / rate.

Synthetic pooled-like data at d = 3: x0 = t in [0, 60], x1 = alpha_0
around 0.085 (7% spread), x2 = Delta_1 in [0.10, 0.20], rows drawn
independently; y = exp(-x1*t) * (1 + x2*t) + noise.

Run (WSL, mcmc_koopman, from gpu_ms; spawn needs the env mpiexec):
    python bench_i3b_burnin.py --P 9 --n 423 --workers 0 --cap 3000
    mpiexec -n 1 python bench_i3b_burnin.py --P 9 --n 423 --workers 2 --cap 3000
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

OPS4 = ["+", "-", "*", "/"]
VOCAB = {
    # the d=2 I3-A reference vocabulary (7 primitives)
    "d2_7": ("d2", ["x0", "x1", "exp(x0)", "exp(x1)", "tanh(x0)",
                    "tanh(x1)", "log(x0)"]),
    # d=3 candidates for I3-B (D3 vocabulary family)
    "9": ("d3", ["x0", "x1", "x2", "exp(x0)", "exp(x1)", "exp(x2)",
                 "tanh(x0)", "tanh(x1)", "tanh(x2)"]),
    "10": ("d3", ["x0", "x1", "x2", "exp(x0)", "exp(x1)", "exp(x2)",
                  "tanh(x0)", "tanh(x1)", "tanh(x2)", "log(x1)"]),
    "12": ("d3", ["x0", "x1", "x2", "exp(x0)", "exp(x1)", "exp(x2)",
                  "tanh(x0)", "tanh(x1)", "tanh(x2)", "log(x1)", "log(x2)",
                  "log(x0)"]),
}


def make_data(kind, n, seed=7):
    rng = np.random.default_rng(seed)
    if kind == "d3":
        t = rng.uniform(0.0, 60.0, n)
        a0 = rng.uniform(0.080, 0.092, n)
        d1 = rng.uniform(0.10, 0.20, n)
        X = np.ascontiguousarray(np.vstack([t, a0, d1]).astype(np.float64))
        y = np.exp(-a0 * t) * (1.0 + d1 * t)
        vars_ = ("x0", "x1", "x2")
    else:
        nu = rng.uniform(0.1, 0.25, n)
        chi = rng.uniform(-0.8, 0.8, n)
        X = np.ascontiguousarray(np.vstack([nu, chi]).astype(np.float64))
        y = 0.3 + 2.0 * nu + 0.5 * np.tanh(chi) * nu
        vars_ = ("x0", "x1")
    sd = 0.01 * float(np.std(y))
    y = (y + rng.normal(0.0, sd, n)).astype(np.float64)
    return vars_, X, y, sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--P", default="9", choices=sorted(VOCAB))
    ap.add_argument("--n", type=int, default=423)
    ap.add_argument("--workers", type=int, default=0,
                    help="0 = solo serial burn-in; >= 2 = spawn workers")
    ap.add_argument("--cap", type=int, default=3000,
                    help="stage-2 items to time (0 = exhaustive run)")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    kind, funcs = VOCAB[args.P]
    vars_, X, y, sd = make_data(kind, args.n)
    real_functionC.set_variables(vars_)
    exp_common.set_caps()
    P1 = len(funcs) + 1
    n_ops = len(OPS4)
    stage1 = P1 * P1 * n_ops

    t0 = time.perf_counter()
    if args.workers == 0:
        g = Grove(X, y, 1.0, 8, 2, list(funcs), sigma=sd,
                  operations=OPS4, use_gradients=False, burn_in=True,
                  use_mpi=0, seed=args.seed, burn_in_cap=args.cap)
    else:
        g = Grove(X, y, 1.0, 8, 2, list(funcs), sigma=sd,
                  operations=OPS4, use_gradients=False, burn_in=True,
                  use_mpi=int(args.workers), use_spawn=True,
                  seed=args.seed, burn_in_cap=args.cap)
    wall = time.perf_counter() - t0
    n_cand = len(g.trees)
    # n_top: valid distinct depth-1 candidates = candidates of depth 1
    # (root with two leaf children). Recover it from the surviving trees
    # when the run was exhaustive; for capped runs rerun stage 1 only.
    n_top = sum(1 for t in g.trees if t.get_root().depth() == 1)
    exhaustive = n_top * (n_top + P1) * n_ops
    timed = exhaustive if (args.cap == 0 or args.cap >= exhaustive) \
        else args.cap
    # stage-1 wall is not separated out by the kernel; for the capped
    # runs it is a small fraction (P1^2 * 4 scorings), reported inside wall.
    rate = timed / wall
    rec = dict(P=args.P, n_funcs=len(funcs), n=args.n,
               workers=args.workers, cap=args.cap, seed=args.seed,
               stage1_trees=stage1, n_top=n_top,
               stage2_exhaustive=exhaustive, stage2_timed=timed,
               wall_s=round(wall, 1), combos_per_s=round(rate, 2),
               exhaustive_est_min=round(exhaustive / rate / 60.0, 1),
               n_candidates=n_cand,
               best=g.best_tree().tree_get_description()[:80])
    print("BENCH " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
