"""exp_mr_i3b_fullspace.py -- I3-B: FULL-SPACE symbolic regression on the
pooled merger-ringdown catalogue (template driver; see
I3B_PREP_REPORT.md for the certification and the go/no-go settings).

Instead of one expression per simulation in t followed by an
interpolation of its coefficients (I3-A), the machine searches ONE
expression over the three-variable domain

    x0 = t            time inside a simulation
    x1 = alpha_0      fundamental damping rate of the remnant
    x2 = Delta_1      first-overtone offset, alpha_1 - alpha_0

pooled over all simulations, so the binary-dependence is part of the
discovered structure. x1 and x2 are constant per simulation and are
known functions of the binary: at deployment the expression is
evaluated with k = 0 per-binary free parameters.

Physical coordinates (D6). Remnant ringing rates rather than raw binary
parameters, because the anchors found on the science side are SHORT in
them (the leading phase rate is exactly 2*alpha_0; catalogue spread of
alpha_0 is 7 %), so the search space is one where the answer is simple.
No kernel change is involved: they are ordinary registered variables.

R3 accounting. Every constant fitted on the pooled data is a parameter
of the pooled fit: it pays BIC against n_pooled (the number of TRAIN
rows), once, at discovery. Deployed evaluation on a new binary has
k = 0. The structure prior (Polya, frozen vocabulary) is therefore the
ONLY Occam term at deployment; i3b_polya_pricing prices it, and every
archive entry stores its skeleton so the price is exact. The output
JSON carries n_pooled, n_full, k, held_fold and fold_sims for the
post-run three-term tables.

Protocol. The P1/SR-1 ladder-EMC recipe (same ladder, steps, swaps,
crossovers per round, spawn burn-in at depth 2, running archive), on
the FD-certified d = 3 kernel: variables registered BEFORE any
construction, compose 'o' excluded, structural tilt OFF (untested at
d > 1), vocabulary per D3

    per variable {x, exp(x), tanh(x)}; log(x) for the positive ones
    (x1, x2 by default; x0 = t is 0 at the peak, so log(x0) is off)

and ops {+, -, *, /}. Burn-in is budgeted (burn_in_top_k / burn_in_cap,
deterministic under the seed), because the exhaustive depth-2 stage 2
of this vocabulary is ~5e5 combinations on 1e4+ rows.

Data. A pooled file written by i3b_pool.save_pooled (--pool), or the
synthetic catalogue of i3b_pool.synthetic_sims (--synthetic, the
default under --smoke). Thinning is stratified by simulation (--n-t
rows per simulation, --n-sims simulations, --thin-seed). The held-out
set is a BY-SIMULATION fold (--held-fold, deterministic hash of the
simulation id); the machine never sees its rows in the likelihood.

Robustness (D5):
  --sigma-floor-frac F  lift every sigma to >= F * robust spread of the
                        TRAIN targets (IQR/1.349); default 0.02, 0 = off
  --winsor P            clip TRAIN targets to [P, 100-P] percentiles
  --full-guard          the chain's domain is the FULL row set with
                        sigma = inf on held-out rows: zero likelihood
                        weight, but the kernel's validity gate then
                        rejects any candidate non-finite on ANY row of
                        the full domain. Side effect: the BIC charge is
                        0.5*k*log(n_full) instead of 0.5*k*log(n_train),
                        a constant per parameter recorded in the JSON.
  archive hook          predictions on ALL rows at offer time, chi2 on
                        train rows only, held-out chi2 reported too.

Run (WSL, mcmc_koopman env, from gpu_ms; the spawn burn-in needs the
env's mpiexec; PYTHONHASHSEED pinned so that a run is bit-reproducible
from its seed -- set iteration order otherwise reaches the optimiser's
parameter ordering, see I3B_PREP_REPORT.md section 8):
    ENV=/home/el_bardo/anaconda3/envs/mcmc_koopman
    PYTHONHASHSEED=0 "$ENV/bin/mpiexec" -n 1 "$ENV/bin/python" \
        exp_mr_i3b_fullspace.py --smoke --seed 1
    PYTHONHASHSEED=0 "$ENV/bin/mpiexec" -n 1 "$ENV/bin/python" \
        exp_mr_i3b_fullspace.py \
        --pool results/mr_i3b/datasets/pool_v1.npz --seed 101 \
        --n-t 40 --burn-top-k 150 --burn-cap 60000 --full-guard \
        [--atom 'E=c0*exp(-c1*x0*x1)']

Output: results/mr_i3b/runs/<tag>__seed<seed>.json (refuses overwrite).
This template does not launch production runs: use --smoke first.
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

import numpy as np

import real_functionC
import exp_common
import i3b_pool
import i3b_polya_pricing as pricing
from function_treeC import tree_to_skel

OUT_DIR = os.path.join(_HERE, "results", "mr_i3b", "runs")

# P1 protocol constants (PLAN §4.2 family), unchanged from I3-A
_BETAS = (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)
LADDER = [1.0 / b for b in _BETAS]
SIZE_PER_GROVE = 4
WARMUP_STEPS = 100
STEPS_PER_ROUND = 6
SWAPS_PER_ROUND = 4
XOVERS_PER_ROUND = 1
ROUNDS_DEFAULT = 300
G_TAU = 1.0

ARCHIVE_CAP = 50
ARCHIVE_KEEP = 200

VARIABLES = i3b_pool.VARIABLES
OPS_SAFE4 = pricing.OPS_SAFE4


class ArchiveTop:
    """All-time archive, full-space flavour: predictions on the FULL row
    set (train first, then held-out) at offer time; chi2 on train rows;
    held-out chi2 reported; skeleton stored for exact Polya pricing."""

    def __init__(self, X_full, y_full, s_full, n_tr, polya_t0, funcs,
                 atom_specs=None, cap=ARCHIVE_CAP, keep=ARCHIVE_KEEP):
        self.X_full = X_full
        self.y_full, self.s_full = y_full, s_full
        self.n_tr = int(n_tr)
        self.polya_t0 = polya_t0
        self.funcs = list(funcs)
        self.atom_specs = atom_specs
        self.log_n_total = float(np.log(
            pricing.n_total_cells(funcs, len(atom_specs) if atom_specs else 0)))
        assert polya_t0.n_prims + pricing.N_OPS_POLYA == \
            pricing.n_total_cells(funcs, len(atom_specs) if atom_specs else 0)
        self.cap, self.keep = int(cap), int(keep)
        self.pool = {}
        self.n_offered = self.n_evals = self.n_rejected_nonfinite = 0

    def offer(self, tree, post, rnd):
        self.n_offered += 1
        desc = tree.tree_get_description()
        cur = self.pool.get(desc)
        if cur is not None and post <= cur["post"]:
            return
        try:
            pred = np.asarray(tree.evaluate(self.X_full), dtype=np.float64)
        except Exception:
            return
        if pred.shape != (self.X_full.shape[1],) \
                or not np.all(np.isfinite(pred)):
            # definedness guard at the archive: a candidate non-finite on
            # any row of the full domain is never archived
            self.n_rejected_nonfinite += 1
            return
        self.n_evals += 1
        nv = dict(tree.get_param_dicts()[1])
        k = len(nv)
        n_tr = self.n_tr
        r = (pred - self.y_full) / self.s_full
        chi2_tr = float(np.sum(r[:n_tr] ** 2) / max(n_tr - k, 1))
        n_ho = pred.shape[0] - n_tr
        chi2_ho = (float(np.sum(r[n_tr:] ** 2) / max(n_ho, 1))
                   if n_ho > 0 else float("nan"))
        skel = tree_to_skel(tree.get_root())
        # t = 0 structure price by the closed form -N*log(n_total), which
        # test_i3b_prep P-e certifies equal to the kernel's own
        # log_prior_from_tree under the frozen vocabulary (no throwaway
        # tree per offer). The skeleton is stored for re-pricing.
        n_nodes = pricing.skel_n_nodes(skel)
        lp0 = -float(n_nodes) * self.log_n_total
        self.pool[desc] = {
            "desc": desc, "post": float(post), "theta": nv, "k": k,
            "n_nodes": int(n_nodes), "log_prior_t0": float(lp0),
            "skel": pricing.skel_to_json(skel),
            "chi2_dof_train": chi2_tr, "chi2_dof_heldout": chi2_ho,
            "pred_full": [round(float(v), 10) for v in pred],
            "first_round": (cur["first_round"] if cur is not None
                            else int(rnd)),
            "best_round": int(rnd)}
        if len(self.pool) > 2 * self.keep:
            ranked = sorted(self.pool.values(), key=lambda e: -e["post"])
            self.pool = {e["desc"]: e for e in ranked[:self.keep]}

    def update_from_forest(self, forest, rnd):
        for g in forest.groves:
            posts = list(g.root_posteriors)
            for i, tr in enumerate(g.trees):
                p = posts[i]
                if p is not None and np.isfinite(p):
                    self.offer(tr, float(p), rnd)

    def top(self):
        return sorted(self.pool.values(),
                      key=lambda e: -e["post"])[:self.cap]


def parse_atoms(specs):
    """--atom NAME=TEMPLATE entries -> AtomSpec list (templates in the
    registered variables; slots c0, c1, ...)."""
    if not specs:
        return None
    out = []
    for s in specs:
        name, tmpl = s.split("=", 1)
        out.append(real_functionC.AtomSpec(name.strip(), tmpl.strip()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=None,
                    help="pooled .npz from i3b_pool.save_pooled")
    ap.add_argument("--synthetic", action="store_true",
                    help="use i3b_pool.synthetic_sims (default under "
                         "--smoke when --pool is absent)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    # data handling
    ap.add_argument("--held-fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--n-t", type=int, default=0,
                    help="rows per simulation after thinning (0 = all)")
    ap.add_argument("--n-sims", type=int, default=0,
                    help="stratified subset of simulations (0 = all)")
    ap.add_argument("--thin-seed", type=int, default=0)
    # vocabulary
    ap.add_argument("--log-vars", default="x1,x2",
                    help="variables declared positive (get log(x))")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--atom", action="append", default=None,
                    help="NAME=TEMPLATE opaque atom, e.g. "
                         "E=c0*exp(-c1*x0*x1) (repeatable)")
    # burn-in budget
    ap.add_argument("--burn-top-k", type=int, default=0)
    ap.add_argument("--burn-cap", type=int, default=0)
    # robustness
    ap.add_argument("--sigma-floor-frac", type=float, default=0.02)
    ap.add_argument("--winsor", type=float, default=0.0)
    ap.add_argument("--full-guard", action="store_true")
    args = ap.parse_args()

    if args.pool is None and not args.synthetic:
        if args.smoke:
            args.synthetic = True
        else:
            raise SystemExit("pass --pool PATH (or --synthetic)")
    tag = args.tag or ("synthetic" if args.synthetic
                       else os.path.splitext(os.path.basename(args.pool))[0])
    out = os.path.join(OUT_DIR, "%s%s__fold%d__seed%d.json"
                       % ("smoke_" if args.smoke else "", tag,
                          args.held_fold, args.seed))
    if os.path.exists(out):
        raise SystemExit("REFUSING to overwrite %s" % out)
    os.makedirs(OUT_DIR, exist_ok=True)
    rounds = 25 if args.smoke else args.rounds
    warmup = 20 if args.smoke else WARMUP_STEPS
    workers = int(os.environ.get("I3B_WORKERS", "2"))
    burn_top_k = args.burn_top_k
    burn_cap = args.burn_cap
    if args.smoke and burn_top_k == 0 and burn_cap == 0:
        burn_top_k, burn_cap = 60, 3000

    # variables FIRST -- before any tree/atom/Grove construction
    real_functionC.set_variables(VARIABLES)
    exp_common.set_caps()

    positive = () if args.no_log else tuple(
        v.strip() for v in args.log_vars.split(",") if v.strip())
    for v in positive:
        if v not in VARIABLES:
            raise SystemExit("--log-vars: %r is not a variable" % v)
    funcs = pricing.vocabulary_i3b(VARIABLES, positive=positive,
                                   with_log=bool(positive))
    atoms = parse_atoms(args.atom)

    # ---- data ------------------------------------------------------------
    if args.synthetic:
        n_sims = 24 if args.smoke else 120
        m = 40 if args.smoke else 300
        pool = i3b_pool.build_pooled(
            i3b_pool.synthetic_sims(n_sims=n_sims, m=m, seed=args.thin_seed),
            n_folds=args.n_folds)
        src = "synthetic(n_sims=%d, m=%d)" % (n_sims, m)
    else:
        pool = i3b_pool.load_pooled(args.pool)
        src = args.pool
    sims_subset = None
    if args.n_sims > 0:
        sims_subset = i3b_pool.select_sims_stratified(
            pool, args.n_sims, args.thin_seed)
    if args.n_t > 0 or sims_subset is not None:
        pool = i3b_pool.thin_pooled(
            pool, n_t=(args.n_t if args.n_t > 0 else 10 ** 9),
            seed=args.thin_seed, sims_subset=sims_subset)
    pool, n_tr = i3b_pool.split_train_first(pool, held_fold=args.held_fold)
    X_full, y_full, s_full = pool["X"], pool["y"], pool["sigma"]
    n_full = int(y_full.shape[0])
    if n_tr == 0 or n_tr == n_full:
        raise SystemExit("degenerate split: train %d of %d rows"
                         % (n_tr, n_full))
    fold_sims = sorted(set(pool["sid"][n_tr:].tolist()))

    # robustness: sigma floor (train spread, applied to all rows),
    # winsorised TRAIN targets
    y_tr = np.ascontiguousarray(y_full[:n_tr])
    if args.sigma_floor_frac > 0:
        s_full, floor, n_lifted = i3b_pool.sigma_with_floor(
            s_full, y_tr, args.sigma_floor_frac)
    else:
        floor, n_lifted = 0.0, 0
    y_tr = i3b_pool.winsorise(y_tr, args.winsor)
    s_tr = np.ascontiguousarray(s_full[:n_tr])
    X_tr = np.ascontiguousarray(X_full[:, :n_tr])
    y_full_scored = np.concatenate([y_tr, y_full[n_tr:]])

    # the chain's data: train rows, or the full set with weightless
    # held-out rows under --full-guard
    if args.full_guard:
        X_chain = X_full
        y_chain = np.ascontiguousarray(y_full_scored)
        s_chain = np.ascontiguousarray(np.concatenate(
            [s_tr, np.full(n_full - n_tr, np.inf)]))
        n_bic = n_full
    else:
        X_chain, y_chain, s_chain = X_tr, y_tr, s_tr
        n_bic = n_tr

    print("[i3b] %s  rows %d (train %d, held-out %d in fold %d: %d sims)"
          "  seed %d" % (src, n_full, n_tr, n_full - n_tr, args.held_fold,
                         len(fold_sims), args.seed), flush=True)
    print("[i3b] vocabulary (%d): %s ; ops %s ; atoms %s"
          % (len(funcs), " ".join(funcs), "".join(OPS_SAFE4),
             [a.name for a in atoms] if atoms else None), flush=True)
    print("[i3b] sigma floor %.3g (%d rows lifted), winsor %.2f, "
          "full-guard %s, burn top_k %d cap %d"
          % (floor, n_lifted, args.winsor, args.full_guard, burn_top_k,
             burn_cap), flush=True)

    # ---- search ------------------------------------------------------------
    t0 = time.perf_counter()
    roots, posts, wall_b = exp_common.spawn_burn_in(
        X_chain, y_chain, funcs, OPS_SAFE4, s_chain.copy(), args.seed,
        atoms=atoms, n_workers=workers, burn_in_top_k=burn_top_k,
        burn_in_cap=burn_cap)
    print("[i3b] burn-in: %d candidates (%.1f s)" % (len(roots), wall_b),
          flush=True)
    forest = exp_common.seeded_forest(
        X_chain, y_chain, LADDER, SIZE_PER_GROVE, funcs, OPS_SAFE4,
        s_chain.copy(), args.seed, roots, atoms=atoms)
    forest.run_all(y_chain, warmup)

    polya_t0 = pricing.make_t0_polya(funcs, atoms=atoms)
    atom_specs = ({a.name: a for a in atoms} if atoms else None)
    archive = ArchiveTop(X_full, y_full_scored, s_full, n_tr, polya_t0,
                         funcs, atom_specs)
    archive.update_from_forest(forest, -1)
    for rnd in range(rounds):
        forest.run_all(y_chain, STEPS_PER_ROUND)
        for _ in range(SWAPS_PER_ROUND):
            forest.attempt_swaps(y_chain)
        for _ in range(XOVERS_PER_ROUND):
            forest.attempt_crossovers(y_chain)
        archive.update_from_forest(forest, rnd)
        if (rnd + 1) % 25 == 0:
            g0 = forest.groves[0]
            arr = np.asarray([p if p is not None else np.nan
                              for p in g0.root_posteriors], dtype=np.float64)
            print("    i3b round %d/%d (%.1f s)  best cold %.3f  archive %d"
                  "  rejected-nonfinite %d"
                  % (rnd + 1, rounds, time.perf_counter() - t0,
                     float(np.nanmax(arr)), len(archive.pool),
                     archive.n_rejected_nonfinite), flush=True)
    wall = time.perf_counter() - t0

    stats = forest.ladder_stats()
    rec = dict(
        source=src, seed=args.seed, rounds=rounds, smoke=bool(args.smoke),
        variables=dict(zip(VARIABLES, i3b_pool.COORD_NAMES)),
        funcs=funcs, ops=OPS_SAFE4,
        atoms=([[a.name, a.template] for a in atoms] if atoms else None),
        ladder=LADDER,
        rounds_protocol=dict(steps=STEPS_PER_ROUND, swaps=SWAPS_PER_ROUND,
                             xovers=XOVERS_PER_ROUND, warmup=warmup,
                             size_per_grove=SIZE_PER_GROVE),
        thinning=dict(n_t=args.n_t, n_sims=args.n_sims,
                      thin_seed=args.thin_seed, sims_subset=sims_subset),
        robustness=dict(sigma_floor_frac=args.sigma_floor_frac,
                        sigma_floor=floor, n_sigma_lifted=n_lifted,
                        winsor=args.winsor, full_guard=bool(args.full_guard)),
        # R3 bookkeeping
        n_pooled=int(n_tr), n_full=n_full, n_bic=int(n_bic),
        bic_per_param_nats=0.5 * float(np.log(n_bic)),
        polya_per_node_nats=float(np.log(pricing.n_total_cells(
            funcs, len(atoms) if atoms else 0))),
        held_fold=args.held_fold, n_folds=args.n_folds, fold_sims=fold_sims,
        n_sims_train=len(set(pool["sid"][:n_tr].tolist())),
        row_order_sids=pool["sid"].tolist(),
        burn_in=dict(n_candidates=len(roots), wall_s=wall_b,
                     workers=workers, top_k=burn_top_k, cap=burn_cap),
        ladder_stats=stats,
        archive_stats=dict(offered=archive.n_offered, evals=archive.n_evals,
                           rejected_nonfinite=archive.n_rejected_nonfinite),
        wall_s=wall,
        archive_top=archive.top())
    with open(out, "w") as fp:
        json.dump(rec, fp)
    best = rec["archive_top"][0] if rec["archive_top"] else None
    print("[i3b] DONE wall %.1f s  best post %.3f  k=%d  nodes=%d  "
          "chi2/dof train %.3g held-out %.3g  %s"
          % (wall, best["post"] if best else float("nan"),
             best["k"] if best else -1, best["n_nodes"] if best else -1,
             best["chi2_dof_train"] if best else float("nan"),
             best["chi2_dof_heldout"] if best else float("nan"),
             (best["desc"][:90] if best else "-")), flush=True)
    print("[i3b] wrote %s" % out, flush=True)


if __name__ == "__main__":
    main()
