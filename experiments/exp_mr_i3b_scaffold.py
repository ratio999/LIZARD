"""exp_mr_i3b_scaffold.py -- wave-level law search on an immortal
multi-slot scaffold (I3-B D9, deliverable K5).

One kernel run per channel on a pooled multivariate domain. The deployed
closed-form template of the channel is installed as an IMMORTAL SCAFFOLD
(multi-slot FlexAtomSpec): its shape never mutates, its four knob
positions are live subtrees the machine grows, restricted to the two
remnant coordinates {x1, x2} (never time, never pin columns), and the
score is the ordinary pooled sigma-weighted likelihood + BIC over all
fitted constants + Polya on the grown structure — the standard
three-term bill.

Channels (templates verified against the science-side fitters; the
symbolic identity is asserted in test_i3b_scaffold.py):

  phase  d = 4: x0 = t, x1 = alpha_0, x2 = Delta_1, x3 = c99 slope read.
         LP2c with c1 = g*W, c2 = 2*x1*r1, c3 = (1-g)*W, c4 = 2*x2*r2.
         Slots (declaration order): r1, r2, W, g.
  amp    d = 6: x0 = t, x1, x2 as above, x3 = c98, x4 = c99, x5 = Aref.
         TS3c at the deployed two-term conventions (amp_kn8: t1 = 0,
         r3 = 1, t3 = 0, f -> 1), which reduces exactly to
             A(t) = x3 - c2*tanh(c4)
                    + ((x4 - c2*c3*(1 - tanh(c4)**2))/c0)*tanh(c0*x0)
                    + c2*tanh(c3*x0 + c4)
         with c0 = 2*x1*r1, c2 = s1*s*x5, c3 = x1*r2, c4 = -x1*r2*tau2.
         Slots: r1, r2, tau2, s. s1 is the frozen catalogue sign,
         supplied via --start-values (smoke default +1). NB the deployed
         amp_kn8 uses f = 1 - 1.5e-6; the scaffold takes the exact
         f = 1 limit (relative difference ~1e-6, recorded here for
         honesty).

Pooled data: an .npz from the science side's i16_wavepool builder —
X (d, n) C-contiguous float64, y, sigma, sid, fold — with every column
except x0 CONSTANT within each simulation (validated; refused
otherwise). --smoke builds a synthetic pool with planted knob laws.

Seeding: no burn-in (refused with flex anyway); every rung's slots start
as constant parameter leaves at the --start-values (defaults: r-knobs
1.0; phase W = 5.0, g = 0.5; amp tau2 = 0.5, s = 1.0). Serial EMC only
(spawn is refused with flex — asserted early). Structural tilt OFF.

K6 diagnostic: every archive entry records, per knob, the fraction of
TRAIN simulations whose predicted knob falls outside the science-side
sanity bounds (phase: r1 [0.5,1.5], r2 [0.75,1.5], W [0.3,60],
g [0.02,0.98]; amp: r1, r2 as phase, tau2/s unspecified -> not
checked). A diagnostic column, never a rejection. The in-chain
definedness story is the kernel's own validity gate, which runs through
the composed template (a non-finite scaffold evaluation is an all-NaN
vector and a hard reject).

Run (WSL, mcmc_koopman, from gpu_ms; PYTHONHASHSEED pinned for
bit-reproducibility):
    PYTHONHASHSEED=0 python exp_mr_i3b_scaffold.py --channel phase \
        --smoke --seed 1
    PYTHONHASHSEED=0 python exp_mr_i3b_scaffold.py --channel phase \
        --pool results/mr_i3b/datasets/wavepool_phase.npz --seed 101 \
        [--rounds 600] [--full-guard] [--start-values sv.json]

Output: results/mr_i3b/runs/scaffold__<channel>__<tag>__fold<f>__seed<s>.json
(refuses overwrite). This template does not launch production runs.
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
import sympy as sp

import real_functionC
import function_treeC
import exp_common
import i3b_pool
import i3b_polya_pricing as pricing
from exp_mr_i3b_fullspace import ArchiveTop

OUT_DIR = os.path.join(_HERE, "results", "mr_i3b", "runs")

_BETAS = (1.0, 0.82, 0.64, 0.46, 0.28, 0.10)
LADDER = [1.0 / b for b in _BETAS]
SIZE_PER_GROVE = 4
WARMUP_STEPS = 100
STEPS_PER_ROUND = 6
SWAPS_PER_ROUND = 4
XOVERS_PER_ROUND = 1
ROUNDS_DEFAULT = 300
OPS_SAFE4 = ["+", "-", "*", "/"]

# slot symbols must be plain identifiers (no letter+digits); the knob
# names they realize, in declaration order:
SLOT_SYMS = ["Ka", "Kb", "Kc", "Kd"]
KNOBS = {"phase": ["r1", "r2", "W", "g"],
         "amp": ["r1", "r2", "tau2", "s"]}
VARS = {"phase": ("x0", "x1", "x2", "x3"),
        "amp": ("x0", "x1", "x2", "x3", "x4", "x5")}
COORD_NAMES = {"phase": ("t", "alpha0", "delta1", "c99"),
               "amp": ("t", "alpha0", "delta1", "c98", "c99", "Aref")}
BOUNDS = {"phase": {"r1": (0.5, 1.5), "r2": (0.75, 1.5),
                    "W": (0.3, 60.0), "g": (0.02, 0.98)},
          "amp": {"r1": (0.5, 1.5), "r2": (0.75, 1.5)}}
START_DEFAULTS = {"phase": {"r1": 1.0, "r2": 1.0, "W": 5.0, "g": 0.5},
                  "amp": {"r1": 1.0, "r2": 1.0, "tau2": 0.5, "s": 1.0,
                          "s1": 1.0}}
SLOT_MASK = ("x1", "x2")

# the science-side pinned-algebra strings (i2_interp_judge.TPL), time
# variable 'x'; verified 2026-09-16 against the fitters
TPL_LP2C = ("(-c99*(1 + c1 + c3)/(c1*c2 + c3*c4))"
            "*log((1 + c1*exp(-c2*x) + c3*exp(-c4*x))/(1 + c1 + c3))")
TPL_TS3C = ("(c98 - ((c99 - (c2*c3*(1 - tanh(c4)**2)"
            " + c5*c6*(1 - tanh(c7)**2)))"
            "/(c0*(1 - tanh(c1)**2)))*tanh(c1) - c2*tanh(c4)"
            " - c5*tanh(c7))"
            " + ((c99 - (c2*c3*(1 - tanh(c4)**2)"
            " + c5*c6*(1 - tanh(c7)**2)))"
            "/(c0*(1 - tanh(c1)**2)))*tanh(c0*x + c1)"
            " + c2*tanh(c3*x + c4) + c5*tanh(c6*x + c7)")


def channel_template(channel, s1=1.0):
    """The frozen template EXPRESSION STRING of a channel: the deployed
    pinned-algebra with the knob map substituted, over the registered
    variables and the four slot symbols Ka..Kd (== KNOBS order)."""
    Ka, Kb, Kc, Kd = [sp.Symbol(s) for s in SLOT_SYMS]
    x = sp.Symbol('x')
    if channel == "phase":
        c = {sp.Symbol('c1'): Kd * Kc,
             sp.Symbol('c2'): 2 * sp.Symbol('x1') * Ka,
             sp.Symbol('c3'): (1 - Kd) * Kc,
             sp.Symbol('c4'): 2 * sp.Symbol('x2') * Kb,
             sp.Symbol('c99'): sp.Symbol('x3'),
             x: sp.Symbol('x0')}
        expr = sp.sympify(TPL_LP2C).subs(c, simultaneous=True)
    elif channel == "amp":
        c0 = 2 * sp.Symbol('x1') * Ka
        c2 = sp.Float(s1, 15) * Kd * sp.Symbol('x5')
        c3 = sp.Symbol('x1') * Kb
        c4 = -sp.Symbol('x1') * Kb * Kc
        c = {sp.Symbol('c0'): c0, sp.Symbol('c1'): sp.Integer(0),
             sp.Symbol('c2'): c2, sp.Symbol('c3'): c3,
             sp.Symbol('c4'): c4, sp.Symbol('c5'): sp.Integer(0),
             sp.Symbol('c6'): sp.Symbol('x1'),
             sp.Symbol('c7'): sp.Integer(0),
             sp.Symbol('c98'): sp.Symbol('x3'),
             sp.Symbol('c99'): sp.Symbol('x4'),
             x: sp.Symbol('x0')}
        expr = sp.sympify(TPL_TS3C).subs(c, simultaneous=True)
    else:
        raise ValueError("channel must be 'amp' or 'phase'")
    return str(sp.simplify(expr) if channel == "amp" else expr)


def build_spec(channel, s1=1.0):
    return real_functionC.FlexAtomSpec(
        "Scaf_%s" % channel, channel_template(channel, s1),
        slot_symbols=list(SLOT_SYMS), allow_variables=True,
        slot_variables={s: SLOT_MASK for s in SLOT_SYMS})


def seed_root(spec, X, starts, channel):
    """The seeded scaffold: every slot a constant parameter leaf at its
    start value, distinct 'k' names (outer canonicalization renames
    freely later)."""
    slots = []
    vals = []
    for i, kn in enumerate(KNOBS[channel]):
        nm = "k%d" % i
        slots.append(function_treeC.node_from_skel(('L', nm), X, None))
        vals.append((nm, float(starts[kn])))
    root = function_treeC.make_flex_node(spec, slots, [])
    for (nm, v), sl in zip(vals, root.flex_slots):
        sl.set_value(v)
    root.update_func()
    return root


def validate_pool(X, sid):
    """Every column except x0 must be constant within each simulation."""
    for s in np.unique(sid):
        rows = sid == s
        for j in range(1, X.shape[0]):
            col = X[j][rows]
            if not np.all(col == col[0]):
                raise SystemExit(
                    "pool violates the pin contract: column x%d varies "
                    "inside simulation %r" % (j, s))


def synthetic_pool(channel, n_sims=20, m=60, seed=0, noise_frac=0.03):
    """Smoke fixture: planted smooth knob laws of (x1, x2), waveform
    rows from the exact channel template."""
    rng = np.random.default_rng(seed)
    spec_expr = sp.sympify(channel_template(channel, 1.0))
    fn = sp.lambdify([sp.Symbol(v) for v in VARS[channel]]
                     + [sp.Symbol(s) for s in SLOT_SYMS], spec_expr,
                     "numpy")
    laws = {"r1": lambda a, b: 0.9 + 0.2 * a * b,
            "r2": lambda a, b: 1.0 + 0.1 * np.tanh(b),
            "W": lambda a, b: 4.0 + 2.0 * a,
            "g": lambda a, b: 0.4 + 0.1 * np.tanh(a * b),
            "tau2": lambda a, b: 0.4 + 0.2 * a,
            "s": lambda a, b: 1.0 + 0.1 * b}
    cols, ys, sids = [], [], []
    for i in range(n_sims):
        x1 = rng.uniform(0.8, 1.2)
        x2 = rng.uniform(0.8, 1.2)
        t = np.sort(rng.uniform(0.05, 6.0, m))
        pins = {"x3": rng.uniform(0.5, 1.5), "x4": rng.uniform(-2.0, -0.5),
                "x5": rng.uniform(1.0, 2.0)}
        row = [t] + [np.full(m, {"x1": x1, "x2": x2}.get(
            v, pins.get(v, 0.0))) for v in VARS[channel][1:]]
        kn = [laws[k](x1, x2) for k in KNOBS[channel]]
        yv = np.asarray(fn(*(row + kn)), dtype=np.float64)
        cols.append(np.vstack(row))
        ys.append(yv)
        sids.extend(["SYN%03d" % i] * m)
    X = np.ascontiguousarray(np.hstack(cols).astype(np.float64))
    y = np.concatenate(ys)
    sd = noise_frac * float(np.std(y))
    y = np.ascontiguousarray(y + rng.normal(0.0, sd, y.size))
    sid = np.asarray(sids, dtype=str)
    return dict(X=X, y=y, sigma=np.full(y.size, sd), sid=sid,
                fold=i3b_pool.folds_of(sid, 5))


class ScaffoldArchive(ArchiveTop):
    """ArchiveTop + the K6 knob-bounds diagnostic and the slot
    skeletons/laws per entry."""

    def __init__(self, *args, channel=None, sim_coords=None, **kw):
        super().__init__(*args, **kw)
        self.channel = channel
        self.sim_coords = sim_coords  # (d, n_sims) per-sim column matrix

    def offer(self, tree, post, rnd):
        n_before = self.n_evals
        super().offer(tree, post, rnd)
        if self.n_evals == n_before:      # rejected or not better
            return
        e = self.pool.get(tree.tree_get_description())
        if e is None:
            return
        fxs = function_treeC.collect_flex_nodes(tree.get_root(), None)
        if not fxs:
            return
        nv = dict(tree.get_param_dicts()[1])
        knobs = {}
        oob = {}
        for i, kn in enumerate(KNOBS[self.channel]):
            slot = fxs[0].flex_slots[i]
            expr = slot.get_fun().get_description()
            vals = np.asarray(slot.get_fun().get_eval(
                self.sim_coords, nv), dtype=np.float64)
            knobs[kn] = {"expr": expr,
                         "per_sim": [round(float(v), 8) for v in vals]}
            b = BOUNDS[self.channel].get(kn)
            if b is not None:
                oob[kn] = float(np.mean((vals < b[0]) | (vals > b[1])))
        e["knobs"] = knobs
        e["oob_fraction"] = oob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", required=True, choices=("amp", "phase"))
    ap.add_argument("--pool", default=None)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=ROUNDS_DEFAULT)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--held-fold", type=int, default=0)
    ap.add_argument("--start-values", default=None,
                    help="JSON {knob: value} (+ 's1' for amp)")
    ap.add_argument("--sigma-floor-frac", type=float, default=0.02)
    ap.add_argument("--winsor", type=float, default=0.0)
    ap.add_argument("--full-guard", action="store_true")
    args = ap.parse_args()

    ch = args.channel
    tag = args.tag or ("smoke" if args.smoke else
                       os.path.splitext(os.path.basename(args.pool))[0]
                       if args.pool else "synthetic")
    out = os.path.join(OUT_DIR, "scaffold__%s__%s__fold%d__seed%d.json"
                       % (ch, tag, args.held_fold, args.seed))
    if os.path.exists(out):
        raise SystemExit("REFUSING to overwrite %s" % out)
    os.makedirs(OUT_DIR, exist_ok=True)
    rounds = 25 if args.smoke else args.rounds
    warmup = 20 if args.smoke else WARMUP_STEPS

    starts = dict(START_DEFAULTS[ch])
    if args.start_values:
        with open(args.start_values) as fp:
            starts.update(json.load(fp))
    s1 = float(starts.get("s1", 1.0))

    # variables FIRST
    real_functionC.set_variables(VARS[ch])
    exp_common.set_caps()
    spec = build_spec(ch, s1)

    # slot vocabulary: the D3 declared set restricted to {x1, x2}
    funcs = pricing.vocabulary_i3b(("x1", "x2"), positive=("x1", "x2"))

    if args.pool:
        z = np.load(args.pool, allow_pickle=False)
        X = np.ascontiguousarray(z["X"], dtype=np.float64)
        y = np.ascontiguousarray(z["y"], dtype=np.float64)
        sigma = np.ascontiguousarray(z["sigma"], dtype=np.float64)
        sid = np.asarray(z["sid"], dtype=str)
        fold = np.asarray(z["fold"], dtype=np.int64)
        pool = dict(X=X, y=y, sigma=sigma, sid=sid, fold=fold)
    else:
        if not args.smoke:
            raise SystemExit("pass --pool PATH (or --smoke)")
        pool = synthetic_pool(ch, seed=args.seed)
    if pool["X"].shape[0] != len(VARS[ch]):
        raise SystemExit("pool has d=%d, channel %s needs d=%d"
                         % (pool["X"].shape[0], ch, len(VARS[ch])))
    validate_pool(pool["X"], pool["sid"])

    ordered, n_tr = i3b_pool.split_train_first(
        dict(pool, sims=sorted(set(pool["sid"].tolist())),
             coords={}, n_folds=5), held_fold=args.held_fold)
    X_full, y_full, s_full = ordered["X"], ordered["y"], ordered["sigma"]
    n_full = int(y_full.shape[0])
    if n_tr == 0 or n_tr == n_full:
        raise SystemExit("degenerate split: train %d of %d" % (n_tr, n_full))
    y_tr = np.ascontiguousarray(y_full[:n_tr])
    if args.sigma_floor_frac > 0:
        s_full, floor, n_lift = i3b_pool.sigma_with_floor(
            s_full, y_tr, args.sigma_floor_frac)
    else:
        floor, n_lift = 0.0, 0
    y_tr = i3b_pool.winsorise(y_tr, args.winsor)
    s_tr = np.ascontiguousarray(s_full[:n_tr])
    X_tr = np.ascontiguousarray(X_full[:, :n_tr])
    y_scored = np.concatenate([y_tr, y_full[n_tr:]])
    if args.full_guard:
        X_ch, y_ch = X_full, np.ascontiguousarray(y_scored)
        s_ch = np.ascontiguousarray(np.concatenate(
            [s_tr, np.full(n_full - n_tr, np.inf)]))
        n_bic = n_full
    else:
        X_ch, y_ch, s_ch = X_tr, y_tr, s_tr
        n_bic = n_tr

    fold_sims = sorted(set(ordered["sid"][n_tr:].tolist()))
    tr_sims = sorted(set(ordered["sid"][:n_tr].tolist()))
    print("[scaffold:%s] rows %d (train %d over %d sims; held-out %d "
          "sims) seed %d  template %d chars, slots %s"
          % (ch, n_full, n_tr, len(tr_sims), len(fold_sims), args.seed,
             len(spec.template), KNOBS[ch]), flush=True)
    print("[scaffold:%s] slot vocabulary: %s ; sigma floor %.3g "
          "(%d lifted); full-guard %s"
          % (ch, " ".join(funcs), floor, n_lift, args.full_guard),
          flush=True)

    root = seed_root(spec, X_ch, starts, ch)
    t0 = time.perf_counter()
    forest = exp_common.seeded_forest(
        X_ch, y_ch, LADDER, SIZE_PER_GROVE, funcs, OPS_SAFE4,
        s_ch.copy(), args.seed, [root], flex_atoms=[spec],
        flex_root_lock=True)
    # spawn is refused with flex; make the serial-only contract loud
    assert forest.groves and forest.groves[0].flex_atoms is not None
    forest.run_all(y_ch, warmup)

    # per-sim coordinate matrix for the K6 knob read-out (train sims)
    sim_rows = []
    for j in range(len(VARS[ch])):
        sim_rows.append(np.asarray(
            [X_full[j][ordered["sid"] == s][0] for s in tr_sims]))
    sim_coords = np.ascontiguousarray(np.vstack(sim_rows))

    polya_t0 = pricing.make_t0_polya(funcs, atoms=[spec])
    archive = ScaffoldArchive(X_full, y_scored, s_full, n_tr, polya_t0,
                              funcs, {spec.name: spec}, channel=ch,
                              sim_coords=sim_coords)
    archive.update_from_forest(forest, -1)
    for rnd in range(rounds):
        forest.run_all(y_ch, STEPS_PER_ROUND)
        for _ in range(SWAPS_PER_ROUND):
            forest.attempt_swaps(y_ch)
        for _ in range(XOVERS_PER_ROUND):
            forest.attempt_crossovers(y_ch)
        archive.update_from_forest(forest, rnd)
        if (rnd + 1) % 25 == 0:
            g0 = forest.groves[0]
            arr = np.asarray([p if p is not None else np.nan
                              for p in g0.root_posteriors],
                             dtype=np.float64)
            print("    %s round %d/%d (%.1f s) best cold %.3f archive %d"
                  % (ch, rnd + 1, rounds, time.perf_counter() - t0,
                     float(np.nanmax(arr)), len(archive.pool)),
                  flush=True)
    wall = time.perf_counter() - t0

    rec = dict(
        channel=ch, seed=args.seed, rounds=rounds, smoke=bool(args.smoke),
        source=args.pool or "synthetic",
        variables=dict(zip(VARS[ch], COORD_NAMES[ch])),
        template=spec.template, knobs=KNOBS[ch], s1=s1,
        slot_mask=list(SLOT_MASK), funcs=funcs, ops=OPS_SAFE4,
        start_values={k: starts[k] for k in KNOBS[ch]},
        ladder=LADDER,
        rounds_protocol=dict(steps=STEPS_PER_ROUND,
                             swaps=SWAPS_PER_ROUND,
                             xovers=XOVERS_PER_ROUND, warmup=warmup,
                             size_per_grove=SIZE_PER_GROVE),
        robustness=dict(sigma_floor_frac=args.sigma_floor_frac,
                        sigma_floor=floor, n_sigma_lifted=n_lift,
                        winsor=args.winsor,
                        full_guard=bool(args.full_guard)),
        n_pooled=int(n_tr), n_full=n_full, n_bic=int(n_bic),
        bic_per_param_nats=0.5 * float(np.log(n_bic)),
        polya_per_node_nats=float(np.log(
            pricing.n_total_cells(funcs, 1))),
        held_fold=args.held_fold, fold_sims=fold_sims,
        n_sims_train=len(tr_sims),
        knob_bounds=BOUNDS[ch],
        ladder_stats=forest.ladder_stats(),
        archive_stats=dict(offered=archive.n_offered,
                           evals=archive.n_evals,
                           rejected_nonfinite=archive.n_rejected_nonfinite),
        wall_s=wall,
        archive_top=archive.top())
    with open(out, "w") as fp:
        json.dump(rec, fp)
    best = rec["archive_top"][0] if rec["archive_top"] else None
    if best:
        print("[scaffold:%s] DONE wall %.1f s best post %.3f k=%d "
              "chi2/dof tr %.3g ho %.3g oob %s"
              % (ch, wall, best["post"], best["k"],
                 best["chi2_dof_train"], best["chi2_dof_heldout"],
                 best.get("oob_fraction")), flush=True)
        for kn in KNOBS[ch]:
            print("    %-5s = %s" % (kn, best["knobs"][kn]["expr"]),
                  flush=True)
    print("[scaffold:%s] wrote %s" % (ch, out), flush=True)


if __name__ == "__main__":
    main()
