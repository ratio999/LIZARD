"""i3b_pool.py -- pooled full-space dataset for I3-B (preparation D4/D5).

The pooled dataset
------------------
Rows (x0, x1, x2) -> y with per-point sigma, where x0 = t is the time
coordinate inside one simulation and x1, x2 are CONSTANT PER SIMULATION
(the remnant ringing rates of D6: x1 = alpha_0, x2 = Delta_1 =
alpha_1 - alpha_0, both known functions of the binary). The science
side supplies, per simulation, the time series (t, y, sigma) and the
two coordinates; `build_pooled` stacks them into the kernel's (d, n)
C-contiguous layout, points along the last axis, and keeps the
simulation id of every row.

Splitting is BY SIMULATION, never by row: `fold_of_sid` is a
deterministic hash of the id (SHA-1, first 8 hex digits, modulo
n_folds), so the fold of a simulation is a property of its name, the
same on every machine, in every run, for every thinning.

Thinning is stratified by simulation: N_t rows per simulation, drawn
without replacement under a fixed seed (all rows when a simulation has
fewer), and optionally a stratified subset of the simulations
themselves (equal counts per quantile bin of a stratification key,
x1 by default), also under the seed.

Sigma convention (D5 i)
-----------------------
The kernel's Gaussian likelihood is only as honest as sigma. The
coefficient-law pilot overfit catastrophically when sigma was too
small by orders of magnitude (chi2/dof ~ 1e4: the BIC charge vanished
next to the likelihood gain of any extra term). `sigma_with_floor`
lifts every sigma to at least `floor_frac` times a robust spread of
the TRAIN targets, spread = IQR / 1.349 (the sigma of a Gaussian with
that IQR), so that the floor is in the units of the target and is
insensitive to outliers. Default floor_frac = 0.02: a point can never
count as known to better than 2 % of the target's spread. Held-out rows
keep the same per-point sigma (the floor is computed on train rows and
applied to all rows, so no row is treated as more precise than train
data could be).

Winsorisation (D5 ii) clips TRAIN targets to [p, 100 - p] percentiles;
held-out rows are never touched.

File format: np.savez with X (3, n), y, sigma, sid (unicode), plus a
JSON-serialisable meta dict (variable names, coordinate names, per-sim
coordinate values) stored as a JSON string under 'meta'.
"""
import hashlib
import json

import numpy as np

VARIABLES = ("x0", "x1", "x2")
COORD_NAMES = ("t", "alpha0", "delta1")


# ---------------------------------------------------------------------------
# folds
# ---------------------------------------------------------------------------
def fold_of_sid(sid, n_folds=5):
    """Deterministic fold label of a simulation id (SHA-1 of the UTF-8
    string, first 8 hex digits, modulo n_folds)."""
    h = hashlib.sha1(str(sid).encode("utf-8")).hexdigest()[:8]
    return int(h, 16) % int(n_folds)


def folds_of(sids, n_folds=5):
    return np.asarray([fold_of_sid(s, n_folds) for s in sids],
                      dtype=np.int64)


# ---------------------------------------------------------------------------
# building
# ---------------------------------------------------------------------------
def build_pooled(sims, n_folds=5):
    """Stack per-simulation products into one pooled dataset.

    sims : iterable of dicts with keys
        'sid'   : simulation id (string)
        't'     : (m,) time coordinate            -> x0
        'y'     : (m,) target
        'sigma' : (m,) or scalar per-point sigma
        'x1'    : scalar, constant per simulation -> x1
        'x2'    : scalar, constant per simulation -> x2
    Returns a dict pool with X (3, n) C-contiguous float64, y (n,),
    sigma (n,), sid (n,) unicode, fold (n,) int64, sims (list of sid in
    order), coords {sid: (x1, x2)}.
    """
    Xs, ys, ss, ids = [], [], [], []
    coords = {}
    order = []
    for s in sims:
        sid = str(s["sid"])
        t = np.asarray(s["t"], dtype=np.float64).ravel()
        y = np.asarray(s["y"], dtype=np.float64).ravel()
        if t.shape != y.shape:
            raise ValueError("sim %s: t %s vs y %s" % (sid, t.shape, y.shape))
        sg = np.asarray(s["sigma"], dtype=np.float64)
        if sg.ndim == 0:
            sg = np.full(t.shape, float(sg))
        sg = sg.ravel()
        if sg.shape != t.shape:
            raise ValueError("sim %s: sigma %s vs t %s" % (sid, sg.shape,
                                                           t.shape))
        if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
            raise ValueError("sim %s: non-finite t or y" % sid)
        if not np.all(sg > 0):
            raise ValueError("sim %s: sigma must be > 0" % sid)
        x1 = float(s["x1"])
        x2 = float(s["x2"])
        if sid in coords:
            raise ValueError("duplicate simulation id %r" % sid)
        coords[sid] = (x1, x2)
        order.append(sid)
        m = t.shape[0]
        Xs.append(np.vstack([t, np.full(m, x1), np.full(m, x2)]))
        ys.append(y)
        ss.append(sg)
        ids.extend([sid] * m)
    if not Xs:
        raise ValueError("no simulations")
    X = np.ascontiguousarray(np.hstack(Xs).astype(np.float64))
    y = np.ascontiguousarray(np.concatenate(ys))
    sigma = np.ascontiguousarray(np.concatenate(ss))
    sid = np.asarray(ids, dtype=str)
    return dict(X=X, y=y, sigma=sigma, sid=sid,
                fold=folds_of(sid, n_folds), sims=order, coords=coords,
                n_folds=int(n_folds))


def _subset(pool, mask):
    idx = np.where(mask)[0]
    out = dict(pool)
    out["X"] = np.ascontiguousarray(pool["X"][:, idx])
    out["y"] = np.ascontiguousarray(pool["y"][idx])
    out["sigma"] = np.ascontiguousarray(pool["sigma"][idx])
    out["sid"] = pool["sid"][idx]
    out["fold"] = pool["fold"][idx]
    out["sims"] = [s for s in pool["sims"] if s in set(out["sid"].tolist())]
    return out


# ---------------------------------------------------------------------------
# thinning
# ---------------------------------------------------------------------------
def select_sims_stratified(pool, n_sims, seed, key="x1", n_bins=5):
    """Deterministic stratified subset of simulation ids: the coordinate
    `key` ('x1' or 'x2') is binned into n_bins quantile bins and equal
    counts are drawn from each bin without replacement (remainders go
    to the bins in order). Returns the sorted list of chosen ids."""
    sims = list(pool["sims"])
    if n_sims >= len(sims):
        return sorted(sims)
    k = 0 if key == "x1" else 1
    vals = np.asarray([pool["coords"][s][k] for s in sims])
    edges = np.quantile(vals, np.linspace(0.0, 1.0, n_bins + 1))
    bins = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0,
                   n_bins - 1)
    rng = np.random.default_rng(seed)
    per = [n_sims // n_bins + (1 if b < n_sims % n_bins else 0)
           for b in range(n_bins)]
    chosen = []
    for b in range(n_bins):
        members = sorted(s for s, bb in zip(sims, bins) if bb == b)
        take = min(per[b], len(members))
        if take:
            pick = rng.choice(len(members), size=take, replace=False)
            chosen.extend(members[i] for i in sorted(pick))
    # If some bins were short, top up from the remaining ids.
    if len(chosen) < n_sims:
        rest = sorted(set(sims) - set(chosen))
        extra = rng.choice(len(rest), size=n_sims - len(chosen),
                           replace=False)
        chosen.extend(rest[i] for i in sorted(extra))
    return sorted(chosen)


def thin_pooled(pool, n_t, seed, sims_subset=None):
    """N_t rows per simulation (all rows when a simulation is shorter),
    drawn without replacement under `seed`; optionally restricted to the
    simulation ids in `sims_subset`. Deterministic: the per-simulation
    draw uses a child seed derived from (seed, sid), so adding or
    removing OTHER simulations does not change a simulation's rows."""
    keep = np.zeros(pool["sid"].shape[0], dtype=bool)
    allowed = None if sims_subset is None else set(map(str, sims_subset))
    sid = pool["sid"]
    for s in pool["sims"]:
        if allowed is not None and s not in allowed:
            continue
        rows = np.where(sid == s)[0]
        if rows.shape[0] <= n_t:
            keep[rows] = True
            continue
        child = int(hashlib.sha1(("%d:%s" % (int(seed), s)).encode("utf-8"))
                    .hexdigest()[:8], 16)
        rng = np.random.default_rng(child)
        pick = rng.choice(rows.shape[0], size=int(n_t), replace=False)
        keep[rows[np.sort(pick)]] = True
    return _subset(pool, keep)


# ---------------------------------------------------------------------------
# split + ordering
# ---------------------------------------------------------------------------
def split_train_first(pool, held_fold=0):
    """Reorder rows so TRAIN rows (fold != held_fold) come first; returns
    (pool_ordered, n_train). The archive stores predictions on all rows
    and scores chi2 on pred[:n_train] -- the exp_mr_i3a_laws.py layout."""
    tr = pool["fold"] != int(held_fold)
    order = np.concatenate([np.where(tr)[0], np.where(~tr)[0]])
    out = dict(pool)
    out["X"] = np.ascontiguousarray(pool["X"][:, order])
    out["y"] = np.ascontiguousarray(pool["y"][order])
    out["sigma"] = np.ascontiguousarray(pool["sigma"][order])
    out["sid"] = pool["sid"][order]
    out["fold"] = pool["fold"][order]
    out["held_fold"] = int(held_fold)
    return out, int(tr.sum())


# ---------------------------------------------------------------------------
# robustness (D5)
# ---------------------------------------------------------------------------
def robust_spread(y):
    """IQR / 1.349: the sigma of a Gaussian with the sample's IQR."""
    q1, q3 = np.percentile(np.asarray(y, dtype=np.float64), [25.0, 75.0])
    return float((q3 - q1) / 1.349)


def sigma_with_floor(sigma, y_train, floor_frac=0.02):
    """max(sigma, floor_frac * robust_spread(y_train)) for every row.
    Returns (sigma_eff, floor_value, n_lifted)."""
    floor = float(floor_frac) * robust_spread(y_train)
    sg = np.asarray(sigma, dtype=np.float64)
    eff = np.maximum(sg, floor)
    return np.ascontiguousarray(eff), floor, int(np.sum(sg < floor))


def winsorise(y_train, p):
    """Clip to [p, 100 - p] percentiles (p in percent; 0 = off)."""
    if p <= 0:
        return np.ascontiguousarray(np.asarray(y_train, dtype=np.float64))
    lo, hi = np.percentile(y_train, [p, 100.0 - p])
    return np.ascontiguousarray(np.clip(y_train, lo, hi))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def save_pooled(path, pool, extra_meta=None):
    meta = dict(variables=list(VARIABLES), coord_names=list(COORD_NAMES),
                sims=list(pool["sims"]),
                coords={s: list(pool["coords"][s]) for s in pool["sims"]},
                n_folds=int(pool.get("n_folds", 5)))
    if extra_meta:
        meta.update(extra_meta)
    np.savez(path, X=pool["X"], y=pool["y"], sigma=pool["sigma"],
             sid=pool["sid"], fold=pool["fold"], meta=json.dumps(meta))


def load_pooled(path):
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta"]))
    pool = dict(X=np.ascontiguousarray(z["X"], dtype=np.float64),
                y=np.ascontiguousarray(z["y"], dtype=np.float64),
                sigma=np.ascontiguousarray(z["sigma"], dtype=np.float64),
                sid=np.asarray(z["sid"], dtype=str),
                fold=np.asarray(z["fold"], dtype=np.int64),
                sims=list(meta["sims"]),
                coords={s: tuple(v) for s, v in meta["coords"].items()},
                n_folds=int(meta.get("n_folds", 5)), meta=meta)
    return pool


# ---------------------------------------------------------------------------
# synthetic pooled data (smoke tests and benchmarks)
# ---------------------------------------------------------------------------
def synthetic_sims(n_sims=40, m=300, seed=0, noise_frac=0.02):
    """Per-simulation products with the D6 coordinate ranges and a known
    law y = exp(-x1 t) (1 + x2 t): x1 = alpha_0 around 0.085 (7 %
    spread), x2 = Delta_1 in [0.10, 0.20], t in [0, 60]."""
    rng = np.random.default_rng(seed)
    sims = []
    for i in range(n_sims):
        x1 = float(rng.uniform(0.080, 0.092))
        x2 = float(rng.uniform(0.10, 0.20))
        t = np.sort(rng.uniform(0.0, 60.0, m))
        truth = np.exp(-x1 * t) * (1.0 + x2 * t)
        sd = noise_frac * float(np.std(truth))
        y = truth + rng.normal(0.0, sd, m)
        sims.append(dict(sid="SIM%04d" % i, t=t, y=y, sigma=sd, x1=x1,
                         x2=x2))
    return sims
