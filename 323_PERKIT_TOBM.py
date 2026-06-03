#!/usr/bin/env python3
"""
PERKIT / ParkPass — Network-Effects Simulation (plain-Python version)
=====================================================================
Converted from the Jupyter notebook for ease of review.

HOW IT BEHAVES
  * Each figure opens in its own interactive window. Close a window to
    let the script continue to the next stage.
  * Nothing is written to disk — no image files are saved.

PERFORMANCE
  * Iteration counts and DAU ranges are set to light "reviewer" values
    so the whole script finishes quickly on any machine. To reproduce
    the reporting-grade results from the thesis, raise ITERATIONS,
    DAU_AXIS, SOBOL_N and the per-section *_ITER values in the config.

REQUIREMENTS
  pip install numpy pandas matplotlib tqdm SALib
"""


# ==========================================================================
# SECTION 0
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  PARKPASS — Network Effects Simulation  v10                  ║
# ║  CELL 0: Imports, Plot Style & All Knobs                     ║
# ║                                                              ║
# ║  THIS IS THE ONLY CELL YOU NEED TO EDIT.                     ║
# ║  Change a number here, then hit  Run All.                    ║
# ╚══════════════════════════════════════════════════════════════╝

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.ticker as mticker

from tqdm import tqdm
import time

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze



# ══════════════════════════════════════════════════════════════════
#  A — GEOGRAPHY
#  The simulated city zone and how far apart a seeker and leaver
#  can be and still be matched.
# ══════════════════════════════════════════════════════════════════
ZONE_W_KM  = 2.0    # zone width  (km)  — Thessaloniki centre ≈ 2 km wide
ZONE_H_KM  = 1.0    # zone height (km)
AREA_KM2   = ZONE_W_KM * ZONE_H_KM
RADIUS_KM  = 0.3    # match radius (km) — must be within this distance to pair


# ══════════════════════════════════════════════════════════════════
#  B — TIME & BEHAVIOUR
#  How long each side waits, how long the active day lasts, and the
#  baseline chance a seeker finds a spot without the app.
# ══════════════════════════════════════════════════════════════════
S_WAIT             = 50     # seeker gives up after this many minutes
L_WAIT             = 1      # leaver broadcasts for this many minutes
WINDOW_MINS        = 1020   # operating day length: 07:00–24:00
LUCKY_FIND_PER_MIN = 1/20   # 5 %/min organic street-find chance
                             # → E[find without app] = 20 min

# ══════════════════════════════════════════════════════════════════
#  C — DAU SWEEP
#  Which DAU points to simulate and how many Monte-Carlo
#  iterations to run at each one (more = tighter confidence bands,
#  slower runtime).
# ══════════════════════════════════════════════════════════════════
DAU_AXIS   = [250, 500, 1000, 2000, 4000]            # daily active users (seekers = leavers = DAU)
ITERATIONS = 15                 # light/reviewer value       # Monte-Carlo iterations per DAU point


# ══════════════════════════════════════════════════════════════════
#  D — SPATIAL CLUSTERING
#  SEE CLUSTERING CELL (SELF CONTAINED)
#  Set N_CLUSTERS = 0 and CLUSTER_FRAC = 0.0 for pure uniform
#  (identical to the V5 baseline — no clustering effect).
# ══════════════════════════════════════════════════════════════════
N_CLUSTERS    = 0      # number of hotspot centres (0 = clustering off)
CLUSTER_FRAC  = 0.0    # fraction of users drawn to hotspots  (0.0–1.0)
CLUSTER_SIGMA = 0.15   # hotspot radius in km (1 std dev of Gaussian)
                        # 0.05 = one block | 0.15 = few blocks | 0.3 = neighbourhood
CLUSTER_SEED  = 42     # seed for reproducible hotspot placement

# To set hotspots manually instead of random, replace HOTSPOTS below with e.g.:
# HOTSPOTS = [
#     {'cx': 0.65, 'cy': 0.55, 'weight': 0.45},   # Aristotelous Square
#     {'cx': 1.00, 'cy': 0.30, 'weight': 0.35},   # Egnatia corridor
#     {'cx': 0.35, 'cy': 0.70, 'weight': 0.20},   # Tsimiski west end
# ]
_rng     = np.random.default_rng(seed=CLUSTER_SEED)
HOTSPOTS = [
    {'cx': float(_rng.uniform(0, ZONE_W_KM)),
     'cy': float(_rng.uniform(0, ZONE_H_KM)),
     'weight': 1.0}
    for _ in range(N_CLUSTERS)
]



# ══════════════════════════════════════════════════════════════════
#  E — TEMPORAL PATTERN
#  Controls whether arrivals are spread evenly through the day or
#  front/back-loaded (e.g. morning commuters vs. evening workers).
#  0.5 / 0.5 = uniform all day
# ══════════════════════════════════════════════════════════════════
SEEKER_EARLY_FRAC = 0.70    # fraction of seekers arriving before TIME_SPLIT
LEAVER_EARLY_FRAC = 0.70    # fraction of leavers arriving before TIME_SPLIT
TIME_SPLIT        = 510    # minute from day-start that splits peak/offpeak
                            # 510 min from 07:00 = 15:30:00 (70/30%)


# ══════════════════════════════════════════════════════════════════
#  F — SENSITIVITY SWEEP SETTINGS
#  Parameters for the area, radius, temporal, and clustering
#  sweeps in Cell 4. You can narrow or widen these ranges freely.
# ══════════════════════════════════════════════════════════════════

# — Area sweep —
AREA_VALUES = [1.5, 2.0]   # zone areas to test (km²)
AREA_DAU    = 1000                          # fixed DAU for area sweep
AREA_ITER   = 15                           # iterations per area point
ASPECT      = 2.0                           # width:height ratio kept constant

# — Radius sweep —
RADIUS_VALUES = [0.3, 0.4, 0.5]   # match radii to test (km)
RAD_DAU       = 1000
RAD_ITER      = 15

# — Temporal mismatch sweep —
T_DAU_VALUES = [500, 1000, 2000]
T_ITER       = 15
T_SCENARIOS  = {
    'Uniform (baseline)':           {'seeker_early_frac': 0.765,  'leaver_early_frac': 0.765},
    'Seekers early / Leavers late': {'seeker_early_frac': 0.80, 'leaver_early_frac': 0.20},
}

# — Clustering sweep —
CLUST_DAU_VALUES  = [500, 1000, 2000]
CLUST_FRAC_VALUES = np.linspace(0, 1, 11)   # 0 %, 10 %, … 100 %
CLUST_ITER        = 15

# — Sanity / diagnostic DAU —
SANITY_DAU  = 1000
SANITY_ITER = 20


# ── Plot style ────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor':   'white',
    'axes.edgecolor':   'black', 'axes.labelcolor':  'black',
    'text.color':       'black', 'xtick.color':      'black',
    'ytick.color':      'black', 'grid.color':       '#cccccc',
    'legend.facecolor': 'white', 'legend.edgecolor': 'black',
    'legend.labelcolor':'black', 'figure.titlesize':  15,
    'axes.titlesize':   13,      'axes.labelsize':    11,
})

# ── Colour palette ────────────────────────────────────────────────
C_SEEK   = '#2ecc71'   # green   — seekers / app match
C_LEAVE  = '#e67e22'   # orange  — leavers
C_INST   = '#3498db'   # blue    — instant matches
C_THEORY = '#95a5a6'   # grey    — theoretical benchmark
C_MDIST  = '#9b59b6'   # purple  — match distance
C_LUCKY  = '#f1c40f'   # yellow  — lucky / organic finds
C_FAIL   = '#e74c3c'   # red     — failures



# ── Config summary ────────────────────────────────────────────────
print('Config loaded — ParkPass Sim v10')
print(f'  Zone : {ZONE_W_KM} × {ZONE_H_KM} km = {AREA_KM2} km²  |  Radius: {RADIUS_KM} km')
print(f'  Day  : {WINDOW_MINS} min  |  S_WAIT: {S_WAIT} min  |  L_WAIT: {L_WAIT} min')
print(f'  Unassisted: {LUCKY_FIND_PER_MIN*100:.0f}%/min  →  E[unassisted search]: {1/LUCKY_FIND_PER_MIN:.0f} min')
print(f'  DAU axis   : {DAU_AXIS}  |  Iterations: {ITERATIONS}')
clust_status = 'OFF (uniform)' if CLUSTER_FRAC == 0 else f'{CLUSTER_FRAC:.0%} of users at {N_CLUSTERS} hotspot(s), σ={CLUSTER_SIGMA} km'
print(f'  Clustering : {clust_status}')
temp_status  = 'OFF (uniform)' if SEEKER_EARLY_FRAC == 0.5 and LEAVER_EARLY_FRAC == 0.5 else f'seekers {SEEKER_EARLY_FRAC:.0%} early | leavers {LEAVER_EARLY_FRAC:.0%} early'
print(f'  Temporal   : {temp_status}')


# ==========================================================================
# SECTION 1
# ==========================================================================
# ╔══════════════╗
# ║  CELL 1: Shared Helpers  (do not edit)                       ║
# ║  ci95()        — 95% confidence half-width of a sample mean  ║
# ║  excel_table() — prints a TAB-separated table; paste straight║
# ║                  into Excel and it splits into columns.       ║
# ╚══════════════╝

def ci95(arr):
    """95% confidence half-width of the mean (1.96 * standard error)."""
    a = np.asarray(arr, dtype=float)
    if a.size < 2:
        return 0.0
    return 1.96 * a.std(ddof=1) / np.sqrt(a.size)

def excel_table(headers, rows, title=None, pretty=False):
    """pretty=False -> TAB-separated; pasting into Excel splits into columns.
       pretty=True  -> space-aligned for on-screen reading only."""
    fmt  = lambda v: '' if v is None else (f'{v:.4g}' if isinstance(v, float) else str(v))
    head = [str(h) for h in headers]
    body = [[fmt(v) for v in r] for r in rows]
    if title:
        print(f"\n# === {title}  (tab-separated — paste into Excel) ===")
    if pretty:
        w = [max(len(r[i]) for r in [head] + body) for i in range(len(head))]
        for r in [head] + body:
            print('  '.join(s.rjust(w[i]) for i, s in enumerate(r)))
    else:
        print('\t'.join(head))
        for r in body:
            print('\t'.join(r))

print('Helpers defined: ci95(), excel_table()')


# ==========================================================================
# SECTION 2
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 2: Simulation Engine                                   ║
# ║                                                              ║
# ║  Seeker result codes stored in s_waits:                      ║
# ║    >= 0.0  →  app match  (0 = instant, >0 = waited N min)   ║
# ║    -3.0    →  unassisted find  (no leaver consumed)             ║
# ║                                                              ║
# ║  Leaver result codes stored in l_waits:                      ║
# ║    >= 0.0  →  matched  (0 = instant, >0 = waited N min)     ║
# ║    -1.0    →  expired after L_WAIT min, no seeker found      ║
# ╚══════════════════════════════════════════════════════════════╝

def run_sim(dau, iterations=200, zone_w=None, zone_h=None,
            radius=None, s_wait=None, l_wait=None, window=None,
            lucky_rate=None, hotspots=None, cluster_frac=None,
            cluster_sigma=None, seeker_early_frac=None,
            leaver_early_frac=None, time_split=None,
            master_seed=12345):
    """
    Run the discrete-event parking match simulation.
    All parameters default to the global knobs set in Cell 0.
    Pass any parameter explicitly to override (used by sensitivity sweeps).
    """
    zone_w            = zone_w            if zone_w            is not None else ZONE_W_KM
    zone_h            = zone_h            if zone_h            is not None else ZONE_H_KM
    radius            = radius            if radius            is not None else RADIUS_KM
    s_wait            = s_wait            if s_wait            is not None else S_WAIT
    l_wait            = l_wait            if l_wait            is not None else L_WAIT
    window            = window            if window            is not None else WINDOW_MINS
    lucky_rate        = lucky_rate        if lucky_rate        is not None else LUCKY_FIND_PER_MIN
    hotspots          = hotspots          if hotspots          is not None else HOTSPOTS
    cluster_frac      = cluster_frac      if cluster_frac      is not None else CLUSTER_FRAC
    cluster_sigma     = cluster_sigma     if cluster_sigma     is not None else CLUSTER_SIGMA
    seeker_early_frac = seeker_early_frac if seeker_early_frac is not None else SEEKER_EARLY_FRAC
    leaver_early_frac = leaver_early_frac if leaver_early_frac is not None else LEAVER_EARLY_FRAC
    time_split        = time_split        if time_split        is not None else TIME_SPLIT

    r2 = radius ** 2
    all_s_waits, all_l_waits             = [], []
    all_arrival_dists, all_match_dists   = [], []
    iter_s_rates, iter_l_rates, iter_s_parked = [], [], []
    zone_s_waits = [[], []]
    zone_l_waits = [[], []]

    master_rng = np.random.default_rng(seed=master_seed)

    for _ in range(iterations):
        rng = np.random.default_rng(master_rng.integers(0, 2**31))
        n   = int(dau)

        s_t = np.sort(rng.uniform(0, window, n)) if seeker_early_frac == 0.5 else _split_times(rng, n, window, seeker_early_frac, time_split)
        l_t = np.sort(rng.uniform(0, window, n)) if leaver_early_frac == 0.5 else _split_times(rng, n, window, leaver_early_frac, time_split)

        s_x, s_y = _positions(rng, n, zone_w, zone_h, hotspots, cluster_frac, cluster_sigma)
        l_x, l_y = _positions(rng, n, zone_w, zone_h, hotspots, cluster_frac, cluster_sigma)

        s_zone = (s_t >= time_split).astype(int)
        l_zone = (l_t >= time_split).astype(int)

        types = np.concatenate([np.zeros(n), np.ones(n)])
        times = np.concatenate([s_t, l_t])
        xs    = np.concatenate([s_x, l_x])
        ys    = np.concatenate([s_y, l_y])
        ids   = np.concatenate([np.arange(n), np.arange(n)])
        order = np.argsort(times, kind='stable')
        types, times, xs, ys, ids = (a[order] for a in (types, times, xs, ys, ids))

        sw     = np.full(n, -3.0)   # default = unassisted find (overwritten on app match)
        lw     = np.full(n, -1.0)
        act_s, act_l = [], []
        prev_t = 0.0

        for i in range(len(times)):
            t, x, y, typ, eid = times[i], xs[i], ys[i], types[i], int(ids[i])

            # ── Purge expired seekers → unassisted find ──
            surviving_s = []
            for s in act_s:
                if t - s[0] > s_wait:
                    sw[int(s[3])] = -3.0
                else:
                    surviving_s.append(s)
            act_s = surviving_s

            act_l = [l for l in act_l if t - l[0] <= l_wait]

            # ── Lucky finds: per-minute organic chance ──
            if lucky_rate > 0 and act_s:
                dt = t - prev_t
                if dt > 0:
                    p_still = (1 - lucky_rate) ** dt
                    surviving = []
                    for s in act_s:
                        if rng.random() < p_still:
                            surviving.append(s)
                        else:
                            sw[int(s[3])] = -3.0
                    act_s = surviving
            prev_t = t

            if typ == 0:   # — Seeker arrives —
                matched = False
                if act_l:
                    lc = np.array([(l[1], l[2]) for l in act_l])
                    sd = (lc[:, 0] - x)**2 + (lc[:, 1] - y)**2
                    mi = int(np.argmin(sd))
                    all_arrival_dists.append(np.sqrt(sd[mi]))
                    if sd[mi] <= r2:
                        all_match_dists.append(np.sqrt(sd[mi]))
                        sw[eid]               = 0.0
                        lw[int(act_l[mi][3])] = t - act_l[mi][0]
                        act_l.pop(mi)
                        matched = True
                if not matched:
                    act_s.append((t, x, y, eid))

            else:          # — Leaver arrives —
                matched = False
                if act_s:
                    sc = np.array([(s[1], s[2]) for s in act_s])
                    sd = (sc[:, 0] - x)**2 + (sc[:, 1] - y)**2
                    mi = int(np.argmin(sd))
                    all_arrival_dists.append(np.sqrt(sd[mi]))
                    if sd[mi] <= r2:
                        all_match_dists.append(np.sqrt(sd[mi]))
                        lw[eid]               = 0.0
                        sw[int(act_s[mi][3])] = t - act_s[mi][0]
                        act_s.pop(mi)
                        matched = True
                if not matched:
                    act_l.append((t, x, y, eid))

        # ── Day ended: remaining seekers → unassisted find ──
        for s in act_s:
            sw[int(s[3])] = -3.0

        all_s_waits.extend(sw)
        all_l_waits.extend(lw)
        iter_s_rates.append(np.sum(sw >= 0) / n)
        iter_l_rates.append(np.sum(lw >= 0) / n)
        iter_s_parked.append(np.sum((sw >= 0) | (sw == -3.0)) / n)
        for z in range(2):
            zone_s_waits[z].extend(sw[s_zone == z])
            zone_l_waits[z].extend(lw[l_zone == z])

    return {
        's_waits':           np.array(all_s_waits),
        'l_waits':           np.array(all_l_waits),
        'arrival_distances': all_arrival_dists,
        'match_distances':   all_match_dists,
        'iter_s_rates':      np.array(iter_s_rates),
        'iter_l_rates':      np.array(iter_l_rates),
        'iter_s_parked':     np.array(iter_s_parked),
        's_waits_early':     np.array(zone_s_waits[0]),
        's_waits_late':      np.array(zone_s_waits[1]),
        'l_waits_early':     np.array(zone_l_waits[0]),
        'l_waits_late':      np.array(zone_l_waits[1]),
    }


# ── Seeded helper: split arrival times ────────────────────────────
def _split_times(rng, n, window, early_frac, split):
    n_early = int(round(n * early_frac))
    n_late  = n - n_early
    early   = rng.uniform(0,     split,  n_early) if n_early > 0 else np.array([])
    late    = rng.uniform(split, window, n_late)  if n_late  > 0 else np.array([])
    return np.sort(np.concatenate([early, late]))


# ── Seeded helper: sample positions (no extend_border) ────────────
def _positions(rng, n, zone_w, zone_h, hotspots, cluster_frac, cluster_sigma):
    if not hotspots or cluster_frac <= 0.0:
        return rng.uniform(0, zone_w, n), rng.uniform(0, zone_h, n)
    n_cluster = int(round(n * cluster_frac))
    n_uniform = n - n_cluster
    xs, ys = [], []
    if n_uniform > 0:
        xs.append(rng.uniform(0, zone_w, n_uniform))
        ys.append(rng.uniform(0, zone_h, n_uniform))
    if n_cluster > 0:
        centers = np.array([(h['cx'], h['cy']) for h in hotspots])
        weights = np.array([h['weight'] for h in hotspots], dtype=float)
        weights /= weights.sum()
        chosen = rng.choice(len(centers), size=n_cluster, p=weights)
        cx = np.clip(centers[chosen, 0] + rng.normal(0, cluster_sigma, n_cluster), 0, zone_w)
        cy = np.clip(centers[chosen, 1] + rng.normal(0, cluster_sigma, n_cluster), 0, zone_h)
        xs.append(cx); ys.append(cy)
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    idx = rng.permutation(n)
    return x_all[idx], y_all[idx]


print('Engine ready: run_sim()')


# ==========================================================================
# SECTION 3
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 3: DAU Sweep  —  computation only, no plots            ║
# ╚══════════════════════════════════════════════════════════════╝

sweep = []

print(f'DAU sweep — {ITERATIONS} iters/point | Lucky: {LUCKY_FIND_PER_MIN*100:.0f}%/min | S_WAIT: {S_WAIT} min')
print(f"{'DAU':>6}  {'App%':>6} {'±':>4}  {'Unassist%':>10}  {'Leave%':>7} {'±':>4}  {'S_inst':>7}  {'L_inst':>7}  {'MatchD':>7}")
print('-' * 84)

for d in DAU_AXIS:
    res = run_sim(d, iterations=ITERATIONS)
    sw, lw = res['s_waits'], res['l_waits']
    n = len(sw)

    p_app   = np.sum(sw >= 0)    / n * 100
    p_lucky = np.sum(sw == -3.0) / n * 100
    p_park  = (np.sum(sw >= 0) + np.sum(sw == -3.0)) / n * 100
    pi_s    = np.sum(sw == 0)    / n * 100
    p_l     = np.sum(lw >= 0)    / n * 100
    pi_l    = np.sum(lw == 0)    / n * 100
    match_d = float(np.mean(res['match_distances'])) if res['match_distances'] else 0.0
    ci_s    = 1.96 * np.std(res['iter_s_rates'], ddof=1) / np.sqrt(ITERATIONS) * 100
    ci_l    = 1.96 * np.std(res['iter_l_rates'], ddof=1) / np.sqrt(ITERATIONS) * 100

    sweep.append({
        'dau': d, 'p_app': p_app, 'ci_s': ci_s, 'p_s_inst': pi_s,
        'p_lucky': p_lucky, 'p_parked': p_park,
        'p_l_match': p_l, 'ci_l': ci_l, 'p_l_inst': pi_l,
        'match_dist': match_d,
        's_waits': sw, 'l_waits': lw,
        'match_distances': res['match_distances'],
        'iter_s_rates': res['iter_s_rates'],
        'iter_l_rates': res['iter_l_rates'],
        'iter_s_parked': res['iter_s_parked'],
    })
    print(f"{d:>6}  {p_app:>5.1f}% {ci_s:>3.1f}  {p_lucky:>9.1f}%  {p_l:>6.1f}% {ci_l:>3.1f}  {pi_s:>6.1f}%  {pi_l:>6.1f}%  {match_d:>6.3f}")

df = pd.DataFrame([{k: v for k, v in r.items()
                     if k not in ('s_waits', 'l_waits', 'match_distances',
                                  'iter_s_rates', 'iter_l_rates', 'iter_s_parked')}
                    for r in sweep])

print('\nSweep complete. Results stored in `sweep` (list) and `df` (DataFrame).')


# ── Excel-ready table ──
excel_table(
    ['DAU','App%','App_CI95','Unassisted%','Leave%','Leave_CI95','S_inst%','L_inst%','MatchD_km'],
    [[r['dau'], r['p_app'], r['ci_s'], r['p_lucky'], r['p_l_match'], r['ci_l'],
      r['p_s_inst'], r['p_l_inst'], r['match_dist']] for r in sweep],
    title='Cell 3 — DAU sweep')


# ==========================================================================
# SECTION 4
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 4: Sensitivity Sweeps  —  computation only, no plots   ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Helper: shared axis style ─────────────────────────────────────
def _ci(iter_rates, n_iter):
    return 1.96 * np.std(iter_rates, ddof=1) / np.sqrt(n_iter) * 100


# ════════════════════════════════════════════════════════════════
#  4A — AREA SWEEP
#  How does match rate change as the city zone grows?
# ════════════════════════════════════════════════════════════════
area_results = []
print(f'Area sweep — DAU={AREA_DAU}, {AREA_ITER} iters, aspect {ASPECT}:1')
print(f"{'Area':>6}  {'W×H':>10}  {'Seek%':>7} {'±':>4}  {'Leave%':>7} {'±':>4}  {'MatchD':>7}")
print('-' * 56)

for area in AREA_VALUES:
    h   = np.sqrt(area / ASPECT)
    w   = ASPECT * h
    res = run_sim(AREA_DAU, iterations=AREA_ITER, zone_w=w, zone_h=h)
    sw, lw = res['s_waits'], res['l_waits']
    n   = len(sw)
    ps  = np.sum(sw >= 0) / n * 100
    pl  = np.sum(lw >= 0) / n * 100
    ci_s = _ci(res['iter_s_rates'], AREA_ITER)
    ci_l = _ci(res['iter_l_rates'], AREA_ITER)
    md  = float(np.mean(res['match_distances'])) if res['match_distances'] else 0.0
    area_results.append({'area': area, 'w': w, 'h': h,
                          'p_s': ps, 'ci_s': ci_s, 'p_l': pl, 'ci_l': ci_l, 'match_dist': md})
    print(f"{area:>5.1f}  {w:.2f}×{h:.2f}  {ps:>6.1f}% {ci_s:>3.1f}  {pl:>6.1f}% {ci_l:>3.1f}  {md:>6.3f}")

adf = pd.DataFrame(area_results)
print()


# ════════════════════════════════════════════════════════════════
#  4B — RADIUS SWEEP
#  How sensitive are match rates to the match radius?
#  Also computes the effective overlap heatmap (used in Cell 6).
# ════════════════════════════════════════════════════════════════
rad_results = []
print(f'Radius sweep — DAU={RAD_DAU}, {RAD_ITER} iters')
print(f"{'Radius':>8}  {'Seek%':>7} {'±':>4}  {'Leave%':>7} {'±':>4}")
print('-' * 38)

for r in RADIUS_VALUES:
    res  = run_sim(RAD_DAU, iterations=RAD_ITER, radius=r)
    sw, lw = res['s_waits'], res['l_waits']
    n    = len(sw)
    ps   = np.sum(sw >= 0) / n * 100
    pl   = np.sum(lw >= 0) / n * 100
    ci_s = _ci(res['iter_s_rates'], RAD_ITER)
    ci_l = _ci(res['iter_l_rates'], RAD_ITER)
    rad_results.append({'radius': r, 'p_s': ps, 'ci_s': ci_s, 'p_l': pl, 'ci_l': ci_l})
    print(f"  r={r:.1f} km  →  Seek: {ps:>5.1f}% ±{ci_s:.1f}  Leave: {pl:>5.1f}% ±{ci_l:.1f}")

rdf = pd.DataFrame(rad_results)

# Effective overlap heatmap grid (Monte-Carlo)
def _overlap(sx, sy, zone_w, zone_h, radius, n_mc=3000):
    angles = np.random.uniform(0, 2*np.pi, n_mc)
    radii  = radius * np.sqrt(np.random.uniform(0, 1, n_mc))
    px, py = sx + radii * np.cos(angles), sy + radii * np.sin(angles)
    return np.mean((px >= 0) & (px <= zone_w) & (py >= 0) & (py <= zone_h))

W, H, R  = ZONE_W_KM, ZONE_H_KM, RADIUS_KM
_res     = 60
_sx      = np.linspace(-R, W + R, _res)
_sy      = np.linspace(-R, H + R, _res)
overlap_grid = np.zeros((_res, _res))
np.random.seed(12345)   # reproducible heatmap
print()
print('Computing overlap heatmap…', end=' ')
for i, sy in enumerate(_sy):
    for j, sx in enumerate(_sx):
        overlap_grid[i, j] = _overlap(sx, sy, W, H, R)
print('done.')
print()


# ════════════════════════════════════════════════════════════════
#  4C — TEMPORAL MISMATCH SWEEP
#  Uniform arrivals vs. seekers-early / leavers-late scenario.
# ════════════════════════════════════════════════════════════════
t_results = {label: [] for label in T_SCENARIOS}
print(f'Temporal sweep — {T_ITER} iters/point')
print(f"{'DAU':>6}  {'Scenario':<34}  {'App%':>6} {'±':>4}  {'Unas%':>6}  {'Leave%':>7}")
print('-' * 68)

for dau in T_DAU_VALUES:
    for label, params in T_SCENARIOS.items():
        res    = run_sim(dau, iterations=T_ITER, **params)
        sw, lw = res['s_waits'], res['l_waits']
        n      = len(sw)
        p_app  = np.sum(sw >= 0)    / n * 100
        p_unas = np.sum(sw == -3.0) / n * 100
        p_l    = np.sum(lw >= 0)    / n * 100
        ci     = _ci(res['iter_s_rates'], T_ITER)
        t_results[label].append({'dau': dau, 'p_app': p_app, 'ci': ci,
                                  'p_unas': p_unas, 'p_l': p_l})
        print(f"{dau:>6}  {label:<34}  {p_app:>5.1f}% {ci:>3.1f}  {p_unas:>5.1f}%  {p_l:>6.1f}%")
    print()


# ════════════════════════════════════════════════════════════════
#  4D — CLUSTERING SWEEP
#  Does concentrating users around hotspots help match rates?
#  Skipped automatically if N_CLUSTERS == 0.
# ════════════════════════════════════════════════════════════════
clust_sweep = {}

if N_CLUSTERS == 0:
    print('Clustering sweep skipped (N_CLUSTERS = 0 in Cell 0).')
else:
    print(f'Clustering sweep — σ={CLUSTER_SIGMA} km | {CLUST_ITER} iters/point')
    print(f"{'DAU':>6}  {'Frac':>6}  {'App%':>7} {'±':>4}  {'Unas%':>6}  {'Leave%':>7}")
    print('-' * 52)
    for dau in CLUST_DAU_VALUES:
        rows = []
        for cf in CLUST_FRAC_VALUES:
            res    = run_sim(dau, iterations=CLUST_ITER,
                             cluster_frac=cf, cluster_sigma=CLUSTER_SIGMA)
            sw, lw = res['s_waits'], res['l_waits']
            n      = len(sw)
            p_app  = np.sum(sw >= 0)    / n * 100
            p_unas = np.sum(sw == -3.0) / n * 100
            p_l    = np.sum(lw >= 0)    / n * 100
            ci     = _ci(res['iter_s_rates'], CLUST_ITER)
            rows.append({'cf': cf, 'p_app': p_app, 'ci': ci,
                         'p_unas': p_unas, 'p_l': p_l})
            print(f"{dau:>6}  {cf:>5.0%}  {p_app:>6.1f}% {ci:>3.1f}  {p_unas:>5.1f}%  {p_l:>6.1f}%")
        clust_sweep[dau] = rows
        print()

print('\nSensitivity sweeps complete.')


# ── Excel-ready tables ──
excel_table(['Area_km2','W_km','H_km','Seek%','Seek_CI95','Leave%','Leave_CI95','MatchD_km'],
            [[r['area'], r['w'], r['h'], r['p_s'], r['ci_s'], r['p_l'], r['ci_l'], r['match_dist']] for r in area_results],
            title='Cell 4A — area sweep')
excel_table(['Radius_km','Seek%','Seek_CI95','Leave%','Leave_CI95'],
            [[r['radius'], r['p_s'], r['ci_s'], r['p_l'], r['ci_l']] for r in rad_results],
            title='Cell 4B — radius sweep')
_trows = []
for label, rows in t_results.items():
    for r in rows:
        _trows.append([label, r['dau'], r['p_app'], r['ci'], r['p_unas'], r['p_l']])
excel_table(['Scenario','DAU','App%','App_CI95','Unassisted%','Leave%'], _trows,
            title='Cell 4C — temporal mismatch sweep')
if clust_sweep:
    _crows = [[dau, r['cf']*100, r['p_app'], r['ci'], r['p_unas'], r['p_l']]
              for dau, rows in clust_sweep.items() for r in rows]
    excel_table(['DAU','ClusterFrac%','App%','App_CI95','Unassisted%','Leave%'], _crows,
                title='Cell 4D — clustering sweep')


# ==========================================================================
# SECTION 5
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 5: Diagnostics & Metrics                               ║
# ╚══════════════════════════════════════════════════════════════╝

SEP  = '=' * 70
SEP2 = '-' * 70

# ════════════════════════════════════════════════════════════════
#  5A — UNASSISTED-FIND BASELINE STATS
# ════════════════════════════════════════════════════════════════
print(SEP)
print('  UNASSISTED-FIND BASELINE')
print(SEP)
print(f'  Rate                : {LUCKY_FIND_PER_MIN*100:.0f}% / min')
print(f'  E[unassisted search]   : {1/LUCKY_FIND_PER_MIN:.0f} min  (without app)')
print(f'  P(find ≤  5 min)    : {(1-(1-LUCKY_FIND_PER_MIN)**5)*100:.1f}%')
print(f'  P(find ≤ 10 min)    : {(1-(1-LUCKY_FIND_PER_MIN)**10)*100:.1f}%')
print(f'  P(find ≤ {S_WAIT} min)   : {(1-(1-LUCKY_FIND_PER_MIN)**S_WAIT)*100:.1f}%  (full seeker window)')
print()


# ════════════════════════════════════════════════════════════════
#  5B — BILATERAL SANITY CHECK
#  Verifies the simulation's internal consistency:
#    - all seekers and leavers account for 100 %
#    - seeker app-match count == leaver app-match count
#    - no wait times exceed their allowed windows
#    - no match distances exceed the match radius
# ════════════════════════════════════════════════════════════════
print(SEP)
print(f'  BILATERAL SANITY CHECK  (DAU={SANITY_DAU}, {SANITY_ITER} iters)')
print(SEP)

sc     = run_sim(SANITY_DAU, iterations=SANITY_ITER)
sc_s, sc_l = sc['s_waits'], sc['l_waits']
total  = len(sc_s)
md     = sc['match_distances']

s_app  = int(np.sum(sc_s >= 0))
s_inst = int(np.sum(sc_s == 0))
s_wait = int(np.sum(sc_s > 0))
s_luck = int(np.sum(sc_s == -3.0))
l_mat  = int(np.sum(sc_l >= 0))
l_inst = int(np.sum(sc_l == 0))
l_wait = int(np.sum(sc_l > 0))
l_miss = int(np.sum(sc_l < 0))

ok_cons_s    = (s_app + s_luck) == total
ok_cons_l    = (l_mat + l_miss)         == total
ok_bilateral = s_app == l_mat
bad_s        = int(np.sum((sc_s > S_WAIT) & (sc_s > 0)))
bad_l        = int(np.sum((sc_l > L_WAIT) & (sc_l > 0)))
bad_dist     = sum(1 for d in md if d > RADIUS_KM + 1e-9)

print(f"  Total seekers sampled  : {total:,}")
print()
print(f"  SEEKERS")
print(f"    App match (instant)  : {s_inst:>8,}")
print(f"    App match (waited)   : {s_wait:>8,}")
print(f"    App match (total)    : {s_app:>8,}  ({s_app/total*100:.1f}%)")
print(f"    Unassisted find      : {s_luck:>8,}  ({s_luck/total*100:.1f}%)")
print()
print(f"  LEAVERS")
print(f"    Matched (instant)    : {l_inst:>8,}")
print(f"    Matched (waited)     : {l_wait:>8,}")
print(f"    Matched (total)      : {l_mat:>8,}  ({l_mat/total*100:.1f}%)")
print(f"    Expired              : {l_miss:>8,}")
print(SEP2)

def _flag(cond): return 'PASS ✓' if cond else 'FAIL ✗'
print(f"  Seeker conservation    : {_flag(ok_cons_s)}")
print(f"  Leaver conservation    : {_flag(ok_cons_l)}")
print(f"  Bilateral symmetry     : {_flag(ok_bilateral)}  (s_app={s_app} | l_mat={l_mat})")
print(f"  Seeker waits ≤ {S_WAIT} min  : {_flag(bad_s == 0)}")
print(f"  Leaver waits ≤  {L_WAIT} min  : {_flag(bad_l == 0)}")
print(f"  Match dist  ≤ {RADIUS_KM} km  : {_flag(bad_dist == 0)}")
print(f"  Mean match distance    : {np.mean(md):.3f} km")
print()


# ════════════════════════════════════════════════════════════════
#  5C — ZONE SPLIT (PEAK vs OFF-PEAK)
#  Breaks results by time-of-day half to show how match quality
#  varies between the early and late portions of the day.
# ════════════════════════════════════════════════════════════════
def _zone_report(label, sw, lw):
    n_s = len(sw);  n_l = len(lw)
    if n_s == 0:
        print(f'  {label}: no data'); return
    s_inst  = np.sum(sw == 0.0)  / n_s * 100
    s_del   = np.sum(sw >  0.0)  / n_s * 100
    s_luck  = np.sum(sw == -3.0) / n_s * 100
    s_park  = s_inst + s_del + s_luck
    l_inst  = np.sum(lw == 0.0)  / n_l * 100 if n_l else 0.0
    l_del   = np.sum(lw >  0.0)  / n_l * 100 if n_l else 0.0
    l_fail  = np.sum(lw == -1.0) / n_l * 100 if n_l else 0.0
    print(f'\n  {label}  (n_seekers≈{n_s//SANITY_ITER}  n_leavers≈{n_l//SANITY_ITER})')
    print(f'  SEEKERS   instant={s_inst:>5.1f}%  waited={s_del:>5.1f}%  unassisted={s_luck:>5.1f}%  parked={s_park:>5.1f}%')
    print(f'  LEAVERS   instant={l_inst:>5.1f}%  waited={l_del:>5.1f}%  unmatched={l_fail:>5.1f}%')

split_h = 7 + TIME_SPLIT // 60
split_m = TIME_SPLIT % 60
print(SEP)
print(f'  ZONE SPLIT  |  split at {split_h:02d}:{split_m:02d}  |  seekers {SEEKER_EARLY_FRAC:.0%}/{1-SEEKER_EARLY_FRAC:.0%}  |  leavers {LEAVER_EARLY_FRAC:.0%}/{1-LEAVER_EARLY_FRAC:.0%}')
print(SEP)
_zone_report(f'PEAK     07:00–{split_h:02d}:{split_m:02d}', sc['s_waits_early'], sc['l_waits_early'])
print(SEP2)
_zone_report(f'OFF-PEAK {split_h:02d}:{split_m:02d}–24:00', sc['s_waits_late'],  sc['l_waits_late'])
print(SEP2)
_zone_report('FULL DAY combined',                          sc['s_waits'],       sc['l_waits'])
print(SEP)
print()


# ════════════════════════════════════════════════════════════════
#  5D — SUMMARY TABLE  (from DAU sweep in Cell 3)
# ════════════════════════════════════════════════════════════════
print(SEP)
print('  DAU SWEEP SUMMARY')
print(f'  Zone: {ZONE_W_KM}×{ZONE_H_KM} km | Radius: {RADIUS_KM} km | Lucky: {LUCKY_FIND_PER_MIN*100:.0f}%/min | Iters: {ITERATIONS}')
print(SEP)
print(f"{'DAU':>6}  {'App%':>6} {'±':>4}  {'Unassist%':>10}  {'Leave%':>7} {'±':>4}  {'S_inst':>7}  {'L_inst':>7}  {'MatchD':>7}")
print('-' * 84)
for row in sweep:
    print(f"{row['dau']:>6}  {row['p_app']:>5.1f}% {row['ci_s']:>3.1f}  {row['p_lucky']:>9.1f}%  {row['p_l_match']:>6.1f}% {row['ci_l']:>3.1f}  {row['p_s_inst']:>6.1f}%  {row['p_l_inst']:>6.1f}%  {row['match_dist']:>6.3f}")
print()
print('  App%       = matched via app          |  Unassist% = parked unassisted (incl. >50 min)')
print('  App% + Unassist% = 100% (every seeker eventually parks)')
print('  Leave%     = leavers who handed off    |  MatchD    = mean match distance (km)')


# ── Excel-ready table (DAU summary) ──
excel_table(
    ['DAU','App%','App_CI95','Unassisted%','Leave%','Leave_CI95','S_inst%','L_inst%','MatchD_km'],
    [[r['dau'], r['p_app'], r['ci_s'], r['p_lucky'], r['p_l_match'], r['ci_l'],
      r['p_s_inst'], r['p_l_inst'], r['match_dist']] for r in sweep],
    title='Cell 5 — DAU summary')


# ==========================================================================
# SECTION 6
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 6: All Plots                                           ║
# ╚══════════════════════════════════════════════════════════════╝

MILESTONES   = [100, 300, 500, 1000, 2000, 4000]
C_GREEN      = '#27ae60'
C_PURPLE     = '#8e44ad'

def _style(ax):
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('black'); sp.set_linewidth(0.8)
    ax.grid(True, ls='--', alpha=0.3, color='grey')
    ax.set_axisbelow(True)

def _milestones(ax):
    for m in MILESTONES:
        ax.axvline(m, color='#aaaaaa', ls=':', alpha=0.5, lw=1)

daus    = df['dau'].values
idx_1k  = min(range(len(sweep)), key=lambda i: abs(sweep[i]['dau'] - 1000))


# ════════════════════════════════════════════════════════════════
#  FIG 1 — Five-Panel Main Plot
# ════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 18), facecolor='white')
gs  = fig.add_gridspec(3, 2, hspace=0.42, wspace=0.32)
fig.suptitle(
    f'ParkPass — Network Effects Simulation\n'
    f'Zone {ZONE_W_KM}×{ZONE_H_KM} km  |  Radius {RADIUS_KM} km  '
    f'|  Lucky {LUCKY_FIND_PER_MIN*100:.1f}%/min  '
    f'|  S_WAIT {S_WAIT} min  |  L_WAIT {L_WAIT} min',
    fontsize=13, fontweight='bold', color='black', y=1.01
)

# Panel 1 — Seeker outcomes
ax1 = fig.add_subplot(gs[0, 0]); _style(ax1); _milestones(ax1)
ax1.fill_between(daus, 0,             df['p_app'],    alpha=0.18, color=C_SEEK)
ax1.fill_between(daus, df['p_app'],   df['p_parked'], alpha=0.18, color=C_LUCKY)
ax1.plot(daus, df['p_app'],    'o-', color=C_SEEK,  lw=2.5, ms=6, label='App match')
ax1.plot(daus, df['p_parked'], 's-', color=C_LUCKY, lw=2,   ms=5, label='Total parked (app + unassisted)')
ax1.errorbar(daus, df['p_app'], yerr=df['ci_s'], fmt='none', ecolor=C_SEEK, capsize=4, lw=1.2, alpha=0.5)
ax1.set_title('Fig 1 — Seeker Outcomes vs DAU', fontweight='bold', fontsize=11, pad=10)
ax1.set_xlabel('DAU (seekers & leavers each)'); ax1.set_ylabel('% of All Seekers')
ax1.set_ylim(0, 105); ax1.set_xlim(daus[0]*0.9, daus[-1]*1.05)
ax1.legend(fontsize=9, loc='center right', framealpha=0.9, edgecolor='black')

# Panel 2 — CDF at closest DAU to 1000
ax2   = fig.add_subplot(gs[0, 1]); _style(ax2)
tbins = np.linspace(0, S_WAIT, 400)
sw_1k = sweep[idx_1k]['s_waits']; lw_1k = sweep[idx_1k]['l_waits']; n_1k = len(sw_1k)
cum_app = [(np.sum((sw_1k >= 0) & (sw_1k <= t)) / n_1k) * 100 for t in tbins]
cum_all = [(np.sum(((sw_1k >= 0) & (sw_1k <= t)) | (sw_1k == -3.0)) / n_1k) * 100 for t in tbins]
cum_l   = [(np.sum((lw_1k >= 0) & (lw_1k <= t)) / len(lw_1k)) * 100 for t in tbins]
ax2.fill_between(tbins, cum_app, color=C_SEEK,  alpha=0.10)
ax2.fill_between(tbins, cum_l,   color=C_LEAVE, alpha=0.10)
ax2.plot(tbins, cum_app, lw=2.5, color=C_SEEK,  label='Seeker: app match')
ax2.plot(tbins, cum_all, lw=2,   color=C_LUCKY, ls='--', label='Seeker: total parked')
ax2.plot(tbins, cum_l,   lw=2.5, color=C_LEAVE, label='Leaver: matched')
ax2.axvline(L_WAIT, color=C_LEAVE, ls=':', lw=1.5, alpha=0.7)
ax2.text(L_WAIT + 0.5, 8, f'Leaver\nexpiry\n({L_WAIT} min)', color=C_LEAVE, fontsize=8, va='bottom')
ax2.set_title(f'Fig 2 — Cumulative Match Rate  (DAU={sweep[idx_1k]["dau"]})', fontweight='bold', fontsize=11, pad=10)
ax2.set_xlabel('Minutes Since Opening App'); ax2.set_ylabel('Cumulative % Matched')
ax2.set_xlim(0, S_WAIT); ax2.set_ylim(0, 100)
ax2.legend(fontsize=8.5, framealpha=0.9, edgecolor='black')

# Panel 3 — Match rate: seeker vs leaver
ax3 = fig.add_subplot(gs[1, 0]); _style(ax3); _milestones(ax3)
ax3.plot(daus, df['p_s_inst'],  'v--', color=C_GREEN,  lw=1.8, ms=4, label='Seeker: instant')
ax3.plot(daus, df['p_app'],     'o-',  color=C_GREEN,  lw=2.5, ms=5, label='Seeker: total')
ax3.errorbar(daus, df['p_app'], yerr=df['ci_s'], fmt='none', ecolor=C_GREEN, capsize=3, lw=1, alpha=0.4)
ax3.plot(daus, df['p_l_inst'],  '^--', color=C_PURPLE, lw=1.8, ms=4, label='Leaver: instant')
ax3.plot(daus, df['p_l_match'], 's-',  color=C_PURPLE, lw=2.5, ms=5, label='Leaver: total')
ax3.errorbar(daus, df['p_l_match'], yerr=df['ci_l'], fmt='none', ecolor=C_PURPLE, capsize=3, lw=1, alpha=0.4)
ax3.set_title('Fig 3 — App Match Rate: Seeker vs Leaver', fontweight='bold', fontsize=11, pad=10)
ax3.set_xlabel('DAU (seekers & leavers each)'); ax3.set_ylabel('Match Rate (%)')
ax3.set_ylim(0, 100); ax3.set_xlim(daus[0]*0.9, daus[-1]*1.05)
ax3.legend(fontsize=8.5, framealpha=0.9, edgecolor='black')

# Panel 4 — Mean match distance
ax4 = fig.add_subplot(gs[1, 1]); _style(ax4); _milestones(ax4)
ax4.fill_between(daus, df['match_dist'], RADIUS_KM, alpha=0.10, color=C_MDIST)
ax4.plot(daus, df['match_dist'], 's-', color=C_MDIST, lw=2.5, ms=5, label='Mean match distance')
ax4.axhline(RADIUS_KM, color=C_FAIL, ls='--', lw=1.8, alpha=0.7, label=f'Match radius cap ({RADIUS_KM} km)')
ax4.set_title('Fig 4 — Mean Match Distance vs DAU', fontweight='bold', fontsize=11, pad=10)
ax4.set_xlabel('DAU (seekers & leavers each)'); ax4.set_ylabel('Distance (km)')
ax4.set_ylim(0, RADIUS_KM * 1.2); ax4.set_xlim(daus[0]*0.9, daus[-1]*1.05)
ax4.legend(fontsize=9, framealpha=0.9, edgecolor='black')


# Panel 5 — Match distance distribution
ax5  = fig.add_subplot(gs[2, :]); _style(ax5)
md   = np.array(sweep[idx_1k]['match_distances'])
if len(md) > 0:
    bins   = np.linspace(0, RADIUS_KM, 50)
    counts, edges = np.histogram(md, bins=bins)
    pct    = counts / len(md) * 100
    centers = (edges[:-1] + edges[1:]) / 2
    ax5.bar(centers, pct, width=(edges[1]-edges[0])*0.88,
            color=C_MDIST, alpha=0.75, edgecolor='white', lw=0.4, label='Distribution')
    ax5.axvline(np.mean(md),   color=C_FAIL,  ls='--', lw=2.2, label=f'Mean: {np.mean(md):.3f} km')
    ax5.axvline(np.median(md), color='black', ls=':',  lw=2.2, label=f'Median: {np.median(md):.3f} km')
    ax5.axvline(RADIUS_KM,     color=C_LUCKY, ls='-',  lw=1.5, alpha=0.7, label=f'Radius cap: {RADIUS_KM} km')
ax5.set_xlim(0, RADIUS_KM)
ax5.set_title(f'Fig 5 — Match Distance Distribution  (DAU={sweep[idx_1k]["dau"]})', fontweight='bold', fontsize=11, pad=10)
ax5.set_xlabel('Distance Between Seeker and Leaver at Match (km)')
ax5.set_ylabel('% of All Matches')
ax5.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
ax5.legend(fontsize=10, framealpha=0.9, edgecolor='black')

plt.show()


# ════════════════════════════════════════════════════════════════
#  FIG 2 — Area & Radius Sensitivity (2 panels side by side)
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
fig.suptitle('ParkPass — Sensitivity: Zone Area & Match Radius', fontsize=12, fontweight='bold', y=1.02)

ax = axes[0]; _style(ax)
ax.errorbar(adf['area'], adf['p_s'], yerr=adf['ci_s'], fmt='o-', color=C_SEEK,  lw=2.5, ms=8, capsize=4, label='Seeker match %')
ax.errorbar(adf['area'], adf['p_l'], yerr=adf['ci_l'], fmt='s-', color=C_LEAVE, lw=2.5, ms=8, capsize=4, label='Leaver match %')
afine  = np.linspace(0.3, 5.0, 200)
ax.axvline(AREA_KM2, color='#888888', ls=':', alpha=0.5)
ax.text(AREA_KM2 + 0.05, 25, f'Baseline\n{AREA_KM2} km²', color='#666666', fontsize=9)
ax.set_title(f'Fig 6 — Area Sensitivity  (DAU={AREA_DAU})', fontweight='bold')
ax.set_xlabel('Zone Area (km²)'); ax.set_ylabel('Match Rate (%)')
ax.set_ylim(0, 105); ax.legend(fontsize=10)

ax = axes[1]; _style(ax)
ax.errorbar(rdf['radius'], rdf['p_s'], yerr=rdf['ci_s'], fmt='o-', color=C_SEEK,  lw=2.5, ms=8, capsize=4, label='Seeker')
ax.errorbar(rdf['radius'], rdf['p_l'], yerr=rdf['ci_l'], fmt='s-', color=C_LEAVE, lw=2.5, ms=8, capsize=4, label='Leaver')
ax.axvline(RADIUS_KM, color='#888888', ls=':', alpha=0.5)
ax.text(RADIUS_KM + 0.02, 25, f'Baseline\n{RADIUS_KM} km', color='#666666', fontsize=9)
ax.set_title(f'Fig 7 — Radius Sensitivity  (DAU={RAD_DAU})', fontweight='bold')
ax.set_xlabel('Match Radius (km)'); ax.set_ylabel('Match Rate (%)')
ax.set_ylim(0, 105); ax.legend(fontsize=10)

plt.tight_layout()
plt.show()


# ════════════════════════════════════════════════════════════════
#  FIG 3 — Effective Overlap Heatmap & Radius Sensitivity
# ════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5), facecolor='white')
im = ax.imshow(overlap_grid, origin='lower',
               extent=[-RADIUS_KM, ZONE_W_KM+RADIUS_KM, -RADIUS_KM, ZONE_H_KM+RADIUS_KM],
               cmap='YlGnBu', vmin=0, vmax=1, aspect='auto')
rect = plt.Rectangle((0, 0), ZONE_W_KM, ZONE_H_KM,
                      fill=False, edgecolor='#e74c3c', lw=2.5, ls='--', label='Leaver zone')
ax.add_patch(rect)
ax.set_title('Fig 8 — Effective Search Overlap Heatmap', fontweight='bold')
ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)')
ax.legend(fontsize=9, loc='upper left')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Fraction of search disc inside zone', color='black')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black')
plt.tight_layout()
plt.show()


# ════════════════════════════════════════════════════════════════
#  FIG 4 — Temporal Mismatch
# ════════════════════════════════════════════════════════════════
_colors_t  = {'Uniform (baseline)': '#2ecc71', 'Seekers early / Leavers late': '#e74c3c'}
_markers_t = {'Uniform (baseline)': 'o',       'Seekers early / Leavers late': 's'}

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
fig.suptitle('ParkPass — Temporal Mismatch: Seekers Early / Leavers Late',
             fontsize=11, fontweight='bold', y=1.02)

ax_a, ax_b = axes
for ax in axes: _style(ax)

for label, rows in t_results.items():
    y   = [r['p_app']  for r in rows]
    yf  = [r['p_unas'] for r in rows]
    ci  = [r['ci']     for r in rows]
    col = _colors_t[label]; mk = _markers_t[label]
    ax_a.plot(T_DAU_VALUES, y,  mk+'-', color=col, lw=2.5, ms=7, label=label)
    ax_a.fill_between(T_DAU_VALUES, [a-c for a,c in zip(y,ci)], [a+c for a,c in zip(y,ci)], alpha=0.12, color=col)
    ax_b.plot(T_DAU_VALUES, yf, mk+'-', color=col, lw=2.5, ms=7, label=label)

ax_a.set_title('Fig 9a — Seeker App Match Rate vs DAU', fontweight='bold', fontsize=11, pad=10)
ax_a.set_xlabel('DAU'); ax_a.set_ylabel('Seeker App Match Rate (%)'); ax_a.set_ylim(0, 100)
ax_a.legend(fontsize=10, framealpha=0.9, edgecolor='black')

ax_b.set_title('Fig 9b — Seeker Unassisted-Find Rate vs DAU', fontweight='bold', fontsize=11, pad=10)
ax_b.set_xlabel('DAU'); ax_b.set_ylabel('Seeker Unassisted-Find Rate (%)')
ax_b.legend(fontsize=10, framealpha=0.9, edgecolor='black')

plt.tight_layout()
plt.show()


# ════════════════════════════════════════════════════════════════
#  FIG 5 — Clustering Sweep  (only if clustering was run)
# ════════════════════════════════════════════════════════════════
if clust_sweep:
    _colors_c = {500: '#e74c3c', 1000: '#2ecc71', 2000: '#3498db'}

    def _style_c(ax):
        ax.set_facecolor('white')
        for sp in ax.spines.values(): sp.set_edgecolor('black'); sp.set_linewidth(0.8)
        ax.tick_params(colors='black')
        ax.grid(True, ls='--', alpha=0.3, color='grey'); ax.set_axisbelow(True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='white')
    fig.suptitle(f'ParkPass — Geographic Clustering Sweep  (σ={CLUSTER_SIGMA} km)',
                 fontsize=11, fontweight='bold', y=1.02)
    ax_a, ax_b = axes
    for ax in axes: _style_c(ax)

    for dau in CLUST_DAU_VALUES:
        rows   = clust_sweep[dau]
        cf_pct = [r['cf'] * 100   for r in rows]
        p_app  = [r['p_app']       for r in rows]
        p_fail = [r['p_unas']      for r in rows]
        ci     = [r['ci']          for r in rows]
        col    = _colors_c[dau]
        ax_a.plot(cf_pct, p_app,  'o-', color=col, lw=2.5, ms=6, label=f'DAU={dau}')
        ax_a.fill_between(cf_pct, [a-c for a,c in zip(p_app,ci)], [a+c for a,c in zip(p_app,ci)], alpha=0.12, color=col)
        ax_b.plot(cf_pct, p_fail, 's-', color=col, lw=2.5, ms=6, label=f'DAU={dau}')

    ax_a.set_title('Fig 10a — Seeker Match Rate vs Cluster Fraction', fontweight='bold', fontsize=11, pad=10)
    ax_a.set_xlabel('Cluster Fraction (% of users at hotspots)'); ax_a.set_ylabel('Seeker App Match Rate (%)')
    ax_a.set_xlim(0, 100); ax_a.set_ylim(0, 100); ax_a.legend(fontsize=10, framealpha=0.9, edgecolor='black')

    max_fail = max(r['p_unas'] for rows in clust_sweep.values() for r in rows)
    ax_b.set_title('Fig 10b — Seeker Unassisted-Find Rate vs Cluster Fraction', fontweight='bold', fontsize=11, pad=10)
    ax_b.set_xlabel('Cluster Fraction (% of users at hotspots)'); ax_b.set_ylabel('Seeker Unassisted-Find Rate (%)')
    ax_b.set_xlim(0, 100); ax_b.set_ylim(0, max_fail * 1.2); ax_b.legend(fontsize=10, framealpha=0.9, edgecolor='black')

    plt.tight_layout()
    plt.show()
else:
    print('Clustering plot skipped (N_CLUSTERS = 0 — set in Cell 0).')


# ── Excel-ready table (temporal mismatch behind Fig 9) ──
_t9 = [[label, r['dau'], r['p_app'], r['ci'], r['p_unas'], r['p_l']]
       for label, rows in t_results.items() for r in rows]
excel_table(['Scenario','DAU','App%','App_CI95','Unassisted%','Leave%'], _t9,
            title='Cell 6 — temporal mismatch (Fig 9)')


# ==========================================================================
# SECTION 7
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 7: Six-Panel Single-DAU Diagnostic Chart              ║
# ║  Self-contained — runs its own simulation internally.       ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Knobs for this cell ───────────────────────────────────────────
CHART_DAU  = 1000
CHART_ITER = 30

# ── Run simulation ────────────────────────────────────────────────
res  = run_sim(CHART_DAU, iterations=CHART_ITER)
sw   = res['s_waits']
lw   = res['l_waits']
n    = len(sw)
dists = res['match_distances']

# ── Seeker metrics ────────────────────────────────────────────────
p_inst_s   = np.sum(sw == 0.0)   / n * 100
p_waited_s = np.sum(sw >  0.0)   / n * 100
p_lucky    = np.sum(sw == -3.0)  / n * 100

matched_sw  = sw[sw >= 0]
avg_wait_s  = float(np.mean(matched_sw))   if len(matched_sw) > 0 else 0.0
med_wait_s  = float(np.median(matched_sw)) if len(matched_sw) > 0 else 0.0

# ── Leaver metrics ────────────────────────────────────────────────
p_inst_l   = np.sum(lw == 0.0)   / n * 100
p_waited_l = np.sum(lw >  0.0)   / n * 100
p_l_miss   = np.sum(lw == -1.0)  / n * 100


matched_lw  = lw[lw >= 0]
avg_wait_l  = float(np.mean(matched_lw))   if len(matched_lw) > 0 else 0.0
med_wait_l  = float(np.median(matched_lw)) if len(matched_lw) > 0 else 0.0

# ── Distance metrics ──────────────────────────────────────────────
avg_dist = float(np.mean(dists))   if dists else 0.0
med_dist = float(np.median(dists)) if dists else 0.0

# ── global lettering sizes (restored at end of cell) ──
_saved_rc = mpl.rcParams.copy()
plt.rcParams.update({
    'font.size':        13,
    'axes.titlesize':   14,
    'axes.labelsize':   15,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'legend.fontsize':  12,
})

# ── Colour palette for this chart ────────────────────────────────
COLORS = {
    'instant' : '#2ecc71',
    'waited'  : '#3498db',
    'lucky'   : '#f1c40f',
    'fail'    : '#e74c3c',
    'unmatch' : '#e67e22',
}


# ── Build Figure 1: Outcomes & Distance ──────────────────────────
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
fig1.suptitle(
    f'Match Success Rates and Distances',
    fontsize=16, fontweight='bold', color='black', y=1.01
)
for ax in axes1:
    ax.set_facecolor('white')
    for sp in ax.spines.values():
        sp.set_edgecolor('black')

ax1 = axes1[0]  # Seeker pie
ax2 = axes1[1]  # Leaver pie
ax5 = axes1[2]  # Match distance

# ── Panel 1.1: Seeker outcomes — pie ───────────────────────────────
s_vals    = [p_inst_s, p_waited_s, p_lucky]
s_labels  = [f'Instant match\n{p_inst_s:.1f}%',
             f'Waited match\n{p_waited_s:.1f}%',
             f'Unassisted find\n{p_lucky:.1f}%']
s_colors  = [COLORS['instant'], COLORS['waited'], COLORS['lucky']]
ax1.pie(s_vals, labels=s_labels, colors=s_colors,
        explode=[0.03, 0.03, 0.03], startangle=90,
        textprops={'color': 'black', 'fontsize': 12},
        wedgeprops={'linewidth': 1, 'edgecolor': 'black'})
ax1.set_title('Seeker Outcomes',
              fontsize=15, fontweight='bold', color='black', pad=12)

# ── Panel 1.2: Leaver outcomes — pie ───────────────────────────────
l_vals    = [p_inst_l, p_waited_l, p_l_miss]
l_labels  = [f'Instant match\n{p_inst_l:.1f}%',
             f'Waited match\n{p_waited_l:.1f}%',
             f'Unmatched\n{p_l_miss:.1f}%']
l_colors  = [COLORS['instant'], COLORS['waited'], COLORS['unmatch']]
ax2.pie(l_vals, labels=l_labels, colors=l_colors,
        explode=[0.03, 0.03, 0.03], startangle=90,
        textprops={'color': 'black', 'fontsize': 12},
        wedgeprops={'linewidth': 1, 'edgecolor': 'black'})
ax2.set_title('Leaver Outcomes',
              fontsize=15, fontweight='bold', color='black', pad=12)

# ── Panel 1.3: Match distance distribution ─────────────────────────
if dists:
    counts_d, edges_d = np.histogram(dists, bins=35)
    pct_d = counts_d / len(dists) * 100
    centers_d = (edges_d[:-1] + edges_d[1:]) / 2
    ax5.bar(centers_d, pct_d, width=(edges_d[1]-edges_d[0])*0.85,
            color=COLORS['instant'], edgecolor='white', lw=0.4,
            alpha=0.88, label='Match distances')
    ax5.axvline(avg_dist,  color='#e74c3c', lw=2, ls='--',
                label=f'Mean: {avg_dist:.3f} km')
    ax5.axvline(med_dist,  color='black',   lw=2, ls=':',
                label=f'Median: {med_dist:.3f} km')

ax5.set_title('Seeker ↔ Leaver Match Distance',
              fontsize=15, fontweight='bold', color='black', pad=10)
ax5.set_xlabel('Distance (km)', fontsize=12)
ax5.set_ylabel('% of All Matches', fontsize=12)
ax5.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
ax5.legend(fontsize=10, framealpha=0.9, edgecolor='black', facecolor='white')
ax5.yaxis.grid(True, ls='--', alpha=0.4, color='grey')
ax5.set_axisbelow(True)

plt.tight_layout()
plt.show()

fig2, axes2 = plt.subplots(1, 3, figsize=(18, 7), facecolor='white')
ax3 = axes2[0]
ax4 = axes2[1]
ax6 = axes2[2]

# ── Build Figure 2: Wait Times ────────────────────────────────────
fig2.suptitle(
    f'Matching Simulation\nDAU={CHART_DAU}  |  Area {ZONE_W_KM}×{ZONE_H_KM} km²  '
    f'|  Max. Distance {RADIUS_KM * 1000:.0f} m\nExpected unassisted park 20 minutes  '
    f'|  Leaver window {L_WAIT} minutes  |  Monte Carlo trials per DAU = {CHART_ITER}',
    fontsize=17, fontweight='bold', color='black', y=1.01
)





plt.figure(fig2.number)

# ── Panel 2.1: Seeker wait time distribution ───────────────────────
if len(matched_sw) >= 0:
    counts, edges = np.histogram(matched_sw, bins=60)
    pct = counts / n * 100
    centers = (edges[:-1] + edges[1:]) / 2
    ax3.bar(centers, pct, width=(edges[1]-edges[0])*0.85,
            color=COLORS['waited'], edgecolor='white', lw=0.4,
            alpha=0.88, label='Matched Seekers')
    ax3.axvline(avg_wait_s, color='#e74c3c', lw=2, ls='--',
                label=f'Mean: {avg_wait_s:.1f} min')
    ax3.axvline(med_wait_s, color='black',   lw=2, ls=':',
                label=f'Median: {med_wait_s:.1f} min')
ax3.set_title('Seeker Wait Time Distribution (App-matched only)',
              fontsize=12, fontweight='bold', color='black', pad=10)
ax3.set_xlabel('Wait Time (minutes)', fontsize=12)
ax3.set_ylabel('% of All Seekers',    fontsize=12)
ax3.set_xlim(-0.05, 30)
ax3.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
ax3.legend(fontsize=10, framealpha=0.9, edgecolor='black', facecolor='white')
ax3.yaxis.grid(True, ls='--', alpha=0.4, color='grey')
ax3.set_axisbelow(True)

# ── Panel 2.2: Leaver wait time distribution ───────────────────────
if len(matched_lw) >= 0:
    counts_l, edges_l = np.histogram(matched_lw, bins=18, range=(-0.05, 3))
    pct_l = counts_l / n * 100
    centers_l = (edges_l[:-1] + edges_l[1:]) / 2
    ax4.bar(centers_l, pct_l, width=(edges_l[1]-edges_l[0])*0.85,
            color=COLORS['unmatch'], edgecolor='white', lw=0.4,
            alpha=0.88, label='Matched Leavers')
    ax4.axvline(avg_wait_l, color='#e74c3c', lw=2, ls='--',
                label=f'Mean: {avg_wait_l:.2f} min')
    ax4.axvline(med_wait_l, color='black',   lw=2, ls=':',
                label=f'Median: {med_wait_l:.2f} min')
ax4.set_title('Leaver Wait Time Distribution',
              fontsize=12, fontweight='bold', color='black', pad=10)
ax4.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
ax4.set_xlabel('Wait Time (minutes)', fontsize=10)
ax4.set_ylabel('% of All Leavers',    fontsize=10)
ax4.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
ax4.legend(fontsize=10, framealpha=0.9, edgecolor='black', facecolor='white')
ax4.yaxis.grid(True, ls='--', alpha=0.4, color='grey')
ax4.set_axisbelow(True)


# ── Panel 2.3: Wait time boxplot — seekers vs leavers ──────────────
plot_data, plot_labels, plot_colors = [], [], []
if len(matched_sw) > 0:
    plot_data.append(matched_sw[matched_sw > 0])
    plot_labels.append('Seekers')
    plot_colors.append(COLORS['waited'])
if len(matched_lw) > 0:
    plot_data.append(matched_lw[matched_lw > 0])
    plot_labels.append('Leavers')
    plot_colors.append(COLORS['unmatch'])

if plot_data:
    bp = ax6.boxplot(
        plot_data, tick_labels=plot_labels, patch_artist=True, widths=0.45,
        medianprops  = dict(color='black',  linewidth=2.5),
        whiskerprops = dict(color='black',  linewidth=1.5),
        capprops     = dict(color='black',  linewidth=1.5),
        flierprops   = dict(marker='o', markersize=2, alpha=0.25,
                            markerfacecolor='grey', markeredgecolor='grey'),
        boxprops     = dict(linewidth=1.5)
    )
    for patch, col in zip(bp['boxes'], plot_colors):
        patch.set_facecolor(col); patch.set_alpha(0.75)
    for i, data in enumerate(plot_data, start=1):
        med = np.median(data); avg = np.mean(data)
        ax6.plot(i, avg, marker='D', color='#e74c3c', markersize=6, zorder=5,
                 label='Mean' if i == 1 else '')

ax6.set_title('Wait Time Comparison\nSeekers vs Leavers (Excluding instant)',
              fontsize=12, fontweight='bold', color='black', pad=10)
ax6.set_yticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24])          # labelled ticks
ax6.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                15, 16, 17, 18, 19, 20, 21, 22, 23, 24], minor=True)      # grid lines
ax6.yaxis.grid(True, which='minor', ls='--', alpha=0.2, color='grey')
ax6.yaxis.grid(True, which='major', ls='--', alpha=0.4, color='grey')
ax6.set_ylim(0, 25)
ax6.set_ylabel('Wait Time (minutes)', fontsize=12)
ax6.set_xlabel('User Type', fontsize=12)
ax6.legend(handles=[
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#e74c3c',
               markersize=7, label='Mean'),
    plt.Line2D([0], [0], color='black', lw=2.5, label='Median'),
], fontsize=10, framealpha=0.9, edgecolor='black', facecolor='white')
ax6.yaxis.grid(True, ls='--', alpha=0.4, color='grey')
ax6.set_axisbelow(True)

for ax in axes2:
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(axis='both', labelsize=11)
    if ax.get_legend():
        for item in ax.get_legend().get_texts():
            item.set_fontsize(10)

plt.tight_layout()
excel_table(
    ['Role','Instant%','Waited%','Unassisted/Unmatched%','MeanWait_min','MedianWait_min'],
    [['Seekers', p_inst_s, p_waited_s, p_lucky, avg_wait_s, med_wait_s],
     ['Leavers', p_inst_l, p_waited_l, p_l_miss, avg_wait_l, med_wait_l]],
    title='Cell 7 — single-DAU outcome breakdown')
excel_table(['Metric','Mean_km','Median_km'],
            [['Match distance', avg_dist, med_dist]],
            title='Cell 7 — match distance')

mpl.rcParams.update(_saved_rc)  # restore global plot style
print(f'Rendered both figures  |  DAU={CHART_DAU}  |  iters={CHART_ITER}')


# ==========================================================================
# SECTION 8
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 8: Seeker Parking Time Breakdown                       ║
# ║  Self-contained — uses res from Cell 8 if already run,      ║
# ║  otherwise re-runs simulation.                               ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Reuse Cell 8 results if available, else re-run ───────────────
try:
    sw, n, matched_sw
except NameError:
    res        = run_sim(CHART_DAU, iterations=CHART_ITER)
    sw         = res['s_waits']
    n          = len(sw)
    matched_sw = sw[sw >= 0]

# ── Generate lucky find times synthetically ───────────────────────
n_lucky   = int(np.sum(sw == -3.0))
N_SAMPLES = max(n_lucky, 10000)

rng         = np.random.default_rng(seed=42)
lucky_times = []
for _ in range(N_SAMPLES):
    for minute in range(1, S_WAIT + 1):
        if rng.random() < LUCKY_FIND_PER_MIN:
            lucky_times.append(minute)
            break
    else:
        lucky_times.append(S_WAIT)

lucky_times  = np.array(lucky_times)

# ── Graph 2 data: app-matched + lucky combined ────────────────────
lucky_prop   = n_lucky / n
n_lucky_draw = int(lucky_prop * n)
lucky_sample = rng.choice(lucky_times, size=n_lucky_draw, replace=True)
combined     = np.concatenate([matched_sw, lucky_sample])

avg_combined = float(np.mean(combined))
med_combined = float(np.median(combined))
avg_lucky    = float(np.mean(lucky_times))
med_lucky    = float(np.median(lucky_times))

# ── Build figure ──────────────────────────────────────────────────
fig9, axes9 = plt.subplots(1, 2, figsize=(14, 7), facecolor='white')
fig9.suptitle('Seeker Parking Time: With App vs Unassisted\n'
              f'DAU={CHART_DAU}  |  {CHART_ITER} Monte Carlo iterations',
              fontsize=14, fontweight='bold', y=1.02)
ax_b = axes9[0]
ax_c = axes9[1]


# ── Panel B: App-matched + lucky combined ────────────────────────
counts_b, edges_b = np.histogram(combined, bins=60, range=(0, 60))
pct_b    = counts_b / len(combined) * 100
centers_b = (edges_b[:-1] + edges_b[1:]) / 2
ax_b.bar(centers_b, pct_b, width=(edges_b[1]-edges_b[0])*0.85,
         color='#2ecc71', edgecolor='white', lw=0.4,
         alpha=0.88, label='App-matched + unassisted seekers')
ax_b.axvline(avg_combined, color='#e74c3c', lw=2, ls='--',
             label=f'Mean: {avg_combined:.1f} min')
ax_b.axvline(med_combined, color='black',   lw=2, ls=':',
             label=f'Median: {med_combined:.1f} min')
ax_b.set_title('Total Parking Time with App\n(app-matched + unassisted finds)',
               fontsize=13, fontweight='bold', color='black', pad=10)
ax_b.set_xlabel('Time to Park (minutes)', fontsize=12)
ax_b.set_ylabel('% of Parked Seekers',    fontsize=12)
ax_b.set_xlim(-0.5, 60)
ax_b.set_ylim(0, 22)
ax_b.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
ax_b.legend(fontsize=10, framealpha=0.9, edgecolor='black', facecolor='white')
ax_b.yaxis.grid(True, ls='--', alpha=0.4, color='grey')
ax_b.set_axisbelow(True)

# ── Panel C: Organic only — geometric distribution, no app ───────
counts_c, edges_c = np.histogram(lucky_times, bins=60, range=(0, 60))
pct_c    = counts_c / len(lucky_times) * 100
centers_c = (edges_c[:-1] + edges_c[1:]) / 2
ax_c.bar(centers_c, pct_c, width=(edges_c[1]-edges_c[0])*0.85,
         color='#f1c40f', edgecolor='white', lw=0.4,
         alpha=0.88, label='No-app seekers')
ax_c.axvline(avg_lucky, color='#e74c3c', lw=2, ls='--',
             label=f'Mean: {avg_lucky:.1f} min')
ax_c.axvline(med_lucky, color='black',   lw=2, ls=':',
             label=f'Median: {med_lucky:.1f} min')
ax_c.set_title('Parking Time Without App\n(unassisted find only, 5%/min)',
               fontsize=13, fontweight='bold', color='black', pad=10)
ax_c.set_xlabel('Time to Park (minutes)', fontsize=12)
ax_c.set_ylabel('% of Seekers',           fontsize=12)
ax_c.set_xlim(-0.5, 60)
ax_c.set_ylim(0, 22)
ax_c.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f%%'))
ax_c.legend(fontsize=10, framealpha=0.9, edgecolor='black', facecolor='white')
ax_c.yaxis.grid(True, ls='--', alpha=0.4, color='grey')
ax_c.set_axisbelow(True)

for ax in axes9:
    ax.title.set_fontsize(14)
    ax.xaxis.label.set_fontsize(12)
    ax.yaxis.label.set_fontsize(12)
    ax.tick_params(axis='both', labelsize=11)
    if ax.get_legend():
        for item in ax.get_legend().get_texts():
            item.set_fontsize(10)

plt.tight_layout()
plt.show()
excel_table(['Scenario','Mean_min','Median_min'],
            [['With app (matched + unassisted)', avg_combined, med_combined],
             ['Without app (unassisted only)',   avg_lucky,    med_lucky]],
            title='Cell 8 — parking time with vs without app')
print(f'Rendered  |  DAU={CHART_DAU}  |  iters={CHART_ITER}')


# ==========================================================================
# SECTION 9
# ==========================================================================
# ============================================================
# CELL 9 — FOUR-SCENARIO ORGANIC SEARCH SWEEP (extended to 8,000 DAU)
# ============================================================




# --- Sweep configuration ---
DAU_SWEEP  = [ 200, 500, 1000, 2000, 4000]
MILESTONES = [ 200, 500, 1000, 2000, 4000]

SC_ITERS  = 10        # light/reviewer value
SC_S_WAIT = 50
SC_L_WAIT = 2
SC_WINDOW = 1020
SC_AREA_W = 2.0
SC_AREA_H = 1.0
SC_RADIUS = 0.3

SCENARIOS = {
    'STRESS TEST (~10 min)':  1/10,
    'BASE (~20 min)':   1/20,
    'MODERATE (~30 min)':     1/30,
    'HARD (~40 min)':  1/40,
}

COLORS     = ['#999999', '#4878CF', '#6ACC65', '#D65F5F']
LINESTYLES = ['--', '-', '-', '-']

# --- Run sweep ---
results    = {name: [] for name in SCENARIOS}
results_ci = {name: [] for name in SCENARIOS}

t0 = time.time()
total_runs = len(SCENARIOS) * len(DAU_SWEEP)
pbar = tqdm(total=total_runs, desc="Scenario sweep")

for sc_name, lucky_rate in SCENARIOS.items():
    for dau in DAU_SWEEP:
        sim = run_sim(
            dau               = dau,
            iterations        = SC_ITERS,
            zone_w            = SC_AREA_W,
            zone_h            = SC_AREA_H,
            radius            = SC_RADIUS,
            s_wait            = SC_S_WAIT,
            l_wait            = SC_L_WAIT,
            window            = SC_WINDOW,
            lucky_rate        = lucky_rate,
            hotspots          = None,
            seeker_early_frac = SEEKER_EARLY_FRAC,
            leaver_early_frac = LEAVER_EARLY_FRAC,
            time_split        = TIME_SPLIT,
        )
        results[sc_name].append(float(np.mean(sim['iter_s_rates'])))
        results_ci[sc_name].append(ci95(sim['iter_s_rates']))
        pbar.update(1)

pbar.close()
elapsed = time.time() - t0
print(f"\nSweep complete in {elapsed/60:.1f} minutes.")

# --- Print results table ---
col_names = list(SCENARIOS.keys())
header = f"{'DAU':>6}  " + "  ".join(f"{n:>24}" for n in col_names)
print("\n" + header)
print("-" * len(header))
for i, dau in enumerate(DAU_SWEEP):
    row = f"{dau:>6}  "
    row += "  ".join(f"{results[sc][i]*100:>21.1f}%   " for sc in col_names)
    marker = " ◄" if dau in MILESTONES else ""
    print(row + marker)

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5))

for (sc_name, _), color, ls in zip(SCENARIOS.items(), COLORS, LINESTYLES):
    y  = [v * 100 for v in results[sc_name]]
    ci = [c * 100 for c in results_ci[sc_name]]
    # Plot the line and points
    ax.plot(DAU_SWEEP, y, marker='o', markersize=4, color=color, linestyle=ls, linewidth=2, label=sc_name, zorder=3)
    ax.fill_between(DAU_SWEEP, [a-b for a,b in zip(y,ci)], [a+b for a,b in zip(y,ci)],
                    color=color, alpha=0.12, zorder=2)  # 95% CI band

    # Annotate the exact value at each milestone
    for x_val, y_val in zip(DAU_SWEEP, y):
        ax.annotate(f"{y_val:.1f}%",
                    xy=(x_val, y_val),
                    xytext=(0, 6),  # 6 points vertical offset so text sits above the line
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, color=color, weight='bold', zorder=4)

for ms in MILESTONES:
    ax.axvline(x=ms, color='black', linewidth=0.8, linestyle=':', zorder=2)

ax.set_xlabel('Daily Active Users (DAU)', fontsize=11)
ax.set_ylabel('App Match Rate (%)', fontsize=11)
ax.set_title('ParkPass Match Rate by DAU — Four Organic Search Scenarios', fontsize=12, pad=14)

# Slightly expanded limits so the new annotations don't get cut off at the edges
ax.set_xlim(-100, 8500)
ax.set_ylim(0, 115)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))

# Display ONLY your specific milestones on the X-axis
ax.set_xticks(MILESTONES)
# Rotate slightly to prevent 200 and 500 from overlapping
plt.xticks(rotation=45)

ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
# Updated the filename to reflect your new 8000 DAU cap
plt.show()

# ── Excel-ready table (error bars = 95% CI) ──
_rows9 = [[sc, DAU_SWEEP[k], results[sc][k]*100, results_ci[sc][k]*100]
          for sc in SCENARIOS for k in range(len(DAU_SWEEP))]
excel_table(['Scenario','DAU','AppMatch%','CI95'], _rows9,
            title='Cell 9 — four-scenario DAU sweep')


# ==========================================================================
# SECTION 10
# ==========================================================================
# ╔══════════════════════════════════════════════════════════════╗
# ║  CELL 10: Sobol Global Sensitivity Analysis                  ║
# ║  Quantifies each parameter's independent contribution        ║
# ║  to output variance in seeker app match rate.                ║
# ╚══════════════════════════════════════════════════════════════╝


# ── Problem definition ────────────────────────────────────────────
problem = {
    'num_vars': 4,
    'names': ['DAU', 'Match Radius', 'Leaver Window', 'Unassisted Find Rate'],
    'bounds':   [
        [200,  2000],   # DAU
        [0.3,  0.5],    # RADIUS (km)
        [0.5,  3.0],    # L_WAIT (min)
        [0.02, 0.20],   # LUCKY_RATE (per min)
    ]
}
# AREA fixed at 2.0 km², WINDOW fixed at 1020 min, S_WAIT fixed at 50 min

# ── Sampling ──────────────────────────────────────────────────────
SOBOL_N    = 8      # light/reviewer value — raise to 256 for final. Saltelli N × (4+2) calls
SOBOL_ITER = 4      # TEST value — raise to ~25 for final to cut MC noise

param_values = sobol_sample.sample(problem, SOBOL_N, calc_second_order=False)
print(f"Sobol sampling: N={SOBOL_N} → {len(param_values)} total simulation calls")
print("Light/reviewer settings — runs in well under a minute. Raise SOBOL_N for reporting grade.\n")

# ── Evaluate model ────────────────────────────────────────────────
Y = np.zeros(len(param_values))
t0 = time.time()
pbar = tqdm(total=len(param_values), desc="Sobol evaluation")

for i, params in enumerate(param_values):
    dau_val, radius_val, lw_val, lucky_val = params
    sim = run_sim(
        dau               = int(round(dau_val)),
        iterations        = SOBOL_ITER,
        zone_w            = ZONE_W_KM,
        zone_h            = ZONE_H_KM,
        radius            = radius_val,
        s_wait            = S_WAIT,
        l_wait            = lw_val,
        window            = WINDOW_MINS,
        lucky_rate        = lucky_val,
        hotspots          = None,
        cluster_frac      = 0.0,
        seeker_early_frac = SEEKER_EARLY_FRAC,
        leaver_early_frac = LEAVER_EARLY_FRAC,
        time_split        = TIME_SPLIT,
    )
    Y[i] = float(np.mean(sim['iter_s_rates']))
    pbar.update(1)

pbar.close()
print(f"\nComplete in {(time.time()-t0)/60:.1f} minutes.")

# ── Analyze ───────────────────────────────────────────────────────
Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)

print("\n" + "="*60)
print("  SOBOL SENSITIVITY INDICES")
print("="*60)
print(f"{'Parameter':>12}  {'S1':>8}  {'S1_conf':>9}  {'ST':>8}  {'ST_conf':>9}")
print("-"*55)
for name, s1, s1c, st, stc in zip(
        problem['names'],
        Si['S1'], Si['S1_conf'],
        Si['ST'], Si['ST_conf']):
    print(f"{name:>12}  {s1:>8.3f}  {s1c:>9.3f}  {st:>8.3f}  {stc:>9.3f}")
print(f"\nSum of S1 = {sum(Si['S1']):.3f}  (near 1.0 = low parameter interactions)")
print("="*60)

# ── Excel-ready table ──
excel_table(['Parameter','S1','S1_conf','ST','ST_conf'],
            [[problem['names'][k], Si['S1'][k], Si['S1_conf'][k], Si['ST'][k], Si['ST_conf'][k]]
             for k in range(len(problem['names']))],
            title='Cell 10 — Sobol sensitivity indices')

# ── Plot ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
names     = problem['names']
s1_vals   = Si['S1']
s1_conf   = Si['S1_conf']
st_vals   = Si['ST']
st_conf   = Si['ST_conf']

colors = ['#4878CF', '#6ACC65', '#D65F5F', '#E8A838']

order = np.argsort(s1_vals)[::-1]
names   = [names[i] for i in order]
s1_vals = s1_vals[order]
s1_conf = s1_conf[order]
st_vals = st_vals[order]
st_conf = st_conf[order]
colors  = [colors[i] for i in order]

x_pos  = np.arange(len(names))
width  = 0.35


bars = ax.bar(x_pos - width/2, s1_vals, width, label='First-order S1',
              color=colors, alpha=0.85)
ax.errorbar(x_pos - width/2, s1_vals, yerr=s1_conf,
            fmt='none', color='black', capsize=4, lw=1.2)
bars2 = ax.bar(x_pos + width/2, st_vals, width, label='Total-order ST',
               color=colors, alpha=0.45, edgecolor=colors, linewidth=1.5)
ax.errorbar(x_pos + width/2, st_vals, yerr=st_conf,
            fmt='none', color='black', capsize=4, lw=1.2)
ax.set_xticks(x_pos)
ax.set_xticklabels(names, fontsize=11)
ax.set_ylabel('Sensitivity Index', fontsize=11)
ax.set_title('Sobol Global Sensitivity Analysis — Match Rate Variance Drivers',
             fontsize=13, fontweight='bold', pad=14)
ax.text(0.5, 1.02,
        'S1 = direct effect   |   ST = total effect including interactions',
        transform=ax.transAxes, ha='center', fontsize=9, color='gray', style='italic')
ax.set_ylim(0, 0.90)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)



plt.tight_layout()
plt.show()


# ==========================================================================
# SECTION 11
# ==========================================================================
# ==============================================================================
# CELL 11 — LOCAL vs GLOBAL STRUCTURAL STRESS TEST (spatiotemporal clustering)
# Fully reproducible: every random draw is seeded per iteration via a master RNG.
# Error bars are 95% confidence intervals.
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

# ── Simulation parameters ─────────────────────────────────────────────────────
CL_DAU        = 1_000
CL_ITERS      = 12            # light/reviewer value
CL_S_WAIT     = 50
CL_L_WAIT     = 2
CL_WINDOW     = 1020
CL_ZONE_W     = 2.0
CL_ZONE_H     = 1.0
CL_RADIUS     = 0.3
CL_LUCKY_RATE = 1 / 20
CL_SEED       = 12345         # master seed → full reproducibility

# ── Hotspot (cluster) parameters ──────────────────────────────────────────────
HS_RADIUS     = 0.3          # cluster radius 0.3 km = 300 m  (area ≈ 0.283 km²)
HS_DURATION   = 240.0        # each hotspot lasts 4 hours (240 min)
HS_MAGNETISM  = 4.0          # hotspots draw 4x their natural volume
N_HOTSPOTS    = 10           # number of simultaneous hotspots

_S_EARLY = globals().get('SEEKER_EARLY_FRAC', 0.70)
_L_EARLY = globals().get('LEAVER_EARLY_FRAC', 0.70)
_T_SPLIT = globals().get('TIME_SPLIT', 510)

SHARED_SEED = 42
_rng_h = np.random.default_rng(SHARED_SEED)
SHARED_HOTSPOTS = []
for _ in range(N_HOTSPOTS):
    _st = float(_rng_h.uniform(0, CL_WINDOW - HS_DURATION))
    SHARED_HOTSPOTS.append({
        'cx': float(_rng_h.uniform(0.1, CL_ZONE_W - 0.1)),
        'cy': float(_rng_h.uniform(0.1, CL_ZONE_H - 0.1)),
        'start_time': _st,
        'end_time': _st + HS_DURATION,
    })
print(f"Shared hotspots initialized: n={N_HOTSPOTS}, seed={SHARED_SEED}")

def _times(rng, n, window, early_frac, split):
    n_early = int(round(n * early_frac)); n_late = n - n_early
    early = rng.uniform(0, split, n_early) if n_early > 0 else np.array([])
    late  = rng.uniform(split, window, n_late) if n_late > 0 else np.array([])
    return np.concatenate([early, late])

ASYM_CONDITIONS = [
    ('100 / 0\n(Leaver Abs.)', 1.0), ('90 / 10', 0.9), ('80 / 20', 0.8),
    ('70 / 30', 0.7), ('60 / 40', 0.6), ('50 / 50\n(Balanced)', 0.5),
    ('40 / 60', 0.4), ('30 / 70', 0.3), ('20 / 80', 0.2),
    ('10 / 90', 0.1), ('0 / 100\n(Seeker Abs.)', 0.0),
]
UNIFORM_LABEL = '0 / 0 (Pure Uniform Background)'

def run_sim_coordinated(dau, iterations, zone_w, zone_h, radius, s_wait, l_wait,
                        window, lucky_rate, hotspots, hotspot_radius, seeker_share,
                        use_filter=True, master_seed=CL_SEED):
    r2 = radius ** 2
    global_s_rates, local_l_rates, local_s_rates = [], [], []
    hs_s_counts, hs_l_counts = [], []
    master_rng = np.random.default_rng(master_seed)

    for _ in range(iterations):
        rng = np.random.default_rng(master_rng.integers(0, 2**31))
        n = int(dau)
        s_t = _times(rng, n, window, _S_EARLY, _T_SPLIT)
        l_t = _times(rng, n, window, _L_EARLY, _T_SPLIT)
        s_x, s_y = rng.uniform(0, zone_w, n), rng.uniform(0, zone_h, n)
        l_x, l_y = rng.uniform(0, zone_w, n), rng.uniform(0, zone_h, n)
        s_evicted = np.zeros(n, dtype=bool); l_evicted = np.zeros(n, dtype=bool)

        if use_filter:
            for h in hotspots:
                s_active = (s_t >= h['start_time']) & (s_t <= h['end_time'])
                s_spat   = (((s_x - h['cx'])**2 + (s_y - h['cy'])**2) <= hotspot_radius**2) & ~s_evicted
                idx_s    = np.where(s_active & s_spat)[0]
                l_active = (l_t >= h['start_time']) & (l_t <= h['end_time'])
                l_spat   = (((l_x - h['cx'])**2 + (l_y - h['cy'])**2) <= hotspot_radius**2) & ~l_evicted
                idx_l    = np.where(l_active & l_spat)[0]
                total_pull = (len(idx_s) + len(idx_l)) * HS_MAGNETISM
                target_s_j = int(round(total_pull * seeker_share))
                target_l_j = int(round(total_pull)) - target_s_j
                if len(idx_s) > target_s_j:
                    ev = rng.choice(idx_s, size=(len(idx_s) - target_s_j), replace=False)
                    s_x[ev], s_y[ev] = rng.uniform(0, zone_w, len(ev)), rng.uniform(0, zone_h, len(ev))
                    s_evicted[ev] = True
                elif len(idx_s) < target_s_j:
                    out = np.where(s_active & ~s_spat & ~s_evicted)[0]
                    pull = min(target_s_j - len(idx_s), len(out))
                    if pull > 0:
                        pi = rng.choice(out, size=pull, replace=False)
                        ang = rng.uniform(0, 2*np.pi, pull); rr = hotspot_radius*np.sqrt(rng.uniform(0,1,pull))
                        s_x[pi] = np.clip(h['cx'] + rr*np.cos(ang), 0, zone_w)
                        s_y[pi] = np.clip(h['cy'] + rr*np.sin(ang), 0, zone_h)
                if len(idx_l) > target_l_j:
                    ev = rng.choice(idx_l, size=(len(idx_l) - target_l_j), replace=False)
                    l_x[ev], l_y[ev] = rng.uniform(0, zone_w, len(ev)), rng.uniform(0, zone_h, len(ev))
                    l_evicted[ev] = True
                elif len(idx_l) < target_l_j:
                    out = np.where(l_active & ~l_spat & ~l_evicted)[0]
                    pull = min(target_l_j - len(idx_l), len(out))
                    if pull > 0:
                        pi = rng.choice(out, size=pull, replace=False)
                        ang = rng.uniform(0, 2*np.pi, pull); rr = hotspot_radius*np.sqrt(rng.uniform(0,1,pull))
                        l_x[pi] = np.clip(h['cx'] + rr*np.cos(ang), 0, zone_w)
                        l_y[pi] = np.clip(h['cy'] + rr*np.sin(ang), 0, zone_h)

        s_in_hs = np.zeros(n, dtype=bool); l_in_hs = np.zeros(n, dtype=bool)
        for h in hotspots:
            s_in_hs |= ((s_t >= h['start_time']) & (s_t <= h['end_time']) &
                        (((s_x - h['cx'])**2 + (s_y - h['cy'])**2) <= hotspot_radius**2) & ~s_evicted)
            l_in_hs |= ((l_t >= h['start_time']) & (l_t <= h['end_time']) &
                        (((l_x - h['cx'])**2 + (l_y - h['cy'])**2) <= hotspot_radius**2) & ~l_evicted)
        hs_s_counts.append(np.sum(s_in_hs) / len(hotspots))
        hs_l_counts.append(np.sum(l_in_hs) / len(hotspots))

        types = np.concatenate([np.zeros(n), np.ones(n)])
        times = np.concatenate([s_t, l_t])
        xs, ys = np.concatenate([s_x, l_x]), np.concatenate([s_y, l_y])
        ids = np.concatenate([np.arange(n), np.arange(n)])
        order = np.argsort(times, kind='stable')
        types, times, xs, ys, ids = (a[order] for a in (types, times, xs, ys, ids))
        sw, lw = np.full(n, -1.0), np.full(n, -1.0)
        act_s, act_l = [], []; prev_t = 0.0
        for i in range(len(times)):
            t, x, y, typ, eid = times[i], xs[i], ys[i], types[i], int(ids[i])
            act_s = [s for s in act_s if t - s[0] <= s_wait]
            act_l = [l for l in act_l if t - l[0] <= l_wait]
            if lucky_rate > 0 and act_s:
                dt = t - prev_t
                if dt > 0:
                    keep = []
                    for s in act_s:
                        if rng.random() < ((1 - lucky_rate) ** dt): keep.append(s)
                        else: sw[int(s[3])] = -3.0
                    act_s = keep
            prev_t = t
            if typ == 0:
                matched = False
                if act_l:
                    sd = (np.array([l[1] for l in act_l]) - x)**2 + (np.array([l[2] for l in act_l]) - y)**2
                    mi = int(np.argmin(sd))
                    if sd[mi] <= r2:
                        sw[eid], lw[int(act_l[mi][3])] = 0.0, t - act_l[mi][0]
                        act_l.pop(mi); matched = True
                if not matched: act_s.append((t, x, y, eid))
            else:
                matched = False
                if act_s:
                    sd = (np.array([s[1] for s in act_s]) - x)**2 + (np.array([s[2] for s in act_s]) - y)**2
                    mi = int(np.argmin(sd))
                    if sd[mi] <= r2:
                        lw[eid], sw[int(act_s[mi][3])] = 0.0, t - act_s[mi][0]
                        act_s.pop(mi); matched = True
                if not matched: act_l.append((t, x, y, eid))

        global_s_rates.append(float(np.sum(sw >= 0) / n))
        if use_filter:
            local_l_rates.append(float(np.sum((lw >= 0) & l_in_hs) / np.sum(l_in_hs)) if np.sum(l_in_hs) > 0 else 0.0)
            local_s_rates.append(float(np.sum((sw >= 0) & s_in_hs) / np.sum(s_in_hs)) if np.sum(s_in_hs) > 0 else 0.0)
        else:
            local_l_rates.append(float(np.sum(lw >= 0) / n))
            local_s_rates.append(float(np.sum(sw >= 0) / n))

    return {
        'match_rate':         float(np.mean(global_s_rates)),
        'match_rate_ci':      ci95(global_s_rates),
        'local_match_rate':   float(np.mean(local_l_rates)),
        'local_match_ci':     ci95(local_l_rates),
        'local_s_match_rate': float(np.mean(local_s_rates)),
        'local_s_match_ci':   ci95(local_s_rates),
        'avg_hs_seekers':     float(np.mean(hs_s_counts)),
        'avg_hs_leavers':     float(np.mean(hs_l_counts)),
    }

# ── Run Sweep ─────────────────────────────────────────────────────────────────
results = {}
pbar = tqdm(total=len(ASYM_CONDITIONS) + 1, desc='Structural Symmetrical Sweep')

for label, seeker_share in ASYM_CONDITIONS:
    results[label] = run_sim_coordinated(
        dau=CL_DAU, iterations=CL_ITERS, zone_w=CL_ZONE_W, zone_h=CL_ZONE_H, radius=CL_RADIUS,
        s_wait=CL_S_WAIT, l_wait=CL_L_WAIT, window=CL_WINDOW, lucky_rate=CL_LUCKY_RATE,
        hotspots=SHARED_HOTSPOTS, hotspot_radius=HS_RADIUS, seeker_share=seeker_share, use_filter=True
    )
    pbar.update(1)

results[UNIFORM_LABEL] = run_sim_coordinated(
    dau=CL_DAU, iterations=CL_ITERS, zone_w=CL_ZONE_W, zone_h=CL_ZONE_H, radius=CL_RADIUS,
    s_wait=CL_S_WAIT, l_wait=CL_L_WAIT, window=CL_WINDOW, lucky_rate=CL_LUCKY_RATE,
    hotspots=SHARED_HOTSPOTS, hotspot_radius=HS_RADIUS, seeker_share=0.50, use_filter=False
)
pbar.update(1)
pbar.close()

# ── 2-Panel Plot Generation ───────────────────────────────────────────────────
labels    = [lbl for lbl, _ in ASYM_CONDITIONS]
x_pos     = np.arange(len(labels))

global_mr  = [results[lbl]['match_rate'] * 100 for lbl, _ in ASYM_CONDITIONS]
global_err = [results[lbl]['match_rate_ci'] * 100 for lbl, _ in ASYM_CONDITIONS]

local_l_mr  = [results[lbl]['local_match_rate'] * 100 for lbl, _ in ASYM_CONDITIONS]
local_l_err = [results[lbl]['local_match_ci'] * 100 for lbl, _ in ASYM_CONDITIONS]

local_s_mr  = [results[lbl]['local_s_match_rate'] * 100 for lbl, _ in ASYM_CONDITIONS]
local_s_err = [results[lbl]['local_s_match_ci'] * 100 for lbl, _ in ASYM_CONDITIONS]

uniform_mr = results[UNIFORM_LABEL]['match_rate'] * 100

# 1. Initialize the clean 2-Panel Canvas
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Match Success Rates Under Clustering Conditions  (error bars = 95% CI)', fontsize=12, weight='bold', y=1.02)

# 2. Reconfigure REGIONS boundaries to match your 11 fine-grained data points
REGIONS = [
    ((-0.5, 4.5), '#ffeeee', 'Demand Excess\n(Leaver Shortage)'),
    (( 4.5, 5.5), '#eeffee', 'Market Equilibrium\n(Optimal Co-location)'),
    (( 5.5, 10.5), '#ddeeff', 'Supply Excess\n(Seeker Shortage)'),
]

# 3. Apply rock-solid formatting to both axes panels
for ax in (ax1, ax2):
    ax.set_xlim(-0.5, 10.5) # Span perfectly across all 11 tick locations (0 to 10)
    ax.set_ylim(0, 100)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([l.replace('\n', ' ') for l in labels], rotation=45, ha='right', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5, zorder=0)

    for (xmin, xmax), color, region_label in REGIONS:
        ax.axvspan(xmin, xmax, color=color, alpha=0.35, zorder=0)
        # Dynamic text placement locked exactly to the center of each span
        ax.text((xmin + xmax) / 2, 96, region_label, ha='center', va='top',
                fontsize=7.5, color='#333333', weight='bold')

# 4. Plot Global Data (Panel 1)
ax1.errorbar(x_pos, global_mr, yerr=global_err, fmt='o-', color='#8e44ad', linewidth=2.0, markersize=6, capsize=3, zorder=3)
ax1.axhline(uniform_mr, color='#444444', linewidth=1.2, linestyle='--', zorder=2, label=f'Baseline ({uniform_mr:.1f}%)')
ax1.set_title("City-wide Match Rates Under Clustering", fontsize=10, pad=15, weight='bold')
ax1.set_ylabel('Match Rate (%)', fontsize=10)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax1.legend(fontsize=8, loc='lower center')

# 5. Plot Cluster Local Data (Panel 2)
ax2.errorbar(x_pos, local_l_mr, yerr=local_l_err, fmt='o-', color='#b83b3b', linewidth=2.2, markersize=7, capsize=4, zorder=3, label='Leaver Match Rates (Supply Clearance)')
ax2.errorbar(x_pos, local_s_mr, yerr=local_s_err, fmt='s-', color='#2980b9', linewidth=2.2, markersize=7, capsize=4, zorder=3, label='Seeker Match Rates')
ax2.axhline(uniform_mr, color='#444444', linewidth=1.2, linestyle='--', zorder=2, label=f'City Baseline ({uniform_mr:.1f}%)')
ax2.set_title("Match Rates Within Clusters", fontsize=10, pad=15, weight='bold')
ax2.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax2.legend(fontsize=8, loc='lower center')

plt.tight_layout()
plt.show()

# ── Appendix Table Generation ─────────────────────────────────────────────────
print("\n" + "="*95)
print(f"{'APPENDIX: HOTSPOT MICRO-MARKET DATA TABLE':^95}")
print("="*95)
print(f"{'Condition':<15} | {'Global Match':<13} | {'Leaver Success':<14} | {'Seeker Success':<14} | {'Avg Seekers/HS':<14} | {'Avg Leavers/HS'}")
print("-" * 95)
for lbl, _ in ASYM_CONDITIONS:
    clean_lbl = lbl.split('\n')[0]
    g_match   = f"{results[lbl]['match_rate']*100:.1f}%"
    l_clear   = f"{results[lbl]['local_match_rate']*100:.1f}%"
    s_clear   = f"{results[lbl]['local_s_match_rate']*100:.1f}%"
    s_count   = f"{results[lbl]['avg_hs_seekers']:.1f}"
    l_count   = f"{results[lbl]['avg_hs_leavers']:.1f}"
    print(f"{clean_lbl:<15} | {g_match:<13} | {l_clear:<14} | {s_clear:<14} | {s_count:<14} | {l_count}")
print("="*95)

# ── Excel-ready table (error bars = 95% CI) ──
excel_table(
    ['Condition','GlobalSeekerMatch%','GlobalCI95','LocalLeaverMatch%','LocalLeaverCI95','LocalSeekerMatch%','LocalSeekerCI95'],
    [[lbl.replace(chr(10),' '),
      results[lbl]['match_rate']*100, results[lbl]['match_rate_ci']*100,
      results[lbl]['local_match_rate']*100, results[lbl]['local_match_ci']*100,
      results[lbl]['local_s_match_rate']*100, results[lbl]['local_s_match_ci']*100]
     for lbl in results],
    title='Cell 11 — clustering stress test')


# ==========================================================================
# SECTION 12
# ==========================================================================
# ============================================================
# CELL 12 — BILATERAL WAIT OPTIMIZATION SWEEP
# Tests L_WAIT from 0.5 to 15 min across DAU milestones
# Measures seeker match rate, leaver match rate, and
# mean wait times to find the optimal broadcast duration
# ============================================================

# --- Configuration ---
OPT_DAU_VALUES  = [300, 500, 1000, 2000]
OPT_L_WAITS     = [0.5, 1, 2, 3, 5, 8, 10, 13, 15]
OPT_ITERS       = 15        # light/reviewer value
OPT_S_WAIT      = 60
OPT_WINDOW      = 1020
OPT_ZONE_W      = 2.0
OPT_ZONE_H      = 1.0
OPT_RADIUS      = 0.4
OPT_LUCKY_RATE  = 1/20    # base scenario

# Theoretical optimal L_WAIT from analytical formula:
# L_WAIT_opt(DAU) = WINDOW / (DAU × Π₁)
PI_1 = np.pi * OPT_RADIUS**2 / (OPT_ZONE_W * OPT_ZONE_H)
theoretical_opt = {dau: OPT_WINDOW / (dau * PI_1) for dau in OPT_DAU_VALUES}

print("Π₁ (spatial reach fraction) = {:.4f}".format(PI_1))
print("\nTheoretical optimal L_WAIT by DAU:")
for dau, lw in theoretical_opt.items():
    print(f"  DAU={dau:>5}: L_WAIT_opt = {lw:.2f} min")

# --- Results structure ---
# results[dau][l_wait] = dict of metrics
sweep_results = {dau: {} for dau in OPT_DAU_VALUES}

total_runs = len(OPT_DAU_VALUES) * len(OPT_L_WAITS)
pbar = tqdm(total=total_runs, desc="L_WAIT optimization sweep")
t0 = time.time()

for dau in OPT_DAU_VALUES:
    for lw in OPT_L_WAITS:
        sim = run_sim(
            dau               = dau,
            iterations        = OPT_ITERS,
            zone_w            = OPT_ZONE_W,
            zone_h            = OPT_ZONE_H,
            radius            = OPT_RADIUS,
            s_wait            = OPT_S_WAIT,
            l_wait            = lw,
            window            = OPT_WINDOW,
            lucky_rate        = OPT_LUCKY_RATE,
            hotspots          = None,
            cluster_frac      = 0.0,
            seeker_early_frac = SEEKER_EARLY_FRAC,
            leaver_early_frac = LEAVER_EARLY_FRAC,
            time_split        = TIME_SPLIT,
        )

        s_waits  = sim['s_waits']
        l_waits  = sim['l_waits']
        n        = dau

        s_match_rate  = float(np.mean(sim['iter_s_rates']))
        l_match_rate  = float(np.mean(sim['iter_l_rates']))
        mean_s_wait   = float(np.mean(s_waits[s_waits >= 0])) if np.any(s_waits >= 0) else np.nan
        mean_l_wait   = float(np.mean(l_waits[l_waits >= 0])) if np.any(l_waits >= 0) else np.nan
        bilateral_sum = (mean_s_wait if not np.isnan(mean_s_wait) else 0) + \
                        (mean_l_wait if not np.isnan(mean_l_wait) else 0)

        sweep_results[dau][lw] = {
            's_match_rate':  s_match_rate,
            's_match_ci':    ci95(sim['iter_s_rates']),
            'l_match_rate':  l_match_rate,
            'l_match_ci':    ci95(sim['iter_l_rates']),
            'mean_s_wait':   mean_s_wait,
            'mean_l_wait':   mean_l_wait,
            'bilateral_sum': bilateral_sum,
        }
        pbar.update(1)

pbar.close()
print(f"\nComplete in {time.time()-t0:.1f} seconds.")

# --- Print table (wait times, not match rates) ---
print(f"\n{'L_WAIT':>8}  ", end='')
for dau in OPT_DAU_VALUES:
    print(f"{'DAU='+str(dau):>24}", end='  ')
print()
print(f"{'(min)':>8}  ", end='')
for dau in OPT_DAU_VALUES:
    print(f"{'E[sw] / E[lw] / sum':>24}", end='  ')
print()
print("-" * (10 + len(OPT_DAU_VALUES) * 26))

for lw in OPT_L_WAITS:
    print(f"{lw:>8.1f}  ", end='')
    for dau in OPT_DAU_VALUES:
        r = sweep_results[dau][lw]
        print(f"{r['mean_s_wait']:>6.1f} / {r['mean_l_wait']:>5.2f} / {r['bilateral_sum']:>5.1f}  ", end='')
    print()

# Theoretical optimum marker
print(f"\nTheoretical L_WAIT* (formula):")
for dau in OPT_DAU_VALUES:
    print(f"  DAU={dau}: {theoretical_opt[dau]:.1f} min")

# --- Plot (wait times) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
colors = ['#D65F5F', '#E8A838', '#4878CF', '#6ACC65', '#9b59b6']

# Panel 1: Mean seeker wait time vs L_WAIT
ax = axes[0]
for dau, color in zip(OPT_DAU_VALUES, colors):
    s_waits = [sweep_results[dau][lw]['mean_s_wait'] for lw in OPT_L_WAITS]
    l_waits = [sweep_results[dau][lw]['mean_l_wait'] for lw in OPT_L_WAITS]
    ax.plot(OPT_L_WAITS, s_waits, color=color, linewidth=2,
            label=f'Seeker DAU={dau}', zorder=3)
    ax.plot(OPT_L_WAITS, l_waits, color=color, linewidth=1.5,
            linestyle='--', zorder=3)
    ax.axvline(theoretical_opt[dau], color=color, linewidth=0.7,
               linestyle=':', alpha=0.7)

ax.set_xlabel('Leaver broadcast duration L_WAIT (min)', fontsize=11)
ax.set_ylabel('Mean wait time (min)', fontsize=11)
ax.set_title('Mean Seeker Wait (solid) and Leaver Wait (dashed) vs L_WAIT\n'
             'Dotted verticals = theoretical optimal L_WAIT per DAU',
             fontsize=10, pad=10)
ax.set_xlim(0, 15)
ax.legend(fontsize=8.5, framealpha=0.9)
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

# Panel 2: Bilateral sum vs L_WAIT
ax2 = axes[1]
for dau, color in zip(OPT_DAU_VALUES, colors):
    bilateral = [sweep_results[dau][lw]['bilateral_sum'] for lw in OPT_L_WAITS]
    ax2.plot(OPT_L_WAITS, bilateral, color=color, linewidth=2,
             label=f'DAU={dau}', zorder=3)
    ax2.axvline(theoretical_opt[dau], color=color, linewidth=0.7,
                linestyle=':', alpha=0.7)

ax2.set_xlabel('Leaver broadcast duration L_WAIT (min)', fontsize=11)
ax2.set_ylabel('Mean bilateral wait — E[sw] + E[lw] (min)', fontsize=11)
ax2.set_title('Total Bilateral Wait vs L_WAIT\n'
              'Minimum = optimal broadcast duration · dotted = theoretical prediction',
              fontsize=10, pad=10)
ax2.set_xlim(0, 15)
ax2.legend(fontsize=8.5, framealpha=0.9)
ax2.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
ax2.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

# ── Excel-ready table (match rates with 95% CI; wait times in minutes) ──
_rows12 = []
for dau in OPT_DAU_VALUES:
    for lw in OPT_L_WAITS:
        r = sweep_results[dau][lw]
        _rows12.append([dau, lw, r['s_match_rate']*100, r['s_match_ci']*100,
                        r['l_match_rate']*100, r['l_match_ci']*100,
                        r['mean_s_wait'], r['mean_l_wait'], r['bilateral_sum']])
excel_table(['DAU','L_WAIT_min','SeekMatch%','Seek_CI95','LeaveMatch%','Leave_CI95',
             'MeanS_wait_min','MeanL_wait_min','BilateralSum_min'], _rows12,
            title='Cell 12 — L_WAIT optimization')


# ==========================================================================
# SECTION 13
# ==========================================================================
# ==============================================================================
# CELL 13 — TEMPORAL ASYMMETRY STRESS TEST
# TIME_SPLIT = 510 = 15:30 = mid-day
# X-axis shows seeker / leaver proportions active before split
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from tqdm import tqdm

TA_DAU    = 1_000
TA_ITERS  = 15        # light/reviewer value
TA_SPLIT  = 510

# Seekers fixed at 70% before split across all scenarios
TA_SEEKER = 0.70

# Leaver scenarios — fracs avoid 0.5 to bypass engine uniform shortcut
TA_SCENARIOS = [
    {'l_frac': 0.70, 'color': '#27ae60'},
    {'l_frac': 0.50, 'color': '#f39c12'},
    {'l_frac': 0.30, 'color': '#e67e22'},
    {'l_frac': 0.10, 'color': '#c0392b'},
]

# ── Run ───────────────────────────────────────────────────────────────────────
pbar = tqdm(total=len(TA_SCENARIOS), desc='Temporal asymmetry scenarios')
for sc in TA_SCENARIOS:
    sim = run_sim(
        dau               = TA_DAU,
        iterations        = TA_ITERS,
        hotspots          = None,
        cluster_frac      = 0.0,
        seeker_early_frac = TA_SEEKER,
        leaver_early_frac = sc['l_frac'],
        time_split        = TA_SPLIT,
    )
    rates       = sim['iter_s_rates']
    sc['mean']  = float(np.mean(rates)) * 100
    sc['std']   = float(np.std(rates))  * 100
    sc['ci']    = ci95(rates) * 100
    pbar.update(1)
pbar.close()

baseline = TA_SCENARIOS[0]['mean']

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Seekers <15:30':>16}  {'Leavers <15:30':>16}  {'Match Rate':>12}  {'95% CI':>8}")
print("-" * 60)
for sc in TA_SCENARIOS:
    print(f"{TA_SEEKER*100:>15.0f}%  {sc['l_frac']*100:>15.0f}%  "
          f"{sc['mean']:>11.1f}%  {sc['ci']:>7.1f}%")
print(f"\n  Aligned reference: {baseline:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
x      = np.arange(len(TA_SCENARIOS))
means  = [sc['mean'] for sc in TA_SCENARIOS]
cis    = [sc['ci']   for sc in TA_SCENARIOS]
colors = [sc['color'] for sc in TA_SCENARIOS]

# X-axis labels: proportions only, no names
x_labels = [
    f"S: {int(TA_SEEKER*100)}%  /  L: {int(sc['l_frac']*100)}%"
    for sc in TA_SCENARIOS
]

fig, ax = plt.subplots(figsize=(9, 5))

bars = ax.bar(x, means, 0.45, color=colors, alpha=0.88, zorder=3)

ax.errorbar(x, means, yerr=cis, fmt='none', color='#2c3e50',
            capsize=5, capthick=1.2, linewidth=1.2, zorder=4)

for bar, mean, ci in zip(bars, means, cis):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci + 1.0,
            f'{mean:.1f}%', ha='center', va='bottom',
            fontsize=9.5, fontweight='bold', color='#2c3e50')

ax.axhline(baseline, color='#27ae60', linewidth=1.4, linestyle='--',
           zorder=2, label=f'Aligned reference ({baseline:.1f}%)')

ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=9.5)
ax.set_xlabel('Proportion of each role active before 15:30  (S = Seekers, L = Leavers)',
              fontsize=9, labelpad=10)
ax.set_ylim(0, 80)
ax.set_ylabel('App Match Rate (%)', fontsize=10)
ax.set_title(
    'Match Rates Under Temporal Role Asymmetry\n'
    f'DAU = {TA_DAU}  ·  Operating day 07:00–24:00  ·  split at 15:30',
    fontsize=10, pad=12
)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
ax.legend(fontsize=8.5, framealpha=0.95, edgecolor='#cccccc', loc='upper right')
ax.grid(axis='y', linestyle='--', linewidth=0.4, alpha=0.5)
ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
plt.show()

# ── Excel-ready table (error bars = 95% CI) ──
excel_table(['Seekers<15:30_%','Leavers<15:30_%','SeekerMatch%','CI95'],
            [[TA_SEEKER*100, sc['l_frac']*100, sc['mean'], sc['ci']] for sc in TA_SCENARIOS],
            title='Cell 13 — temporal asymmetry')


# ==========================================================================
# SECTION 14
# ==========================================================================
# ==============================================================================
# CELL 14 — COMBINED STRESS TEST  (corrected, with 95% confidence intervals)
# ------------------------------------------------------------------------------
# Three adverse conditions imposed simultaneously:
#   1. Population deficit:  1,000 seekers vs 800 leavers  (demand-led adoption)
#   2. Spatial clustering:  10 clusters, 70% seekers / 30% leavers inside them
#   3. Temporal mismatch:   700S / 400L active before 15:30  (70% / 50% split)
#
# DESIGN DECISION — clustering is SPATIAL-ONLY:
#   In this combined test the temporal stressor is already supplied, cleanly
#   and exactly, by the 70/50 pre-15:30 split (condition 3). Giving clusters
#   their own 4-hour windows would place a SECOND temporal mechanism on the
#   same agents, conflicting with that split and blurring which lever drives
#   the result. Clusters are therefore spatial concentrations only, keeping
#   the three stressors orthogonal and individually interpretable. (The
#   4-hour spatiotemporal clusters belong to the composition-sweep cell,
#   where time-localised hotspots ARE the object of study.)
#
# Clustering = SOFT GAUSSIAN: 10 centres placed at random (seed=42). A fixed
# number of each role is assigned to a centre and positioned at
# N(centre, sigma=0.10 km) -> ~95% within 200 m (2-sigma). The remainder are
# uniform. Composition inside clusters is fixed at 70/30 by clustered COUNTS,
# not by a per-role fraction (the previous bug: equal 50% fractions produced a
# 55.6/44.4 mix, never 70/30).
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ── Population & engine parameters ────────────────────────────────────────────
ST_N_SEEKERS   = 1_000
ST_N_LEAVERS   = 800
ST_ITERS       = 20             # light/reviewer value

ST_ZONE_W      = 2.0            # km
ST_ZONE_H      = 1.0            # km
ST_RADIUS      = 0.3            # km  (match radius)
ST_S_WAIT      = 50             # min (seeker patience)
ST_L_WAIT      = 1              # min (leaver patience)
ST_WINDOW      = 1020           # min (07:00-24:00)
ST_LUCKY_RATE  = 1 / 20         # per-min hazard of an unassisted find

# ── Clustering: spatial, fixed 70/30 composition ──────────────────────────────
ST_N_CLUSTERS         = 10
ST_CLUST_SIGMA        = 0.10    # km Gaussian spread (~200 m at 2-sigma)
ST_CLUST_TOTAL_FRAC   = 0.50    # share of ALL agents pulled into clusters
ST_CLUST_SEEKER_SHARE = 0.70    # of clustered agents: 70% seekers / 30% leavers

_n_clust_total = int(round((ST_N_SEEKERS + ST_N_LEAVERS) * ST_CLUST_TOTAL_FRAC))
ST_N_CLUST_S   = min(int(round(_n_clust_total * ST_CLUST_SEEKER_SHARE)), ST_N_SEEKERS)
ST_N_CLUST_L   = min(_n_clust_total - ST_N_CLUST_S, ST_N_LEAVERS)        # 630 S / 270 L

# ── Temporal mismatch: 70% of seekers, 50% of leavers before 15:30 ─────────────
ST_TIME_SPLIT  = 510            # min from 07:00 -> 15:30
ST_S_EARLY     = 700 / 1_000    # 700 seekers before / 300 after
ST_L_EARLY     = 400 / 800      # 400 leavers before / 400 after

_rng_c = np.random.default_rng(42)
ST_CLUSTERS = [
    {'cx': float(_rng_c.uniform(0.2, ST_ZONE_W - 0.2)),
     'cy': float(_rng_c.uniform(0.2, ST_ZONE_H - 0.2))}
    for _ in range(ST_N_CLUSTERS)
]

# ── Helpers ────────────────────────────────────────────────────────────────────
def _st_times(rng, n, window, early_frac, split):
    n_early = int(round(n * early_frac))
    n_late  = n - n_early
    early = rng.uniform(0,     split,  n_early) if n_early > 0 else np.array([])
    late  = rng.uniform(split, window, n_late)  if n_late  > 0 else np.array([])
    return np.sort(np.concatenate([early, late]))

def _st_positions(rng, n, zone_w, zone_h, clusters, n_clust, clust_sigma):
    n_clust = min(int(n_clust), n)
    n_unif  = n - n_clust
    xs, ys  = [], []
    if n_unif > 0:
        xs.append(rng.uniform(0, zone_w, n_unif))
        ys.append(rng.uniform(0, zone_h, n_unif))
    if n_clust > 0:
        centers = np.array([(c['cx'], c['cy']) for c in clusters])
        chosen  = rng.choice(len(centers), size=n_clust)
        cx = np.clip(centers[chosen, 0] + rng.normal(0, clust_sigma, n_clust), 0, zone_w)
        cy = np.clip(centers[chosen, 1] + rng.normal(0, clust_sigma, n_clust), 0, zone_h)
        xs.append(cx); ys.append(cy)
    idx = rng.permutation(n)
    return np.concatenate(xs)[idx], np.concatenate(ys)[idx]

def _ci95(arr):
    arr = np.asarray(arr, dtype=float)
    m   = float(arr.mean())
    se  = float(arr.std(ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return m, 1.96 * se

# ── Engine ────────────────────────────────────────────────────────────────────
def run_stress_test(n_s, n_l, iters, zone_w, zone_h, radius, s_wait, l_wait,
                    window, lucky_rate, clusters, n_clust_s, n_clust_l, clust_sigma,
                    s_early, l_early, time_split):
    r2 = radius ** 2
    s_instant, s_waited, s_failed = [], [], []
    l_instant, l_waited, l_failed = [], [], []

    for it in tqdm(range(iters), desc='Stress test', leave=False):
        rng = np.random.default_rng(it)

        s_t = _st_times(rng, n_s, window, s_early, time_split)
        l_t = _st_times(rng, n_l, window, l_early, time_split)
        s_x, s_y = _st_positions(rng, n_s, zone_w, zone_h, clusters, n_clust_s, clust_sigma)
        l_x, l_y = _st_positions(rng, n_l, zone_w, zone_h, clusters, n_clust_l, clust_sigma)

        sw = np.full(n_s, -1.0)
        lw = np.full(n_l, -1.0)

        types = np.concatenate([np.zeros(n_s), np.ones(n_l)])
        times = np.concatenate([s_t, l_t])
        xs    = np.concatenate([s_x, l_x])
        ys    = np.concatenate([s_y, l_y])
        ids   = np.concatenate([np.arange(n_s), np.arange(n_l)])

        order = np.argsort(times, kind='stable')
        types, times, xs, ys, ids = (a[order] for a in (types, times, xs, ys, ids))

        act_s, act_l = [], []
        prev_t = 0.0

        for i in range(len(times)):
            t, x, y, typ, eid = times[i], xs[i], ys[i], types[i], int(ids[i])
            act_s = [s for s in act_s if t - s[0] <= s_wait]
            act_l = [l for l in act_l if t - l[0] <= l_wait]

            if lucky_rate > 0 and act_s:
                dt = t - prev_t
                if dt > 0:
                    surviving = []
                    for s in act_s:
                        if rng.random() < (1 - lucky_rate) ** dt:
                            surviving.append(s)
                        else:
                            sw[int(s[3])] = -3.0   # unassisted find -> not an app match
                    act_s = surviving
            prev_t = t

            if typ == 0:   # seeker arrives
                matched = False
                if act_l:
                    sd = (np.array([l[1] for l in act_l]) - x)**2 + \
                         (np.array([l[2] for l in act_l]) - y)**2
                    mi = int(np.argmin(sd))
                    if sd[mi] <= r2:
                        sw[eid] = 0.0
                        lw[int(act_l[mi][3])] = t - act_l[mi][0]
                        act_l.pop(mi); matched = True
                if not matched:
                    act_s.append((t, x, y, eid))
            else:          # leaver arrives
                matched = False
                if act_s:
                    sd = (np.array([s[1] for s in act_s]) - x)**2 + \
                         (np.array([s[2] for s in act_s]) - y)**2
                    mi = int(np.argmin(sd))
                    if sd[mi] <= r2:
                        lw[eid] = 0.0
                        sw[int(act_s[mi][3])] = t - act_s[mi][0]
                        act_s.pop(mi); matched = True
                if not matched:
                    act_l.append((t, x, y, eid))

        s_instant.append(np.sum(sw == 0.0) / n_s * 100)
        s_waited.append( np.sum(sw >  0.0) / n_s * 100)
        s_failed.append( np.sum(sw <  0.0) / n_s * 100)
        l_instant.append(np.sum(lw == 0.0) / n_l * 100)
        l_waited.append( np.sum(lw >  0.0) / n_l * 100)
        l_failed.append( np.sum(lw <  0.0) / n_l * 100)

    s_total = (np.array(s_instant) + np.array(s_waited)).tolist()
    l_total = (np.array(l_instant) + np.array(l_waited)).tolist()
    keys = ['s_instant','s_waited','s_failed','l_instant','l_waited','l_failed','s_total','l_total']
    arrs = [s_instant,s_waited,s_failed,l_instant,l_waited,l_failed,s_total,l_total]
    out = {}
    for k, a in zip(keys, arrs):
        m, h = _ci95(a)
        out[k] = m; out[k + '_ci'] = h
    return out

# ── Run ───────────────────────────────────────────────────────────────────────
res = run_stress_test(
    n_s=ST_N_SEEKERS, n_l=ST_N_LEAVERS, iters=ST_ITERS,
    zone_w=ST_ZONE_W, zone_h=ST_ZONE_H, radius=ST_RADIUS,
    s_wait=ST_S_WAIT, l_wait=ST_L_WAIT, window=ST_WINDOW,
    lucky_rate=ST_LUCKY_RATE, clusters=ST_CLUSTERS,
    n_clust_s=ST_N_CLUST_S, n_clust_l=ST_N_CLUST_L, clust_sigma=ST_CLUST_SIGMA,
    s_early=ST_S_EARLY, l_early=ST_L_EARLY, time_split=ST_TIME_SPLIT,
)

# ── Table with 95% CIs ──────────────────────────────────────────────────────────
def _row(label, sm, sc, lm, lc):
    print(f"  {label:<15}{sm:>7.1f}%  +/- {sc:<4.1f}{lm:>11.1f}%  +/- {lc:<4.1f}")

print("\n" + "=" * 60)
print(f"{'COMBINED STRESS TEST  -  OUTCOME BREAKDOWN (95% CI)':^60}")
print("=" * 60)
print(f"  {'':<15}{'Seekers (n=1000)':>16}{'Leavers (n=800)':>21}")
print(f"  {'-' * 54}")
_row('Instant match', res['s_instant'], res['s_instant_ci'], res['l_instant'], res['l_instant_ci'])
_row('Waited match',  res['s_waited'],  res['s_waited_ci'],  res['l_waited'],  res['l_waited_ci'])
_row('Unassist/Unmat',res['s_failed'],  res['s_failed_ci'],  res['l_failed'],  res['l_failed_ci'])
print(f"  {'-' * 54}")
_row('Total matched', res['s_total'],   res['s_total_ci'],   res['l_total'],   res['l_total_ci'])
print("=" * 60)
print(f"  Clustered composition: {ST_N_CLUST_S} seekers / {ST_N_CLUST_L} leavers "
      f"({ST_N_CLUST_S/(ST_N_CLUST_S+ST_N_CLUST_L)*100:.0f}/{ST_N_CLUST_L/(ST_N_CLUST_S+ST_N_CLUST_L)*100:.0f})")
print(f"  Matched pairs ~ {res['s_total']/100*ST_N_SEEKERS:.0f}  "
      f"(seekers {res['s_total']:.1f}% of 1000  ==  leavers {res['l_total']:.1f}% of 800)\n")

# ── Two polished pies ────────────────────────────────────────────────────────────
COLORS = ['#27ae60', '#f39c12', '#e74c3c']   # instant / waited / failed

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.8))
fig.suptitle(
    'Combined Stress Test  —  Match Outcome Breakdown by Role\n'
    f'1,000 Seekers vs 800 Leavers   ·   10 Clusters (σ=100 m, 70/30 inside)   ·   '
    f'{round(ST_N_SEEKERS*ST_S_EARLY)}S / {round(ST_N_LEAVERS*ST_L_EARLY)}L before 15:30',
    fontsize=13, weight='bold', y=1.05
)

for ax, role, im, wm, fm, tm, tci, n, third in [
    (ax1, 'Seekers', res['s_instant'], res['s_waited'], res['s_failed'], res['s_total'], res['s_total_ci'], ST_N_SEEKERS, 'Unassisted'),
    (ax2, 'Leavers', res['l_instant'], res['l_waited'], res['l_failed'], res['l_total'], res['l_total_ci'], ST_N_LEAVERS, 'Unmatched'),
]:
    wedges, texts, autotexts = ax.pie(
        [im, wm, fm],
        labels=['Instant match', 'Waited match', third],
        colors=COLORS,
        autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
        startangle=90,
        wedgeprops=dict(linewidth=2.0, edgecolor='white'),
        textprops=dict(fontsize=12, weight='bold'),
        pctdistance=0.72,
    )
    for at in autotexts:
        at.set_fontsize(12); at.set_fontweight('bold'); at.set_color('black')
    ax.set_title(f'{role}  (n = {n:,})\nmatched {tm:.1f}%  ±{tci:.1f}',
                 fontsize=12, pad=16, weight='bold')

plt.tight_layout()
plt.show()

# ── Excel-ready table (error bars = 95% CI) ──
excel_table(['Role','Instant%','Waited%','Unassisted/Unmatched%','TotalMatched%','Matched_CI95'],
            [['Seekers', res['s_instant'], res['s_waited'], res['s_failed'], res['s_total'], res['s_total_ci']],
             ['Leavers', res['l_instant'], res['l_waited'], res['l_failed'], res['l_total'], res['l_total_ci']]],
            title='Cell 14 — combined stress test')

