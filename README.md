# Perkit — Two-Sided Online Bipartite Matching (TOBM) Simulation

Reproducible code for the feasibility simulation in the BBA thesis
**"Perkit"** (course MNGT 323), by Michalis Kostopoulos.

> **Authorship and AI assistance.** The experiment in this repository,
> the research question, the model design, the parameter choices and their
> justification, the validation strategy, and the interpretation of the
> results, was conceived, directed, and is owned by the author. Parts of
> the code plate (boilerplate, plotting scaffolding, and refactoring) were
> drafted with the assistance of Claude Code (Anthropic). Every generated
> line was specified, reviewed, audited, and tested by the author, who
> takes full responsibility for the final content and its correctness.

This repository contains the stochastic discrete-event Monte Carlo
simulation that underpins the **Feasibility Study** of the thesis. It
reproduces the figures and the numbers reported there; it is **not** the
application itself. Because no empirical data for the proposition exist
yet, the model simulates the user journey up to the matching proposal
using synthetic agents.

---

## What is in this repository

| File | Purpose |
| --- | --- |
| `323_PERKIT_TOBM.py` | Plain-Python script. Runs top to bottom; each figure opens in its own window. Recommended for review — needs no Jupyter installation. |
| `PERKIT_TOBM_JUPYTER.ipynb` | The same model as a Jupyter notebook. Figures render inline under each cell. |
| `README.md` | This file. |

The script and the notebook contain the **same model and the same section
numbering**: `SECTION N` in the script corresponds to cell `N` in the
notebook. Use whichever is more convenient.

---

## Methodology in brief

Street-parking coordination is framed as a **Two-Sided Online Bipartite
Matching (TOBM) problem in 2D space**, the framework used in operational
research on taxi dispatch and food-delivery platforms. Two agent
populations — *seekers* (drivers searching for a spot) and *leavers*
(drivers about to vacate one) — arrive stochastically over a 07:00–24:00
operating day inside a bounded 2 km² zone representing Thessaloniki's city
centre (the irregular pilot area is modelled as a rectangle of equivalent
dimensions for tractability).

The core of the model is a **stochastic discrete-event simulator with a
bilateral exit mechanism**. Agents are processed in arrival order; a
**greedy algorithm** pairs each arrival instantly with its nearest active
counterpart within the match radius, mirroring the launch mechanism of
the product. Agents whose patience window lapses exit the pool: leavers
expire unmatched after the leaver window, while seekers may also resolve
through an *unassisted find* — an independent per-minute chance of
locating a spot without the app, which acts as the competing baseline the
platform must beat. Consistent with traffic reports for Thessaloniki,
70% of arrivals are placed before mid-day (15:30).

Outcomes are estimated by **Monte Carlo iteration**: every configuration
is simulated repeatedly and metrics are reported with 95% confidence
intervals (`1.96 × std / √n`). All stochastic draws — agent placement,
arrival times, unassisted finds, cluster geometry, and the Sobol sample —
run through seeded generators, so identical configurations reproduce
identical results.

The analysis proceeds in four stages, matching the thesis narrative:

1. **Global sensitivity (Sobol).** Variance-based decomposition of the
   match rate over the plausible launch parameter space, separating
   first-order effects (S1) from total effects including interactions
   (ST). This ranks the levers: user density dominates, the unassisted
   find rate comes second, and the spatial/temporal design knobs account
   for the remainder.
2. **DAU sweeps.** Match-rate curves across user-density levels for four
   expected unassisted-parking-time scenarios, establishing the shape of
   the adoption curve and its diminishing returns.
3. **Scenario analysis.** A reference operating point (DAU = 1000,
   UPT = 20 min) is decomposed into outcome shares, match distances, and
   wait-time distributions for both roles.
4. **Stress tests.** Three adverse regimes probe structural resilience:
   spatial asymmetry (spatiotemporal clusters with skewed role
   compositions), temporal asymmetry (leaver availability shifted away
   from seeker demand), and a combined test imposing a supply deficit,
   clustering, and temporal mismatch simultaneously.

## Model parameters

The seven fundamental variables match the thesis table "Core Simulation
Variables". Three are fixed: the operating window (07:00–24:00 = 1020
min, when the vast majority of driving trips in Thessaloniki occur), the
service area (2 km²), and the seeker-window cap (50 min, which clears the
pool of edge cases that would misrepresent density). The remaining four
are swept over the ranges of Appendix B:

| Parameter | Range | Rationale (abridged) |
| --- | --- | --- |
| Daily Active Users (DAU) | 200 – 2000 | Pre-launch marketing targets, with margin for surprises |
| Unassisted Find Rate | 2 – 20 % per minute | Equivalent to 5–50 min of expected unassisted parking time; below 5 min the value proposition is redundant, above 50 min seekers are removed as edge cases |
| Match Radius | 300 – 500 m | Product design, motivated by the literature and primary interview data |
| Leaver Window | 0.5 – 3 min | Product design; longer windows withhold spots from the public for diminishing returns |

---

## How to run

Requirements: Python 3.10+, then

```bash
pip install numpy pandas matplotlib tqdm SALib
```

**Script (recommended for reviewers):**

```bash
python 323_PERKIT_TOBM.py
```

Each figure opens in an interactive window. **Close a window to let the
script continue** to the next analysis. Summary tables print to the
console in tab-separated form (they paste directly into a spreadsheet).
No files are written to disk.

**Notebook:**

Open `PERKIT_TOBM_JUPYTER.ipynb` in Jupyter and choose *Run All*. Figures
appear inline beneath each cell.

---

## Mapping: thesis figures → code

Section/cell numbers are identical across the script and the notebook.

| Thesis figure | Section / Cell | What it produces |
| --- | --- | --- |
| **Fig. 3** — Sobol Analysis Results | **10** | Global sensitivity (first-order S1 and total-order ST) of match rate to DAU, Unassisted Find Rate, Match Radius, Leaver Window |
| **Fig. 4** — Match Rates at different DAU and UPT | **9** | Match-rate curves across DAU for four Unassisted-Parking-Time scenarios |
| **Fig. 5** — Match Success Rates and Distances (DAU = 1000, UPT = 20) | **7**, first figure | Seeker- and leaver-outcome breakdowns and the seeker–leaver match-distance distribution |
| **Fig. 6** — Waiting Times of Successful Matches | **7**, second figure | Seeker and leaver wait-time distributions |
| **Fig. 7** — Match Rates under Clustering | **11** | City-wide and within-cluster match rates across seeker/leaver compositions (Stress Test 1, spatial asymmetry) |
| **Fig. 8** — Match Rates under Temporal Asymmetry | **13** | Match rate as leaver timing diverges from the fixed seeker profile (Stress Test 2) |
| **Fig. 9** — Multivariable / Combined Stress Test | **14** | Seeker and leaver outcome breakdowns under three adverse conditions imposed simultaneously (Stress Test 3) |

**Supporting sections (not numbered figures in the thesis):** Section 0
holds all configuration knobs; 1–2 define the helpers and the matching
engine; 3–5 run the DAU and sensitivity sweeps and the diagnostics that
feed the narrative (including a bilateral sanity check of conservation,
symmetry, and window/radius bounds); 6 and 8 are exploratory plots; 12 is
the bilateral wait (leaver-window) optimization that motivates the
2-minute leaver window.

---

## Reproducing the reporting-grade results

To keep the default run fast and light on any reviewer's machine, the
iteration counts and DAU ranges in **Section 0** (and the per-test values
in later sections) are set to small "reviewer" values. They are
sufficient to confirm that the model runs and that the curves take the
shape reported in the thesis, but they are **noisier** than the
reporting-grade numbers and do not extend to the full DAU axis used in
some figures.

To reproduce the thesis values, raise the following before running:

| Setting | Light default | Thesis / reporting grade |
| --- | --- | --- |
| `ITERATIONS` (main DAU sweep) | 15 | 1000 – 2000 |
| `DAU_AXIS` ceiling | 4000 | up to 16000 |
| `DAU_SWEEP` (Fig. 4) ceiling | 4000 | 8000 |
| `SOBOL_N` (Fig. 3) | 8 | 256 |
| `SOBOL_ITER` (Fig. 3) | 4 | ~25 |
| per-test `*_ITER` values | 10 – 20 | ~25 – 50 |

Higher settings increase runtime substantially (the Sobol analysis in
particular scales as N × (2D + 2) simulation calls).

---

## Notes

- Confidence intervals are reported as 95 % half-widths:
  `1.96 × std / √n`.
- The matching uses a greedy, instantaneous assignment, consistent with
  the thesis description of the launch mechanism. Accumulation-based
  matching (e.g. the Hungarian algorithm) is discussed in the thesis as a
  later-stage option, once density permits.
- The simulation is a **stochastic discrete-event model**, not a
  statistical analysis of observed data.

## License / use

This code accompanies an academic thesis and is provided for review and
reproducibility. Please cite the thesis if you reference or reuse it.
