# Resonant-triad tools

Three scripts in `rsw_sphere/plotting/` compute and visualize three-wave
(triad) interactions — built on `rsw_sphere.dynamics.dynamic_triads.TRIAD`
and the same Hough-harmonic eigensolver used by the dispersion-relation and
Hough-mode tools (see [`dispersion_relation.md`](dispersion_relation.md),
[`hough_modes.md`](hough_modes.md)). They were written to back paper §2.2
("Resonant Triads"): every number quoted there should come from one of
these, not be hand-typed, so the paper and the code cannot drift apart.

| Script | Shows |
|--------|-------|
| `rsw_sphere/plotting/triad_table.py` | Batch table: per-mode frequency/period/coupling coefficient, per-triad mismatch `δ`, pump mode, energy-conservation residual |
| `rsw_sphere/plotting/triad_dynamics.py` | Energy-integration time series for one triad |
| `rsw_sphere/plotting/triad_efficiency.py` | 2D efficiency-of-energy-transfer sweep over two modes' initial velocities |

All three load their triads from a **registry YAML** rather than
hardcoding mode numbers — by default
[`examples/triads_section_2_2.yaml`](../examples/triads_section_2_2.yaml),
via `rsw_sphere.dynamics.triad_specs.load_triad_specs()`. Pass `--specs
path/to/other.yaml` to point any of the three scripts at a different
registry.

Output convention: every script defaults to printing/showing (`stdout` or
`plt.show()`) when no path is given, and otherwise writes exactly where you
tell it to — by convention `outputs/figures/triads/` (gitignored, mirrors
the existing `outputs/figures/<mode-tag>/` convention for Hough-mode
figures). **None of these scripts ever write into the paper repository**;
copying a finished figure into
`paper-nonlinear-interactions-SWE-sphere/Figures/` is a separate, manual
step.

---

## 1. The registry YAML

```yaml
rossby_pump:
  label: "Rossby-only, pump mode, short period"
  role: "..."          # one-line note on why this triad is included
  h_e: 10000
  mode_a: {m: 3, n: 4, alpha: 3, u: 0.0}     # alpha: 1=EIG, 2=WIG, 3=RH
  mode_b: {m: 1, n: 2, alpha: 3, u: 58.0}    # u = initial zonal velocity, m/s
  mode_c: {m: 4, n: 5, alpha: 3, u: 100.0}
```

Keys are **semantic role labels**, not mode numbers (`rossby_pump`, not
`rh-34`) — these triads get reused as building blocks in later paper
sections (Multiple Triads, Five-Wave), so a name tied to only one of the
three modes wouldn't scale. The four currently registered:

| key | modes (a, b, c) |
|---|---|
| `rossby_near_resonant` | RH(3,10), RH(1,2), RH(4,5) |
| `rossby_pump` | RH(3,4), RH(1,2), RH(4,5) |
| `kelvin_rh_flow` | EG(1,1), RH(3,4), RH(4,5) |
| `gravity_catalyst` | EG(6,9), RH(1,7), EG(7,9) |

To add or edit a triad, edit the YAML — no Python changes needed. The
initial velocities (`u`) in the shipped file are illustrative starting
points, not yet re-verified against the specific efficiency-maximizing
values reported in the dissertation for every triad (see
`paper-nonlinear-interactions-SWE-sphere/.claude/NUMBERS-CHECK-section-2.2.md`
for what has and hasn't been cross-checked so far).

---

## 2. `triad_table.py` — batch properties table

```bash
python rsw_sphere/plotting/triad_table.py outputs/figures/triads/table.tex
python rsw_sphere/plotting/triad_table.py outputs/figures/triads/table.csv --fmt csv
python -m rsw_sphere.plotting.triad_table outputs/figures/triads/table.md --fmt markdown --include-table1
```

or from Python:

```python
from rsw_sphere.dynamics.triad_specs import load_triad_specs
from rsw_sphere.plotting.triad_table import triad_properties, triad_table

specs = load_triad_specs()
props = triad_properties(specs['gravity_catalyst'].modes)   # single triad, as a dict
triad_table(specs, fmt='latex', path='outputs/figures/triads/table.tex')  # full registry
```

`triad_properties` is also the **correctness check**: it returns the
energy-conservation residual `(α^a_bc + α^b_ac − α^c_ab) + δ·S_abc`, which
must be ≈0 for every physically consistent triad. `--include-table1` adds
the four zero-coupling quasi-resonant triads from dissertation Table 2.1.

Console script: `rsw-triad-table` (after `pip install -e .`). Full flags:
`rsw-triad-table --help`.

---

## 3. `triad_dynamics.py` — energy integration

```bash
python rsw_sphere/plotting/triad_dynamics.py outputs/figures/triads/rossby_pump_energy.png --triad rossby_pump --tf 3
```

Console script: `rsw-triad`. `--triad` selects a role key from the
registry (default `gravity_catalyst`); `--tf` is the integration horizon in
days, `--h` the RK33 step (nondimensional time). Full flags: `rsw-triad
--help`.

---

## 4. `triad_efficiency.py` — efficiency sweep

Sweeps the initial zonal velocities of two of the three modes over a grid,
integrates each combination, and records the efficiency of a target mode
(`max − min` of its normalized kinetic energy). This is the expensive one —
a 40×40 grid is ~1.6×10⁶ RK33 steps at the defaults (~10⁷ at the
dissertation's own 100×100 grid) — so the compute (`efficiency_sweep`) is
split from the plotting (`plot_efficiency_map`) and **cached to `.npz`**:
the first run computes and saves, every subsequent run with the same
`--cache` path just reloads.

```bash
python rsw_sphere/plotting/triad_efficiency.py outputs/figures/triads/rossby_pump_efficiency.png \
    --triad rossby_pump --n-grid 20 --cache outputs/figures/triads/rossby_pump_eff.npz
```

Console script: `rsw-triad-efficiency`. `--target {0,1,2}` picks which
mode's efficiency is plotted (a/b/c). Full flags: `rsw-triad-efficiency
--help`.

---

## 5. `examples/make_section22_figures.py` — paper composite figures

Builds the two composite figures used directly in paper §2.2: a 2×2
Rossby-only panel (`rossby_near_resonant`, `rossby_pump`) and a 2×2
combined Rossby-gravity panel (`kelvin_rh_flow`, `gravity_catalyst`), each
row an (efficiency map, energy integration) pair built from `triad_table.py`.
Regenerated as a normal `outputs/`-writing script, then copied into the
paper repo's `Figures/` by hand (the script prints the exact `cp` commands
it needs rather than performing the copy itself).

```bash
# full run, both composite panels, current tuned settings
python examples/make_section22_figures.py

# fast/coarse look while iterating (few minutes instead of tens of minutes)
python examples/make_section22_figures.py --n-grid 6 --tf-scale 0.5

# regenerate just one triad's (efficiency, energy) pair, e.g. after
# editing its velocities in the registry YAML
python examples/make_section22_figures.py --triad gravity_catalyst --clear-cache
```

Per-triad grid resolution and integration horizon live in the script's
`TRIAD_SETTINGS` dict (fallback: `DEFAULT_SETTINGS`, with a printed
warning, for any registry triad not listed there — e.g. a newly-added
one). **Read the module docstring's "CALIBRATION NOTES" section before
trusting a new/edited triad's numbers** — the two most common silent
mistakes are: (1) `tf_days` too short relative to the triad's own
nonlinear exchange period, which under-reports efficiency without
erroring (this is exactly what happened for `kelvin_rh_flow`'s fixed
velocity, caught and documented in
`paper-nonlinear-interactions-SWE-sphere/.claude/NUMBERS-CHECK-section-2.2.md`);
and (2) the `.npz` sweep cache is keyed by triad name only, not by
parameters, so changing a triad's settings requires `--clear-cache` (or
deleting `outputs/figures/triads/<key>_sweep.npz` by hand) or the stale
result is silently reused.

Full flags: `python examples/make_section22_figures.py --help`.

---

## 6. References

See [`dispersion_relation.md`](dispersion_relation.md) and the main
[`README.md`](../README.md) for the underlying eigenvalue problem, and
`.claude/THESIS_FIGURES.md` for how thesis figures/tables map to code.
