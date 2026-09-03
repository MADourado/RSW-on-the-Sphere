# Quartet/quintet ("wave set") tools

Five modules (3 in `rsw_sphere/plotting/`, 2 in `rsw_sphere/utilities/`)
compute and visualize coupled multi-triad interactions (quartets,
quintets, and — as a degenerate case — single triads), built on
`rsw_sphere.dynamics.wave_sets.WaveSet`, a generalization of the
single-triad `rsw_sphere.dynamics.dynamic_triads.TRIAD` to an arbitrary
set of Hough modes coupled through an arbitrary set of resonant triads.
They back the paper's "Coupled Triads" section: every number quoted there
should come from one of these, not be hand-typed. `TRIAD` itself is
untouched and serves as the independent reference implementation
`WaveSet` is checked against (§0).

| Module | Shows |
|--------|-------|
| `rsw_sphere/plotting/wave_set_table.py` | Batch table: per-mode frequency/period, one coupling-coefficient column per constituent triad, per-triad mismatch `δ` and pump mode |
| `rsw_sphere/plotting/energy_evolution.py` | Energy-integration time series — one mode's own trajectory, or a "triad 1 / triad 2 [/ triad 3] / full wave set" comparison row |
| `rsw_sphere/plotting/period_panel.py` | Power spectrum (dominant periods) of a wave set's kinetic-energy time series |
| `rsw_sphere/utilities/pmeasure.py` | Efficiency variation (%): how much a wave set's extra mode(s) enhance/inhibit one constituent triad's own energy exchange |
| `rsw_sphere/utilities/precession.py` | Precession-frequency + efficiency (+ individual-mode phase) sweep over one mode's driving velocity |

Two root drivers sit on top of these: `run_dynamics.py` (one wave set,
one integration, full diagnostics report) and `run_sweep.py` (§6.1, a
YAML-driven dispatcher sweeping 1-2 modes' velocities over the same
diagnostics engine). Prefer them over calling a module directly when the
run is tied to a specific figure, so the registry entry — not a new
script — is what's committed.

All five modules load their wave sets from a **registry YAML** rather
than hardcoding mode numbers — by default
[`wave_sets_default.yaml`](../wave_sets_default.yaml), via
`rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs()`. Pass
`--specs path/to/other.yaml` to point any of them at a different
registry.

**Every function's core signature takes an explicit configuration**
(`modes`, `triads`, `velocities`, ...), never only a registry key — the
`*_from_spec(spec, ...)` variants (`wave_set_energy_evolution_from_spec`,
etc.) are thin convenience wrappers layered on top. This is what makes it
possible to test a one-off quartet/quintet/triad that isn't in the YAML at
all: build `modes`/`triads` by hand (§1) and call the underlying function
directly, or pass `--modes`/`--triads` to
`rsw_sphere/utilities/check_wave_set_physics.py` (§0). You do not need to
edit a config file to try a new idea.

Output convention: every script defaults to printing/showing (`stdout` or
`plt.show()`) when no path is given, and otherwise writes exactly where
you tell it to — by convention `outputs/figures/wave_sets/` (gitignored).
**None of these scripts ever write into the paper repository**; copying a
finished figure into `paper-nonlinear-interactions-SWE-sphere/Figures/` is
a separate, manual step — see §7 for the dedicated per-figure scripts
that cover §4.3/§5's composite wave-set figures.

---

## 0. Before trusting a new configuration: the physics gate

`rsw_sphere/utilities/check_wave_set_physics.py` is a **hard gate** — run it on any
new or edited wave set (registered or ad-hoc) before trusting a figure
generated from it:

```bash
# every registered wave set, all checks
python rsw_sphere/utilities/check_wave_set_physics.py

# one registered wave set, a subset of checks
python rsw_sphere/utilities/check_wave_set_physics.py --wave-set quartet_rossby_kelvin --check C1,C2,C4,C5,C8

# an AD-HOC configuration not in any registry -- e.g. a new quintet idea
python rsw_sphere/utilities/check_wave_set_physics.py \
    --modes "4,5,3" "3,4,3" "1,2,3" "1,1,1" "7,9,1" \
    --triads "0,1,2" "0,1,3" "4,0,1"
```

`--modes` is a list of `"m,n,alpha"` triples (alpha: 1=EIG/EG, 2=WIG/WG,
3=RH). `--triads` is a list of `"i_sum,i_p,i_q"` index triples into
`--modes`, **sum mode first** — the paper's own convention (`eq :4sys1`):
the sum mode's zonal wavenumber must equal its two members' sum
(`m_sum == m_p + m_q`), checked at load time with an error naming exactly
which triad and role failed if not.

Checks C1-C5 and C8 are hard (a failure aborts with a nonzero exit code);
C6 (energy-drift validity) and C7 are advisory. **C6 in particular is
worth reading even though it never fails the run**: it reports
`drift / ΔEK` per mode, and any mode with a large ratio means a
`P`-measure or `ΔEK`-derived number for that mode is measuring truncation
error, not physics — this happens for low-energy "catalyst" modes more
often than you'd expect, and it will not be flagged for you anywhere else.

See the module's own docstring for the full C1-C8 table and what each one
proves (not just what it computes).

---

## 1. The registry YAML

```yaml
quartet_rossby_kelvin:
  label: "RH(4,5)+RH(3,4)+RH(1,2) with the Kelvin mode EG(1,1)"
  display_label: "Quartet C"
  role: "..."          # one-line note on why this wave set is included
  h_e: 10000
  modes:
    a: {m: 4, n: 5, alpha: 3, u: 30.0}   # alpha: 1=EIG, 2=WIG, 3=RH
    b: {m: 3, n: 4, alpha: 3, u: 30.0}   # u = initial zonal velocity, m/s
    c: {m: 1, n: 2, alpha: 3, u: 30.0}
    d: {m: 1, n: 1, alpha: 1, u: 0.0}
  triads:
    - {sum: a, members: [b, c], display_label: "Triad 1 (RH-only)"}
    - {sum: a, members: [b, d], display_label: "Triad 2 (with EG(1,1))"}
  reference_triad: 0
  settings:
    tf_days: 20
    h: 0.01
```

Mode keys are **symbolic letters** (`a`, `b`, `c`, ...), not positional
indices, matching the paper's own lettering (`eq :4sys1`: `a` is always a
constituent triad's "sum" mode). Each entry in `triads` names its sum mode
and two members by those same letters — the YAML reads the same way the
paper's equations do, and the load-time validator (§0) checks the
wavenumber constraint for you.

| field | meaning |
|---|---|
| `modes` | one entry per Hough mode, each with `m`, `n`, `alpha`, `u` (initial zonal velocity, m/s) |
| `triads` | list of constituent triads; each names `sum` + `members` by mode letter, plus an optional `display_label` and documentary `triad_key` (pointing at a §2.2 triad registry role-key, if this constituent triad happens to also be independently registered there — not resolved automatically) |
| `reference_triad` | index into `triads` — the default efficiency-variation denominator for a mode that belongs to it (see `rsw_sphere/utilities/pmeasure.py`'s module docstring for the full per-mode rule) |
| `h_e` | equivalent height, m |
| `settings` | per-wave-set `tf_days`/`h`, read by every driver and the `examples/figures/paper_figureNNN_*.py` scripts (§7) — the single source of truth for how long/finely to integrate this particular configuration, not a shared default. (Not the Hough-eigenproblem truncation order `N` -- every driver hardcodes `N=10`, independent of any registry entry.) |
| `sweep` | optional; read by `run_sweep.py` (§6.1) -- which mode(s) to sweep and which diagnostics to compute |
| `alternative_modes` | optional; read by `run_sweep_sets.py` -- candidate-mode substitution (below) |

**Velocity caps**: Rossby (RH) mode velocities up to 100 m/s, gravity
(EG/WG) mode velocities up to 50 m/s
(`rsw_sphere.utilities.efficiency.default_velocity_range`).

### `alternative_modes:` — candidate-mode substitution

An optional top-level block per entry, one sub-block per swapped mode
key, read directly by `run_sweep_sets.py` (a raw YAML passthrough like
`sweep:`/`plot:` above — not a `WaveSetSpec` field, since only that one
driver consumes it):

```yaml
quartet_rossby_kelvin:
  ...
  alternative_modes:
    d:                              # mode key being substituted
      target_mode: d                # optional, default: the slot itself
      diagnostics: [efficiency_var] # subset of {efficiency_var, spectral_dev_var,
                                     # novelty_period, efficiency,
                                     # low_frequency_energy}
      candidate_velocity: 30.0      # optional: drive every candidate at
                                     # this velocity instead of the slot's
                                     # own registered one
      tf_days: 20                   # optional override (default: the
      h: 0.01                       # wave set's own settings block)
      candidates:                   # m/n/alpha triples -- paste straight
        - {m: 1, n: 1, alpha: 1}    # from run_mode_search.py's own output
        - {m: 1, n: 3, alpha: 1}    # (drop role/required_m/coup_*/pump)
```

`python run_sweep_sets.py --wave-set quartet_rossby_kelvin --slot d` runs
one static (never swept) `run_dynamics()` + `compute_diagnostics_report()`
point per candidate, at the base wave set's own registered velocities
(unless `candidate_velocity` overrides the slot's own). Every requested
diagnostic, plus each candidate's own static `delta` (frequency
mismatch), `coeff` (coupling coefficient), isolated-triad `efficiency`,
and dimensional `total_energy_joules` (the full wave set's own
time-averaged total energy, `rsw_sphere.physics.total_energy_joules` --
always computed, not gated behind `diagnostics:`), becomes both a CSV
column (`outputs/sweep_sets/<wave_set_key>/alternatives_to_mode_<slot>/candidates.csv`)
and its own point-wise figure (one point per candidate, no connecting
line, since candidates have no natural ordering unlike a swept continuous
velocity). `target_mode_override` (function parameter, or `--target-mode`
on the CLI) evaluates the same candidate list against a different target
than the registry's own `target_mode`, without editing the registry twice
-- e.g. a figure wanting both the swept slot's own diagnostics and
another mode's, from one candidate list.

Candidate-mode screening is entirely registry-driven this way, matching
every other driver: no separate config file, and no ad-hoc "family" YAML
of its own. `quartet_rh_preference`'s own `alternative_modes.d` (RH(3,n),
n=4..16) backs `examples/tables/paper_table03_rh_partner_family.py`.

### Adding a new quartet or quintet

1. Add an entry to `wave_sets_default.yaml` (or your own YAML —
   `--specs` works everywhere). No Python changes needed.
2. Run the physics gate on it (§0) — `python rsw_sphere/utilities/check_wave_set_physics.py --wave-set your_new_key`.
   Fix any hard-check failure before proceeding; read the C6 report even
   though it can't fail the run.
3. Add `tf_days`/`h` to its `settings` block based on its own nonlinear
   exchange period, not a copy-pasted default — an integration horizon
   shorter than the true exchange period silently under-reports
   `ΔEK`/`P` without erroring.
4. Regenerate its table (§2) and figures (§3-5) in isolation first
   (`--wave-set your_new_key`) before wiring it into a dedicated
   `examples/figures/paper_figureNNN_*.py` script (§7) or any paper
   composite.

### Testing a triad, quartet, or quintet that isn't in any registry at all

Every plotting function's core signature is `(modes, triads, velocities,
...)` — you never have to touch a YAML file to try an idea:

```python
from rsw_sphere.plotting.energy_evolution import wave_set_energy_evolution

modes = [(4, 5, 3), (3, 4, 3), (1, 2, 3), (1, 1, 1), (7, 9, 1)]   # a, b, c, d, e
triads = [(0, 1, 2), (0, 1, 3), (4, 0, 1)]   # sum-mode-first index triples
velocities = [30.0, 30.0, 30.0, 0.0, 0.0]

result = wave_set_energy_evolution(modes, triads, velocities, h_e=10000,
                                    tf_days=10, h=0.01,
                                    path="outputs/figures/wave_sets/scratch.png")
print(result['drift'], result['dEK'])
```

Run the physics gate on the same `modes`/`triads` first (§0's `--modes`/
`--triads` flags take the identical format) — a configuration that fails
C1/C2/C5 is not a valid resonant-triad network and no figure built from it
means anything.

---

## 2. `wave_set_table.py` — batch properties table

```bash
python rsw_sphere/plotting/wave_set_table.py outputs/figures/wave_sets/table.tex
python rsw_sphere/plotting/wave_set_table.py outputs/figures/wave_sets/table.csv --fmt csv
python -m rsw_sphere.plotting.wave_set_table outputs/figures/wave_sets/table.md --fmt markdown --wave-set quartet_rossby_kelvin
```

or from Python:

```python
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_properties, wave_set_table

specs = load_wave_set_specs()
props = wave_set_properties(specs['quartet_rossby_kelvin'])   # single wave set, as a dict
wave_set_table(specs, fmt='latex', path='outputs/figures/wave_sets/table.tex')  # full registry
```

One `\begin{table}...\end{table}` block per wave set (unlike §2.2's single
uniform master table, since wave sets have differing mode/triad counts):
one coefficient column per constituent triad, `-` where a mode isn't in
that triad, a `Pump` column flagging each triad's dominant-coupling mode.
The per-triad energy-conservation residual is computed and warned on
internally but never rendered.

Console script: `rsw-waveset-table`. Full flags: `rsw-waveset-table --help`.

`wave_set_master_table(specs, fmt='latex', path=None)` (same module) is
the alternative for wave sets that all share the same triad count (every
quartet here: 2 each) -- ONE combined `\begin{table}` across several wave
sets, `\midrule`-separated with a bold group-name + `$\delta$` header row
per wave set, mirroring `tab: master`'s own style for the single triads.
Backs `examples/tables/paper_table02_quartet_master.py` (Quartets A-E,
`tab: quartet_master`).

---

## 3. `energy_evolution.py` — energy integration

```bash
# one wave set's own energy trajectory
python rsw_sphere/plotting/energy_evolution.py outputs/figures/wave_sets/quartet_rossby_kelvin_energy.png --wave-set quartet_rossby_kelvin

# "triad 1 / triad 2 [/ triad 3] / wave set" comparison row
python rsw_sphere/plotting/energy_evolution.py outputs/figures/wave_sets/quartet_rossby_kelvin_panel.png --wave-set quartet_rossby_kelvin --panel
```

Console script: `rsw-waveset`. `--wave-set` selects a role key from the
registry; `--tf`/`--h` override the registry's own `settings` (default:
`tf_days=10`/`h=0.01` if the wave set has no `settings` block at all).
Full flags: `rsw-waveset --help`.

**Panels plot raw, unnormalized `|A|^2`** — not normalized by initial
total energy as the single-triad tool does, because a wave set with 2+
constituent triads does not conserve energy in general. The grey dotted
total-energy line's own drift away from its initial value is the
diagnostic for how much truncation error is present;
`wave_set_energy_evolution`'s return dict also reports it numerically as
`drift`.

The **"triad 1 / triad 2 / wave set" comparison panel** builds each
sub-triad panel from its own 3-mode `WaveSet`, *not* by calling the §2.2
single-triad tool — this keeps every panel in the row in the same
(unnormalized) units. `highlight` (not `target`) picks which mode is drawn
solid vs. dashed: "target held at rest" is a triad-sweep concept with no
wave-set meaning, so the name is deliberately not reused.

Each mode is drawn in a persistent color regardless of which wave
set/panel it appears in — `rsw_sphere.plotting.style.MODE_COLORS`. If you
add a mode not yet in that dict, it falls back to grey rather than
erroring; **extend the dict, don't fork it**.

---

## 4. `period_panel.py` — dominant-period analysis

```bash
python rsw_sphere/plotting/period_panel.py outputs/figures/wave_sets/quartet_rossby_kelvin_periods.png --wave-set quartet_rossby_kelvin
```

Console script: `rsw-waveset-periods`. Same `--wave-set`/`--tf`/`--h`
flags as `rsw-waveset`. Prints, per mode, `period_global` (the single
largest spectral peak) and `period_local_max` (the largest-*period* local
maximum, if distinct from the global one — a real, lower-power
periodicity the global max alone hides), flagging `[HORIZON-LIMITED]` for
any period within 10% of the integration horizon (unresolvable, not
necessarily a genuine long periodicity — read `--max-period`'s docstring
in `dominant_periods()` before citing a period near this boundary).

Input to the FFT is the kinetic-energy series `|A_j(t)|^2`, not the raw
complex amplitude. Peak-finding uses `scipy.signal.find_peaks`, not a
manual float-equality scan.

**Not yet built**: a period-*difference* sweep (this wave set's dominant
period minus its reference triad's, gridded over two swept velocities).
Only the single-IC power spectrum above exists so far.

Run `python -m rsw_sphere.utilities.periods` (no args) for a fast,
registry-independent synthetic self-check instead of the registry CLI.

---

## 5. `rsw_sphere/utilities/pmeasure.py` — efficiency-variation diagnostics

Compares one target mode's own trajectory in the full wave set against
its trajectory in one constituent triad alone (the paper's own
`ΔE^a_{b,c,...}` notation, §4.1). Two related diagnostics live here and
in the sibling `periods.py` module:

- **Efficiency variation** (`Δ𝓔`, paper eq. `effvar`) -- raw
  energy-variation percent change: `Δ𝓔 = 100 * (ΔEK_wave_set -
  ΔEK_triad) / ΔEK_triad`. Positive means the extra mode(s) enhance that
  target's energy exchange; negative means they inhibit it. What an
  earlier version of this codebase computed and reported separately as
  "P-measure" (raw swing) and "efficiency variation" (each side's own raw
  swing divided by *its own* configuration's mean total energy first,
  `rsw_sphere.utilities.efficiency.wave_set_efficiency`) are the same
  quantity (2026-09-03): naively normalizing each side by its own budget
  is wrong in general, since the two configurations' total-energy budgets
  can differ for reasons unrelated to the target's actual response (e.g.
  a fixed driving velocity needs more energy at a higher wavenumber), so
  that ratio could drift purely from the denominator. Normalizing both
  sides by the SAME (reference) energy budget instead cancels that
  denominator out of the ratio entirely, leaving exactly the raw-swing
  formula above -- "P-measure" was already computing the right thing; the
  name is retired, "efficiency variation" is the one diagnostic reported
  in the paper and by every driver.
- **Spectral deviation** (`𝓓₂ᵃ`, `rsw_sphere.utilities.periods.spectral_deviation`)
  -- compares the two trajectories' own power spectra (of amplitude
  share `q = |A|/sqrt(E_total.mean())`, not raw amplitude or energy)
  rather than a raw time-domain difference, normalized by whichever of
  the two has the larger own spectral power. This avoids two failure
  modes a naive comparison has: penalizing a reference simply for having
  fewer active modes and a smaller energy budget, and never converging
  with `t_f` for a near-resonant sub-triad whose dynamical phase drifts
  on a timescale far longer than any `t_f` used in this paper.

**Computing several diagnostics together**: `run_sweep.py`'s own engine
(`rsw_sphere.dynamics.diagnostics_report.compute_diagnostics_report`)
computes every diagnostic from **one** `run_dynamics()` call per grid
point, whichever `sweep.diagnostics: [...]` (§6.1) requests.

```bash
python -c "
from rsw_sphere.dynamics.run_config import RunConfig, SweepAxis, SweepConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from run_sweep import run_sweep
spec = load_wave_set_specs()['quartet_rossby_kelvin']
sweep = SweepConfig(axes=(SweepAxis(mode='c', min=0.0, max=30.0),
                           SweepAxis(mode='d', min=0.0, max=30.0)),
                     n_grid=5, diagnostics=('efficiency_var', 'novel_period'))
config = RunConfig.from_wave_set(spec, tf_days=20, h=0.01, sweep=sweep)
result = run_sweep(config)
print(result['efficiency_var']['series'])
"
```

Two things live in `_default_triad_index_for_mode`, not a plotting
special case: **per-mode reference triad** (different target modes can
use different denominator triads — the default follows the dissertation's
own rule, see the module docstring), and **`dEK_triad < MIN_REFERENCE_DEK`
(1e-4) → `NaN`**, guarded explicitly rather than raising (paper eq. `Pa`'s
own stated rule — not just the exact-zero corner, the wider near-zero
band where the ratio is ill-conditioned).

**`efficiency_variation_final`**: comparing a target against ONE fixed
reference triad (`pairwise_target_diagnostics`) is fine for a private
mode, but for a mode shared across triads that one triad can happen to
leave it weakly excited even while a *different* containing triad drives
it hard, making the percentage dominated by how small that one triad's
own `dEK` was rather than by how much the mode's dynamics actually
changed. `efficiency_variation_final` instead compares against whichever
containing triad gives the target its own LARGEST `dEK`, integrating
every containing triad per grid point. It is a small, pure, broadcastable
function (`dEK_full`/each candidate's `dEK` can be a scalar or a whole
sweep-grid ndarray) — `efficiency_variation_combined_for_target`/
`_for_all_targets` wrap it around `run_dynamics.py`'s own results dict for
single-run reporting; `compute_diagnostics_report`'s own `final` list
wraps it for a sweep (`run_sweep.py`'s `efficiency_var` diagnostic).

`pairwise_target_diagnostics` (single named reference triad) reports
`spectral_deviation`/`efficiency_var`/`novelty_period` for any target,
private or shared — unlike the "final"/combined versions above, this
pairwise form is exactly the effect of one specific, named mode removal,
valid whether the target is private or shared.

**Standalone legacy engine** (`p_measure`/`p_measure_sweep`, bottom of
`pmeasure.py`, plus `rsw_sphere/plotting/pmeasure_map.py` and its
`rsw-waveset-pmeasure` console script): an older, standalone
modes/triads/velocities API predating `WaveSetSpec`/`run_dynamics()`, with
its own single-fixed-reference-triad 2D sweep + diverging-colormap
plotting, kept only for
`examples/figures/legacy/paper_figure011_quintet_gravity_star_pmeasure.py`
-- not part of the current pipeline (no current `paper_figure*.py` script
calls it). Computes the same formula as `pairwise_target_diagnostics`
above, just against always the one fixed reference triad, never the
"final"/largest-of-every-containing-triad selection.

---

## 6. `precession.py` — precession-frequency + efficiency sweep

Sweeps one mode's driving velocity for a registered wave set, reporting
every constituent triad's dynamical-phase (`Φ`,
`rsw_sphere.dynamics.dynamical_phase`) libration statistics, plus
(optionally) one target mode's own time-averaged-total-energy efficiency
and/or one or more modes' own raw `individual_phase` slope (Raphaldini et
al. 2022's individual-mode phase-reversal diagnostic, their Section
III.A/Fig. 3).

Every swept trajectory is cached via
`rsw_sphere.dynamics.trajectory_cache.run_and_cache` under
`<output_root>/trajectories/<topology>/<ic_label>_tf<days>_h<h>_<hash8>.npz`
(§6.2) — grouped by topology (`triads`/`quartets`/`quintets`), named by
initial condition so the *same physical configuration* from a different
script/sweep lands in the same cache entry, not by `wave_set_key`.
Re-running the same sweep a second time reads from cache rather than
re-integrating. `precession_frequency_efficiency`'s own optional
`sweep_cache_path` additionally caches the sweep's summary arrays
(frequency/efficiency/individual-phase-slope vs. swept velocity) so a
re-plot never re-derives them from many trajectory loads. `run_sweep.py`
(§6.1) does not call this function -- its own `dynamical_phase`/
`efficiency` diagnostics go through the unified `compute_diagnostics_report`
engine instead; `individual_phase`/`low_freq_period_cutoff_days` (the
Raphaldini-reproduction extras) remain available only through this
module directly, e.g. `examples/raphaldini2022_compare/precession_comparison.py`.

```bash
python rsw_sphere/plotting/precession_plot.py outputs/figures/wave_sets/quartet_rh_preference_precession.png --wave-set quartet_rh_preference --sweep-mode d --target c
```

Console script: `rsw-waveset-precession`. Full flags:
`rsw-waveset-precession --help`.

Also exposes `plot_phase_trace` — one or more `phi(t)` vs. time traces
(works for both the combined `Φ` and an individual raw mode phase
`phi_j~`), for a low-vs-high driving-amplitude comparison (Raphaldini et
al. 2022's own Fig. 3 layout).

### 6.1. `run_sweep.py` (repo root) — general sweep driver

Reads a `RunConfig` (registry key, `rsw_sphere.dynamics.run_config`)
with a `sweep:` block naming 1-2 modes to sweep and which diagnostic(s)
to compute. One `run_dynamics()` call per grid point
(`rsw_sphere.dynamics.diagnostics_report.compute_diagnostics_report`),
parallelized across grid points, feeds every diagnostic -- 1D and 2D
share the exact same per-point compute; only the rendering differs (line
plot vs. heatmap). `--wave-set KEY` reads the `sweep`/`tf_days`/`h`/`plot`
keys straight from that wave set's own `wave_sets_default.yaml` entry --
no separate config file needed; a wave set not yet worth adding to the
default registry can be swept via `--specs path.yaml` instead (same
registry schema, e.g. `examples/wave_sets_custom.yaml`).

```bash
python run_sweep.py --wave-set quartet_rossby_kelvin
```

**Diagnostic vocabulary** (`sweep.diagnostics: [...]`, same names for 1D
and 2D):

- `efficiency`/`dominant_freq`/`dominant_period`/`low_frequency_energy`
  -- 1D: one line per (mode, unit). 2D: one heatmap per mode (`full`
  unit's own value only -- a heatmap grid doesn't have a line plot's
  spare room for a per-unit breakdown).
- `dynamical_phase` -- 1D: one line per triad. 2D: one heatmap per triad.
- `efficiency_var` (alias `energy_var`), `spectral_dev_var`,
  `novel_freq`/`novelty_freq`, `novel_period`/`novelty_period` -- one
  line/heatmap per mode, combining every containing sub-triad the same
  way `run_dynamics.py --diagnostics`'s own "final diagnostics" table
  does. Undefined for a plain triad with no sub-triad to compare against
  -- warned and skipped there.
- `total_energy` -- one line/heatmap per UNIT (`full` plus every
  sub-triad, not per mode): that unit's own time-averaged total energy,
  in Joules (`rsw_sphere.physics.total_energy_joules`). Useful context
  alongside `efficiency_var` when a sweep axis also changes the wave
  set's own energy budget -- `efficiency_var` itself no longer needs it
  as a correction (both sides of its ratio share one reference energy
  budget), but it's still informative to see whether the budget moved.

`diagnostics: [all]` expands to every name above. Output:
`outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png` + a
matching long/tidy `.csv`, one pair per requested diagnostic -- there is
no single "the" output for a sweep to override, every diagnostic manages
its own path, and each write prints a `figure ->`/`table ->` line.
`--no-plot-per-point` skips every per-grid-point file output (each grid
point otherwise also writes its own full `run_dynamics.py
--diagnostics`-equivalent bundle under its own
`outputs/dynamics/<key>/<run_label>/`).

If `sweep.axes` is omitted entirely, it auto-derives from the wave set's
own "private" modes (`WaveSetSpec.shared_and_private_modes()` -- a mode
common to every constituent triad is "shared" and held fixed at its own
registered velocity; a mode private to exactly one triad is a swept
axis/target), so an ordinary quartet needs no `axes` at all:

```yaml
# inside wave_sets_default.yaml's own quartet_rossby_kelvin entry:
sweep:
  diagnostics: [efficiency_var]
  n_grid: 10
```

Only the 2-private-mode case (an ordinary quartet) auto-derives --
otherwise (a quintet's 3 private modes, or any other pair) pass
`sweep.axes` explicitly, e.g.
`axes: [{mode: c, min: 0.0, max: 100.0}, {mode: d, min: 0.0, max: 50.0}]`.
These figures are a draft for analysis, not copied into the paper
repository (see the output convention above).

**A composite figure combining two or more diagnostics** (e.g. the
paper's own precession-frequency-and-efficiency figure) is a separate,
paper-specific script's job, not something `run_sweep.py` special-cases
-- see `examples/figures/paper_figure007_quartet_a_precession.py`, which
calls `run_sweep()` once for `[dynamical_phase, efficiency]` (one shared
computation) and composes its own dual-axis figure from the two results,
reusing `rsw_sphere.plotting.precession_plot.plot_dual_axis_frequency_efficiency`
unchanged.

See `run_sweep.py`'s own module docstring for the full config schema.

### 6.2. `rsw_sphere/dynamics/trajectory_cache.py` — raw trajectory caching

`run_and_cache(ws, A0, t_f, h, velocities=None, output_root="outputs/trajectories", label=None)`
caches the raw `Y(t)` ODE solution itself, not just a derived summary --
every other cache in this codebase (the legacy `p_measure_sweep`/
`efficiency_sweep`, `rsw_sphere.plotting.sweeps`) stores summary/sweep
arrays only, so any new
diagnostic on an already-run trajectory would otherwise mean
re-integrating from scratch. Cache path:
`outputs/trajectories/<topology>/<label>_<hash8>.npz`; the hash covers
everything that changes the numerical result (modes, triads, `gamma`,
`N`, `deg`, `A0`, `t_f`, `h`). `topology` is auto-derived from the wave
set's own mode count (`triads`/`quartets`/`quintets`/`n<k>modes`,
`topology_folder(ws.n_modes)`), and `label` defaults to
`ic_label(ws.modes, velocities)` (every mode's own family/wavenumber +
initial velocity, canonically sorted) plus `t_f`/`h` -- **not** a
caller-supplied key, so the same physical configuration lands in the same
cache entry regardless of which script built it. Pass an explicit `label`
instead of `velocities` when a run is driven by something other than
per-mode velocities (e.g. a single overall amplitude scale -- see
`examples/raphaldini2022_compare/precession_comparison.py`'s own comment
on this). Any script computing a `WaveSet` trajectory that might be
revisited should go through this rather than calling `RK44` directly.

---

## 7. §4.3/§5 paper composite figures

Each composite figure for the gravity-Rossby quartets (§4.3) and the star
quintet (§5) has its own dedicated `examples/figures/paper_figureNNN_*.py`
script. Script numbers are kept in sync with each script's own actual
compiled figure number (`JFM-template.aux`'s `\newlabel{fig: ...}`, not
creation order) -- a script whose own figure is removed from the paper
entirely moves to `examples/figures/legacy/` instead of being renumbered
into a slot that no longer means anything (see `examples/README.md`).

| Script | Figure | Wave set |
|---|---|---|
| `paper_figure008_quartet_rossby_kelvin_panel.py` | `fig: cap4ex1` | `quartet_rossby_kelvin` (3x2: Triad 1/Triad 2/full-quartet evolution on top, one novelty spectrum per Rossby mode on bottom) |
| `paper_figure009_quartet_rossby_kelvin_gravity_wavenumber.py` | `fig: rossby_kelvin_wavenumber` | `quartet_rossby_kelvin` (single panel, twin y-axes: candidate's own coupling coefficient (log, left) + RH(1,2)'s own efficiency variation (right), over the registered `alternative_modes.d` candidate list EG(1,n)/WG(1,n); `run_sweep_sets.run_sweep_sets`'s `target_mode_override` run twice over the same candidates -- redesigned 2026-09-03, dropped the old 2x2 efficiency_var/p_measure grid once both were recognized as the same quantity) |
| `paper_figure010_quartet_rossby_gravity_influence_panel.py` | `fig: quartet_rossby_gravity_influence_panel` (Quartet D) | `quartet_rossby_gravity_influence` (2x2: both constituent-triad evolutions + full-quartet evolution + RH(3,4) novelty spectrum against Triad 1, its only containing triad) |
| `paper_figure011_quartet_rossby_gravity_influence_efficiency.py` | `fig: quartet_rossby_gravity_influence_efficiency` (Quartet D efficiency sweep) | `quartet_rossby_gravity_influence` (1D efficiency_var sweep over WG(3,9), Rossby modes only + 2D heatmap for RH(4,5)) |
| `paper_figure012_quartet_rossby_gravity_influence_high_panel.py` | `fig: quartet_rossby_gravity_influence_high_panel` (Quartet E) | `quartet_rossby_gravity_influence_high` (same 2x2 layout as Quartet D's panel; EG(7,9) plays sum role for Triad 1 and member role for Triad 2, closed by EG(11,11)) |
| `paper_figure014_quintet_gravity_star_panel.py` | `fig: quintetpanel` (Quintet A) | `quintet_gravity_star` (3x2: one evolution panel per constituent triad on top, full-quintet evolution + 2 novelty spectra (RH(4,5)/RH(1,2)) on bottom -- same layout family as `paper_figure008`'s, extended by one triad/column) |
| `paper_figure015_quintet_gravity_influence_star_panel.py` | `fig: quintetpanel_b` (Quintet B) | `quintet_gravity_influence_star` (same 3x2 layout as `paper_figure014`'s, spectra for RH(3,4)/RH(4,5)) |

Each is runnable standalone and writes under
`outputs/figures/wave_sets/<key>/`; copying the finished PNG into
`paper-nonlinear-interactions-SWE-sphere/Figures/` under the filename the
paper's own `\includegraphics` expects is a separate, manual step (each
script's own module docstring names the target file).

```bash
python examples/figures/paper_figure008_quartet_rossby_kelvin_panel.py
python examples/figures/paper_figure009_quartet_rossby_kelvin_gravity_wavenumber.py
python examples/figures/paper_figure010_quartet_rossby_gravity_influence_panel.py
python examples/figures/paper_figure011_quartet_rossby_gravity_influence_efficiency.py
python examples/figures/paper_figure012_quartet_rossby_gravity_influence_high_panel.py
python examples/figures/paper_figure014_quintet_gravity_star_panel.py
python examples/figures/paper_figure015_quintet_gravity_influence_star_panel.py
```

The Quartet A/B precession-frequency figure (JFM-template.tex
`fig: precession_frequency`) has no analogous composite script: it is two
independent PNGs (`run_sweep.py --wave-set quartet_rh_preference` for
Quartet A, `examples/raphaldini2022_compare/precession_comparison.py` for
Quartet B, i.e. the registered `quartet_rh_borrowed_topology`) combined by
LaTeX `subfigure`, not a Python-composited image -- each source script
already writes its own complete, publication-ready PNG, so no separate
assembly step exists or is needed.

Retired scripts and superseded tables/figures live in
`examples/{figures,tables}/legacy/` -- see each retired script's own
module docstring and `examples/README.md` for what superseded it, rather
than repeating that history here.

---

## 8. References

See [`triads.md`](triads.md) for the single-triad tools this generalizes,
and [`dispersion_relation.md`](dispersion_relation.md) and the main
[`README.md`](../README.md) for the underlying eigenvalue problem. See
`rsw_sphere/dynamics/wave_sets.py`'s own module docstring for the design
rationale (the permutation between `TRIAD`'s and `WaveSet`'s conventions,
the `fat`/gauge argument, why quartets don't conserve energy).
