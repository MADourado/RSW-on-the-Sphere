# Quartet/quintet ("wave set") tools

**Driver interface note (updated 2026-08-26):** `run_sweep.py`'s
diagnostic switch is `sweep.diagnostics: [...]` (values: `p_measure`,
`p_measure_final`, `novelty_period`, `efficiency`,
`low_frequency_energy` for a 2D sweep; `precession` for 1D) -- any `quartet_diagnostics`
mentioned below is stale, from before that switch existed. The wave-set
registry moved to repo-root `wave_sets_default.yaml` (was
`examples/wave_sets_section_3.yaml`) and now also covers plain triads
(the §2.2 triad registry's own 4 headline triads are registered there
too). `run_linear_modes.py`/`run_dynamics.py` read the registry directly
(`--wave-set KEY`), no separate config file. A new driver,
`run_mode_search.py`, finds candidate modes completing a triad with a
given edge/pivot -- see `docs/code_guide.md`'s "Entry points" for the
current driver interface; the per-script sections below (§1-§6) describe
the underlying compute/plot functions each driver calls, corrected for
the `rsw_sphere/utilities/` vs. `rsw_sphere/plotting/` split (pmeasure.py,
periods.py, precession.py moved to `utilities/`, compute-only; the
`plotting/` counterparts are render-only).

Five modules (3 in `rsw_sphere/plotting/`, 2 in `rsw_sphere/utilities/`)
compute and visualize coupled multi-triad interactions (quartets,
quintets, and — as a degenerate case — single triads) — built on
`rsw_sphere.dynamics.wave_sets.WaveSet`, a
generalization of the single-triad `rsw_sphere.dynamics.dynamic_triads.TRIAD`
to an arbitrary set of Hough modes coupled through an arbitrary set of
resonant triads. They were written to back the paper's "Coupled Triads"
section: every number quoted there should come from one of these, not be
hand-typed. `TRIAD` itself is untouched and serves as the independent
reference implementation `WaveSet` is checked against (see §0 below).

| Script | Shows |
|--------|-------|
| `rsw_sphere/plotting/wave_set_table.py` | Batch table: per-mode frequency/period, one coupling-coefficient column per constituent triad, per-triad mismatch `δ` and pump mode |
| `rsw_sphere/plotting/energy_evolution.py` | Energy-integration time series — one mode's own trajectory, or a "triad 1 / triad 2 [/ triad 3] / full wave set" comparison row |
| `rsw_sphere/plotting/period_panel.py` | Power spectrum (dominant periods) of a wave set's kinetic-energy time series |
| `rsw_sphere/utilities/pmeasure.py` | P-measure (%) and filtering error ($\mathcal{F}_2^a$): how much a wave set's extra mode(s) enhance/inhibit one constituent triad's own energy exchange, as a single value or a 2D sweep over two initial velocities. Both diagnostics share one sweep loop (`wave_set_diagnostics_sweep`) so computing both costs one integration pass, not two |
| `rsw_sphere/utilities/precession.py` | Precession-frequency + efficiency (+ individual-mode phase) sweep over one mode's driving velocity, with every swept trajectory cached (§6) |

`run_sweep.py` (repo root, §6.1) is a thin YAML-driven dispatcher over
these scripts' own sweep functions — prefer it over calling a
`wave_set_*.py` script directly when the sweep is a one-off tied to a
specific figure, so the config (not a new script) is what's committed.
Its `sweep.diagnostics: [p_measure]` is the entry point for §5's
P-measure panel ($\mathcal{F}_2^a$/filtering error was retired
2026-08-27, replaced by the single-run-only spectral, share-normalized
`spectral_deviation` -- not yet wired into this 2D sweep engine, see
`rsw_sphere.utilities.periods.spectral_deviation`'s own docstring).

All five load their wave sets from a **registry YAML** rather than
hardcoding mode numbers — by default
[`wave_sets_default.yaml`](../wave_sets_default.yaml), via
`rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs()`. Pass
`--specs path/to/other.yaml` to point any of the five scripts at a
different registry.

**Every function's core signature takes an explicit configuration**
(`modes`, `triads`, `velocities`, ...), never only a registry key — the
`*_from_spec(spec, ...)` variants (`wave_set_energy_evolution_from_spec`,
etc.) are thin convenience wrappers layered on top. This is what makes it
possible to test a one-off quartet/quintet/triad that isn't in the YAML at
all: build `modes`/`triads` by hand (see §1) and call the underlying
function directly, or pass `--modes`/`--triads` to
`rsw_sphere/utilities/check_wave_set_physics.py` (§0). You do not need to edit a
config file to try a new idea.

Output convention: every script defaults to printing/showing (`stdout` or
`plt.show()`) when no path is given, and otherwise writes exactly where you
tell it to — by convention `outputs/figures/wave_sets/` (gitignored).
**None of these scripts ever write into the paper repository**; copying a
finished figure into `paper-nonlinear-interactions-SWE-sphere/Figures/` is
a separate, manual step (`examples_legacy/make_section3_figures.py`, §7, prints
the exact `cp` commands it needs rather than performing the copy itself).

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
    n_grid: 10
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
| `reference_triad` | index into `triads` — the default P-measure denominator for a mode that belongs to it (see `rsw_sphere/utilities/pmeasure.py`'s module docstring for the full per-mode rule) |
| `h_e` | equivalent height, m |
| `settings` | per-wave-set `tf_days`/`h`/`n_grid`, read by `examples_legacy/make_section3_figures.py` — the single source of truth for how long/finely to integrate this particular configuration, not a shared default |

**Velocity caps**: Rossby (RH) mode velocities up to 100 m/s, gravity
(EG/WG) mode velocities up to 50 m/s
(`rsw_sphere.utilities.efficiency.default_velocity_range`).

### Adding a new quartet or quintet

1. Add an entry to `wave_sets_default.yaml` (or your own YAML —
   `--specs` works everywhere). No Python changes needed.
2. Run the physics gate on it (§0) — `python rsw_sphere/utilities/check_wave_set_physics.py --wave-set your_new_key`.
   Fix any hard-check failure before proceeding; read the C6 report even
   though it can't fail the run.
3. Add `tf_days`/`h`/`n_grid` to its `settings` block based on its own
   nonlinear exchange period, not a copy-pasted default — an integration
   horizon shorter than the true exchange period silently under-reports
   `ΔEK`/`P` without erroring (the exact failure mode logged for §2.2's
   triad registry; the same risk applies here).
4. Regenerate its table (§2) and figures (§3-5) in isolation first
   (`--wave-set your_new_key`) before adding it to
   `examples_legacy/make_section3_figures.py`'s `PMEASURE_WAVE_SETS` dict or any
   paper composite.

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
uniform master table) since wave sets have differing mode/triad counts.
Layout matches the dissertation's own `tab: cap41`/`cap42`/`cap43`: one
coefficient column per constituent triad, `-` where a mode isn't in that
triad, a `Pump` column flagging each triad's dominant-coupling mode. The
per-triad energy-conservation residual is computed and warned on
internally but never rendered.

Console script: `rsw-waveset-table`. Full flags: `rsw-waveset-table --help`.

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

**Panels plot raw, unnormalized `|A|^2`** —
not normalized by initial total energy as the single-triad tool does,
because a wave set with 2+ constituent triads does not conserve energy in
general. The grey dotted total-energy line's own drift away from its
initial value is the diagnostic for how much truncation error is present;
`wave_set_energy_evolution`'s return dict also reports it numerically as
`drift`.

The **"triad 1 / triad 2 / wave set" comparison panel** builds each
sub-triad panel from its own 3-mode `WaveSet`, *not* by calling the §2.2
single-triad tool — this keeps every panel in the row in the same
(unnormalized) units. `highlight` (not `target`) picks which mode is drawn
solid vs. dashed: "target held at rest" is a triad-sweep concept with no
wave-set meaning, so the name was deliberately not reused (the identical
naming collision caused a real bug twice in the §2.2 triad tools before
this convention was adopted).

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
complex amplitude — a deliberate correction of the legacy dissertation
scripts' own inconsistency between their code (FFT of amplitude) and their
prose ("FFT of the kinetic energy field"). Peak-finding uses
`scipy.signal.find_peaks`, not a manual float-equality scan.

**Not yet built**: a period-*difference* sweep (this wave set's dominant
period minus its reference triad's, gridded over two swept velocities —
the paper's `fig: power 4`/`domper`/`five`). Only the single-IC power
spectrum above exists so far.

Run `python -m rsw_sphere.utilities.periods` (no args) for a fast,
registry-independent synthetic self-check instead of the registry CLI.

---

## 5. `rsw_sphere/utilities/pmeasure.py` — P-measure and filtering error ($\mathcal{F}_2^a$)

Two per-target diagnostics live here, both comparing one target mode's
own trajectory in the full wave set against its trajectory in one
constituent triad alone (the paper's own $\Delta E^a_{b,c,\ldots}$
notation, §4.1). The P-measure (paper eq. `Pa`) compares kinetic-energy
variation: `P = 100 * (ΔEK_wave_set - ΔEK_triad) / ΔEK_triad`. Positive
means the extra mode(s) enhance that target's energy exchange; negative
means they inhibit it. $\mathcal{F}_2^a$ (paper eq. `F2`) instead compares
amplitude trajectories directly and is unsigned: `F2 = RMS_t(|A_full(t)| -
|A_triad(t)|) / RMS_t(|A_triad(t)|)`.

```bash
# single point (no sweep) -- fast, useful while iterating
python -c "
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.utilities.pmeasure import p_measure
spec = load_wave_set_specs()['quartet_rossby_kelvin']
triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
print(p_measure(spec.modes, triads, spec.velocities, h_e=spec.h_e,
                 reference_triad=spec.reference_triad, tf_days=20, h=0.01))
"

# full 2D sweep + plot (expensive -- see the runtime note below)
python rsw_sphere/plotting/pmeasure_map.py outputs/figures/wave_sets/quartet_rossby_kelvin_pmeasure.png --wave-set quartet_rossby_kelvin --n-grid 10
```

Console script: `rsw-waveset-pmeasure`. Full flags: `rsw-waveset-pmeasure --help`.

Three things live in the function signature, not a plotting special case
(the exact way a scalar-only API caused a real bug in the §2.2 tools):
**per-mode `triad_index`** (different target modes can use different
denominator triads — the default follows the dissertation's own rule, see
the module docstring); **`dEK_triad < MIN_REFERENCE_DEK` (1e-4) → `NaN`**,
guarded explicitly rather than raising (paper eq. `Pa`'s own stated rule —
not just the exact-zero corner, the wider near-zero band where the ratio
is ill-conditioned); and **row-level denominator caching** in the sweep
(free speedup whenever a target's own denominator triad doesn't depend on
one of the two swept modes).

**Runtime**: a 2D sweep costs roughly `n_grid²` full-wave-set integrations
plus (with caching) up to `n_grid` sub-triad integrations per target mode.
At `n_grid=10` this is a few minutes; at the dissertation's own `n_grid=50`
it is tens of minutes to hours depending on the wave set. **Start coarse
(`n_grid=8-10`)** while iterating; a hi-res final pass is a separate,
deliberate step (same deferral pattern as §2.2's triad efficiency sweeps).

**Colormap: diverging** (`RdBu_r` + `TwoSlopeNorm(vcenter=0)`, convention
14), not the sequential `cividis` the single-triad efficiency maps use — a
sequential map cannot show the zero-crossing between inhibition and
enhancement, which is this figure's entire point. `plot_p_measure_map`
returns `n_clipped` (grid points whose `|P|` exceeded the ±100% display
ceiling) so a caption can state it rather than clip silently.

**A negative-P (inhibition) region can be real but too small in magnitude
to see by eye** against a color scale dominated by saturated color
elsewhere in the domain — this exact mistake was made and caught twice in
one session. **Always check `p_measure_sweep`'s returned array directly** (e.g.
`(result['P'] < 0).mean()`) rather than answering "does this figure show
inhibition" by looking at the rendered PNG.

**Computing several diagnostics together: `wave_set_diagnostics_sweep`.**
Calling `p_measure_sweep` and a separate diagnostic-only sweep back to
back would integrate the same full-wave-set and reference-triad
trajectories twice for no reason. `wave_set_diagnostics_sweep(...,
diagnostics=("p_measure", "novelty_period"))` shares **one** grid loop
across whichever diagnostics are requested — the switch `run_sweep.py`'s
own `sweep.diagnostics: [...]` (§6.1) exposes as YAML. `p_measure_sweep`
itself is kept as its own separate, unchanged function rather than
replaced, since its `.npz` cache format is pinned to figures already on
disk (`rsw_sphere.plotting.sweeps`'s own docstring) — `wave_set_diagnostics_sweep`
is for a new combined-panel use case, not a drop-in replacement.

```bash
python -c "
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.utilities.pmeasure import wave_set_diagnostics_sweep
spec = load_wave_set_specs()['quartet_rossby_kelvin']
triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
result = wave_set_diagnostics_sweep(spec.modes, triads, spec.h_e, (2, 3),
                                     {0: 30.0, 1: 30.0}, [2, 3], n_grid=5, tf_days=20, h=0.01)
print(result['P'])
"
```

**`p_measure_final`** (`final_p_measure`, `rsw_sphere/utilities/pmeasure.py`):
plain `p_measure` compares a target against ONE fixed reference triad
(`reference_triad`/`triad_index`) — fine for a private mode, but for a
mode shared across triads that one triad can happen to leave it weakly
excited (off-resonance, or a weak coupling role in that specific triad)
even while a *different* containing triad drives it hard. The resulting
percentage is then dominated by how small that one triad's own `dEK` was,
not by how much the mode's dynamics actually changed (found 2026-08-27,
`quartet_rossby_gravity_influence`: both WG(7,9) and RH(4,5) showed P in
the hundreds-to-thousands of percent against one specific sub-triad,
while the (then time-domain, now-retired) filtering error against that
same reference stayed modest). `p_measure_final`
instead compares against whichever containing triad gives the target its
own LARGEST `dEK` — integrating every containing triad per grid point,
not just one, so it costs more when a target belongs to more than one.
Companion array **`PFinalRefIdx`** (float, same shape) records which
triad (its index into the wave set's own `triads`) was picked at each
point, NaN where every candidate's own `dEK` stayed below
`MIN_REFERENCE_DEK`. `final_p_measure` itself is a small, pure,
broadcastable function (`dEK_full`/each candidate's `dEK` can be a scalar
or a whole sweep-grid ndarray) — `p_measure_combined_for_target`/
`_for_all_targets` wrap it around `run_dynamics.py`'s own results dict
for single-run reporting (mirrors
`rsw_sphere.utilities.novelty_frequency.novelty_combined_for_target`'s
convention); `wave_set_diagnostics_sweep` wraps it for a 2D sweep.
`run_dynamics.py --diagnostics`'s "final diagnostics" table section uses
the single-run wrapper, alongside the equivalent multi-sub-triad-combined
novelty-period result.

**`spectral_deviation`** (`rsw_sphere.utilities.periods.spectral_deviation`,
gated wrapper `_spectral_deviation` in `pmeasure.py`): retired the
time-domain "filtering error"/F2 entirely 2026-08-27 (`plot_filtering_error_map`
and the `"filtering_error"` sweep diagnostic are both gone -- not yet
replaced in `wave_set_diagnostics_sweep`, single-run reporting only for
now). F2 had two real problems: (1) comparing raw `|A_full(t)|` against
`|A_sub(t)|` penalizes a sub-triad simply for having fewer active modes
and therefore a smaller total-energy budget, not for behaving
differently; (2) as a point-by-point time-domain RMS error, it never
converged with `t_f` for a near-resonant sub-triad whose own dynamical
phase drifts on an ~2000-day timescale (`quartet_rossby_gravity_influence`,
`triad_wg39_rh45`: F2 bounced between 0.176 and 0.439 from `t_f`=20d to
160d, never settling). `spectral_deviation` fixes both by (1) comparing
each run's own AMPLITUDE share `q = |A_target|/sqrt(E_total.mean())`
instead of raw amplitude -- linear in amplitude, unlike the energy share
`E/E_total.mean()`, which would quadratically amplify a given relative
difference before it ever reaches the ratio (same fix family as
`wave_set_efficiency`'s own, just without that extra amplification), and
(2) comparing POWER SPECTRA (`periods._power_spectrum`, the exact
primitive `novel_frequency_content` already uses) instead of a raw
time-domain difference -- a spectral comparison integrates coherently
over the whole window and isn't dominated by wherever the two
trajectories happen to be out of phase at the arbitrary `t_f` cutoff.
A third fix, added after the first version of this metric still showed
huge values for a weakly-excited reference (`WG(7,9)` vs
`triad_rh34_rh45`: 155.4, vs. 0.9998 after the fix): the ratio's own
denominator normalizes by whichever of {full, sub} has the LARGER
spectral power, not the reference alone -- otherwise a reference that
leaves the target only weakly excited (almost no spectral power to
normalize by) inflates the ratio by orders of magnitude, the exact same
"weak fixed reference" pathology `final_p_measure` fixes for P/efficiency,
just showing up in the spectral metric's own denominator instead of a
"which triad to pick" choice.

`pairwise_target_diagnostics` (single named reference triad) reports
`spectral_deviation` for any target, private or shared -- unlike the
"final"/combined versions of every other diagnostic here, this pairwise
form is exactly the "filtering error" of one specific, named mode
removal, valid whether the target is private or shared. `p_measure_combined_for_target`'s
own `spectral_deviation` (against the SAME winning reference
`final_p_measure` already picked, so every "final" column agrees on
which sub-triad a shared mode is being read against) does NOT carry that
framing -- for a shared mode there is no single canonical "the" removal
experiment (there are two, one per containing triad), so it instead
reports how much the target deviates from whichever single triad best
explains it, not the effect of removing one specific known mode.

---

## 6. `precession.py` — precession-frequency + efficiency sweep

Sweeps one mode's driving velocity for a registered wave set, reporting
every constituent triad's dynamical-phase (`Φ`, `rsw_sphere.dynamics.dynamical_phase`)
libration statistics, plus (optionally) one target mode's own
time-averaged-total-energy efficiency and/or one or more modes' own raw
`individual_phase` slope (Raphaldini et al. 2022's individual-mode
phase-reversal diagnostic, their Section III.A/Fig. 3). Generalizes what
were previously two bespoke scripts (`quartet_precession_sweep.py`,
`precession_sweep_figure.py`, both retired 2026-08-26 -- see
`examples_legacy/README.md`) into this repo's own registry-driven
`wave_set_<topic>.py` pattern.

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
re-plot never re-derives them from 45+ trajectory loads --
`run_sweep.py` (§6.1) does not currently pass this (each point still
reads its own cached trajectory, just doesn't re-derive the summary
arrays cheaply); wire it up if a sweep's own re-plot turnaround matters.

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

Reads a `RunConfig` (registry key or inline `modes:`, `rsw_sphere.dynamics.run_config`)
with a `sweep:` block naming 1-2 modes to sweep and which
diagnostic(s) to compute, and produces a cached `.npz` + a figure — a
dispatcher over the sweep functions above (`registry.sweep_2d` for 2D,
`precession_frequency_efficiency` for 1D), not a reimplementation of
their math. Replaces the need for a new bespoke `examples/*.py` script
every time someone wants a new sweep combination. `--wave-set KEY` reads
the `sweep`/`tf_days`/`h`/`plot`/`output`/`target_mode`/`plot_triad`
keys straight from that wave set's own `wave_sets_default.yaml` entry
(e.g. `quartet_rh_preference`'s own sweep, the migrated replacement for
`examples_legacy/legacy/precession_sweep_figure.py`'s own
`precession_sweep_figures.yaml` entry, verified to reproduce that
script's output pixel-for-pixel) — no separate config file needed;
`--config path.yaml` (a standalone YAML) is only for an ad-hoc sweep not
worth registering.

```bash
python run_sweep.py --wave-set quartet_rh_preference
```

**2D sweeps** (`sweep.axes` has 2 entries) dispatch to
`rsw_sphere.utilities.registry.sweep_2d`, wrapping §5's
`wave_set_diagnostics_sweep` (`p_measure`/`p_measure_final`/
`novelty_period`) and §-less `functional_diagnostics_sweep`
(`efficiency`/`low_frequency_energy`) behind one shared diagnostic list
-- one combined panel (one row per requested diagnostic, one column per
swept mode, which doubles as the target). If `sweep.axes` is omitted
entirely, it auto-derives from the wave set's own "private" modes
(`WaveSetSpec.shared_and_private_modes()` -- a mode common to every
constituent triad is "shared" and held fixed at its own registered
velocity; a mode private to exactly one triad is a swept axis/target),
so an ordinary quartet needs no `axes` at all:

```yaml
# inside wave_sets_default.yaml's own quartet_rossby_kelvin entry:
sweep:
  diagnostics: [p_measure]
  n_grid: 10
output: outputs/figures/wave_sets/quartet_rossby_kelvin_diagnostics.png
```

```bash
python run_sweep.py --wave-set quartet_rossby_kelvin
```

Only the 2-private-mode case (an ordinary quartet) auto-derives --
otherwise (a quintet's 3 private modes, or any other pair) pass
`sweep.axes` explicitly, e.g.
`axes: [{mode: c, min: 0.0, max: 100.0}, {mode: d, min: 0.0, max: 50.0}]`.
This panel is a draft for analysis, not copied into the paper repository
(see the module docstring convention above).

**1D sweeps** (`sweep.axes` has 1 entry) only support
`sweep.diagnostics: [precession]` -- `precession_frequency_efficiency`,
already natively 1D. `target_mode`/`plot_triad` (top-level keys, sibling
to `sweep:`) pick which mode's own efficiency to report and which
constituent triad to plot, respectively -- see `quartet_rh_preference`'s
own `wave_sets_default.yaml` entry for a worked example. See
`run_sweep.py`'s own module docstring for the full config schema.

### 6.2. `rsw_sphere/dynamics/trajectory_cache.py` — raw trajectory caching

`run_and_cache(ws, A0, t_f, h, velocities=None, output_root="outputs/trajectories", label=None)`
caches the raw `Y(t)` ODE solution itself (not just a derived summary) —
the one piece with no analogue elsewhere in this repo before this was
added: every other cache in this codebase (this file's own `p_measure_sweep`/
`efficiency_sweep`, `rsw_sphere.plotting.sweeps`) stores summary/sweep
arrays only, so any new diagnostic on an already-run trajectory used to
mean re-integrating from scratch. Cache path:
`outputs/trajectories/<topology>/<label>_<hash8>.npz`; hash covers
everything that changes the numerical result (modes, triads, `gamma`,
`N`, `deg`, `A0`, `t_f`, `h`). `topology` is auto-derived from the wave
set's own mode count (`triads`/`quartets`/`quintets`/`n<k>modes`,
`topology_folder(ws.n_modes)`), and `label` defaults to
`ic_label(ws.modes, velocities)` (every mode's own family/wavenumber +
initial velocity, canonically sorted) plus `t_f`/`h` — **not** a
caller-supplied key, so the same physical configuration lands in the
same cache entry regardless of which script built it (2026-08-25 —
previously grouped by a caller-chosen `wave_set_key` string, which a
few scripts had to hand-coordinate a shared tag for just to get this
same reuse). Pass an explicit `label` instead of `velocities` when a run
is driven by something other than per-mode velocities (e.g. a single
overall amplitude scale — see `examples/raphaldini2022_compare/precession_comparison.py`'s
own comment on this). Any script computing a `WaveSet` trajectory that
might be revisited should go through this rather than calling `RK44`
directly.

---

## 7. `examples_legacy/make_section3_figures.py` — paper composite figures

Builds, per registered wave set: a comparison panel (§3), a period panel
(§4), and — for the wave sets listed in `PMEASURE_WAVE_SETS` — a P-measure
sweep (§5). Mirrors `examples_legacy/make_section22_figures.py`'s structure
(`*_SETTINGS` dict, CLI overrides, `outputs/`-only writes, prints the `cp`
commands rather than copying itself).

```bash
# everything, current tuned settings
python examples_legacy/make_section3_figures.py

# fast/coarse look while iterating
python examples_legacy/make_section3_figures.py --n-grid 5 --tf-scale 0.5

# just one wave set
python examples_legacy/make_section3_figures.py --wave-set quartet_rossby_kelvin

# skip the (expensive) P-measure sweeps entirely
python examples_legacy/make_section3_figures.py --skip-pmeasure
```

Full flags: `python examples_legacy/make_section3_figures.py --help`.

The Quartet A/B precession-frequency figure (JFM-template.tex
`fig: precession_frequency`) has no analogous composite script: it is two
independent PNGs (`run_sweep.py --wave-set quartet_rh_preference`
for Quartet A, `examples/raphaldini2022_compare/precession_comparison.py`
for Quartet B, i.e. the registered `quartet_rh_borrowed_topology`) combined
by LaTeX `subfigure`, not a Python-composited image -- each source script
already writes its own complete, publication-ready PNG, so no separate
assembly step exists or is needed (see `docs/code_guide.md`'s note on why
there is no `postproc/` folder).

---

## 8. References

See [`triads.md`](triads.md) for the single-triad tools this generalizes,
and [`dispersion_relation.md`](dispersion_relation.md) and the main
[`README.md`](../README.md) for the underlying eigenvalue problem. See
`rsw_sphere/dynamics/wave_sets.py`'s own module docstring for the design
rationale (the permutation between `TRIAD`'s and `WaveSet`'s conventions,
the `fat`/gauge argument, why quartets don't conserve energy), and
`docs/code_guide.md` for why the `postproc/` folder an earlier reorg plan
called for was dropped after execution.
