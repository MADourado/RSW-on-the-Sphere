# Quartet/quintet ("wave set") tools

Four scripts in `rsw_sphere/plotting/` compute and visualize coupled
multi-triad interactions (quartets, quintets, and — as a degenerate case —
single triads) — built on `rsw_sphere.dynamics.wave_sets.WaveSet`, a
generalization of the single-triad `rsw_sphere.dynamics.dynamic_triads.TRIAD`
to an arbitrary set of Hough modes coupled through an arbitrary set of
resonant triads. They were written to back the paper's "Coupled Triads"
section: every number quoted there should come from one of these, not be
hand-typed. `TRIAD` itself is untouched and serves as the independent
reference implementation `WaveSet` is checked against (see §0 below).

| Script | Shows |
|--------|-------|
| `rsw_sphere/plotting/wave_set_table.py` | Batch table: per-mode frequency/period, one coupling-coefficient column per constituent triad, per-triad mismatch `δ` and pump mode |
| `rsw_sphere/plotting/wave_set_dynamics.py` | Energy-integration time series — one mode's own trajectory, or a "triad 1 / triad 2 [/ triad 3] / full wave set" comparison row |
| `rsw_sphere/plotting/wave_set_periods.py` | Power spectrum (dominant periods) of a wave set's kinetic-energy time series |
| `rsw_sphere/plotting/wave_set_pmeasure.py` | P-measure (%): how much a wave set's extra mode(s) enhance/inhibit one constituent triad's own energy exchange, as a single value or a 2D sweep over two initial velocities |

All four load their wave sets from a **registry YAML** rather than
hardcoding mode numbers — by default
[`examples/wave_sets_section_3.yaml`](../examples/wave_sets_section_3.yaml),
via `rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs()`. Pass
`--specs path/to/other.yaml` to point any of the four scripts at a
different registry.

**Every function's core signature takes an explicit configuration**
(`modes`, `triads`, `velocities`, ...), never only a registry key — the
`*_from_spec(spec, ...)` variants (`wave_set_energy_evolution_from_spec`,
etc.) are thin convenience wrappers layered on top. This is what makes it
possible to test a one-off quartet/quintet/triad that isn't in the YAML at
all: build `modes`/`triads` by hand (see §1) and call the underlying
function directly, or pass `--modes`/`--triads` to
`examples/check_wave_set_physics.py` (§0). You do not need to edit a
config file to try a new idea.

Output convention: every script defaults to printing/showing (`stdout` or
`plt.show()`) when no path is given, and otherwise writes exactly where you
tell it to — by convention `outputs/figures/wave_sets/` (gitignored).
**None of these scripts ever write into the paper repository**; copying a
finished figure into `paper-nonlinear-interactions-SWE-sphere/Figures/` is
a separate, manual step (`examples/make_section3_figures.py`, §5, prints
the exact `cp` commands it needs rather than performing the copy itself).

---

## 0. Before trusting a new configuration: the physics gate

`examples/check_wave_set_physics.py` is a **hard gate** — run it on any
new or edited wave set (registered or ad-hoc) before trusting a figure
generated from it:

```bash
# every registered wave set, all checks
python examples/check_wave_set_physics.py

# one registered wave set, a subset of checks
python examples/check_wave_set_physics.py --wave-set quartet_gravity_kelvin --check C1,C2,C4,C5,C8

# an AD-HOC configuration not in any registry -- e.g. a new quintet idea
python examples/check_wave_set_physics.py \
    --modes "4,5,3" "1,2,3" "3,4,3" "1,1,1" "7,9,1" \
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
quartet_gravity_kelvin:
  label: "RH(4,5)+RH(3,4)+RH(1,2) with the Kelvin mode EG(1,1)"
  display_label: "Quartet B"
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
| `reference_triad` | index into `triads` — the default P-measure denominator for a mode that belongs to it (see `wave_set_pmeasure.py`'s module docstring for the full per-mode rule) |
| `h_e` | equivalent height, m |
| `settings` | per-wave-set `tf_days`/`h`/`n_grid`, read by `examples/make_section3_figures.py` — the single source of truth for how long/finely to integrate this particular configuration, not a shared default |

**Velocity caps** (same convention as the §2.2 triad registry): Rossby
(RH) mode velocities up to 100 m/s, gravity (EG/WG) mode velocities up to
50 m/s (`rsw_sphere.plotting.triad_efficiency.default_velocity_range`,
reused here rather than forked).

### Adding a new quartet or quintet

1. Add an entry to `examples/wave_sets_section_3.yaml` (or your own YAML —
   `--specs` works everywhere). No Python changes needed.
2. Run the physics gate on it (§0) — `python examples/check_wave_set_physics.py --wave-set your_new_key`.
   Fix any hard-check failure before proceeding; read the C6 report even
   though it can't fail the run.
3. Add `tf_days`/`h`/`n_grid` to its `settings` block based on its own
   nonlinear exchange period, not a copy-pasted default — an integration
   horizon shorter than the true exchange period silently under-reports
   `ΔEK`/`P` without erroring (the exact failure mode logged for §2.2's
   triad registry; the same risk applies here).
4. Regenerate its table (§2) and figures (§3-5) in isolation first
   (`--wave-set your_new_key`) before adding it to
   `examples/make_section3_figures.py`'s `PMEASURE_WAVE_SETS` dict or any
   paper composite.

### Testing a triad, quartet, or quintet that isn't in any registry at all

Every plotting function's core signature is `(modes, triads, velocities,
...)` — you never have to touch a YAML file to try an idea:

```python
from rsw_sphere.plotting.wave_set_dynamics import wave_set_energy_evolution

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
python -m rsw_sphere.plotting.wave_set_table outputs/figures/wave_sets/table.md --fmt markdown --wave-set quartet_gravity_kelvin
```

or from Python:

```python
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_properties, wave_set_table

specs = load_wave_set_specs()
props = wave_set_properties(specs['quartet_gravity_kelvin'])   # single wave set, as a dict
wave_set_table(specs, fmt='latex', path='outputs/figures/wave_sets/table.tex')  # full registry
```

One `\begin{table}...\end{table}` block per wave set (unlike §2.2's single
uniform master table) since wave sets have differing mode/triad counts.
Layout matches the dissertation's own `tab: cap41`/`cap42`/`cap43`: one
coefficient column per constituent triad, `-` where a mode isn't in that
triad, a `Pump` column flagging each triad's dominant-coupling mode. The
per-triad energy-conservation residual is computed and warned on
internally but never rendered (same convention as `triad_table.py`).

Console script: `rsw-waveset-table`. Full flags: `rsw-waveset-table --help`.

---

## 3. `wave_set_dynamics.py` — energy integration

```bash
# one wave set's own energy trajectory
python rsw_sphere/plotting/wave_set_dynamics.py outputs/figures/wave_sets/quartet_gravity_kelvin_energy.png --wave-set quartet_gravity_kelvin

# "triad 1 / triad 2 [/ triad 3] / wave set" comparison row
python rsw_sphere/plotting/wave_set_dynamics.py outputs/figures/wave_sets/quartet_gravity_kelvin_panel.png --wave-set quartet_gravity_kelvin --panel
```

Console script: `rsw-waveset`. `--wave-set` selects a role key from the
registry; `--tf`/`--h` override the registry's own `settings` (default:
`tf_days=10`/`h=0.01` if the wave set has no `settings` block at all).
Full flags: `rsw-waveset --help`.

**Panels plot raw, unnormalized `|A|^2`** (convention 15, see
`paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-section-2.2.md`) —
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

## 4. `wave_set_periods.py` — dominant-period analysis

```bash
python rsw_sphere/plotting/wave_set_periods.py outputs/figures/wave_sets/quartet_gravity_kelvin_periods.png --wave-set quartet_gravity_kelvin
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
spectrum above exists so far. See
`paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-section-3.md`'s
Phase C4 for the intended design before building this.

Run `python -m rsw_sphere.plotting.wave_set_periods` (no args) for a fast,
registry-independent synthetic self-check instead of the registry CLI.

---

## 5. `wave_set_pmeasure.py` — P-measure

The P-measure (paper eq. `Pa`) compares one target mode's kinetic-energy
variation in the full wave set against its variation in one constituent
triad alone: `P = 100 * (ΔEK_wave_set - ΔEK_triad) / ΔEK_triad`. Positive
means the extra mode(s) enhance that target's energy exchange; negative
means they inhibit it.

```bash
# single point (no sweep) -- fast, useful while iterating
python -c "
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_pmeasure import p_measure
spec = load_wave_set_specs()['quartet_gravity_kelvin']
triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
print(p_measure(spec.modes, triads, spec.velocities, h_e=spec.h_e,
                 reference_triad=spec.reference_triad, tf_days=20, h=0.01))
"

# full 2D sweep + plot (expensive -- see the runtime note below)
python rsw_sphere/plotting/wave_set_pmeasure.py outputs/figures/wave_sets/quartet_gravity_kelvin_pmeasure.png --wave-set quartet_gravity_kelvin --n-grid 10
```

Console script: `rsw-waveset-pmeasure`. Full flags: `rsw-waveset-pmeasure --help`.

Three things live in the function signature, not a plotting special case
(the exact way a scalar-only API caused a real bug in the §2.2 tools):
**per-mode `triad_index`** (different target modes can use different
denominator triads — the default follows the dissertation's own rule, see
the module docstring); **`dEK_triad == 0` → `NaN`**, guarded explicitly
rather than raising; and **row-level denominator caching** in the sweep
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
one session (see `paper-nonlinear-interactions-SWE-sphere/.claude/NUMBERS-CHECK-section-3.md`
§0). **Always check `p_measure_sweep`'s returned array directly** (e.g.
`(result['P'] < 0).mean()`) rather than answering "does this figure show
inhibition" by looking at the rendered PNG.

---

## 6. `examples/make_section3_figures.py` — paper composite figures

Builds, per registered wave set: a comparison panel (§3), a period panel
(§4), and — for the wave sets listed in `PMEASURE_WAVE_SETS` — a P-measure
sweep (§5). Mirrors `examples/make_section22_figures.py`'s structure
(`*_SETTINGS` dict, CLI overrides, `outputs/`-only writes, prints the `cp`
commands rather than copying itself).

```bash
# everything, current tuned settings
python examples/make_section3_figures.py

# fast/coarse look while iterating
python examples/make_section3_figures.py --n-grid 5 --tf-scale 0.5

# just one wave set
python examples/make_section3_figures.py --wave-set quartet_gravity_kelvin

# skip the (expensive) P-measure sweeps entirely
python examples/make_section3_figures.py --skip-pmeasure
```

Full flags: `python examples/make_section3_figures.py --help`.

---

## 7. References

See [`triads.md`](triads.md) for the single-triad tools this generalizes,
[`dispersion_relation.md`](dispersion_relation.md) and the main
[`README.md`](../README.md) for the underlying eigenvalue problem, and
`paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-section-3.md` for
the design rationale (the permutation between `TRIAD`'s and `WaveSet`'s
conventions, the `fat`/gauge argument, why quartets don't conserve
energy) and what's still open.
