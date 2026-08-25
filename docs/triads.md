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
| `rsw_sphere/plotting/triad_table.py` | Batch table: per-mode frequency/period/coupling coefficient, per-triad mismatch `δ`, pump mode |
| `rsw_sphere/plotting/triad_dynamics.py` | Energy-integration time series for one triad |
| `rsw_sphere/plotting/triad_efficiency.py` | 2D efficiency-of-energy-transfer sweep over two modes' initial velocities |

All three load their triads from a **registry YAML** rather than
hardcoding mode numbers — by default
[`examples/triads_section_2_2.yaml`](../examples/triads_section_2_2.yaml),
via `rsw_sphere.dynamics.triad_specs.load_triad_specs()`. Pass `--specs
path/to/other.yaml` to point any of the three scripts at a different
registry.

**Target mode.** In every efficiency-sweep figure, the *target mode* is
the mode whose efficiency is measured. Following the dissertation's own
methodology (`Chapter2.tex:341`), it is held at rest (zero initial zonal
velocity) while the *other two* modes are swept over a velocity grid — not
the other way around. `triad_efficiency.efficiency_sweep`'s defaults
(`fixed_index=target`, `fixed_velocity=0.0`) implement this directly.

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
triad_rossby_only_non_resonant:
  label: "Rossby-only, non-resonant, short period"
  display_label: "Triad B"
  role: "..."          # one-line note on why this triad is included
  h_e: 10000
  mode_a: {m: 3, n: 4, alpha: 3, u: 0.0}     # alpha: 1=EIG, 2=WIG, 3=RH
  mode_b: {m: 1, n: 2, alpha: 3, u: 58.0}    # u = initial zonal velocity, m/s
  mode_c: {m: 4, n: 5, alpha: 3, u: 100.0}
```

Keys are **semantic role labels**, not mode numbers
(`triad_rossby_only_non_resonant`, not `rh-34`) — these triads get reused
as building blocks in later paper sections (Multiple Triads, Five-Wave),
so a name tied to only one of the three modes wouldn't scale.
`display_label` is the short "Triad A"/"B"/"C"/"D" tag used in the master
table and figure titles. The four currently registered:

| key | display | modes (a, b, c) |
|---|---|---|
| `triad_rossby_only_near_resonant` | Triad A | RH(3,10), RH(1,2), RH(4,5) |
| `triad_rossby_only_non_resonant` | Triad B | RH(3,4), RH(1,2), RH(4,5) |
| `triad_kelvin_rossby_flow` | Triad C | EG(1,1), RH(3,4), RH(4,5) |
| `triad_gravity_with_rossby_catalyst` | Triad D | EG(6,9), RH(1,7), EG(7,9) |

To add or edit a triad, edit the YAML — no Python changes needed. The
initial velocities (`u`) in the shipped file are illustrative starting
points, not yet re-verified against the specific efficiency-maximizing
values reported in the dissertation for every triad. **Velocity ranges**:
Rossby (RH) mode velocities/sweep bounds go up to 100 m/s; gravity (EG/WG)
mode velocities are capped at 50 m/s (see `triad_efficiency.
default_velocity_range`) — realistic Kelvin/inertia-gravity wind-anomaly
amplitudes are much smaller than Rossby-mode wind speeds.

Note `TriadSpec.velocities`' dual meaning: `triad_dynamics.
triad_energy_evolution` (the library function) uses all three entries
literally as the initial condition, but `triad_efficiency.
efficiency_sweep` only uses whichever entry corresponds to the *fixed*
mode (by default the target mode itself, held at 0 regardless of what's
registered here) — the other two are swept over a grid, not read from the
registry. See `TriadSpec`'s own docstring.

`examples/make_section22_figures.py`'s composite-panel builder (`_panel`)
goes one step further for consistency with the efficiency map next to it:
it forces the *target* mode's velocity to 0 for the energy-integration
panel too, by default, overriding the registry's own value for that one
component. This matters because several registry entries set the target
mode's own `u` to something nonzero (e.g. Triad C's EG(1,1), Triad D's
EG(6,9), both 50 m/s) — reusing that literally would make the "target
mode" wording in the figure title/caption false for the trajectory panel
even though the map beside it correctly enforces target-at-rest (caught
in paper review, 2026-08-11). Pass an explicit `energy_velocities` triple
in a `PANELS` row to override this default when the target-zeroed
registry values wouldn't tell a useful story on their own (e.g. Triad D:
with EG(6,9) zeroed and EG(7,9) also registered at 0, only RH(1,7) would
carry any energy, and its coupling is too weak to move it — the row
instead puts the initial energy on EG(7,9) explicitly).

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
props = triad_properties(specs['triad_gravity_with_rossby_catalyst'].modes)   # single triad, as a dict
triad_table(specs, fmt='latex', path='outputs/figures/triads/table.tex')  # full registry
```

`triad_properties` also computes the energy-conservation residual
`(α^a_bc + α^b_ac − α^c_ab) + δ·S_abc`, which must be ≈0 for every
physically consistent triad — kept as an internal correctness check
(`triad_properties` `warnings.warn`s if it is not ~0) but **not rendered**
in any output format (`latex`/`csv`/`markdown`). `--include-table1` adds
the four zero-coupling quasi-resonant triads from dissertation Table 2.1.

The `latex` table's leftmost column (filled once per triad, spanning its 3
mode rows) shows the role-key name word-wrapped to 3 lines (via
`\shortstack`, no extra package required), the "Triad A/B/C/D"
`display_label`, and the mismatch `δ` — replacing the older
`\multicolumn` banner-row-per-triad layout. Column headers are centered
via `\multicolumn{1}{c}{...}`; `ω` is labelled "(dimensionless)" since
this nondimensional convention has no dimensional frequency unit to quote.

Console script: `rsw-triad-table` (after `pip install -e .`). Full flags:
`rsw-triad-table --help`.

---

## 3. `triad_dynamics.py` — energy integration

```bash
python rsw_sphere/plotting/triad_dynamics.py outputs/figures/triads/triad_gravity_with_rossby_catalyst_energy.png --triad triad_gravity_with_rossby_catalyst --tf 3
```

Console script: `rsw-triad`. `--triad` selects a role key from the
registry (default `triad_gravity_with_rossby_catalyst`); `--tf` is the
integration horizon in days, `--h` the RK33 step (nondimensional time).
Full flags: `rsw-triad --help`.

Each mode is drawn in a persistent color regardless of which triad/panel
it appears in — see `rsw_sphere.plotting.style.MODE_COLORS` (warm
red/orange/yellow for RH modes, cool blue/indigo for EG/WG modes, grey for
the non-modal total-energy reference line).

---

## 4. `triad_efficiency.py` — efficiency sweep

Sweeps the initial zonal velocities of the two non-target modes over a
grid (target mode held at rest, see "Target mode" above), integrates each
combination, and records the efficiency of the target mode (`max − min` of
its normalized kinetic energy). This is the expensive one — a 40×40 grid
is ~1.6×10⁶ RK33 steps at the defaults (~10⁷ at the dissertation's own
100×100 grid) — so the compute (`efficiency_sweep`) is split from the
plotting (`plot_efficiency_map`) and **cached to `.npz`**: the first run
computes and saves, every subsequent run with the same `--cache` path just
reloads.

```bash
python rsw_sphere/plotting/triad_efficiency.py outputs/figures/triads/triad_rossby_only_non_resonant_efficiency.png \
    --triad triad_rossby_only_non_resonant --n-grid 20 --cache outputs/figures/triads/triad_rossby_only_non_resonant_eff.npz
```

Console script: `rsw-triad-efficiency`. `--target {0,1,2}` picks which
mode is the target (a/b/c), held at rest while the other two are swept.
Axis labels and the figure title are derived from actual mode identity
(e.g. `"RH(4,5)"`, target shown as `"target: EG(1,1)"`), not generic "mode
a"/"mode b" strings. The colormap is `cividis` with a `PowerNorm`
(gamma≈0.45) on a fixed 0-100% scale and colorbar ticks biased toward the
low end (`0, 1, 5, 10, 25, 50, 100`) — perceptually uniform, colorblind-
and grayscale-safe, and resolves low-efficiency structure that a linear
norm flattens. Full flags: `rsw-triad-efficiency --help`.

---

## 5. `examples/make_section22_figures.py` — paper composite figures

Builds the two composite figures used directly in paper §2.2: a 2×2
Rossby-only panel (`triad_rossby_only_near_resonant`,
`triad_rossby_only_non_resonant`) and a 2×2 combined Rossby-gravity panel
(`triad_kelvin_rossby_flow`, `triad_gravity_with_rossby_catalyst`), each
row an (efficiency map, energy integration) pair, target mode = index 0
("mode a") for every triad in both composite panels. Regenerated as a
normal `outputs/`-writing script, then copied into the paper repo's
`Figures/` by hand (the script prints the exact `cp` commands it needs
rather than performing the copy itself).

```bash
# full run, both composite panels + the extra gravity->Rossby target pair, current tuned settings
python examples/make_section22_figures.py

# fast/coarse look while iterating (few minutes instead of tens of minutes)
python examples/make_section22_figures.py --n-grid 6 --tf-scale 0.5

# regenerate just one triad's (efficiency, energy) pair at target=0, e.g. after
# editing its velocities in the registry YAML
python examples/make_section22_figures.py --triad triad_gravity_with_rossby_catalyst

# regenerate one triad's pair at a specific target mode index
python examples/make_section22_figures.py --triad triad_kelvin_rossby_flow --target 1
```

`triad_kelvin_rossby_flow` gets a second, standalone (efficiency, energy)
pair at `target=1` (RH(3,4), its pump mode, verified against the master
table) beyond the default `target=0` (EG(1,1)) used in the composite
panel — same three modes, same coupling network, illustrating the
reversed (gravity→Rossby) transfer direction. See `EXTRA_TARGETS` in the
script and its module docstring's "TARGET-MODE CONVENTION" section.

Per-triad grid resolution and integration horizon live in the script's
`TRIAD_SETTINGS` dict (fallback: `DEFAULT_SETTINGS`, with a printed
warning, for any registry triad not listed there — e.g. a newly-added
one); velocity ranges are no longer set per-triad here (derived
automatically from each swept mode's family — RH up to 100 m/s, EG/WG up
to 50 m/s). **Read the module docstring's "CALIBRATION NOTES" section
before trusting a new/edited triad's numbers** — the most common silent
mistake is `tf_days` too short relative to the triad's own nonlinear
exchange period, which under-reports efficiency without erroring (this is
exactly what happened historically for `triad_kelvin_rossby_flow`, caught
and documented in
`paper-nonlinear-interactions-SWE-sphere/.claude/NUMBERS-CHECK-section-2.2.md`).
Sweep caches are keyed by a hash of every parameter that affects the
result (`triad_efficiency.cache_key_hash`), not by triad name alone, so
changing settings no longer risks silently serving a stale result;
`--clear-cache` remains available as an escape hatch.

Full flags: `python examples/make_section22_figures.py --help`.

---

## 6. References

See [`dispersion_relation.md`](dispersion_relation.md) and the main
[`README.md`](../README.md) for the underlying eigenvalue problem, and
`.claude/THESIS_FIGURES.md` for how thesis figures/tables map to code.
