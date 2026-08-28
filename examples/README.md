# examples/

Registries (YAML) and driver configs for the root drivers
(`run_linear_modes.py`, `run_dynamics.py`, `run_sweep.py`, `run_sweep_sets.py`).
The investigation scripts that predated the driver refactor lived in
`../examples_legacy/`, deleted 2026-08-27 once everything still valuable
there had a current-driver reproduction path (see the "moved from
examples_legacy/" notes throughout this file for where each one landed).

## Registries

| File | Used by |
|---|---|
| `../wave_sets_default.yaml` (repo root) | The registry (`rsw_sphere.dynamics.wave_set_specs`) -- triads, quartets and quintets alike (a triad is just the 1-triad case), used by all four root drivers, including `run_linear_modes.py`. |
| `wave_sets_custom.yaml` | Example of a non-default registry, passed via `--specs`/`specs_path`. |

## Driver configs

`run_sweep.py`/`run_sweep_sets.py` both read straight from the wave set's own
`wave_sets_default.yaml` entry (`--wave-set KEY`), which carries its own
`sweep:`/`alternative_modes:` block respectively -- no separate config file
for either. A wave set not yet worth adding to the default registry can be
pointed at with `--specs path.yaml` instead (same registry schema, e.g.
`wave_sets_custom.yaml`).

```bash
python run_sweep.py --wave-set quartet_rossby_kelvin
python run_sweep.py --wave-set quartet_rh_preference
python run_sweep_sets.py --wave-set quartet_rossby_kelvin --slot d
python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
```

`triad_families.yaml`/`rh_partner_family.py`/the `candidates_*.yaml` configs
this section used to document were retired 2026-08-28: candidate-mode
screening is now entirely `alternative_modes:`-driven from the registry
itself (see `docs/wave_sets.md`).

## Infrastructure

`rsw_sphere/utilities/check_wave_set_physics.py` is the `WaveSet`-vs-`TRIAD`
physics gate -- run on any new/edited wave set before trusting a figure
from it (see `docs/wave_sets.md` §0).

## `tables/`/`figures/`

Small scripts that call a driver (or a standalone `rsw_sphere/plotting/*.py`
module) for its own data and format/compose one paper table or figure on
top, named `paper_table<NN>_<name>.py`/`paper_figure<NNN>_<name>.py` (the
LaTeX `\label{tab: ...}`/`\label{fig: ...}` number/name, cross-referenced
against the paper's own numbering, plus a stable descriptive name) --
`<NN>`/`<NNN>` is kept in sync with the script's own actual compiled
number (2026-08-28 policy: renamed whenever a figure/table shifts, rather
than letting the number drift from what's actually in the paper; see the
note below on where a script goes if its own figure/table is removed
entirely). Populated so far for everything through §4.2
"Rossby-only quartet" (JFM-template.tex, `sec: quartet_rh`), plus §4.3
"Gravity-Rossby quartets"/§5 "Quintets" -- see each script's own module
docstring for exactly which `\label{...}` it reproduces:
`paper_figure008_quartet_rossby_kelvin_panel.py` (`fig: cap4ex1`) is
Quartet C's own figure -- 3x2 grid: top row is the same "Triad 1 / Triad 2
/ full quartet" evolution row it always was, bottom row is now a
novelty-frequency spectrum panel per Rossby mode (RH(4,5)/RH(3,4)/
RH(1,2)), reusing `rsw_sphere.plotting.novelty_frequency_panel.novelty_frequency_figure`
the same way Quartet D's own panel does. Rebuilt 2026-08-28 around a
symmetric 40 m/s IC (all four modes, replacing the registry's own
30 m/s default) specifically to surface a new, slower period the
RH-only triad doesn't have on its own -- see the module docstring for
the exact numbers. This retires the companion `fig: power1` power-spectrum
figure entirely (its own spectra are now this figure's bottom row) --
its script, `paper_figure009_quartet_rossby_kelvin_periods.py`, moved to
`figures/legacy/`, along with `tables/legacy/paper_headline_quartet_c_phaselag.py`
(its own 30 m/s-IC phase-lag number is no longer cited once the
paragraph citing it was rewritten around the new IC).
`paper_figure013_quintet_gravity_star_panel.py` (`fig: quintetpanel`) --
these retire `make_section3_figures.py` (deleted 2026-08-27, once every
wave set it covered that is actually cited in the paper had its own
dedicated script). `paper_figure009_quartet_rossby_kelvin_gravity_wavenumber.py`
(`fig: rossby_kelvin_wavenumber`) opens `sec: quartet_rossby_gravity_fast`:
`quartet_rossby_kelvin`'s own registered `alternative_modes.d` candidate
list (EG(1,n)/WG(1,n), n=1..15 odd), run twice via
`run_sweep_sets.run_sweep_sets`'s `target_mode_override` -- once against
the candidate's own diagnostics, once against RH(1,2)'s (the registry's
own `alternative_modes.d.target_mode: c`) -- as a 2x2 grid, efficiency
variation and p_measure (rows) x gravity mode and RH(1,2) (columns).
p_measure was reinstated here 2026-08-28 after an earlier pass had
dropped it in favor of efficiency_var alone: RH(1,2)'s own efficiency_var
swings monotonically as candidate wavenumber grows while its p_measure
stays flat and non-monotonic, a share-vs-raw-swing divergence driven by
the quartet's own total energy budget growing with the candidate's own
wavenumber (see `rsw_sphere.physics.total_energy_joules`) -- invisible
without p_measure alongside it.
`paper_figure010_quartet_rossby_gravity_influence_panel.py`
is the current Quartet D figure (2x2: both constituent-triad evolutions,
the full-quartet evolution, and RH(3,4)'s novelty-frequency spectrum
against Triad 1, its only containing triad), replacing the retired
`paper_figure009_quartet_gravity_79_panel.py` (`quartet_gravity_79`/EG(7,9)
-- superseded 2026-08-28 by `quartet_rossby_gravity_influence`, a
different topology where WG(7,9) is the shared sum mode rather than a
private 4th mode; the EG(7,9) result survives only as a contrasting
remark in the prose, no table/figure of its own).
`paper_figure011_quartet_rossby_gravity_influence_efficiency.py` (1D + 2D
`efficiency_var` sweep over WG(3,9)'s driving velocity) is wired into
`sec: quartet_rossby_gravity_fast` right after Quartet D's own panel,
with full interpretive prose (the notch near u=20-28 m/s explained by
WG(7,9)'s own efficiency in Triad 2 passing through a near-cancellation).
`paper_figure012_quartet_rossby_gravity_influence_high_panel.py` is
Quartet E's own panel (`quartet_rossby_gravity_influence_high`,
registered 2026-08-28) -- same 2x2 layout as Quartet D's panel; EG(7,9)
plays a double role (sum mode of the RH-only Triad 1, member of a second
triad closed by EG(11,11)).

Script numbers are kept in sync with each script's OWN actual compiled
figure/table number (checked via `JFM-template.aux`'s `\newlabel{fig:
...}`/`\newlabel{tab: ...}` entries, not the LaTeX source order) --
unlike the general convention above, this repo keeps these current rather
than letting them drift, so a script whose own figure/table is removed
from the paper entirely (not just renumbered) moves to
`examples/figures/legacy/`/`examples/tables/legacy/` instead of being
renumbered into a slot that no longer means anything:
`figures/legacy/paper_figure011_quintet_gravity_star_pmeasure.py`
(`fig: 4eff3`, no longer cited anywhere in `JFM-template.tex`),
`figures/legacy/paper_figure009_quartet_rossby_kelvin_periods.py`
(`fig: power1`, folded into `fig: cap4ex1` above), and
`tables/legacy/paper_table02_quartet_a_properties.py` (`tab: cap41`,
superseded by the combined `tab: quartet_master` below) all moved there
2026-08-28, staged for outright deletion once confirmed unneeded.

Table `precession_comparison` and
Figure `borrowed_topology_precession` (Quartet B, §4.2.2) are also not
covered by this naming convention -- they're generated by
`raphaldini2022_compare/precession_comparison.py` instead, since that
comparison's other side is Raphaldini et al. (2022)'s own barotropic
vorticity equation, which has no `WaveSet`/registry representation (see
`raphaldini2022_compare/README.md`). The RSW side of that same comparison
*is* registered (`quartet_rh_borrowed_topology`).

Three further §4.3-era scripts whose core computation isn't reducible to a
`run_sweep.py`/`run_sweep_sets.py` diagnostic (moved into `examples/` from
`examples_legacy/special_runs/` 2026-08-26, then into `tables/` and renamed
to the `paper_table<NN>_<name>.py`/`paper_headline_<name>.py` convention
2026-08-27; the other 7 scripts that were in that cluster were deleted the
same day along with the paper sections they backed -- see
`examples_legacy/README.md`):

- `tables/paper_table02_quartet_master.py` -- ONE combined table across
  Quartets A/B/C/D/E (`tab: quartet_master`; Quartet E added 2026-08-28),
  replacing the three separate
  per-quartet coefficient tables `tab: cap41`/`cap42`/`cap43` (Quartet
  A/C/D) that used to sit inline in each quartet's own subsection, via
  the new `rsw_sphere.plotting.wave_set_table.wave_set_master_table`
  (mirrors `tab: master`'s own hand-merged, multi-group style, generated
  instead of hand-touched-up). Retires the now-deleted
  `tables/paper_table04_gravity_quartet_coefficients.py` (was
  `regen_gravity_quartet_tables.py`), whose Quartet C numbers moved into
  this combined table unchanged and whose Quartet D (EG(7,9)) numbers
  were dropped along with that quartet's own table/figure (2026-08-28).
  Quartet B's own `tab: precession_comparison` stays separate (different
  shape); this table only adds Quartet B's coefficient properties as a
  4th group.
- `tables/legacy/paper_headline_quartet_c_phaselag.py` (was
  `gate_i5_headline.py`) -- the §Coupled Triads S4 headline number
  (physical-Joules amplitude/phase error from filtering the gravity mode
  at Quartet C's OLD 30 m/s IC), via
  `rsw_sphere.physics.air_density_from_equivalent_depth`. Moved to legacy
  2026-08-28: its own numbers (`-66.5%`/`+14.6%`/`0.9 days` at tf=20d) are
  no longer cited once `fig: cap4ex1`'s own paragraph was rewritten around
  a symmetric 40 m/s IC and a novelty-frequency framing instead.
- `tables/paper_headlines_sec3.3.py` (was `section33_headline_numbers.py`)
  -- generalizes the retired `paper_headline_quartet_c_phaselag.py`'s own
  method (period shift + peak-KE difference, both tf-independent;
  $\mathcal{F}_2$ at the registered tf_days) to both Quartet C and
  Quartet D. Only its Quartet D (EG(7,9)) numbers are still cited in
  `JFM-template.tex` (the brief contrasting remark in `sec:
  quartet_rossby_gravity_fast`) -- its Quartet C numbers were retired
  alongside `paper_headline_quartet_c_phaselag.py` above, see this
  script's own module docstring.

`rh_partner_quartet_family.py` is fully migrated: `quartet_rh_preference`
was already its exact base quartet, so its own registered
`alternative_modes.d` block (`wave_sets_default.yaml`) reproduces it --
`python run_sweep_sets.py --wave-set quartet_rh_preference --slot d` --
with no separate config file.
`short_gravity_long_rossby_example.py` is also migrated, onto the
corrected candidate `quartet_gravity_wg11` (WG(1,1), not the retracted
WG(7,9) claim the script itself made -- see that registry entry's own
`role` note).
`raphaldini2022_compare/` (external barotropic-model comparison, see its
own README) has also moved here from `examples_legacy/` (2026-08-26) and
was later consolidated from seven scripts down to three -- the barotropic
model itself still has no registry entry (per the
`precession_comparison`/`borrowed_topology_precession` note above), but
the RSW side of the same comparison is now registered as
`quartet_rh_borrowed_topology`.
