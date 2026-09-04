# examples/

Registries (YAML) and paper-facing scripts built on top of the root
drivers (`run_linear_modes.py`, `run_dynamics.py`, `run_sweep.py`,
`run_sweep_sets.py`). Everything here either configures a driver (a
registry entry, a `--specs` file) or calls one to produce a single
table/figure/number for the paper.

```
examples/
├── wave_sets_custom.yaml    # starting point for a --specs registry (edit freely)
├── tables/                  # one script per paper table
├── figures/                 # one script per paper figure
└── raphaldini2022_compare/  # standalone comparison against an external model
```

Each script directory carries a small `_bootstrap.py`; a script's first
import is `import _bootstrap`, which puts the repository root on
`sys.path` (so the script runs from an uninstalled checkout) and exposes
`_bootstrap.ROOT` for anchoring output paths. Scripts never resolve a
path against the current working directory, so they can be run from
anywhere.

## Registries

| File | Used by |
|---|---|
| `../wave_sets_default.yaml` (repo root) | The registry (`rsw_sphere.dynamics.wave_set_specs`) -- triads, quartets and quintets alike (a triad is just the 1-triad case), used by all four registry drivers. |
| `wave_sets_custom.yaml` | Starts as a copy of the default registry; edit it freely for your own wave sets -- demonstrates `--specs`/`specs_path` pointing at a registry other than the repo-root default, and is expected to diverge from it over time. |

`run_sweep.py`/`run_sweep_sets.py` read their sweep/candidate-screening
config straight from the wave set's own registry entry (`sweep:`/
`alternative_modes:` blocks) -- no separate config file needed for
either:

```bash
python run_sweep.py --wave-set quartet_rossby_kelvin
python run_sweep_sets.py --wave-set quartet_rossby_kelvin --slot d
python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
python run_dynamics.py --wave-set quartet_rossby_kelvin
```

A wave set not yet worth adding to the default registry can be pointed at
with `--specs path.yaml` instead (same schema; `wave_sets_custom.yaml` starts
as a copy of the default registry for exactly this -- edit it directly for
your own experiments rather than the repo-root one).
Candidate-mode screening (which mode best fills a slot) is entirely
`alternative_modes:`-driven from the registry -- see `docs/wave_sets.md`
for the schema and `rsw_sphere.utilities.check_wave_set_physics` (run it
on any new or edited wave set before trusting a figure from it).

## `tables/` and `figures/`

Each script calls a driver (or a standalone `rsw_sphere/plotting/*.py`
module) for its own data and formats/composes one paper table or figure
on top -- self-contained and runnable standalone:

```bash
python examples/tables/paper_table02_quartet_master.py
python examples/figures/paper_figure008_quartet_rossby_kelvin_panel.py
```

| Script | Reproduces |
|---|---|
| `tables/paper_table01_resonant_triads.py` | `tab: master` |
| `tables/paper_table02_quartet_master.py` | `tab: quartet_master` |
| `tables/paper_table03_rh_partner_family.py` | `tab: rhfamily` |
| `tables/paper_table05_quintet_master.py` | `tab: quintet_master` |
| `figures/paper_figure001_dispersion_relation.py` | `fig: 1` |
| `figures/paper_figure002_hough_examples.py` | `fig: hough_examples` (+ its `hough_rh`/`hough_eg`/`hough_wig`/`hough_rh34` subfigures) |
| `figures/paper_figure003_rossby_only_triads.py` | `fig: rossby_only` |
| `figures/paper_figure004_combined_triads.py` | `fig: combined` |
| `figures/paper_figure006_quartet_rh_preference_panel.py` | `fig: cap42` |
| `figures/paper_figure007_quartet_a_precession.py` | `fig: quartet_a_precession` (panel a of `fig: precession_frequency`) |
| `figures/paper_figure008_quartet_rossby_kelvin_panel.py` | `fig: cap4ex1` |
| `figures/paper_figure009_quartet_rossby_kelvin_gravity_wavenumber.py` | `fig: rossby_kelvin_wavenumber` |
| `figures/paper_figure010_quartet_rossby_gravity_influence_panel.py` | `fig: quartet_rossby_gravity_influence_panel` |
| `figures/paper_figure011_quartet_rossby_gravity_influence_efficiency.py` | `fig: quartet_rossby_gravity_influence_efficiency` |
| `figures/paper_figure012_quartet_rossby_gravity_influence_high_panel.py` | `fig: quartet_rossby_gravity_influence_high_panel` |
| `figures/paper_figure014_quintet_gravity_star_panel.py` | `fig: quintetpanel` |
| `figures/paper_figure015_quintet_gravity_influence_star_panel.py` | `fig: quintetpanel_b` |

`figures/_triad_panel_row.py` is a shared helper, not a figure script:
the "triad 1 / triad 2 / full wave set" panel row several of the above
build on.

Naming: `paper_table<NN>_<name>.py` / `paper_figure<NNN>_<name>.py`,
where `<NN>`/`<NNN>` is the script's own compiled table/figure number in
`JFM-template.tex` (checked via `JFM-template.aux`'s `\newlabel{tab: ...}`
/`\newlabel{fig: ...}` entries, not creation order) and `<name>` is a
stable descriptive name. Renumber a script whenever its own number shifts
in the paper, so the filename can always be trusted to mean "this is
currently Table/Figure N". A script computing a number used only in prose
(not a table or figure) is named `paper_headline_<name>.py` and also
lives in `tables/`. Each script's own module docstring names the exact
`\label{...}` it reproduces and the output PNG filename to copy into
`paper-nonlinear-interactions-SWE-sphere/Figures/`; that docstring, not
this file, is the source of truth for what a script does.
`docs/wave_sets.md` cross-references each wave-set-facing figure's script
alongside its registry entry.

A script whose table or figure is dropped from the paper entirely is
deleted rather than renumbered into a slot that no longer means anything
-- git history keeps it, and a stale script that runs but backs nothing
is worse than no script.

Not every table/figure in the paper has a script here. Table `pump` and
the topology diagrams (`fig: topology_overview`, `fig: quintet_topology`)
are written directly in the LaTeX with no numeric data to regenerate, and
Quartet B's `tab: precession_comparison` / `fig: borrowed_topology_precession`
come from `raphaldini2022_compare/` instead (below), since that
comparison's other side has no `WaveSet`/registry representation.

## Specific experiments

`raphaldini2022_compare/` -- comparison of this paper's RSW quartets
against Raphaldini et al. (2022)'s own barotropic vorticity-equation
model, on an identical four-wave topology (registered here as
`quartet_rh_borrowed_topology`). Self-contained, with its own README
covering usage and the barotropic model's own derivation.
