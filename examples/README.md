# examples/

Registries (YAML) and driver configs for the root drivers
(`run_linear_modes.py`, `run_dynamics.py`, `run_sweep.py`, `run_sweep_sets.py`).
Investigation scripts that predate the driver refactor live in
`../examples_legacy/` (untouched, still runnable, not part of this
folder's own registries).

## Registries

| File | Used by |
|---|---|
| `../wave_sets_default.yaml` (repo root) | The registry (`rsw_sphere.dynamics.wave_set_specs`) -- triads, quartets and quintets alike (a triad is just the 1-triad case), used by all four root drivers, including `run_linear_modes.py`. |
| `wave_sets_custom.yaml` | Example of a non-default registry, passed via `--specs`/`specs_path`. |
| `triad_families.yaml` | RH(3,n) partner-family registry (`rsw_sphere.dynamics.triad_family_specs`), used by `rh_partner_family.py`. |

## Driver configs

`candidates_*.yaml` are `run_sweep_sets.py` configs (a candidate-screening
block). `run_sweep.py` sweeps don't need their own config file when the
wave set's own `wave_sets_default.yaml` entry already carries a `sweep:`
block (`--wave-set KEY`); a standalone `RunConfig`-shaped YAML (`--config
path.yaml`) is only for an ad-hoc/one-off sweep not worth registering.

```bash
python run_sweep.py --wave-set quartet_rossby_kelvin
python run_sweep.py --wave-set quartet_rh_preference
python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
```

## Infrastructure

`rsw_sphere/utilities/check_wave_set_physics.py` is the `WaveSet`-vs-`TRIAD`
physics gate -- run on any new/edited wave set before trusting a figure
from it (see `docs/wave_sets.md` §0).

## `tables/`/`figures/`

Small scripts that call a driver (or a standalone `rsw_sphere/plotting/*.py`
module) for its own data and format/compose one paper table or figure on
top, named `paper_table<NN>_<name>.py`/`paper_figure<NNN>_<name>.py` (the
LaTeX `\label{tab: ...}`/`\label{fig: ...}` number/name, cross-referenced
against the paper's own numbering, plus a stable descriptive name since
LaTeX numbering drifts). Populated so far for everything through §4.2
"Rossby-only quartet" (JFM-template.tex, `sec: quartet_rh`) -- see each
script's own module docstring for exactly which `\label{...}` it
reproduces. §4.3 "Gravity-Rossby quartets" (`sec: gravity`) onward is not
yet covered here (mid-redesign elsewhere); Table `precession_comparison`
and Figure `borrowed_topology_precession` (Quartet B, §4.2.2) are also not
covered -- both depend on `examples_legacy/raphaldini2022_compare/`'s
bespoke comparison against Raphaldini et al. (2022)'s own barotropic
vorticity-equation system, which has no `WaveSet`/registry representation
in this repo and so isn't reproducible via the current drivers alone.

`examples_legacy/special_runs/` (2026-08-26): the 10-script cluster whose
core computation isn't reducible to a `run_sweep.py`/`run_sweep_sets.py`
diagnostic (power-law fits, Hilbert phase lag, dual-estimator agreement
checks, tf-convergence studies, physical-Joules conversion) -- see
`examples_legacy/README.md`'s own note on this move.
`rh_partner_quartet_family.py` is fully migrated: `quartet_rh_preference`
was already its exact base quartet, so `examples/candidates_rh_partner_family.yaml`
(`run_sweep_sets.py --config`) reproduces it with no registry change.
`short_gravity_long_rossby_example.py` is also migrated, onto the
corrected candidate `quartet_gravity_wg11` (WG(1,1), not the retracted
WG(7,9) claim the script itself made -- see that registry entry's own
`role` note).
`examples_legacy/raphaldini2022_compare/` (the 7-script external
barotropic-model comparison cluster) has also been moved into its own
subfolder.
