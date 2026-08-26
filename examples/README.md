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
python run_sweep.py --wave-set quartet_gravity_kelvin
python run_sweep.py --wave-set quartet_rh_preference
python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
```

## Infrastructure

`rsw_sphere/utilities/check_wave_set_physics.py` is the `WaveSet`-vs-`TRIAD`
physics gate -- run on any new/edited wave set before trusting a figure
from it (see `docs/wave_sets.md` §0).

## Not yet populated

`tables/`/`figures/` (small scripts that call a driver for its cached
data and format/compose a paper table or panel on top) is not yet built
-- Table cap41/cap42/cap43 and Figs cap4ex1/power1/quintetpanel are
already reproducible directly via `wave_set_table.py`/`make_section3_figures.py`
(see their own regenerate-comments in `JFM-template.tex`), just not yet
wrapped as named `examples/tables/`/`examples/figures/` scripts.

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
