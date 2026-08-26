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
| `triads_section_2_2.yaml` | §2.2's own parallel single-triad registry (`rsw_sphere.dynamics.triad_specs`), feeding `triad_table.py`/`triad_dynamics.py`/`triad_efficiency.py` -- its 4 headline triads are now ALSO registered in `wave_sets_default.yaml` (added 2026-08-26) for use by the 4 root drivers; this file itself hasn't been retired. |
| `triad_families.yaml` | RH(3,n) partner-family registry, used by `rh_partner_family.py`. |

## Driver configs

`sweep_*.yaml`/`candidates_*.yaml` are `run_sweep.py`/`run_sweep_sets.py`
configs (`RunConfig` + a `sweep:` block, or a candidate-screening block).

```bash
python run_sweep.py --config examples/sweep_quartet_gravity_kelvin_diagnostics.yaml
python run_sweep.py --config examples/sweep_quartet_a_rh36.yaml
python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
```

## Infrastructure

`check_wave_set_physics.py` is the `WaveSet`-vs-`TRIAD` physics gate --
run on any new/edited wave set before trusting a figure from it (see
`docs/wave_sets.md` §0).

## Not yet populated

`tables/`/`figures/` (small scripts that call a driver for its cached
data and format/compose a paper table or panel on top) and
`special_runs/`/`raphaldini2022_compare/` (candidate clusters from
`examples_legacy/` re-attempted against `run_sweep_sets.py`) are planned
per the driver refactor plan but not yet built -- see
`.claude/PLAN-driver-refactor-2026-08-25.md`'s Phase B+C.
