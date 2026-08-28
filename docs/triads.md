# Resonant-triad tools (retired)

The §2.2-specific toolchain this page used to document
(`rsw_sphere/dynamics/triad_specs.py`, `rsw_sphere/plotting/triad_table.py`/
`triad_dynamics.py`/`triad_efficiency.py`, console scripts `rsw-triad-table`/
`rsw-triad`/`rsw-triad-efficiency`) is retired: a plain triad is the
degenerate 1-triad case of `WaveSet`, so every capability it offered is
now covered by the unified drivers, with the four §2.2 triads registered
directly in `wave_sets_default.yaml` (see [`wave_sets.md`](wave_sets.md) §1).

| Old | New |
|---|---|
| `triad_table.py` (batch properties table) | `rsw_sphere/plotting/wave_set_table.py` -- `python -m rsw_sphere.plotting.wave_set_table outputs/figures/table.tex` |
| `triad_dynamics.py` (energy integration) | `run_dynamics.py --wave-set <triad_key>` |
| `triad_efficiency.py` (2D efficiency sweep) | `run_sweep.py --wave-set <triad_key>` with an explicit `sweep.axes` block in the registry entry (a plain triad has no "private" modes for `run_sweep.py` to auto-derive sweep axes from, unlike a quartet/quintet) |

See [`dispersion_relation.md`](dispersion_relation.md) and the main
[`README.md`](../README.md) for the underlying eigenvalue problem.
