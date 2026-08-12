# Example configs

Named `configs.yaml` variants reproducing specific figures/tables from the
thesis (`docs/Marco_Msc.pdf`), runnable with either driver:

```bash
python run_diagnostics.py --config examples/table_2_1_quasi_resonant.yaml
python run_dynamics.py --config examples/table_2_1_quasi_resonant.yaml
```

| Config | Reproduces | Notes |
|--------|-----------|-------|
| `../configs.yaml` (repo root) | Fig. 2.9 / Table 2.2 triad 2 | The shipped default. |
| `table_2_1_quasi_resonant.yaml` | Table 2.1 | Near-zero coupling coefficient, tiny frequency mismatch — expect almost no energy exchange. Initial velocities are placeholders (not given in the thesis table). |
| `table_2_3_kelvin_rh_energy_flow.yaml` | Table 2.3 / Fig. 2.7 | Energy flow from Rossby-Haurwitz to eastward gravity modes via the Kelvin wave `(1,1,EIG)`. |

Each writes its figures under its own `OUTPUT_PATH` (subfolders of
`outputs/figures/`) so they don't overwrite each other or the default run.

See `docs/code_guide.md` and `.claude/THESIS_FIGURES.md` for the full
thesis-figure-to-code map these were drawn from.

## Triad registry

`triads_section_2_2.yaml` is a different kind of config — not a
`run_diagnostics.py`/`run_dynamics.py` input, but the registry of triads
used by `rsw-triad-table` / `rsw-triad` / `rsw-triad-efficiency` (see
[`docs/triads.md`](../docs/triads.md)).

## Wave-set (quartet/quintet) registry

`wave_sets_section_3.yaml` is the same kind of registry, generalized to
quartets and quintets (coupled multi-triad configurations) — used by
`rsw-waveset-table` / `rsw-waveset` / `rsw-waveset-periods` /
`rsw-waveset-pmeasure`, and by `make_section3_figures.py`'s composite
builder (see [`docs/wave_sets.md`](../docs/wave_sets.md), which also
covers how to test a triad/quartet/quintet that isn't in this file at
all). `check_wave_set_physics.py` is the hard physics gate every new or
edited entry must pass before any figure is trusted.

## Precession resonance (2026-08-12)

Two standalone scripts from the same-day precession-resonance
investigation (see `paper-nonlinear-interactions-SWE-sphere/.claude/
INSPECT-phase-I0-I4-2026-08-12.md` for the full writeup):

- `reproduce_raphaldini2022_fig2.py` — direct, dependency-free
  reproduction of Raphaldini, Peixoto, Teruya, Raupp & Bustamante (2022,
  Phys. Fluids)'s own four-wave barotropic-vorticity efficiency peak,
  using their exact published numbers (not re-derived). Confirms the
  precession-resonance mechanism is real and correctly transcribed
  before comparing against RSW.
- `precession_resonance_rsw_vs_barotropic.py` — the identical mode
  topology, built in this repo's RSW/`WaveSet` system instead, for a
  direct barotropic-vs-RSW comparison. See
  `rsw_sphere.dynamics.dynamical_phase` for the reusable
  libration-vs-rotation diagnostic (the textbook precession-resonance
  signature) this investigation also produced.
