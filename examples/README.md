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
