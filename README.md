# RSW on the Sphere — Nonlinear wave interactions

Code accompanying the MSc dissertation

> **Nonlinear wave interactions in rotating shallow water equations on the sphere**
> Marco Antonio Dourado — Institute of Mathematics, Statistics and Computer
> Science, University of São Paulo (IME-USP), 2025.
> Advisor: Prof. Dr. Pedro da Silva Peixoto · Co-advisor: Prof. Dr. Breno Raphaldini.

and paper:

> **Non-linear interactions between slow and fast atmospheric waves on the Sphere **
> Peixoto, Raphaldini, Dourado, Teruya.


**Code Authors:** Marco Antonio Dourado and Pedro da Silva Peixoto

## What this is about

The Rotating Shallow Water (RSW) equations on the sphere are a first global model
whose spectrum contains both **fast** waves (inertia-gravity) and **slow** waves
(Rossby-Haurwitz), connected by the Kelvin and mixed Rossby-gravity modes. This
repository builds the **normal modes** of the linearised RSW equations — the
**Hough harmonics**, obtained as eigenfunctions of Laplace's tidal equations via
the vector-spherical-harmonic method of Swarztrauber & Kasahara (1985) — and uses
them to study the **nonlinear energy exchanges** between waves in reduced systems
of three, four and five interacting modes.

The main physical questions explored (see the thesis for full detail):

- How energy is transferred between Rossby-Haurwitz and gravity waves in a
  **triad**, including the characterization of the *pump mode* and the efficiency
  of resonant / quasi-resonant triads.
- How a single gravity wave alters the kinetic-energy fields and the periods of
  energy exchange of Rossby-Haurwitz waves in **coupled triads** (four- and
  five-wave configurations).

## Current content


- **Hough harmonics & dispersion relation** — assembly and diagonalization of the
  tidal-equation eigenvalue problem, the normalized normal-mode fields
  `(u, v, h)` and their latitudinal derivatives, and the dispersion diagram of
  the three wave families. See [`docs/dispersion_relation.md`](docs/dispersion_relation.md).
- **Hough mode visualization** — latitudinal profiles/derivatives of a single
  mode, and its full spatial pattern (`h` contour + `(u, v)` quiver) on a
  world map. See [`docs/hough_modes.md`](docs/hough_modes.md).
- **Triadic, four-wave and five-wave dynamics** — coupling coefficients,
  frequency mismatch, the amplitude equations, their time integration
  (Runge-Kutta), energy/efficiency diagnostics, and analytic-period
  diagnostics (`rsw_sphere/dynamics/`).
- Four root drivers, all selecting from the single `wave_sets_default.yaml`
  registry: `run_linear_modes.py` (dispersion relation + per-mode Hough
  plots, `--wave-set KEY`/`--run-all`); `run_dynamics.py`, `run_sweep.py`,
  `run_sweep_sets.py` (integration, IC sweeps, and candidate-mode
  screening -- all sharing one config class,
  `rsw_sphere.dynamics.run_config.RunConfig`, and all `--wave-set KEY`-driven,
  no separate config file). A triad is just the registry's 1-triad case, so the
  same registry covers triads, quartets and quintets. Plus a fifth,
  registry-independent driver, `run_mode_search.py`, for finding what
  *could* go in the registry: given 2 fixed modes (an edge) or 1 (a
  pivot), it lists candidate modes completing a valid triad with them.
  See `docs/code_guide.md`'s "Entry points".

## Repository layout

```
rsw_sphere/                 # the installable package
    hough_harmonics/        # eigenvalue problem, normal modes, inner products
    dynamics/                # WaveSet/TRIAD, integrator, trajectory cache, RunConfig
        periods/              # analytic-period / Hamiltonian diagnostics
    utilities/               # diagnostics compute (pmeasure, periods, precession,
                              # efficiency, functional) + the diagnostic registry
    plotting/                # rendering only -- dispersion, Hough, wave-set maps
docs/                      # thesis PDF, general_guide.md (start here), code_guide.md, per-topic docs
examples/                  # registries + driver configs (see examples/README.md)
outputs/                   # generated figures + cached trajectories (gitignored, reproducible)
tests/                     # pytest suite (structural/exact invariants)
run_linear_modes.py         # dispersion relation + per-mode Hough plots
run_dynamics.py            # integrate a wave set (full + sub-triads), cached
run_sweep.py               # IC sweep (1-2 modes) + diagnostics
run_sweep_sets.py          # loop a diagnostic over candidate-mode variants
run_mode_search.py         # find candidate modes completing a triad with a given edge/pivot
wave_sets_default.yaml     # default WaveSet registry (triads, quartets, quintets) -- all four drivers
pyproject.toml             # pip install -e . / console scripts
```

`rsw_sphere/dynamics/wave_sets.py::WaveSet` generalizes the single-triad
`TRIAD` class (same file's `dynamic_triads.py`) to an arbitrary set of
Hough modes coupled through an arbitrary set of resonant triads —
quartets and quintets are instances of it, replacing six earlier
exploratory `FOUR_WAVES`/`FIVE_WAVES` scripts that were deleted in the
2026 paper §3 rebuild (none of them worked as shipped). See
[`docs/wave_sets.md`](docs/wave_sets.md).

## Installation

Requires Python 3.12. Install the package (editable) so `rsw_sphere` is
importable from anywhere and the console scripts are available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For pytests usage do: 
```bash
pip install -e ".[dev]"    # add [dev] to also get pytest for tests/
```

Dependencies are pinned for API compatibility (`numpy<2.0`, `scipy<1.15`,
`matplotlib`, `pyyaml`, `cartopy`) and declared in `pyproject.toml`
(`requirements.txt` mirrors the same pins for non-editable installs).

If `.venv` was created against a base Python that's since changed (e.g. a
conda base-env upgrade), imports may break with an `undefined symbol`/ABI
error from a compiled dependency (scipy, cartopy). Recreate it:

```bash
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

`pytest tests/ -m "not slow"` runs the short suite (pure unit/parsing
tests, a few seconds) -- for everyday iteration. `pytest tests/` (no
marker filter) runs everything, including real `RK44` integration tests
(~1-2 minutes) -- run this before trusting a change to `rsw_sphere/`
itself, not on every edit.

## Usage

```bash
# dispersion relation + per-mode Hough harmonic plots (no dynamics)
python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
python run_linear_modes.py --run-all

# integrate a registered wave set (a triad on its own, or a quartet/quintet
# plus each sub-triad, each cached and plotted separately)
python run_dynamics.py --wave-set quartet_rossby_kelvin

# sweep 1-2 modes' initial velocities + diagnostics (efficiency, dominant
# frequency/period, low-frequency energy, dynamical phase, efficiency
# variation/spectral-deviation/novelty variation) -- one shared
# vocabulary for 1D (line plots) and 2D (heatmaps); reads the wave set's
# own registry entry
python run_sweep.py --wave-set quartet_rossby_kelvin

# screen candidate modes filling one slot of a registered wave set --
# candidates + diagnostics come from that wave set's own registered
# `alternative_modes.<slot>` block (wave_sets_default.yaml)
python run_sweep_sets.py --wave-set quartet_rossby_kelvin --slot d
```

All four drivers select from the same `wave_sets_default.yaml` registry
(`--wave-set KEY [--specs path.yaml]`, or `--run-all`/a `RunConfig`
sweeping every wave set). `run_dynamics.py`/`run_sweep.py`/`run_sweep_sets.py`
additionally share one config class (`rsw_sphere.dynamics.run_config.RunConfig`),
built from a registry key + `specs_path` (`RunConfig.from_registry_entry`/
`from_wave_set`). See `docs/code_guide.md`'s "Entry points" section.

To add a triad/quartet/quintet, add an entry to `wave_sets_default.yaml`
(equivalent height `h_e`, modes `(m, n, alpha)` with `alpha`: 1 = EIG,
2 = WIG, 3 = RH, initial zonal velocities, and the constituent triad(s))
— no separate config file needed, a triad is just the registry's own
1-triad case. A wave set not yet worth adding to the default registry
can be pointed at with `--specs path/to/other.yaml` instead (same schema,
e.g. `examples/wave_sets_custom.yaml`). See [`examples/`](examples/) for
`run_sweep.py` config variants reproducing specific thesis figures/tables.

`run_sweep.py --wave-set KEY` is a general driver for parameter sweeps
over a registered wave set (dynamical phase, efficiency, dominant
frequency/period, low-frequency energy, or efficiency-variation/spectral-
deviation/novelty variation vs. one or two swept velocities) — a registry
`sweep:` block per sweep instead of a new script per sweep; see
`docs/wave_sets.md` §6.1. Every swept trajectory is cached under
`outputs/trajectories/`, so re-running the same sweep is fast. Each
requested diagnostic writes its own
`outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png/.csv` —
there's no single "the" output for a sweep to override. Copying a
finished PNG into the paper repo's `Figures/` is a separate, manual step
(as with every other figure-generating script in this repository).

The standalone dispersion-relation figure is documented separately in
[`docs/dispersion_relation.md`](docs/dispersion_relation.md) (also runnable
directly as `rsw-dispersion output.png` after `pip install -e .`), and the
Hough mode visualization scripts (latitudinal profile and full spatial
pattern) in [`docs/hough_modes.md`](docs/hough_modes.md) (`rsw-hough-mode
output.png --m 3 --n 7 --alpha 3`).

Resonant-triad tools (batch properties table, energy-integration time
series, efficiency-of-energy-transfer sweeps) are covered by the same
unified drivers used for quartets/quintets (a triad is the degenerate
1-triad case) — see [`docs/triads.md`](docs/triads.md) for the old-to-new
command mapping. Quartets and quintets (coupled multi-triad
configurations) are documented in
[`docs/wave_sets.md`](docs/wave_sets.md) (`rsw-waveset-table` /
`rsw-waveset` / `rsw-waveset-periods` / `rsw-waveset-pmeasure` /
`rsw-waveset-precession`) — including how to test a new
triad/quartet/quintet that isn't in either registry YAML at all, and how
`run_sweep.py` builds on top of these.

## References

- Swarztrauber, P. N. & Kasahara, A. (1985). *The vector harmonic analysis of
  Laplace's tidal equations.* SIAM J. Sci. Stat. Comput., 6, 464–491.
- Longuet-Higgins, M. S. (1968). *The eigenfunctions of Laplace's tidal
  equations over a sphere.* Phil. Trans. R. Soc. A, 262, 511–607.
- Raphaldini, B., Peixoto, P., Teruya, A., Raupp, C. & Bustamante, M. (2022).
  *Precession resonance of Rossby wave triads…* Physics of Fluids.
- Dourado, M. A. (2025). *Nonlinear wave interactions in rotating shallow water
  equations on the sphere.* MSc dissertation, IME-USP.
