# RSW on the Sphere — Nonlinear wave interactions

Code accompanying the MSc dissertation

> **Nonlinear wave interactions in rotating shallow water equations on the sphere**
> Marco Antonio Dourado — Institute of Mathematics, Statistics and Computer
> Science, University of São Paulo (IME-USP), 2025.
> Advisor: Prof. Dr. Pedro da Silva Peixoto · Co-advisor: Prof. Dr. Breno Raphaldini.

**Author:** Marco Antonio Dourado.

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
  five-wave configurations) — relevant to whether filtering gravity waves out of
  forecast models is justified.

## Current content

The code so far covers the material of Chapters 1–2 of the thesis (the triadic
part) plus the plotting/driver infrastructure:

- **Hough harmonics & dispersion relation** — assembly and diagonalization of the
  tidal-equation eigenvalue problem, the normalized normal-mode fields
  `(u, v, h)` and their latitudinal derivatives, and the dispersion diagram of
  the three wave families.
- **Hough mode visualization** — latitudinal profiles/derivatives of a single
  mode, and its full spatial pattern (`h` contour + `(u, v)` quiver) on a
  world map. See [`docs/hough_modes.md`](docs/hough_modes.md).
- **Triadic, four-wave and five-wave dynamics** — coupling coefficients,
  frequency mismatch, the amplitude equations, their time integration
  (Runge-Kutta), energy/efficiency diagnostics, and analytic-period
  diagnostics (`rsw_sphere/dynamics/`).
- Two YAML-configured drivers reading the same `configs.yaml`:
  `run_diagnostics.py` (dispersion relation + per-mode Hough plots, no
  dynamics) and `run_dynamics.py` (triad amplitude-equation integration).

## Repository layout

```
rsw_sphere/                 # the installable package
    hough_harmonics/        # eigenvalue problem, normal modes, inner products
    dynamics/                # triad, four-wave and five-wave amplitude dynamics
        periods/              # analytic-period / Hamiltonian diagnostics
    plotting/                # dispersion relation and Hough mode plots
docs/                      # thesis PDF, code guide, per-topic documentation
examples/                  # named configs.yaml variants reproducing thesis figures
outputs/                   # generated figures (gitignored, reproducible)
run_diagnostics.py         # dispersion relation + per-mode Hough plots
run_dynamics.py            # triad amplitude-equation integration
configs.yaml                # shared config for both drivers above
pyproject.toml             # pip install -e . / console scripts
```

`rsw_sphere/dynamics/` currently holds several exploratory `FOUR_WAVES`
implementations side by side (`four_waves_2.py`, `four_waves_79.py`,
`four_waves_pump.py`, `four_waves_basic.py`, `four_waves_rk4_driver.py`) —
consolidating them into one canonical module is future work, not done yet.

## Installation

Requires Python 3.12. Install the package (editable) so `rsw_sphere` is
importable from anywhere and the console scripts are available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Dependencies are pinned for API compatibility (`numpy<2.0`, `scipy<1.15`,
`matplotlib`, `pyyaml`, `cartopy`) and declared in both `pyproject.toml` and
`requirements.txt`.

## Usage

Both drivers read the same `configs.yaml` and write into `OUTPUT_PATH`
(default `outputs/figures/`), each acting only on the section relevant to it:

```bash
# dispersion relation + per-mode Hough harmonic plots (no dynamics)
python run_diagnostics.py --config configs.yaml

# integrate the triad amplitude equations and plot the energy exchange
python run_dynamics.py --config configs.yaml
```

`run_diagnostics.py` plots the dispersion relation (if `dispersion_relation:
true`) and each triad mode's Hough harmonic + derivatives (per-mode
`show_mode: true`). `run_dynamics.py` integrates the triad dynamics (if
`Dynamics.show_dynamics: true`) and plots the energy exchange time series.

Edit `configs.yaml` to change the equivalent height `h_e`, the three modes
`(m, n, alpha)` of the triad (with `alpha`: 1 = EIG, 2 = WIG, 3 = RH), their
initial zonal velocities, and the time-integration parameters. See
[`examples/`](examples/) for named config variants reproducing specific
thesis figures/tables.

The standalone dispersion-relation figure is documented separately in
[`docs/dispersion_relation.md`](docs/dispersion_relation.md) (also runnable
directly as `rsw-dispersion output.png` after `pip install -e .`), and the
Hough mode visualization scripts (latitudinal profile and full spatial
pattern) in [`docs/hough_modes.md`](docs/hough_modes.md) (`rsw-hough-mode
output.png --m 3 --n 7 --alpha 3`).

## References

- Swarztrauber, P. N. & Kasahara, A. (1985). *The vector harmonic analysis of
  Laplace's tidal equations.* SIAM J. Sci. Stat. Comput., 6, 464–491.
- Longuet-Higgins, M. S. (1968). *The eigenfunctions of Laplace's tidal
  equations over a sphere.* Phil. Trans. R. Soc. A, 262, 511–607.
- Raphaldini, B., Peixoto, P., Teruya, A., Raupp, C. & Bustamante, M. (2022).
  *Precession resonance of Rossby wave triads…* Physics of Fluids.
- Dourado, M. A. (2025). *Nonlinear wave interactions in rotating shallow water
  equations on the sphere.* MSc dissertation, IME-USP.
