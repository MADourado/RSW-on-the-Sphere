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
  world map. See [`README_hough_modes.md`](README_hough_modes.md).
- **Triadic interactions** — coupling coefficients, frequency mismatch, the
  three-wave amplitude equations, their time integration (Runge-Kutta), and
  energy/efficiency diagnostics.
- A YAML-configured driver (`main.py` + `configs.yaml`) to reproduce the
  single-triad experiments and mode plots.


## Installation

Requires Python 3.12. Dependencies are pinned for API compatibility:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(`numpy<2.0`, `scipy<1.15`, `matplotlib`, `pyyaml`, `cartopy`.)

## Usage

Run the driver against the default configuration:

```bash
python main.py --config configs.yaml
```

This reads the equivalent height and the triad definition from `configs.yaml`,
then (depending on the flags in the config) plots the dispersion relation, the
Hough harmonics of each mode, and integrates the triad dynamics — writing the
figures to `OUTPUT_PATH` (default `Testes_Marco/figures/`).

Edit `configs.yaml` to change the equivalent height `h_e`, the three modes
`(m, n, alpha)` of the triad (with `alpha`: 1 = EIG, 2 = WIG, 3 = RH), their
initial zonal velocities, and the time-integration parameters.

The standalone dispersion-relation figure is documented separately in
[`README_dispersion.md`](README_dispersion.md), and the Hough mode
visualization scripts (latitudinal profile and full spatial pattern) in
[`README_hough_modes.md`](README_hough_modes.md).

## References

- Swarztrauber, P. N. & Kasahara, A. (1985). *The vector harmonic analysis of
  Laplace's tidal equations.* SIAM J. Sci. Stat. Comput., 6, 464–491.
- Longuet-Higgins, M. S. (1968). *The eigenfunctions of Laplace's tidal
  equations over a sphere.* Phil. Trans. R. Soc. A, 262, 511–607.
- Raphaldini, B., Peixoto, P., Teruya, A., Raupp, C. & Bustamante, M. (2022).
  *Precession resonance of Rossby wave triads…* Physics of Fluids.
- Dourado, M. A. (2025). *Nonlinear wave interactions in rotating shallow water
  equations on the sphere.* MSc dissertation, IME-USP.
