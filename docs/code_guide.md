# Code guide

A file-by-file description of how the code is organized and what each part does.
This complements the physics/maths in the thesis and in `dispersion_relation.md`.

## Big picture

The pipeline goes from the **linear tidal-equation eigenvalue problem** →
**normal modes (Hough harmonics)** → **coupling coefficients of a triad (or
four-/five-wave system)** → **time integration of the nonlinear amplitude
equations** → **energy diagnostics and plots**.

```
run_diagnostics.py               driver: dispersion relation + per-mode Hough plots
run_dynamics.py                  driver: triad amplitude-equation integration
  └─ rsw_sphere/
       ├─ hough_harmonics/       the numerical core (eigenproblem, modes, inner products)
       ├─ dynamics/              triad / four-wave / five-wave objects + ODE integrator
       │    └─ periods/          analytic-period / Hamiltonian diagnostics
       └─ plotting/              thin orchestration + plotting layer
            ├─ dispersion_relation*.py   build & plot the dispersion diagram
            ├─ hough_and_derivatives.py  build & plot one normal mode + derivatives
            ├─ hough_spatial_ev.py       build & plot one mode's full (λ,φ) pattern
            └─ dynamic_three_waves.py    set up a triad and run its dynamics
```

The package is `pip install -e .`-installable (`pyproject.toml`); `rsw_sphere`
resolves from anywhere once installed. Two plotting scripts
(`dispersion_relation_fancy.py`, `hough_spatial_ev.py`) additionally carry a
small `sys.path` bootstrap so they also work run directly without installing.

Everything is non-dimensional (see thesis §1.1): the single control parameter is
`gamma = 1/sqrt(eps)`, with Lamb's number `eps = 4 a² Ω² / (g h_e)`. Time is
scaled by `2Ω`, so a "day" in the plots corresponds to `t = 4π` non-dimensional
time units.

## Entry points

Both scripts below read the same `configs.yaml` (`--config`, default
`configs.yaml`), each using only the section relevant to it. See
`examples/` for named config variants reproducing specific thesis figures.

### `run_diagnostics.py`
Loads the config, creates the output directory, then optionally:
1. plots the dispersion relation (`dispersion_relation: true`);
2. plots each triad mode's Hough harmonic + derivatives (`show_mode` per mode).

### `run_dynamics.py`
Loads the config, creates the output directory, then (if
`Dynamics.show_dynamics: true`) integrates the triad dynamics and plots the
energy exchange.

### `configs.yaml`
All user-facing knobs: output path (`OUTPUT_PATH`, default `outputs/figures`),
equivalent height `h_e`, the three modes of the triad `(m, n, alpha)` with
per-mode initial zonal velocity `u` and a `show_mode` flag, and the dynamics
block (`t0`, `tf` in days, step `h`). Convention: `alpha` = 1 → EIG (eastward
inertia-gravity), 2 → WIG (westward inertia-gravity), 3 → RH
(Rossby-Haurwitz). Mode **c** should be the pump mode (`m_c > m_a, m_b`).

## `rsw_sphere/hough_harmonics/` — the numerical core

### `eigenvalues_and_eigenvectors/matrix_system.py`
Builds the truncated matrices `A(m, γ, N)` (symmetric structure) and
`B(m, γ, N)` (antisymmetric structure) of the tidal-equation eigenvalue problem
for zonal wavenumber `m ≠ 0`. Entries come from the coefficients `p`, `q`, `r`
(meridional coupling, Coriolis, gravity). Each matrix is `3N × 3N`.

### `eigenvalues_and_eigenvectors/matrix_m0.py`
The special `m = 0` case: matrices `C` and `D`. (Used by the dispersion relation;
Rossby modes are degenerate/absent at `m = 0`.)

### `eigenvalues_and_eigenvectors/eigenvectors.py`
The heart of the mode construction:
- `Hough_coef_A/B` — diagonalize `A`/`B`, sort eigenvalues, and select the one
  for the requested `(m, n, alpha)` via the meridional index `l = n − m`
  (index map: WIG below, RH middle, EIG above).
- `Pmn_and_derivative`, `Spherical_vector_harmonics` — normalized associated
  Legendre functions, their φ-derivatives, and the vector spherical harmonics
  `y_1, y_2, y_3` (plus derivatives).
- `symetry(m, n, alpha)` — parity about the equator (RH symmetric when `m−n`
  odd; gravity modes symmetric when `m−n` even).
- `Hough_harmonic(...)` — assembles the mode fields `U, V, Z` (= u, v, h) and
  their derivatives `DU, DV, DZ` at a given latitude φ, returning also the
  eigenvalue (dimensionless frequency).

`eigenvectors_m0.py`, `eigenvector_0.py` — analogous helpers for the `m = 0`
modes.

### `normalization.py`
- `norm_Hough(m, n, alpha, gamma, N, deg)` — evaluates a mode on a
  Gauss-Legendre latitude grid and normalizes it to unit energy inner product,
  returning the normalized `(U, V, Z, DU, DV, DZ)`, the quadrature points, the
  norm, and the eigenvalue. This is the canonical way to obtain a usable mode.
- `norm_component(u)` — norm of just the zonal-velocity component, used to
  convert a desired physical zonal velocity (m/s) into a mode amplitude.

`normalization_m0.py` — the `m = 0` normalization.

### `inner_product.py`
- `inner_product(...)` — the coupling-coefficient inner product (the projection
  of the bilinear term `B` onto a mode; `conj` flag distinguishes projection onto
  a/b vs. onto c). This is what makes a triad interact or not.
- `S_abc(...)` — the integral appearing in the cubic (total) energy.

## `rsw_sphere/dynamics/` — triad / four-wave / five-wave dynamics

### `dynamic_triads.py`
- `class TRIAD` — given `gamma` and the three modes, builds the normalized modes,
  frequencies, the three coupling coefficients (`coef_ABC`, `coef_BAC`,
  `coef_CAB`), the frequency `mismatch`, and the cubic-energy integral `Sabc`.
  `TRIAD.f(AMP)` is the RHS of the three-wave amplitude ODE system.
- `RK33` — Runge-Kutta time integrator for the amplitude equations.
- `Energy_0` — quadratic (E²) and cubic (E³) energy of an amplitude state.
- `Triad_dynamics` — integrates the triad, computes per-mode kinetic energy,
  efficiency (max−min energy), total-energy conservation check, and plots the
  time series.
- `Triad_Precession` — sweeps initial amplitudes over a grid to map the energy
  transfer efficiency (precession-resonance style diagnostics), with a filled
  contour plot.
- `eff_tri`, `period_Fourier` — efficiency-vs-velocity curves and FFT-based
  dominant-period analysis. (Some of these are exploratory helpers.)

### `five_waves.py`
`class FIVE_WAVES` — the sole five-wave implementation (extends the triad
idea to five coupled modes). Not a duplicate of anything else here.

### `four_waves_2.py`, `four_waves_79.py`, `four_waves_pump.py`, `four_waves_basic.py`, `four_waves_rk4_driver.py`
**Five near-duplicate exploratory implementations** of a four-wave `FOUR_WAVES`
class/experiment, kept side by side pending consolidation into one canonical
module (none has been designated canonical yet — ask the user before picking
one). Roughly:
- `four_waves_basic.py` — a standalone `FOUR_WAVES` with no dependency on
  `dynamic_triads.py`/`periods/` (only `hough_harmonics`).
- `four_waves_2.py`, `four_waves_79.py`, `four_waves_pump.py` — variants that
  additionally pull in `periods/period_harris.py` (`PERIOD`, `Amp_change`) and
  `dynamic_triads.py` (`TRIAD`, `Triad_dynamics`).
- `four_waves_rk4_driver.py` — an RK4 integration driver (`FF`, `RK44`) built
  on top of `four_waves_pump.py`'s `FOUR_WAVES`.

### `periods/` — analytic-period diagnostics
Consumes `dynamic_triads.py`'s `TRIAD` (coupling coefficients, mismatch) to
compute the resonant-triad energy-exchange period **analytically** (Jacobi
elliptic integrals), as opposed to reading it off a numerically integrated
time series. Nested inside `dynamics/` because it only makes sense applied to
a `TRIAD` instance, not as a standalone concern.
- `period.py` — Bustamante-style formulation (`Hamiltonian`, `rho`, `nu`, `PERIOD`).
- `period_harris.py` — Harris-style formulation (`UU`, `J`, `E`, `Hamiltonian`,
  `P`, `PERIOD`, `Amp_change`); imports `Energy_0` from `dynamic_triads.py`.
- `period_both.py` — sets up a triad and compares both analytic periods
  (`p_bus` from `period.py`, `p_har` from `period_harris.py`).

## `rsw_sphere/plotting/` — orchestration + plotting

### `dispersion_relation.py`
`dispersion_relation(h_e, path)` — assembles `A`/`B` (and `C`/`D` at `m = 0`) over
`m = 0..10`, sorts the eigenvalues, and plots the combined dispersion diagram of
EIG / WIG / RH families (with Kelvin and mixed Rossby-gravity highlighted) plus a
period axis. Called by `run_diagnostics.py`.

### `dispersion_relation_fancy.py`
Publication-quality variant of the dispersion plot, documented in detail in
`dispersion_relation.md` (adds `ω ≈ c k` and `2Ω` reference curves and the
Matsuno β-plane correspondence). Has its own `argparse` CLI (`main()`), also
installed as the `rsw-dispersion` console script.

### `hough_and_derivatives.py`
`hough_and_derivatives(m, n, alpha, h_e, path)` — builds one normalized mode and
saves two figures: the `(u, v, h)` latitudinal structure and its derivatives.
No CLI; call directly or edit the `if __name__ == "__main__":` example.

### `hough_spatial_ev.py`
`hough_spatial_ev(m, n, alpha, h_e, ..., path)` — reconstructs the full 2D
`(λ, φ)` spatial pattern of a mode (`h` contour + `(u,v)` quiver) on a
PlateCarree cartopy map. Has its own `argparse` CLI (`main()`), also installed
as the `rsw-hough-mode` console script. Documented in detail, including two
worked-around matplotlib/cartopy rendering bugs, in `hough_modes.md`.

### `dynamic_three_waves.py`
`triad_evolution(...)` — the glue for a triad experiment: converts the config's
physical velocities to amplitudes via `norm_component`, constructs a `TRIAD`,
prints coupling coefficients / frequencies / mismatch / energy-conservation
constraint, and calls `Triad_dynamics` to integrate and plot.

## Outputs

Figures are written under `OUTPUT_PATH` from the config (default
`outputs/figures/`): `dispersion_relation.png`, `dispersion_relation_fancy.png`,
`dynamics.png`, and per-mode `<alpha>-<m>-<n>/Hough_harmonic_<alpha>-<m>-<n>.png`
+ `derivatives_<alpha>-<m>-<n>.png` + `Hough_spatial_<alpha>-<m>-<n>.png`
(e.g. `RH-1-2/Hough_harmonic_RH-1-2.png`) — folder and filenames both encode
the mode so files stay identifiable if moved out of their folder. `outputs/`
is gitignored and regenerated by running `run_diagnostics.py` /
`run_dynamics.py`. Standalone dispersion/Hough scripts write wherever their
`path` argument points.

## Conventions & gotchas

- **Units:** all internal quantities are non-dimensional. Multiply time by `4π`
  to get days; frequencies scale by `2Ω`.
- **Mode selection** relies on sorting eigenvalues and indexing by `l = n − m`;
  the ordering (WIG / RH / EIG blocks) is assumed stable — see the index map in
  `eigenvectors.py` and the notes in `dispersion_relation.md` §2.4.
- **Pump mode** is expected to be mode c (`m_c` largest), per the config comment.
- **Dependency pinning** (`numpy<2.0`, `scipy<1.15`) matters: the Legendre /
  eigenvalue APIs used here changed in later releases.
- **No Python identifiers were renamed** in the 2026-07 package refactor
  (`TRIAD`, `FOUR_WAVES`, `matriz_A`, etc. are unchanged) — only directories,
  filenames, and import paths moved to the `rsw_sphere` package layout.
