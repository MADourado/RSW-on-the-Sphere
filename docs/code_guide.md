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
run_dynamics.py                  driver: triad/wave-set amplitude-equation integration
run_sweep.py                     driver: YAML-configured parameter sweep + figure
                                    (incl. multi-diagnostic quartet panels, diagnostic: quartet_diagnostics)
  └─ rsw_sphere/
       ├─ hough_harmonics/       the numerical core (eigenproblem, modes, inner products)
       ├─ dynamics/              triad / wave-set objects, ODE integrator, trajectory cache
       │    └─ periods/          analytic-period / Hamiltonian diagnostics
       └─ plotting/              thin orchestration + plotting layer
            ├─ dispersion_relation*.py   build & plot the dispersion diagram
            ├─ hough_and_derivatives.py  build & plot one normal mode + derivatives
            ├─ hough_spatial_ev.py       build & plot one mode's full (λ,φ) pattern
            ├─ dynamic_three_waves.py    set up a triad and run its dynamics
            └─ wave_set_*.py             registry-driven quartet/quintet tools (docs/wave_sets.md)
```

No `postproc/` folder: an earlier reorg plan called for one (bespoke,
paper-specific figure *assembly*, distinct from the general-analysis
scripts above), but the one candidate use case (combining Quartet A's and
Quartet B's own precession-frequency panels into JFM-template.tex's
`fig: precession_frequency`) turned out not to need it -- that figure is
two independent PNGs combined by LaTeX `subfigure`, not a Python
composite, and each source script (`run_sweep.py` + a config, or
`examples/borrowed_topology_precession_figure.py`) already writes its own
complete, publication-ready PNG on its own. A `postproc/` wrapper that
just re-called those same plotting functions added no assembly work,
only a second code path to keep in sync (2026-08-25 review). Create the
folder if a figure genuinely needs combining two already-computed,
independently-generated outputs into one new image neither source script
can produce alone -- not for "the paper needs this specific PNG," which
the generating script's own config/title/crop already covers.

The package is `pip install -e .`-installable (`pyproject.toml`); `rsw_sphere`
resolves from anywhere once installed. Two plotting scripts
(`dispersion_relation_fancy.py`, `hough_spatial_ev.py`) additionally carry a
small `sys.path` bootstrap so they also work run directly without installing.

Everything is non-dimensional (see thesis §1.1): the single control parameter is
`gamma = 1/sqrt(eps)`, with Lamb's number `eps = 4 a² Ω² / (g h_e)`. Time is
scaled by `2Ω`, so a "day" in the plots corresponds to `t = 4π` non-dimensional
time units.

## Entry points

`run_diagnostics.py`/`run_dynamics.py` read the same `configs.yaml`
(`--config`, default `configs.yaml`), each using only the section relevant
to it. See `examples/` for named config variants reproducing specific
thesis figures. `run_sweep.py` is separate (YAML-per-invocation via
`--config`, no shared `configs.yaml`).

### `run_diagnostics.py`
Loads the config, creates the output directory, then optionally:
1. plots the dispersion relation (`dispersion_relation: true`);
2. plots each triad mode's Hough harmonic + derivatives (`show_mode` per
   mode) -- or, with `--wave-set KEY [--specs path.yaml]`, every mode of
   that registered quartet/quintet (`rsw_sphere.dynamics.wave_set_specs`)
   instead of `configs.yaml`'s `Triad:` block. A plain triad is the
   degenerate 3-modes/1-triad case of a wave set (`wave_sets.py`'s own
   module docstring), so `--wave-set` is a strict addition: without it,
   the original `configs.yaml`-only path is unchanged.

### `run_dynamics.py`
Loads the config, creates the output directory, then (if
`Dynamics.show_dynamics: true`) integrates the triad dynamics and plots the
energy exchange -- or, with `--wave-set KEY [--specs path.yaml]`, integrates
that registered quartet/quintet instead
(`rsw_sphere.plotting.wave_set_dynamics.wave_set_energy_evolution_from_spec`),
sourcing `tf`/`h` from the wave set's own registry `settings` rather than
`configs.yaml`'s `Dynamics:` block. Same strict-addition rule as above.

### `run_sweep.py`
General sweep driver: reads a YAML naming a registered wave set, a
diagnostic (`precession` / `p_measure` / `quartet_diagnostics` /
`efficiency`), and a swept parameter/range, then produces a cached
`.npz` + figure. Dispatches over existing sweep functions
(`rsw_sphere.plotting.wave_set_precession`, `wave_set_pmeasure`,
`triad_efficiency`) rather than reimplementing their math -- replaces the
need for a new bespoke `examples/*.py` script per sweep combination; the
config goes in `examples/` instead (e.g. `examples/sweep_quartet_a_rh36.yaml`).
`quartet_diagnostics` is the "several diagnostics from one shared sweep
pass" mode (`sweep.diagnostics: [p_measure, filtering_error]`, a switch),
computed by `wave_set_pmeasure.wave_set_diagnostics_sweep` -- swept/target
modes default to the wave set's own "private" modes
(`WaveSetSpec.shared_and_private_modes()`), so an ordinary quartet needs
no config beyond `wave_set` + `output`. See `docs/wave_sets.md` §7.1 for
the full schema and `run_sweep.py`'s own module docstring.

    python run_sweep.py --config examples/sweep_quartet_a_rh36.yaml
    python run_sweep.py --config examples/sweep_quartet_gravity_kelvin_diagnostics.yaml

### `configs.yaml`
All user-facing knobs: output path (`OUTPUT_PATH`, default `outputs/figures`),
equivalent height `h_e`, the three modes of the triad `(m, n, alpha)` with
per-mode initial zonal velocity `u` and a `show_mode` flag, and the dynamics
block (`t0`, `tf` in days, step `h`). Convention: `alpha` = 1 → EIG (eastward
inertia-gravity), 2 → WIG (westward inertia-gravity), 3 → RH
(Rossby-Haurwitz). Mode **c** should be the pump mode (`m_c > m_a, m_b`).
Only consulted when `run_diagnostics.py`/`run_dynamics.py` are run
*without* `--wave-set`.

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
- `Triad_Precession` — dead code (commented out since the §2.2 rebuild).
  Superseded by `rsw_sphere/plotting/triad_efficiency.py`.
- `eff_tri`, `period_Fourier` — efficiency-vs-velocity curves and FFT-based
  dominant-period analysis. (Some of these are exploratory helpers.)

### `wave_sets.py` — quartets, quintets (generalized `TRIAD`)
`class WaveSet` — an arbitrary set of Hough modes coupled through an
arbitrary set of resonant triads; a quartet is 4 modes/2 triads sharing
one edge, a quintet is 5 modes/3 triads, a plain triad is the degenerate
3-modes/1-triad case. Replaces six earlier near-duplicate exploratory
scripts (`five_waves.py`'s `FIVE_WAVES`; `four_waves_2.py`,
`four_waves_79.py`, `four_waves_pump.py`, `four_waves_basic.py`,
`four_waves_rk4_driver.py`'s various `FOUR_WAVES` variants), all deleted
in the 2026 paper §3 rebuild — none worked as shipped (all called a
`Triad_dynamics(..., p=...)` kwarg that never existed). `TRIAD` itself
(`dynamic_triads.py`) is untouched and is `WaveSet`'s independent
reference implementation, proven equivalent under a mode-relabeling
permutation (`WaveSet`'s own module docstring;
`examples/check_wave_set_physics.py` checks C1-C3). See
[`../docs/wave_sets.md`](../docs/wave_sets.md) for the plotting/registry
layer built on top and how to test a configuration not yet registered.

### `trajectory_cache.py` — raw trajectory caching
`run_and_cache(ws, A0, t_f, h, velocities=None, output_root="outputs/trajectories", label=None)`
caches a `WaveSet`/`RK33` run's raw `Y(t)` solution itself (not just a
derived summary) under `outputs/trajectories/<topology>/`, `topology`
auto-derived from `ws.n_modes` (`triads`/`quartets`/`quintets`) — the one
piece with no prior analogue in this repo (every other cache here,
`rsw_sphere.plotting.sweeps`/the `wave_set_*.py` sweep functions' own
`cache_path`, stores summary arrays only). The cache filename is built
from every mode's own initial condition (`ic_label(ws.modes, velocities)`,
canonically sorted), not a caller-supplied key, so the same physical
configuration lands in the same cache entry regardless of which script
built it. Used by `rsw_sphere.plotting.wave_set_precession`; any new
script computing a `WaveSet` trajectory that might be revisited should go
through this rather than calling `RK33` directly. See `docs/wave_sets.md`
§7.2.

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

### `wave_set_*.py` — quartet/quintet registry tools
`wave_set_table.py`/`wave_set_dynamics.py`/`wave_set_periods.py`/
`wave_set_pmeasure.py`/`wave_set_precession.py`: the `WaveSet` (above)
analogue of this section's single-triad tools, each registry-driven
(`rsw_sphere.dynamics.wave_set_specs`) and each with its own
`rsw-waveset-*` console script. Fully documented, one section per script,
in [`../docs/wave_sets.md`](../docs/wave_sets.md) rather than duplicated
here — that file is the authoritative reference for this battery of tools.

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

`outputs/trajectories/<topology>/<label>_<hash8>.npz` (`topology`:
`triads`/`quartets`/`quintets`) holds raw cached `WaveSet` trajectories
(`rsw_sphere.dynamics.trajectory_cache`, above) — separate from
`outputs/figures/`, since these are reusable intermediate data (any
diagnostic can be re-derived from one without re-integrating, and the
same physical configuration reuses one entry across scripts), not
figures. Also gitignored.

`examples/legacy/` holds scripts superseded by
`paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-codebase-reorg-2026-08-25.md`'s
reorganization (verified to reproduce their own prior output before the
move) — a holding area pending a later, separate deletion pass, not a
statement that the code there still reflects current practice.

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
