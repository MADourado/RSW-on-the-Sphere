# Code guide

A file-by-file description of how the code is organized and what each part does.
This complements the physics/maths in the thesis and in `dispersion_relation.md`.

## Big picture

The pipeline goes from the **linear tidal-equation eigenvalue problem** →
**normal modes (Hough harmonics)** → **coupling coefficients of a triad (or
four-/five-wave system)** → **time integration of the nonlinear amplitude
equations** → **energy diagnostics and plots**.

```
run_linear_modes.py         driver: dispersion relation + per-mode Hough plots
run_dynamics.py              driver: integrate a wave set (full + sub-triads), cached
run_sweep.py                 driver: IC sweep (1 or 2 modes) + diagnostics, calls run_dynamics per point
run_sweep_sets.py            driver: loop a diagnostic over candidate-mode variants
  └─ rsw_sphere/
       ├─ hough_harmonics/   the numerical core (eigenproblem, modes, inner products)
       ├─ dynamics/          WaveSet/TRIAD, integrator, trajectory cache, RunConfig
       │    └─ periods/      analytic-period / Hamiltonian diagnostics
       ├─ utilities/         compute: diagnostics (pmeasure, periods, precession,
       │                     efficiency, functional), the diagnostic registry
       └─ plotting/          rendering only -- every plot_* fn takes already-computed
                              arrays, returns (fig, ax, ...), never integrates
```

`rsw_sphere/utilities/` vs. `rsw_sphere/plotting/`: a function that returns
numbers goes in `utilities/`; a function that only draws (given
already-computed arrays) goes in `plotting/`. `utilities/registry.py` maps
diagnostic name -> compute engine + plot function, used by
`run_sweep.py`/`run_sweep_sets.py` -- adding a diagnostic is one registry
entry, not a new sweep loop.

No `postproc/` folder: an earlier reorg plan called for one (bespoke,
paper-specific figure *assembly*, distinct from the general-analysis
scripts above), but the one candidate use case (combining Quartet A's and
Quartet B's own precession-frequency panels into JFM-template.tex's
`fig: precession_frequency`) turned out not to need it -- that figure is
two independent PNGs combined by LaTeX `subfigure`, not a Python
composite, and each source script (`run_sweep.py` + a config, or
`examples_legacy/raphaldini2022_compare/borrowed_topology_precession_figure.py`) already writes its own
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

All four drivers select from the same `wave_sets_default.yaml` registry
(a triad is just its own 1-triad case, so there's no separate plain-Triad
config anymore) via `--wave-set KEY [--specs path.yaml]`.
`run_linear_modes.py` needs nothing beyond the `WaveSetSpec` itself (no
time integration). `run_dynamics.py`/`run_sweep.py` build a
`rsw_sphere.dynamics.run_config.RunConfig` (`tf_days`/`h`/`output_root`/
`plot`/`parallel`; a `sweep:` block for `run_sweep.py`) straight from the
registry entry (`RunConfig.from_registry_entry`/`from_wave_set`);
`run_sweep.py` alone also accepts a standalone `RunConfig` YAML
(`--config path.yaml`, `RunConfig.from_yaml`) for an ad-hoc sweep not
worth registering. `run_sweep_sets.py` reads its own, differently-shaped
YAML directly (`--config`, required). See `examples/` for configs.

### `run_linear_modes.py` (renamed from `run_diagnostics.py`)
Creates the output directory (`<output-root>/figures/`), then optionally:
1. plots the dispersion relation (default on; `--no-dispersion-relation`
   to skip) for the wave set's own `h_e` (or every distinct `h_e` across
   the registry, with `--run-all`);
2. plots every mode's Hough harmonic + derivatives for `--wave-set KEY
   [--specs path.yaml]`, or `--run-all [--specs path.yaml]` for every mode
   of every registered wave set (shared modes plotted once, deduplicated
   by `mode_tag`). Per-mode plots go under `<output-root>/figures/linear/<mode_tag>/`.

    python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
    python run_linear_modes.py --run-all

### `run_dynamics.py`
Given a `RunConfig`, integrates the full wave set plus every constituent
triad separately (`WaveSetSpec.has_subtriads()`) -- each integration its
own `rsw_sphere.dynamics.trajectory_cache.run_and_cache` call (trajectory
cache path: `outputs/trajectories/<topology>/<label>_<hash8>.npz`), each
plotted separately (`rsw_sphere.plotting.energy_evolution.plot_energy_evolution`,
unless `config.plot=False`) under `outputs/figures/dynamics/<wave_set_key>/<unit>.png`
(see "Outputs" below for `<unit>`'s own naming). Parallel across units by
default (`ProcessPoolExecutor`, `config.max_workers` or half the CPU
count). `run_dynamics(config) -> dict` is directly importable --
`run_sweep.py` calls it per grid point. `--diagnostics` additionally
prints every pairwise diagnostic (`rsw_sphere.utilities.pmeasure.pairwise_target_diagnostics`:
p_measure, filtering_error, fmax, frequency_shift, novelty_period) for
every target mode against each sub-triad that contains it, and writes
the novelty-frequency spectrum figures (`rsw_sphere.plotting.novelty_panel`).

    python run_dynamics.py --wave-set quartet_rossby_kelvin

### `run_sweep.py`
Sweeps 1 or 2 modes' velocities (`config.sweep.axes`). Calls
`run_dynamics` per grid point (cache + optional per-point figure, per
`sweep.save_point_figures`), then computes/plots every diagnostic in
`sweep.diagnostics`: 1D supports `precession` only
(`rsw_sphere.utilities.precession.precession_frequency_efficiency`,
natively 1D); 2D supports `p_measure`/`filtering_error`/`frequency_shift`/
`fmax`/`efficiency`/`low_frequency_energy` via `rsw_sphere.utilities.registry.sweep_2d`.
Swept/target modes for a 2D sweep default to the wave set's own "private"
modes (`WaveSetSpec.shared_and_private_modes()`). `--wave-set KEY` reads
`sweep`/`tf_days`/`h`/`plot`/`output`/`target_mode`/`plot_triad` straight
from that wave set's own `wave_sets_default.yaml` entry -- `--config
path.yaml` (a standalone `RunConfig` YAML) is only for an ad-hoc sweep
not worth registering.

    python run_sweep.py --wave-set quartet_rossby_kelvin
    python run_sweep.py --wave-set quartet_rh_preference

### `run_sweep_sets.py`
Loops a diagnostic over a LIST of wave-set variants -- substituting which
mode fills one slot (`candidate_slot`), not sweeping a velocity.
`candidates_from: {max_n}` infers the required zonal wavenumber from
`candidate_slot`'s own triad selection rule (`m_sum = m_p + m_q`); `target_mode`
is the (usually different, already-driven) mode whose diagnostic value is
reported. One point per candidate (own registered velocities, unless
`candidate_velocity` overrides the candidate slot's own velocity -- needed
for `frequency_shift`, which a passively/weakly excited candidate won't
resolvably shift), parallel across candidates, writes a CSV (`table:`).
Generalizes the hand-rolled catalogues in
`examples_legacy/gate_i2_map_extension.py` and similar.

    python run_sweep_sets.py --config examples/candidates_quartet_rossby_kelvin.yaml

### `run_mode_search.py`
Registry-independent: given a fixed edge (`--edge MODE_P MODE_Q`) or pivot
(`--pivot MODE`, `m,n,alpha` each), lists candidate modes completing a
valid triad -- for scouting a new `wave_sets_default.yaml` entry or a
`run_sweep_sets.py` `candidates:` block without hand-deriving the
selection rule. Wraps `rsw_sphere.utilities.mode_search`: cheap by
default (wavenumber `m_sum = m_p + m_q` + meridional-symmetry parity,
both O(1)); `--coupling` also computes actual TRIAD coefficients
(slower). `--edge` covers triads/quartets/star-quintets (one edge shared
by every constituent triad); `--pivot` covers an "hourglass" quintet
(two independent triads sharing a single mode, not an edge) -- a valid
`WaveSet` topology not currently used by any registered wave set.

    python run_mode_search.py --edge 4,5,3 3,4,3 --max-n 9 --alphas 1,2
    python run_mode_search.py --edge 4,5,3 3,4,3 --max-n 9 --alphas 1 --coupling --csv outputs/_scratch/candidates.csv
    python run_mode_search.py --pivot 4,5,3 --max-n 6 --alphas 3

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
- `RK44` — Runge-Kutta time integrator for the amplitude equations.
- `Energy_0` — quadratic (E²) and cubic (E³) energy of an amplitude state.
- `Triad_dynamics` — integrates the triad, computes per-mode kinetic energy,
  efficiency (max−min energy), total-energy conservation check, and plots the
  time series.
- `Triad_Precession` — dead code (commented out since the §2.2 rebuild).
  Superseded by `run_sweep.py`'s efficiency diagnostic
  (`rsw_sphere/utilities/efficiency.py`).
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
`rsw_sphere/utilities/check_wave_set_physics.py` checks C1-C3). See
[`../docs/wave_sets.md`](../docs/wave_sets.md) for the plotting/registry
layer built on top and how to test a configuration not yet registered.

### `trajectory_cache.py` — raw trajectory caching
`run_and_cache(ws, A0, t_f, h, velocities=None, output_root="outputs/trajectories", label=None)`
caches a `WaveSet`/`RK44` run's raw `Y(t)` solution itself (not just a
derived summary) under `outputs/trajectories/<topology>/`, `topology`
auto-derived from `ws.n_modes` (`triads`/`quartets`/`quintets`) — the one
piece with no prior analogue in this repo (every other cache here,
`rsw_sphere.plotting.sweeps`/the `wave_set_*.py` sweep functions' own
`cache_path`, stores summary arrays only). The cache filename is built
from every mode's own initial condition (`ic_label(ws.modes, velocities)`,
canonically sorted), not a caller-supplied key, so the same physical
configuration lands in the same cache entry regardless of which script
built it. Used by `run_dynamics.py` and `rsw_sphere.utilities.precession`;
any new script computing a `WaveSet` trajectory that might be revisited
should go through this rather than calling `RK44` directly.

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
period axis. Called by `run_linear_modes.py`.

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

### `wave_set_table.py`, `energy_evolution.py`, `period_panel.py`, `pmeasure_map.py`, `precession_plot.py`, `functional_map.py`
Rendering for quartet/quintet ("wave set") examples, each with its own
`rsw-waveset-*` console script -- the `WaveSet` analogue of this section's
single-triad tools. Compute lives in `rsw_sphere/utilities/` (below), not
here. Fully documented in [`../docs/wave_sets.md`](../docs/wave_sets.md).

## `rsw_sphere/utilities/` — diagnostics compute

`pmeasure.py` (P-measure, filtering error, frequency shift -- pairwise:
full wave set vs. one constituent triad), `functional.py` (efficiency,
low-frequency energy -- full wave set alone), `periods.py`
(`dominant_periods`, `low_frequency_power`), `precession.py`
(`precession_frequency_efficiency`), `efficiency.py`
(`wave_set_efficiency`, drift-gated), `registry.py` (diagnostic name ->
compute engine + plot function, used by `run_sweep.py`/`run_sweep_sets.py`).
Every `*_sweep`/`*_diagnostics_sweep` function here takes a `cache_path`
(`.npz`, cache-if-absent/load-if-present) and reads
`rsw_sphere.utilities.efficiency.default_velocity_range` for its own
default sweep range.

## Outputs

Figures are written under `OUTPUT_PATH` from the config (default
`outputs/figures/`): `dispersion_relation.png`, `dispersion_relation_fancy.png`,
`dynamics.png`, and per-mode `linear/<alpha>-<m>-<n>/Hough_harmonic_<alpha>-<m>-<n>.png`
+ `derivatives_<alpha>-<m>-<n>.png` + `Hough_spatial_<alpha>-<m>-<n>.png`
(e.g. `linear/RH-1-2/Hough_harmonic_RH-1-2.png`) — folder and filenames both
encode the mode so files stay identifiable if moved out of their folder.
`run_dynamics.py`/`run_sweep.py` write under `<output_root>/trajectories/`
and `<output_root>/figures/dynamics/`/`<output_root>/figures/wave_sets/`
instead. `outputs/` is gitignored and regenerated by running the drivers.
Standalone dispersion/Hough scripts write wherever their `path` argument
points.

`outputs/trajectories/<topology>/<label>_<hash8>.npz` (`topology`:
`triads`/`quartets`/`quintets`) holds raw cached `WaveSet` trajectories
(`rsw_sphere.dynamics.trajectory_cache`, above) — separate from
`outputs/figures/`, since these are reusable intermediate data (any
diagnostic can be re-derived from one without re-integrating, and the
same physical configuration reuses one entry across scripts), not
figures. Grouped by topology, not by wave-set registry key, so that two
different registered wave sets sharing the same physical sub-triad
(e.g. two quartets built on the same RH-only edge) land in the same
cache entry instead of duplicating the integration. Also gitignored.

Figures, by contrast, aren't shared/deduplicated the same way, so they're
grouped by wave-set registry key instead (2026-08-26, replacing an
earlier flat/topology-mirrored layout that got hard to browse as the
number of registered wave sets grew):
`outputs/figures/dynamics/<wave_set_key>/<unit>.png` (`unit`: `full`, or
`triad_<member1>_<member2>` per sub-triad -- named from its own two
member modes' filesystem-safe slugs, e.g. `triad_rh34_rh45`, not a
`triad0`/`triad1` index, so the name is meaningful on its own) and
`outputs/figures/wave_sets/<wave_set_key>/...` (sweep/diagnostic
figures, plus `sweep_2d`'s own per-point cache files).
`run_dynamics.py --diagnostics`'s own novelty-frequency spectrum figures
(`rsw_sphere.plotting.novelty_panel`) land alongside the dynamics ones,
in that same `outputs/figures/dynamics/<wave_set_key>/` folder, named
`novelty_<target>_vs_<unit>.png`.

`examples_legacy/legacy/` holds scripts superseded by a codebase reorganization
(verified to reproduce their own prior output before the move) — a
holding area pending a later, separate deletion pass, not a statement
that the code there still reflects current practice.

## Conventions & gotchas

- **Units:** all internal quantities are non-dimensional. Multiply time by `4π`
  to get days; frequencies scale by `2Ω`.
- **Mode selection** relies on sorting eigenvalues and indexing by `l = n − m`;
  the ordering (WIG / RH / EIG blocks) is assumed stable — see the index map in
  `eigenvectors.py` and the notes in `dispersion_relation.md` §2.4.
- **Pump mode** is expected to be mode c (`m_c` largest) -- `TRIAD`'s own convention.
- **Dependency pinning** (`numpy<2.0`, `scipy<1.15`) matters: the Legendre /
  eigenvalue APIs used here changed in later releases.
- **No Python identifiers were renamed** in the 2026-07 package refactor
  (`TRIAD`, `FOUR_WAVES`, `matriz_A`, etc. are unchanged) — only directories,
  filenames, and import paths moved to the `rsw_sphere` package layout.
