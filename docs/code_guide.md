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
run_mode_search.py           driver: find candidate modes completing a triad
  └─ rsw_sphere/
       ├─ paths.py           repo root + default output roots (never CWD-relative)
       ├─ physics.py         constants, gamma/eps, nondimensional-time conversions
       ├─ hough_harmonics/   the numerical core (eigenproblem, modes, inner products)
       ├─ dynamics/          WaveSet/TRIAD, integrator, trajectory cache, RunConfig
       ├─ utilities/         compute: diagnostics (pmeasure, periods, precession,
       │                     efficiency)
       └─ plotting/          rendering only -- every plot_* fn takes already-computed
                              arrays, returns (fig, ax, ...), never integrates
```

`rsw_sphere/utilities/` vs. `rsw_sphere/plotting/`: a function that returns
numbers goes in `utilities/`; a function that only draws (given
already-computed arrays) goes in `plotting/`. `run_sweep.py`'s own
diagnostic vocabulary (`_MODE_UNIT_DIAGNOSTICS`/`_TRIAD_DIAGNOSTICS`/
`_SCALAR_DIAGNOSTICS`) maps diagnostic name -> report field + render
function -- adding a diagnostic is one table entry, not a new sweep loop
(see `docs/wave_sets.md` §6.1).

There is no figure-assembly (`postproc/`) layer, and none is needed
while every paper figure is produced complete by one script: a figure
shown as two panels from two different computations is two independent
PNGs combined by LaTeX `subfigure`, not a Python composite. Add one only
if a figure genuinely requires merging two already-computed,
independently-generated outputs into an image neither source script can
produce alone.

The package is `pip install -e .`-installable (`pyproject.toml`); once
installed, `rsw_sphere` resolves from anywhere. Scripts under `examples/`
additionally `import _bootstrap`, a small per-directory module that puts
the repo root on `sys.path`, so they also run from an uninstalled
checkout.

**Nothing resolves a path against the current working directory.**
`rsw_sphere/paths.py` defines `REPO_ROOT`, `DEFAULT_WAVESETS_PATH`,
`OUTPUT_ROOT`, `TRAJECTORY_ROOT` and `resolve()`; every default output
location and the trajectory cache go through it, and a relative
`output_root` passed by a caller is resolved against the repo root. A
driver or example script therefore writes the same tree whether it is
launched from the repo root or from `examples/figures/`.

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
registry entry (`RunConfig.from_registry_entry`/`from_wave_set`).
`run_sweep_sets.py` similarly reads its own `alternative_modes:` block
straight from the registry entry (`--wave-set KEY --slot LETTER`), a raw
YAML passthrough like `sweep:`/`plot:` rather than a `WaveSetSpec` field
(only this one driver consumes it). See `examples/` for configs.

### `run_linear_modes.py`
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
own `rsw_sphere.dynamics.trajectory_cache.run_and_cache` call, each
plotted separately (`rsw_sphere.plotting.energy_evolution.plot_energy_evolution`,
unless `config.plot=False`) under `outputs/dynamics/<wave_set_key>/<run_label>/`
(see "Outputs" below for the full per-run layout). Parallel across units
by default (`ProcessPoolExecutor`, `config.max_workers` or half the CPU
count). `run_dynamics(config) -> dict` is directly importable --
`run_sweep.py` calls it per grid point. Console output always includes a
linear-vs-observed-FFT-frequency table (`diag_evol_<run_label>.csv`) per
mode per unit; `--diagnostics` additionally prints every pairwise
diagnostic (`rsw_sphere.utilities.pmeasure.pairwise_target_diagnostics`:
efficiency_var, novelty_period) for every target mode against each sub-triad
that contains it, a dynamical-phase precession-frequency table, a "final
diagnostics" section (`efficiency_variation_combined_for_all_targets`/
`rsw_sphere.utilities.novelty_frequency.novelty_combined_for_all_targets`)
with one row per target considering every containing sub-triad at once,
and writes the novelty-frequency spectrum figures
(`rsw_sphere.plotting.novelty_frequency_panel`) -- all four `diag_*`
tables saved as CSV alongside the figures.

    python run_dynamics.py --wave-set quartet_rossby_kelvin

### `run_sweep.py`
Sweeps 1 or 2 modes' velocities (`config.sweep.axes`). One `run_dynamics()`
call per grid point (via `rsw_sphere.dynamics.diagnostics_report.compute_diagnostics_report`,
no separate integration path per dimensionality), then plots/writes every
diagnostic in `sweep.diagnostics` -- one shared vocabulary for 1D and 2D:
`efficiency`/`dominant_freq`/`dominant_period`/`low_frequency_energy`
(1D: one line per (mode, unit); 2D: one heatmap per mode), `dynamical_phase`
(1D: one line per triad; 2D: one heatmap per triad), and the "final"
scalars `efficiency_var` (alias `energy_var`)/`spectral_dev_var`/
`novel_freq`/`novel_period` (one line/heatmap per mode). `diagnostics:
[all]` expands to every name. Swept/target modes for a 2D sweep default
to the wave set's own "private" modes (`WaveSetSpec.shared_and_private_modes()`).
`--wave-set KEY` reads `sweep`/`tf_days`/`h`/`plot` straight from that
wave set's own `wave_sets_default.yaml` entry -- a wave set not yet
worth adding to the default registry can be swept via `--specs
path.yaml` instead (same registry schema). Every diagnostic writes its
own `outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png/.csv`
-- there's no single "the" output for a sweep to override. A composite
figure combining several diagnostics (e.g. the paper's own
precession-frequency-and-efficiency figure) is a separate script's job
(`examples/figures/paper_figure007_quartet_a_precession.py`), not
something this driver special-cases. See `docs/wave_sets.md` §6.1 for the full vocabulary.
`run_sweep_sets.py` and `examples/figures/_triad_panel_row.py` go through
this same unified engine (`run_dynamics()` +
`compute_diagnostics_report()`/`compute_2d_grid()`).

    python run_sweep.py --wave-set quartet_rossby_kelvin
    python run_sweep.py --wave-set quartet_rh_preference

### `run_sweep_sets.py`
Loops a diagnostic over a LIST of wave-set variants -- substituting which
mode fills one slot, not sweeping a velocity. Candidates + diagnostics
come from that wave set's own registered `alternative_modes.<slot>` block
(`wave_sets_default.yaml`, see `docs/wave_sets.md`'s own section on the
schema) -- `target_mode` is the (usually different, already-driven) mode
whose diagnostic value is reported, defaulting to the slot itself. One
point per candidate (own registered velocities, unless
`candidate_velocity` overrides the candidate slot's own velocity),
parallel across candidates. Every requested diagnostic, plus each
candidate's own static `delta`/`coeff`/isolated-triad `efficiency`
(read off the target's own resolved reference triad, no extra
integration), becomes a CSV column (`outputs/sweep_sets/<key>/<slot>/candidates.csv`)
and its own point-wise, no-connecting-line figure (`candidates_<column>.png`)
-- candidates have no natural ordering, unlike a swept continuous velocity.
`run_mode_search.py`'s own `m,n,alpha` output rows paste directly into a
registry's `candidates:` list.

    python run_sweep_sets.py --wave-set quartet_rossby_kelvin --slot d

### `run_mode_search.py`
Registry-independent: given a fixed edge (`--edge MODE_P MODE_Q`) or pivot
(`--pivot MODE`, `m,n,alpha` each), lists candidate modes completing a
valid triad -- for scouting a new `wave_sets_default.yaml` entry or an
`alternative_modes.<slot>.candidates:` block without hand-deriving the
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
The special `m = 0` case: matrices `C` and `D`. Used for the `m = 0`
eigenvalues on the dispersion diagram only -- no `m = 0` eigenvector/mode
construction exists, and no registered wave set uses an `m = 0` mode
(Rossby modes are degenerate/absent there).

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

### `normalization.py`
- `norm_Hough(m, n, alpha, gamma, N, deg)` — evaluates a mode on a
  Gauss-Legendre latitude grid and normalizes it to unit energy inner product,
  returning the normalized `(U, V, Z, DU, DV, DZ)`, the quadrature points, the
  norm, and the eigenvalue. This is the canonical way to obtain a usable mode.
- `norm_component(u)` — norm of just the zonal-velocity component, used to
  convert a desired physical zonal velocity (m/s) into a mode amplitude.

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
- `Energy_0` — quadratic (E²) and cubic (E³) energy of an amplitude state.
- `RK44` — re-exported from `integrators.py` for callers that only import
  this module.

Integration, plotting and efficiency diagnostics for a single triad go
through the same drivers everything else uses (a triad is `WaveSet`'s
1-triad case); `TRIAD` itself only builds the system.

### `wave_sets.py` — quartets, quintets (generalized `TRIAD`)
`class WaveSet` — an arbitrary set of Hough modes coupled through an
arbitrary set of resonant triads; a quartet is 4 modes/2 triads sharing
one edge, a quintet is 5 modes/3 triads, a plain triad is the degenerate
3-modes/1-triad case. `TRIAD` (`dynamic_triads.py`) is deliberately kept
as `WaveSet`'s independent reference implementation, proven equivalent
under a mode-relabeling permutation (`WaveSet`'s own module docstring;
`rsw_sphere/utilities/check_wave_set_physics.py` checks C1-C3). See
[`../docs/wave_sets.md`](../docs/wave_sets.md) for the plotting/registry
layer built on top and how to test a configuration not yet registered.

### `trajectory_cache.py` — raw trajectory caching
`run_and_cache(ws, A0, t_f, h, velocities=None, output_root=None, label=None)`
caches a `WaveSet`/`RK44` run's raw `Y(t)` solution itself (not just a
derived summary) under `outputs/trajectories/<topology>/`, `topology`
auto-derived from `ws.n_modes` (`triads`/`quartets`/`quintets`); every
other cache in the repo stores summary arrays only. The filename is built
from every mode's own initial condition (`ic_label(ws.modes, velocities)`,
canonically sorted), not a caller-supplied key, so the same physical
configuration lands in the same cache entry regardless of which script
built it. `output_root` defaults to `rsw_sphere.paths.TRAJECTORY_ROOT`
and a relative override is resolved against the repo root, so the cache
never fragments by launch directory. Used by `run_dynamics.py` and
`rsw_sphere.utilities.precession`; any new script computing a `WaveSet`
trajectory that might be revisited should go through this rather than
calling `RK44` directly.

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

### `wave_set_table.py`, `energy_evolution.py`, `precession_plot.py`, `functional_map.py`, `novelty_frequency_panel.py`, `sweep_diagnostics.py`
Rendering for quartet/quintet ("wave set") examples, each with its own
`rsw-waveset-*` console script -- the `WaveSet` analogue of this section's
single-triad tools. Compute lives in `rsw_sphere/utilities/` (below), not
here. Fully documented in [`../docs/wave_sets.md`](../docs/wave_sets.md).

## `rsw_sphere/utilities/` — diagnostics compute

`pmeasure.py` (efficiency variation, pairwise/final
full-wave-set-vs-constituent-triad comparisons --
`efficiency_variation_final`, `pairwise_target_diagnostics`),
`periods.py` (`dominant_periods`,
`low_frequency_power`, novelty-frequency content, `spectral_deviation`),
`precession.py` (`precession_frequency_efficiency`), `efficiency.py`
(`wave_set_efficiency`, drift-gated, `default_velocity_range`). The
unified sweep engine itself (`rsw_sphere.dynamics.diagnostics_report.compute_diagnostics_report`,
used by `run_sweep.py`/`run_sweep_sets.py`/`_triad_panel_row.py`) reads
these functions' outputs per grid point rather than duplicating their
formulas. Sweep-style functions take a `cache_path` (`.npz`,
cache-if-absent/load-if-present) as their own, separate sweep cache.

## Outputs

`outputs/` is gitignored and fully regenerated by running the drivers. Three
kinds of output, each with its own layout rationale -- see
[`general_guide.md`](general_guide.md) for the higher-level "why one
run's worth of output lives together" framing; this section has the exact
paths.

**`outputs/trajectories/<topology>/<ic_label>_tf<days>_h<h>_<hash8>.npz`**
(`topology`: `triads`/`quartets`/`quintets`) -- raw cached `WaveSet`
trajectories (`rsw_sphere.dynamics.trajectory_cache.run_and_cache`).
Reusable intermediate data (any diagnostic can be re-derived from one
without re-integrating), so it's kept separate from figures/tables and
grouped by topology, not by wave-set registry key -- two different
registered wave sets sharing the same physical sub-triad (e.g. two
quartets built on the same RH-only edge) land in the same cache entry
instead of duplicating the integration. `ic_label` is built from every
mode's own label + initial velocity, canonically sorted, so the same
physical configuration always produces the same filename regardless of
which script built it.

**`outputs/dynamics/<wave_set_key>/<run_label>/`** (`run_label` = that
run's own `ic_label_tf<days>_h<h>`, i.e. the trajectory filename's own
readable part minus the hash) -- everything ONE `run_dynamics.py`
invocation produces, figures and tables together in one folder:
- `evol_<topology>_<mode1>_<mode2>..._<run_label>.png` -- energy-evolution
  plot per topology unit (`topology`: `quartet`/`triad`/`quintet`; for a
  sub-triad unit, `triad`, always listing that CONSTITUENT triad's own 3
  modes, not the parent wave set's full mode list). Every mode listed by
  its filesystem-safe label (`rsw_sphere.plotting.labels.mode_fs_label`,
  e.g. `RH4_5`), sorted the same way `ic_label` sorts them.
- `diag_freq_novel_<mode>_<run_label>.png` -- novelty-frequency spectrum
  figures (`--diagnostics`, `rsw_sphere.plotting.novelty_frequency_panel`).
- `diag_evol_<run_label>.csv` -- one row per (mode, unit): dEK,
  efficiency, linear period/frequency, top FFT peaks (period/frequency/
  relative power), one column per value (not a packed string) so it's a
  clean CSV to load. Written unconditionally (no `--diagnostics` needed).
- `diag_prec_freq_<run_label>.csv`, `diag_pairwise_<run_label>.csv`,
  `diag_final_<run_label>.csv` -- the `--diagnostics`-only tables
  (dynamical-phase precession frequency; pairwise full-vs-sub-triad
  efficiency-variation/spectral-deviation/novelty-period; the same set
  combined across every containing sub-triad per target mode).

The `<run_label>` suffix on every filename (not just the folder name)
matters because these figures get pulled individually into the paper --
a bare `evol_quartet.png` would collide across different runs once
copied out of its folder, a self-contained filename doesn't.

**`outputs/figures/`** -- `run_linear_modes.py`'s own output:
`dispersion_relation.png`, `dispersion_relation_fancy.png`, and per-mode
`linear/<alpha>-<m>-<n>/Hough_harmonic_<alpha>-<m>-<n>.png` +
`derivatives_<alpha>-<m>-<n>.png` + `Hough_spatial_<alpha>-<m>-<n>.png`
(e.g. `linear/RH-1-2/Hough_harmonic_RH-1-2.png`) -- folder and filenames
both encode the mode so files stay identifiable if moved out of their
folder. `run_sweep.py` writes under
`outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png/.csv`
instead -- one file pair per requested diagnostic, no single "the"
output for a sweep to override (see `docs/wave_sets.md` §6.1).
`run_sweep_sets.py` writes under
`outputs/sweep_sets/<wave_set_key>/<slot>/` -- one `candidates.csv` table
plus one point-wise `candidates_<column>.png` figure per column (every
requested diagnostic and each candidate's own static delta/coeff/
isolated-triad-efficiency properties).

**`outputs/tables/`** -- one `.tex`/`.csv` per
`examples/tables/paper_table<NN>_*.py` script (see `examples/README.md`
for the full list and which paper `\label{...}` each one reproduces).

Standalone dispersion/Hough scripts write wherever their `path` argument
points, outside this layout.

## Conventions & gotchas

- **Units:** all internal quantities are non-dimensional. Multiply time by `4π`
  to get days; frequencies scale by `2Ω`.
- **Mode selection** relies on sorting eigenvalues and indexing by `l = n − m`;
  the ordering (WIG / RH / EIG blocks) is assumed stable — see the index map in
  `eigenvectors.py` and the notes in `dispersion_relation.md` §2.4.
- **Pump mode** is expected to be mode c (`m_c` largest) -- `TRIAD`'s own convention.
- **Dependency pinning** (`numpy<2.0`, `scipy<1.15`) matters: the Legendre /
  eigenvalue APIs used here changed in later releases.
- **Paths are never CWD-relative.** Anything writing under `outputs/`, and
  the trajectory cache, goes through `rsw_sphere/paths.py`; a relative
  `output_root` is resolved against the repo root, not the launch
  directory.
- **`m = 0` modes are not constructed.** `matrix_m0.py` supplies the
  `m = 0` eigenvalues for the dispersion diagram only; there is no `m = 0`
  eigenvector/normalization path, and no registered wave set uses one.
