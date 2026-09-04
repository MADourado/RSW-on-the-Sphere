# RSW on the Sphere — Nonlinear wave interactions

A Python toolkit for studying **nonlinear wave interactions in the rotating
shallow water (RSW) equations on the sphere**.

The RSW equations are the simplest global model whose spectrum contains both
**fast** waves (inertia-gravity) and **slow** waves (Rossby-Haurwitz),
connected by the Kelvin and mixed Rossby-gravity modes. This repository builds
the **normal modes** of the linearised equations (the **Hough harmonics**,
obtained as eigenfunctions of Laplace's tidal equations via the
vector-spherical-harmonic method of Swarztrauber & Kasahara (1985)) and uses
them to analyse the **nonlinear energy exchange** between waves: its
direction, efficiency, periods and spectral signature, for any resonant or
quasi-resonant configuration of three, four or five modes.

Every configuration is a registry entry rather than a bespoke script, so a new
triad, quartet or quintet is added by editing YAML — see
[`docs/general_guide.md`](docs/general_guide.md) to get started.

**Code authors:** Marco Antonio Dourado and Pedro da Silva Peixoto.

## Publications

The code was developed for, and reproduces the results of:

> **Non-linear interactions between slow and fast atmospheric waves on the
> sphere.** Peixoto, Raphaldini, Dourado & Teruya. *In preparation.*

> **Nonlinear wave interactions in rotating shallow water equations on the
> sphere.** Marco Antonio Dourado — MSc dissertation, Institute of
> Mathematics, Statistics and Computer Science, University of São Paulo
> (IME-USP), 2025. Advisor: Prof. Dr. Pedro da Silva Peixoto · Co-advisor:
> Prof. Dr. Breno Raphaldini.

Each paper table and figure is reproduced by one script under
[`examples/`](examples/) — see that directory's README for which
`\label{...}` each one covers.

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
  (Runge-Kutta), and energy/efficiency/spectral diagnostics
  (`rsw_sphere/dynamics/`, `rsw_sphere/utilities/`).
- **Five root drivers.** Four select from the single
  `wave_sets_default.yaml` registry: `run_linear_modes.py` (dispersion
  relation + per-mode Hough plots), `run_dynamics.py`, `run_sweep.py` and
  `run_sweep_sets.py` (integration, IC sweeps, candidate-mode screening --
  all sharing one config class,
  `rsw_sphere.dynamics.run_config.RunConfig`, all `--wave-set KEY`-driven,
  no separate config file). A triad is just the registry's 1-triad case,
  so the same registry covers triads, quartets and quintets. The fifth,
  `run_mode_search.py`, is registry-independent and finds what *could* go
  in the registry: given 2 fixed modes (an edge) or 1 (a pivot), it lists
  candidate modes completing a valid triad with them. See
  `docs/code_guide.md`'s "Entry points".

## Repository layout

```
rsw_sphere/                # the installable package
    paths.py               # repo root + default output roots (never CWD-relative)
    physics.py             # constants, gamma/eps, nondimensional-time conversions
    hough_harmonics/       # eigenvalue problem, normal modes, inner products
    dynamics/              # WaveSet/TRIAD, integrator, trajectory cache, RunConfig
    utilities/             # diagnostics compute (pmeasure, periods, precession,
                           # efficiency, mode search, physics gate)
    plotting/              # rendering only -- dispersion, Hough, wave-set figures
docs/                      # general_guide.md (start here), code_guide.md, per-topic docs, dissertation PDF
examples/                  # registries + one script per paper table/figure (see examples/README.md)
outputs/                   # generated figures + cached trajectories (gitignored, reproducible)
tests/                     # pytest suite (structural/exact invariants)
run_linear_modes.py        # dispersion relation + per-mode Hough plots
run_dynamics.py            # integrate a wave set (full + sub-triads), cached
run_sweep.py               # IC sweep (1-2 modes) + diagnostics
run_sweep_sets.py          # loop a diagnostic over candidate-mode variants
run_mode_search.py         # find candidate modes completing a triad with a given edge/pivot
wave_sets_default.yaml     # default WaveSet registry (triads, quartets, quintets)
pyproject.toml             # pip install -e . / console scripts
```

`rsw_sphere/dynamics/wave_sets.py::WaveSet` generalizes the single-triad
`TRIAD` class (`dynamic_triads.py`) to an arbitrary set of Hough modes
coupled through an arbitrary set of resonant triads; quartets and
quintets are instances of it. `TRIAD` is kept as the independent
reference implementation `WaveSet` is checked against. See
[`docs/wave_sets.md`](docs/wave_sets.md).

Every output path is anchored to the repository root
(`rsw_sphere/paths.py`), so a driver or example script writes the same
tree regardless of the directory it is launched from.

## Installation

Requires Python 3.12. Install the package (editable) so `rsw_sphere` is
importable from anywhere and the console scripts are available:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

To also get pytest for `tests/`:

```bash
pip install -e ".[dev]"
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
(~4 minutes) -- run this before trusting a change to `rsw_sphere/`
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

All four registry drivers select from the same `wave_sets_default.yaml`
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
e.g. `examples/wave_sets_custom.yaml`). Run the physics gate
(`python rsw_sphere/utilities/check_wave_set_physics.py --wave-set KEY`)
on any new or edited entry before trusting a figure from it. See
[`examples/`](examples/) for one script per paper table and figure.

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

Single triads, quartets and quintets all go through the same unified
drivers (a triad is the degenerate 1-triad case) and are documented in
[`docs/wave_sets.md`](docs/wave_sets.md), along with the `rsw-waveset-table`
/ `rsw-waveset` / `rsw-waveset-precession` console scripts, how to test a
configuration that isn't in either registry YAML, and how `run_sweep.py`
builds on top of these.

## References

The literature this implementation builds on (the dissertation and paper it
backs are listed under [Publications](#publications) above):

- Swarztrauber, P. N. & Kasahara, A. (1985). *The vector harmonic analysis of
  Laplace's tidal equations.* SIAM J. Sci. Stat. Comput., 6, 464–491.
- Longuet-Higgins, M. S. (1968). *The eigenfunctions of Laplace's tidal
  equations over a sphere.* Phil. Trans. R. Soc. A, 262, 511–607.
- Raphaldini, B., Peixoto, P., Teruya, A., Raupp, C. & Bustamante, M. (2022).
  *Precession resonance of Rossby wave triads…* Physics of Fluids.
