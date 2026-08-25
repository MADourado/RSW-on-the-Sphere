"""Regenerate the two source panels behind the paper's Quartet A / Quartet B
precession-frequency figure (JFM-template.tex ``fig: precession_frequency``,
a two-panel LaTeX ``subfigure`` combining ``Figures/quartet_a_rh36_precession.png``
and ``Figures/borrowed_topology_precession.png`` -- the "combination" is done
by LaTeX itself, not by compositing a single image here).

Reads ALREADY-SAVED outputs, runs no new dynamics: Quartet A's sweep-level
``.npz`` cache (written by ``run_sweep.py``'s ``precession`` diagnostic,
``rsw_sphere.plotting.wave_set_precession``, Phase 3 of
``paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-codebase-reorg-2026-08-25.md``)
and Quartet B's own ``.npz`` cache (``examples/borrowed_topology_precession_figure.py``'s
``sweep`` -- a genuinely different sweep shape, an amplitude-SCALE sweep
over all 4 modes at once rather than one registered mode's own velocity,
so it was NOT folded into ``run_sweep.py``'s registry-driven schema and
that script is NOT superseded/moved to ``examples/legacy/`` -- see that
script's own module docstring). This is exactly the kind of bespoke,
paper-specific figure assembly ``postproc/`` exists for (Phase 6 of the
plan above) -- generic sweep/dynamics execution stays in the two source
scripts, this script only re-renders from their already-computed caches
and prepares the copy into the paper repo.

Prints the exact ``cp`` commands needed rather than performing the copy
itself (``examples/make_section3_figures.py``'s own established
convention, ``docs/wave_sets.md`` §6) -- run it before regenerating either
panel to compare against what's already checked in.

Run:

    python postproc/precession_quartet_ab_panel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_EXAMPLES = os.path.join(_REPO, "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.plotting.wave_set_precession import (
    precession_frequency_efficiency, plot_dual_axis_frequency_efficiency)

QUARTET_A_SWEEP_CACHE = "outputs/figures/quartet_a_rh36_precession_sweep.npz"
QUARTET_A_OUTPUT = "outputs/figures/quartet_a_rh36_precession.png"
QUARTET_B_SWEEP_CACHE = "outputs/figures/borrowed_topology_precession_cache.npz"
QUARTET_B_OUTPUT = "outputs/figures/borrowed_topology_precession.png"

PAPER_FIGURES_DIR = "paper-nonlinear-interactions-SWE-sphere/Figures"


def build_quartet_a(path=QUARTET_A_OUTPUT, sweep_cache_path=QUARTET_A_SWEEP_CACHE):
    """Re-render Quartet A's panel from its already-computed sweep cache
    (must already exist -- run ``run_sweep.py --config
    examples/sweep_quartet_a_rh36.yaml`` first if not)."""
    if not os.path.exists(sweep_cache_path):
        raise FileNotFoundError(
            f"{sweep_cache_path} does not exist -- run "
            f"`python run_sweep.py --config examples/sweep_quartet_a_rh36.yaml` first.")
    specs = load_wave_set_specs(DEFAULT_WAVESETS_PATH)
    spec = specs["quartet_rh_preference"]
    result = precession_frequency_efficiency(
        spec, "d", [0.0], sweep_cache_path=sweep_cache_path)  # u_values ignored on cache hit
    plot_dual_axis_frequency_efficiency(
        result, spec, plot_triad=0,
        xlabel="RH(3,6) driving velocity $u$ (m/s)",
        title="Quartet A (this paper)", plot_u_max=120.0, path=path)
    return path


def build_quartet_b(path=QUARTET_B_OUTPUT, sweep_cache_path=QUARTET_B_SWEEP_CACHE):
    """Re-render Quartet B's panel from its already-computed sweep cache
    (must already exist -- run
    ``examples/borrowed_topology_precession_figure.py`` first if not)."""
    import borrowed_topology_precession_figure as quartet_b
    if not os.path.exists(sweep_cache_path):
        raise FileNotFoundError(
            f"{sweep_cache_path} does not exist -- run "
            f"`python examples/borrowed_topology_precession_figure.py "
            f"{sweep_cache_path} {path}` first.")
    quartet_b.plot_sweep(sweep_cache_path, path)
    return path


def main():
    written = []
    print("Quartet A (from sweep cache):")
    a_path = build_quartet_a()
    print(f"  wrote {os.path.abspath(a_path)}")
    written.append(a_path)

    print("Quartet B (from sweep cache):")
    b_path = build_quartet_b()
    print(f"  wrote {os.path.abspath(b_path)}")
    written.append(b_path)

    print("\nDone. To copy into the paper repo (JFM-template.tex, fig: precession_frequency):")
    for path in written:
        name = os.path.basename(path)
        print(f"  cp {path} {PAPER_FIGURES_DIR}/{name}")


if __name__ == "__main__":
    main()
