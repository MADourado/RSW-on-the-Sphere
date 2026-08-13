"""Precession frequency vs. amplitude scale for the topology borrowed
unchanged from Raphaldini et al. (2022) -- RH(1,3), RH(3,7), RH(4,5),
RH(2,9) -- comparing their own barotropic vorticity equation against
this paper's RSW system on the same four modes.

Compute and plot are two separate functions (``sweep``/``plot_sweep``),
matching this repo's own established pattern for expensive figures
(``rsw_sphere.plotting.triad_efficiency``'s ``efficiency_sweep``/
``plot_efficiency_map`` split): the sweep always writes its raw result
to a ``.npz`` cache, and plotting always reads from that cache -- labels/
styling can then be iterated without re-paying the sweep's own cost.

Reuses ``examples.precession_resonance_phase_diagnostic``'s own
``barotropic_phases``/``rsw_phases`` plus
``rsw_sphere.dynamics.dynamical_phase.libration_diagnostics``. Tracks
the triad RH(4,5)+RH(1,3)+RH(3,7) (not containing RH(2,9), the mode
whose own growth "efficiency" measures) -- the one whose $\\Phi$
reproduces Raphaldini et al.'s own reported Fig. 3 signature.

Amplitude-scale ranges are capped independently per system, both
checked directly (not assumed) rather than reused from the wider
ranges efficiency-only sweeps use elsewhere in this repository:
barotropic to 0.05 (oscillation amplitude still drifting beyond this,
even where the frequency slope alone looks stable), RSW to 0.25 (its
own frequency flips sign entirely by scale=1.09 between a 1500- and
3000-day window). Efficiency itself may be well-behaved far beyond both
(a different quantity, converged separately elsewhere) -- only the
precession-frequency curves here are capped.

Run:

    python examples/borrowed_topology_precession_figure.py outputs/figures/borrowed_topology_precession_cache.npz outputs/figures/borrowed_topology_precession.png
    python examples/borrowed_topology_precession_figure.py outputs/figures/borrowed_topology_precession_cache.npz outputs/figures/borrowed_topology_precession.png --plot-only
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.dynamics.dynamical_phase import libration_diagnostics
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.physics import days_from_nondim_time

import precession_resonance_phase_diagnostic as phase_diag
import precession_resonance_rsw_vs_barotropic as rsw_comp

SCALES_BARO = np.geomspace(2e-4, 0.05, 20)
SCALES_RSW = np.geomspace(3e-3, 0.25, 20)


def sweep(cache_path, scales_baro=SCALES_BARO, scales_rsw=SCALES_RSW,
          t_f_baro=1500.0, t_f_rsw=1500.0):
    """Pure compute: run both curves and write them to ``cache_path``
    (.npz). Does no plotting. Re-running with an already-existing
    ``cache_path`` is a no-op -- delete the file first to force a
    recompute.

    Returns
    -------
    scales_baro, scales_rsw, f_baro, f_rsw : ndarray
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return d['scales_baro'], d['scales_rsw'], d['f_baro'], d['f_rsw']

    f_baro = []
    for s in scales_baro:
        Phi1, _, T = phase_diag.barotropic_phases(scale=s, t_f=t_f_baro)
        days = days_from_nondim_time(T)
        f_baro.append(libration_diagnostics(Phi1, days)['precession_freq'])

    ws = rsw_comp.build()
    f_rsw = []
    for s in scales_rsw:
        Phi1, _, T = phase_diag.rsw_phases(ws, scale=s, t_f=t_f_rsw)
        days = days_from_nondim_time(T)
        f_rsw.append(libration_diagnostics(Phi1, days)['precession_freq'])

    f_baro, f_rsw = np.array(f_baro), np.array(f_rsw)
    np.savez(cache_path, scales_baro=scales_baro, scales_rsw=scales_rsw, f_baro=f_baro, f_rsw=f_rsw)
    return scales_baro, scales_rsw, f_baro, f_rsw


def plot_sweep(cache_path, path=None):
    """Pure plotting: read ``cache_path`` (must already exist -- see
    ``sweep``) and draw the figure.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} does not exist -- run sweep({cache_path!r}) first.")
    d = np.load(cache_path)
    scales_baro, scales_rsw, f_baro, f_rsw = d['scales_baro'], d['scales_rsw'], d['f_baro'], d['f_rsw']

    apply_house_style()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(scales_baro, np.abs(f_baro), 'o-', ms=3, label='Barotropic (Raphaldini et al. 2022)')
    ax.plot(scales_rsw, np.abs(f_rsw), 's-', ms=3, label='RSW (identical topology)')
    ax.axhline(0.01, color='grey', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('Amplitude scale')
    ax.set_ylabel(r'$|$precession frequency$|$ (rad/day)')
    ax.set_title('Barotropic vs. RSW\nRH(1,3)+RH(3,7)+RH(4,5)+RH(2,9)', fontsize=10)
    ax.legend(fontsize=8)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", help="path to the .npz cache (computed if missing, else reused)")
    parser.add_argument("path", nargs="?", default=None, help="output PNG path")
    parser.add_argument("--plot-only", action="store_true",
                         help="skip sweep() entirely and error if the cache is missing.")
    args = parser.parse_args()

    if not args.plot_only:
        sweep(args.cache)
    plot_sweep(args.cache, args.path)
    print(f"Saved to {args.path}" if args.path else "Shown interactively")
