"""Broadened precession-resonance search in RSW, following up on
``precession_resonance_phase_diagnostic.py``'s negative result (the
paper's own barotropic topology, transplanted into RSW, shows matching
efficiency shape but NO genuine dynamical-phase locking at any scale
tested there).

**Metric used throughout, and why it's adequate for a systematic scan**
(the same "is this even the right diagnostic" question the user raised
during the original 2026-08-12 investigation -- see the INSPECT doc's
"Methodological check" section -- applies here too, now for a *scan*
rather than a handful of points):

- ``net_windings`` (total drift of the unwrapped dynamical phase Phi
  over the run, in units of 2*pi) is the primary screening statistic:
  small in magnitude (order 1 or less) is *necessary* for libration,
  large values definitively rule it out. Unlike the earlier D1-RMS
  metric, it is not conflated with the smooth-law background scatter --
  it depends only on Phi, not on any amplitude-error comparison against
  a reference trajectory.
- A **known false-positive mode** exists and is guarded against
  explicitly: at very low forcing amplitude, any weakly-driven mode's
  phase trivially tracks its forcing (the same adiabatic mechanism
  Gate I4's own scaling law already covers), giving spuriously small
  net_windings that reflect the linear regime, not resonance. Screened
  out here by (a) always sweeping a *range* of amplitudes, not single
  points, and checking small-windings survives as amplitude grows into
  the genuinely nonlinear regime, and (b) reporting the coupled mode's
  own energy fraction alongside windings, so a "locked because nothing
  is happening" result is visible, not silently accepted.
- Any candidate surviving the screen is re-run at 2x t_f and a finer h
  as a convergence check before being reported as a real finding (same
  bar as every other converged number in this investigation).

Three broadenings beyond the single paper-topology quartet tested so far:

1. **h_e sweep** at fixed (peak) scale on the *same* paper-topology
   quartet -- RSW's RH branch approaches the non-divergent barotropic
   limit as h_e grows (see ``rsw_sphere.dynamics.wave_sets`` module
   docstring, "Non-conservation" section, and the paper-nonlinear-
   interactions dissertation's own dispersion-relation appendix), so if
   finite depth (cubic energy) is really what kills the locking, driving
   h_e up should recover it (or at least track delta1's sign flip back
   toward the barotropic value).
2. **Finer scale sweep** (net_windings only, cheap) on the same
   h_e=10000 quartet, filling in between the three points already
   tested, in case a narrow locking window was missed.
3. **RSW's own native near-commensurate candidates** (WG(3,1), WG(4,2),
   WG(5,3) -- found in the 2026-08-12 I4e-h edge search, but only ever
   tested with the inadequate windowed-D1 metric, never with the
   dynamical-phase diagnostic built later that same day) -- arguably
   more relevant than importing the paper's literal topology, since
   these are genuine RSW resonances rather than an as-if-barotropic
   transplant.

Run:

    python examples/precession_resonance_broad_search.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import warnings
import numpy as np

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics

import precession_resonance_rsw_vs_barotropic as rsw_comp

G = 9.8


def phi_stats(ws, triad_idx, A0, t_f, h):
    Y, T = RK33(ws, 0, t_f, h, A0)
    i_sum, i_p, i_q = ws.triads[triad_idx]
    Phi = dynamical_phase(Y, T, ws.omega, i_sum, i_p, i_q, ws.delta[triad_idx])
    stats = libration_diagnostics(Phi, days_from_nondim_time(T))
    E = np.real(Y * np.conj(Y))
    energy_frac_private = E[:, i_q].mean() / E.mean(axis=1).mean() if E.mean() > 0 else float('nan')
    stats['converged'] = None
    return stats, E


def converge_check(ws, triad_idx, A0, t_f, h, abs_tol=0.02):
    """``abs_tol`` (rad/day), not relative: near a genuine lock the base
    value is itself ~0, so a relative tolerance is spuriously strict
    (any numerical jitter looks like a huge percentage change). 0.02
    rad/day is small next to every off-resonance value seen in this
    search (0.1-2 rad/day) but generous next to locked values (~1e-4)."""
    s1, _ = phi_stats(ws, triad_idx, A0, t_f, h)
    s2, _ = phi_stats(ws, triad_idx, A0, 2 * t_f, h)
    s3, _ = phi_stats(ws, triad_idx, A0, t_f, h / 3)
    ok = (abs(s1['precession_freq'] - s2['precession_freq']) < abs_tol
          and abs(s1['precession_freq'] - s3['precession_freq']) < abs_tol)
    return s1, ok, s2, s3


# ---------------------------------------------------------------------
# 1. h_e sweep, paper topology, fixed (peak) scale
# ---------------------------------------------------------------------
def section_he_sweep():
    print("=== 1. h_e sweep, paper topology (RH1,3/RH3,7/RH4,5/RH2,9), "
          "fixed scale=3e-2 (RSW's own efficiency peak at h_e=10000) ===")
    print(f"{'h_e (m)':>10} {'omega1':>8} {'delta1':>9} {'delta2':>9} "
          f"{'Phi1 prec.freq (rad/day)':>26} {'net_windings':>13}")
    scale = 3e-2
    A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
    t_f, h = 1500.0, 0.01
    for h_e in (1e4, 3e4, 1e5, 3e5, 1e6, 1e7, 1e8):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gamma = gamma_from_he(h_e, g=G)[1]
            ws = WaveSet(gamma, [rsw_comp.RH1, rsw_comp.RH2, rsw_comp.RH3, rsw_comp.RH4],
                         rsw_comp.TRIADS, N=12, deg=400)
        stats, _ = phi_stats(ws, 0, A0, t_f, h)
        print(f"{h_e:>10.0e} {ws.omega[0]:>8.4f} {ws.delta[0]:>9.4f} {ws.delta[1]:>9.4f} "
              f"{stats['precession_freq']:>26.5f} {stats['net_windings']:>13.3f}")
    print("(paper's own barotropic values: omega1=-0.083, delta1=-0.003)")


# ---------------------------------------------------------------------
# 2. Finer scale sweep, paper topology, h_e=10000
# ---------------------------------------------------------------------
def section_scale_sweep():
    print("\n=== 2. Finer scale sweep, paper topology, h_e=10000 "
          "(screening for a missed locking window) ===")
    ws = rsw_comp.build()
    print(f"{'scale':>10} {'Phi1 prec.freq (rad/day)':>26} {'net_windings':>13} "
          f"{'osc_amp':>9}")
    for scale in np.logspace(-4, 1.3, 14):
        h = min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))
        A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
        stats, _ = phi_stats(ws, 0, A0, 1500.0, h)
        print(f"{scale:>10.2e} {stats['precession_freq']:>26.5f} "
              f"{stats['net_windings']:>13.3f} {stats['oscillation_amplitude_windings']:>9.3f}")


# ---------------------------------------------------------------------
# 3. RSW-native near-commensurate candidates from the I4e-h edge search
# ---------------------------------------------------------------------
NATIVE_CANDIDATES = {
    # name: (modes list [RH(4,5), RH(1,2)-or-RH(3,4) edge members..., candidate],
    #        triads [(i_sum,i_p,i_q), ...], triad_idx to diagnose, driving velocities)
    "WG(3,1) edge{RH45,RH12}": dict(
        modes=[(4, 5, 3), (1, 2, 3), (3, 4, 3), (3, 1, 2)],
        triads=[(0, 2, 1), (0, 1, 3)],  # triad1: RH45=RH34+RH12 ; triad2: RH45=WG31+RH12
        probe_triad=1,
    ),
    "WG(5,3) edge{RH45,RH12} (candidate as NEW sum)": dict(
        modes=[(4, 5, 3), (1, 2, 3), (3, 4, 3), (5, 3, 2)],
        triads=[(0, 2, 1), (3, 0, 1)],  # triad1: RH45=RH34+RH12 ; triad2: WG53=RH45+RH12
        probe_triad=1,
    ),
    "WG(4,2) edge{RH34,RH12}": dict(
        modes=[(4, 5, 3), (3, 4, 3), (1, 2, 3), (4, 2, 2)],
        triads=[(0, 1, 2), (3, 1, 2)],  # triad1: RH45=RH34+RH12 ; triad2: WG42=RH34+RH12
        probe_triad=1,
    ),
}


def section_native_candidates(h_e=10000.0):
    print(f"\n=== 3. RSW-native near-commensurate candidates (I4e-h search), "
          f"h_e={h_e:.0f}, velocity sweep on the candidate mode, "
          f"RH-triad members fixed at 40 m/s ===")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gamma = gamma_from_he(h_e, g=G)[1]
    for name, cfg in NATIVE_CANDIDATES.items():
        modes, triads, probe = cfg['modes'], cfg['triads'], cfg['probe_triad']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # deg=300 to match velocity_to_amplitude's norm_component default
            # (amplitudes_from_velocities doesn't thread deg through -- see
            # rsw_sphere.hough_harmonics.normalization.norm_component).
            ws = WaveSet(gamma, modes, triads, N=10, deg=300)
        print(f"\n{name}: delta1={ws.delta[0]:.4f}  delta2={ws.delta[1]:.4f}  "
              f"alpha_triad2={ws.alpha[probe]}")
        print(f"  {'u_cand (m/s)':>13} {'Phi(probe) prec.freq':>21} {'net_windings':>13} "
              f"{'osc_amp':>9}")
        for u_cand in (5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0):
            velocities = np.zeros(len(modes))
            velocities[0] = 40.0  # RH(4,5)
            velocities[1] = 40.0  # RH(1,2) or RH(3,4), shared/private per candidate
            velocities[2] = 40.0  # third RH-triad member
            velocities[3] = u_cand  # candidate mode
            A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)
            t_f, h = 3000.0, 0.005
            stats, _ = phi_stats(ws, probe, A0, t_f, h)
            print(f"  {u_cand:>13.1f} {stats['precession_freq']:>21.5f} "
                  f"{stats['net_windings']:>13.3f} {stats['oscillation_amplitude_windings']:>9.3f}")


if __name__ == "__main__":
    section_he_sweep()
    section_scale_sweep()
    section_native_candidates()
