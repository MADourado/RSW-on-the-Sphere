"""Phase 5 of PLAN-codebase-reorg-2026-08-25.md: does this paper's own RSW
quartets show the individual-mode frequency reversal Raphaldini et al.
(2022) report (Section III.A / Fig. 3 -- at high driving amplitude, one
mode's own raw phase can bend enough that its propagation direction
flips sign)?

Three steps, compute-first (report actual slopes/sign-flips before any
figure/paper prose):

1. Calibration -- reproduce their reversal directly in the barotropic
   system (``examples/reproduce_raphaldini2022_fig2.py``'s own ``rhs``/
   ``_rk4``), scanning that script's own established scale range
   (5e-4..1e-2) with ``individual_phase`` + the barotropic-side
   ``phi_j_tilde`` correction (Phase 2 of the plan).
2. Quartet B, RSW -- same topology
   (``examples/precession_resonance_rsw_vs_barotropic.py``'s ``build()``),
   run through ``rsw_sphere.dynamics.trajectory_cache.run_and_cache``, at
   points across that script's own established scale range (1e-4..10).
3. Quartet A, RSW -- the paper's own native quartet
   (``quartet_rh_preference``), via
   ``rsw_sphere.utilities.precession.precession_frequency_efficiency``,
   at u=10,50,80,85,90,120 (spanning before/at/after the confirmed lock
   at u~83-92) -- every constituent mode's own phase slope, not just the
   driven one.

Run:

    python examples/individual_mode_reversal_investigation.py
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

from rsw_sphere.dynamics.dynamical_phase import individual_phase
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.physics import days_from_nondim_time
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.utilities.precession import precession_frequency_efficiency

import reproduce_raphaldini2022_fig2 as baro
import precession_resonance_rsw_vs_barotropic as rsw_b


def step1_barotropic_calibration(scales=None):
    """Mode 3's (index 2, (n,m)=(5,4), triad {1,2,3}'s sum mode) own
    ``phi_j_tilde`` slope across the barotropic script's own established
    scale range -- locate the sign flip, if any.
    """
    if scales is None:
        scales = np.geomspace(5e-4, 1e-2, 20)

    print("=== Step 1: barotropic calibration (mode 3, (n,m)=(5,4)) ===")
    print(f"{'scale':>10} {'slope (rad/nondim-t)':>22} {'efficiency(%)':>14}")
    slopes = []
    prev_sign = None
    for s in scales:
        h = min(0.02, 0.2 / (max(1.0, s) * 6 + 1))
        A0 = s * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
        A1s, A2s, A3s, A4s, ts = [A0[0]], [A0[1]], [A0[2]], [A0[3]], [0.0]
        for t, A in baro._rk4(baro.rhs, 0, 1500.0, h, A0):
            A1s.append(A[0]); A2s.append(A[1]); A3s.append(A[2]); A4s.append(A[3])
            ts.append(t)
        Y = np.stack([np.array(A1s), np.array(A2s), np.array(A3s), np.array(A4s)], axis=1)
        T = np.array(ts)

        phi3_tilde = individual_phase(Y, 2) - baro.OMEGA3 * T
        slope = np.polyfit(T, phi3_tilde, 1)[0]
        slopes.append(slope)

        eff = baro.efficiency(s)
        sign = np.sign(slope)
        flip = ""
        if prev_sign is not None and sign != prev_sign and sign != 0:
            flip = "  <-- SIGN FLIP since previous row"
        prev_sign = sign
        print(f"{s:>10.2e} {slope:>22.6f} {100 * eff:>14.4f}{flip}")

    return np.asarray(scales), np.asarray(slopes)


def step2_quartet_b_rsw(scales=None, t_f=1500.0):
    """Same topology in RSW (``precession_resonance_rsw_vs_barotropic.build``),
    tracking mode 3's RSW analogue (RH3 = RH(4,5), index 2 -- triad1's own
    sum mode, matching the barotropic role) across that script's own
    established scale range, cached via ``run_and_cache``.
    """
    if scales is None:
        scales = np.array([1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0])

    ws = rsw_b.build()
    print("\n=== Step 2: Quartet B, RSW (RH3=RH(4,5), same topology) ===")
    print(f"{'scale':>10} {'slope (rad/day)':>18} {'efficiency(%)':>14}")
    slopes = []
    prev_sign = None
    for s in scales:
        h = min(0.02, 0.2 / (max(1.0, s) * 6 + 1))
        A0 = s * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
        # Explicit scale-based label -- shares a cache namespace (now
        # rsw_sphere.dynamics.trajectory_cache's own "quartets" topology
        # folder) with every other scale-driven Quartet B RSW call site
        # in this repo, see precession_resonance_phase_diagnostic.py's own comment.
        label = f"scale{s:.6g}_tf{t_f:.0f}_h{h:.5f}"
        Y, T, _ = run_and_cache(ws, A0, t_f, h, label=label)
        days = days_from_nondim_time(T)

        phi3 = individual_phase(Y, 2)
        slope = np.polyfit(days, phi3, 1)[0]
        slopes.append(slope)

        E = np.real(Y * np.conj(Y))
        eff = (E[:, 3].max() - E[:, 3].min()) / np.sum(E, axis=1).mean()
        sign = np.sign(slope)
        flip = ""
        if prev_sign is not None and sign != prev_sign and sign != 0:
            flip = "  <-- SIGN FLIP since previous row"
        prev_sign = sign
        print(f"{s:>10.2e} {slope:>18.6f} {100 * eff:>14.4f}{flip}")

    return np.asarray(scales), np.asarray(slopes)


def step3_quartet_a_rsw(u_values=(10.0, 50.0, 80.0, 85.0, 90.0, 120.0)):
    """Quartet A (``quartet_rh_preference``) -- every constituent mode's
    own individual-phase slope, at representative driving velocities
    spanning before/at/after the confirmed lock (u~83-92).
    """
    specs = load_wave_set_specs(DEFAULT_WAVESETS_PATH)
    spec = specs["quartet_rh_preference"]

    print("\n=== Step 3: Quartet A, RSW (quartet_rh_preference) ===")
    print(f"driving mode 'd' (RH(3,6)); reporting every mode's own slope (rad/day)\n")
    result = precession_frequency_efficiency(
        spec, "d", list(u_values), target_mode_key="c",
        individual_mode_keys=list(spec.mode_keys), tf_days=150.0)

    header = f"{'u':>7}" + "".join(f"  {mk}({spec.modes[spec.index(mk)]})".rjust(20) for mk in spec.mode_keys)
    print(header)
    prev_sign = {mk: None for mk in spec.mode_keys}
    for k, u in enumerate(u_values):
        parts = [f"{u:>7.1f}"]
        for mk in spec.mode_keys:
            slope = result["individual_slope"][mk][k]
            sign = np.sign(slope)
            flip = "*" if (prev_sign[mk] is not None and sign != prev_sign[mk] and sign != 0) else " "
            prev_sign[mk] = sign
            parts.append(f"{slope:>19.5f}{flip}")
        print("".join(parts))
    print("(* marks a sign flip relative to the previous row for that mode)")
    print(f"\nefficiency (mode c): {dict(zip(u_values, result['efficiency']))}")
    print(f"energy_drift: {dict(zip(u_values, result['energy_drift']))}")

    return result


if __name__ == "__main__":
    step1_barotropic_calibration()
    step2_quartet_b_rsw()
    step3_quartet_a_rsw()
