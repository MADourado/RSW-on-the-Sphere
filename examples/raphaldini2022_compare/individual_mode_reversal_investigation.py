"""Does either quartet built from Raphaldini et al. (2022)'s topology show
the individual-mode frequency reversal they report (Section III.A / Fig.
3 -- at high driving amplitude, one mode's own raw phase can bend enough
that its propagation direction flips sign)? Backs the live claim at
JFM-template.tex's ``sec: quartet_rh_precession`` ("no such reversal
appears at any scale tested, including through Quartet A's own lock").

Three steps, compute-first (report actual slopes/sign-flips before any
figure/paper prose):

1. Calibration -- reproduce their reversal directly in the barotropic
   system (``rsw_sphere.utilities.barotropic_vort_model``), scanning its
   own established scale range (5e-4..1e-2) with ``individual_phase`` +
   the barotropic-side ``phi_j_tilde`` correction
   (``individual_phase_tilde``).
2. Quartet B, RSW -- same topology
   (``precession_comparison.build_rsw_waveset``), run through
   ``rsw_sphere.dynamics.trajectory_cache.run_and_cache``, at points
   across the same script's own established scale range (1e-4..10).
3. Quartet A, RSW -- the paper's own native quartet
   (``quartet_rh_preference``), via
   ``rsw_sphere.utilities.precession.precession_frequency_efficiency``,
   at u=10,50,80,85,90,120 (spanning before/at/after the confirmed lock
   at u~83-92) -- every constituent mode's own phase slope, not just the
   driven one.

Run:

    python examples/raphaldini2022_compare/individual_mode_reversal_investigation.py
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
import rsw_sphere.utilities.barotropic_vort_model as baro

import precession_comparison as comp


def step1_barotropic_calibration(scales=None):
    """Mode 3's (index 2, (n,m)=(5,4), triad {1,2,3}'s sum mode) own
    ``phi_j_tilde`` slope across the barotropic model's own established
    scale range -- locate the sign flip, if any.
    """
    if scales is None:
        scales = np.geomspace(5e-4, 1e-2, 20)

    print("=== Step 1: barotropic calibration (mode 3, (n,m)=(5,4)) ===")
    print(f"{'scale':>10} {'slope (rad/nondim-t)':>22} {'efficiency(%)':>14}")
    slopes = []
    prev_sign = None
    for s in scales:
        Y, T = baro.integrate(s, t_f=1500.0)
        phi3_tilde = baro.individual_phase_tilde(Y, 2, baro.OMEGA3, T)
        slope = np.polyfit(T, phi3_tilde, 1)[0]
        slopes.append(slope)

        eff = baro.efficiency_from_trajectory(Y)
        sign = np.sign(slope)
        flip = ""
        if prev_sign is not None and sign != prev_sign and sign != 0:
            flip = "  <-- SIGN FLIP since previous row"
        prev_sign = sign
        print(f"{s:>10.2e} {slope:>22.6f} {100 * eff:>14.4f}{flip}")

    return np.asarray(scales), np.asarray(slopes)


def step2_quartet_b_rsw(scales=None, t_f=1500.0):
    """Same topology in RSW (``precession_comparison.build_rsw_waveset``),
    tracking mode 3's RSW analogue (RH3 = RH(4,5), index 2 -- triad1's own
    sum mode, matching the barotropic role) across the same established
    scale range, cached via ``run_and_cache``.
    """
    if scales is None:
        scales = np.array([1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0])

    ws, spec = comp.build_rsw_waveset()
    print("\n=== Step 2: Quartet B, RSW (RH3=RH(4,5), same topology) ===")
    print(f"{'scale':>10} {'slope (rad/day)':>18} {'efficiency(%)':>14}")
    slopes = []
    prev_sign = None
    rh3_idx = spec.index('a')  # RH(4,5), sum mode of triad 1
    target_idx = spec.index(comp.TARGET_MODE_KEY)
    for s in scales:
        Y, T = comp.rsw_trajectory(ws, s, t_f=t_f)
        days = days_from_nondim_time(T)

        phi3 = individual_phase(Y, rh3_idx)
        slope = np.polyfit(days, phi3, 1)[0]
        slopes.append(slope)

        E = np.real(Y * np.conj(Y))
        eff = (E[:, target_idx].max() - E[:, target_idx].min()) / np.sum(E, axis=1).mean()
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
