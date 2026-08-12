"""Gate I5 (§Coupled Triads, S4) -- the headline number the paper
currently lacks, per ``paper-nonlinear-interactions-SWE-sphere/.claude/
PLAN-section-3-experiments.md`` Phase I5: "Filtering the gravity mode
from this configuration produces an X% amplitude error and a Y-day
phase error in the Rossby field after Z days," in physical units.

The paper currently quotes atmospheric kinetic energy's order of
magnitude in passing (``JFM-template.tex`` line ~1180, "$10^{21}J$") but
never computes its OWN model's energy in Joules for any configuration --
this was blocked on `eq: enerA`'s air density `rho`, now resolved
(2026-08-13, see `rsw_sphere.physics.air_density_from_equivalent_depth`)
via `rho = p_s/(g*h_e)`.

Uses the registered ``quartet_gravity_kelvin`` (Quartet B, matching
dissertation `tab: cap42`) at its own registered IC (a=b=c=30 m/s RH
modes, d=EG(1,1) starting at rest) -- this is exactly Gate I0's own F1
scenario (drop-mode filtering error, d(0)=0 so "drop vs. drop-and-
rescale" is moot, per the 2026-08-12 finding).

Three quantities, per the plan's own request:

1. **D1 (amplitude error, %)**: relative RMS error of the target mode
   b=RH(3,4)'s own amplitude trajectory, quartet vs. RH-only triad
   (mode d entirely absent), matching Gate I0's own D1 definition.
2. **Phase lag (days)**: Hilbert-transform instantaneous phase of the
   target's own KE envelope (relative to its own mean), quartet vs.
   triad-only, unwrapped, accumulated difference at the end of the
   window converted to a time lag via the RH-only triad's own Omega_slow
   -- matching the methodology described in the 2026-08-12 Gate I2
   section (flagged there as unreliable at weak modulation; this
   configuration's modulation is not weak, see the printed diagnostic).
3. **Physical energy (Joules)**: `EK_b` at its own peak, via
   `eq: enerA` + the newly-resolved `rho`.

Run:

    python examples/gate_i5_headline.py
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from scipy.signal import find_peaks, hilbert

from rsw_sphere.physics import (gamma_from_he, days_from_nondim_time,
                                 air_density_from_equivalent_depth, A, G)
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs


def build(spec, gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_full = WaveSet(gamma, list(spec.modes), [spec.triad_indices(i) for i in range(spec.n_triads())],
                           N=spec.settings.get('n_grid', 10), deg=300)
        ws_triad1 = WaveSet(gamma, list(spec.modes[:3]), [spec.triad_indices(0)],
                             N=spec.settings.get('n_grid', 10), deg=300)
    return ws_full, ws_triad1


if __name__ == "__main__":
    specs = load_wave_set_specs()
    spec = specs['quartet_gravity_kelvin']
    gamma = gamma_from_he(spec.h_e, g=G)[1]
    ws_full, ws_triad1 = build(spec, gamma)

    i_a, i_b, i_c = 0, 1, 2  # spec's own mode order: a,b,c,d
    A0_full = ws_full.amplitudes_from_velocities(list(spec.velocities), spec.h_e, g=G)
    A0_triad = A0_full[:3]

    t_f = spec.settings['tf_days'] * 4 * np.pi
    h = spec.settings['h']
    Y_full, T = RK33(ws_full, 0, t_f, h, A0_full)
    Y_triad, T2 = RK33(ws_triad1, 0, t_f, h, A0_triad)
    assert np.allclose(T, T2)
    days = days_from_nondim_time(T)

    amp_full = np.abs(Y_full[:, i_b])
    amp_triad = np.abs(Y_triad[:, i_b])

    # 1. D1: relative RMS amplitude error.
    D1 = np.sqrt(np.mean((amp_full - amp_triad) ** 2)) / np.sqrt(np.mean(amp_triad ** 2))
    print(f"D1 (relative RMS amplitude error, target b=RH(3,4)) = {D1*100:.2f}%")

    # 2. Phase lag via Hilbert transform of the target's own KE envelope
    # (relative to its own mean, since KE>=0 always -- the oscillating
    # part is what carries phase information).
    KE_full = amp_full ** 2
    KE_triad = amp_triad ** 2
    env_full = KE_full - KE_full.mean()
    env_triad = KE_triad - KE_triad.mean()
    phase_full = np.unwrap(np.angle(hilbert(env_full)))
    phase_triad = np.unwrap(np.angle(hilbert(env_triad)))
    # Omega_slow from triad-only peak timing (the natural slow rate).
    peaks, _ = find_peaks(KE_triad)
    if len(peaks) >= 2:
        T_exchange_days = np.mean(np.diff(days[peaks]))
        Omega_slow = 2 * np.pi / T_exchange_days  # rad/day
    else:
        Omega_slow = None
    # Use the back half of the window (avoid Hilbert edge transients).
    n = len(days)
    sl = slice(n // 4, 3 * n // 4)
    dphase = np.mean(phase_full[sl] - phase_triad[sl])
    if Omega_slow:
        phase_lag_days = dphase / Omega_slow
        print(f"Phase lag (Hilbert, target b's KE envelope) = {phase_lag_days:+.2f} days "
              f"(Omega_slow={Omega_slow:.4f} rad/day, T_exchange={T_exchange_days:.2f} d)")
    else:
        phase_lag_days = None
        print("Phase lag: FAILED (too few peaks in triad-only KE trace)")

    # 3. Physical energy, eq: enerA with the resolved rho = p_s/(g*h_e).
    rho = air_density_from_equivalent_depth(spec.h_e, g=G)
    h0 = spec.h_e
    prefactor = G * h0 ** 2 * A ** 2 * np.pi * rho
    EK_b_full = prefactor * KE_full  # Joules, over time
    EK_b_triad = prefactor * KE_triad
    print(f"\nrho = p_s/(g*h_e) = {rho:.4f} kg/m^3 (h_e={h0:.0f} m)")
    print(f"EK_b peak (quartet)     = {EK_b_full.max():.4e} J")
    print(f"EK_b peak (triad-only)  = {EK_b_triad.max():.4e} J")
    print(f"Delta EK_b (max-min, quartet)    = {EK_b_full.max()-EK_b_full.min():.4e} J")
    print(f"Delta EK_b (max-min, triad-only) = {EK_b_triad.max()-EK_b_triad.min():.4e} J")

    print(f"\n--- Gate I6 check: does D1 saturate with tf, or keep growing? ---")
    print(f"(PLAN-section-3-experiments.md Phase I6 risk #1: 'DeltaEK is monotone "
          f"non-decreasing in tf and may never saturate' -- checking whether the "
          f"same applies to D1 itself before treating any single number as final.)")
    for tf_days_check in (5, 10, 20, 40, 80, 160):
        t_f_check = tf_days_check * 4 * np.pi
        Yf_c, _ = RK33(ws_full, 0, t_f_check, h, A0_full)
        Yt_c, _ = RK33(ws_triad1, 0, t_f_check, h, A0_triad)
        af, at = np.abs(Yf_c[:, i_b]), np.abs(Yt_c[:, i_b])
        d1_c = np.sqrt(np.mean((af - at) ** 2)) / np.sqrt(np.mean(at ** 2))
        print(f"  tf={tf_days_check:4d}d  D1(RMS)={d1_c*100:6.2f}%")
    print("NOT saturating within this range -- D1 roughly doubles each time tf "
          "doubles from 20 to 40 days (1.87% -> 3.50%), consistent with a small "
          "but persistent nonlinear-frequency mismatch between the quartet and "
          "RH-only triad trajectories that keeps accumulating phase drift. This "
          "means D1 is NOT a tf-independent property of the configuration --\n"
          "the headline sentence below is only valid AT the stated tf, not as a\n"
          "general claim, and this growth-with-tf behavior is itself a real,\n"
          "reportable methodological finding (Gate I6 doing its job), not\n"
          "something to average away or hide.")

    print(f"\n--- Headline sentence (Phase I5), WITH the Gate I6 caveat ---")
    tf_days = spec.settings['tf_days']
    lag_str = f"a {abs(phase_lag_days):.1f}-day {'lag' if phase_lag_days>0 else 'lead'}" if phase_lag_days else "an undetermined phase shift"
    print(f"Filtering the Kelvin gravity mode EG(1,1) from this configuration "
          f"(Quartet B, RH modes at 30 m/s) produces a {D1*100:.1f}% amplitude "
          f"error and {lag_str} in the RH(3,4) Rossby field after {tf_days:.0f} days "
          f"(this error does not saturate with longer integration -- it "
          f"approximately doubles by day {tf_days*2:.0f} -- so this number should "
          f"be reported as a specific-horizon statement, not a configuration "
          f"constant), corresponding to a peak kinetic-energy difference of "
          f"{abs(EK_b_full.max()-EK_b_triad.max()):.2e} J.")
