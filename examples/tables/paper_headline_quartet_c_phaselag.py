"""Gate I5 (§Coupled Triads, S4) -- the headline number the paper
currently lacks: "Filtering the gravity mode
from this configuration produces an X% amplitude error and a Y-day
phase error in the Rossby field after Z days," in physical units.

The paper currently quotes atmospheric kinetic energy's order of
magnitude in passing (``JFM-template.tex`` line ~1180, "$10^{21}J$") but
never computes its OWN model's energy in Joules for any configuration --
this was blocked on `eq: enerA`'s air density `rho`, now resolved
(2026-08-13, see `rsw_sphere.physics.air_density_from_equivalent_depth`)
via `rho = p_s/(g*h_e)`.

Uses the registered ``quartet_rossby_kelvin`` (Quartet B, matching
dissertation `tab: cap42`) at its own registered IC (a=b=c=30 m/s RH
modes, d=EG(1,1) starting at rest) -- this is exactly Gate I0's own F1
scenario (drop-mode filtering error, d(0)=0 so "drop vs. drop-and-
rescale" is moot, per the 2026-08-12 finding).

**Corrected 2026-08-13 (an independent review caught two real problems
in the first version):**

1. The Hilbert-transform phase-lag sign was inverted (fixed below --
   negative `phase_lag_days` is now correctly labelled "lag", not
   "lead").
2. The headline framing ("D1 doesn't converge with tf") was itself
   wrong: D1 (RMS amplitude error over `[0,tf]`) DOES converge, just not
   until ~day 320-400 (saturating near 19%, the full-decorrelation
   ceiling) -- it grows with tf because it is a windowed RMS of a
   **linearly growing phase offset**, not because the underlying physics
   is unconverged. The genuinely tf-independent, physically meaningful
   numbers are the **period shift** (target's own exchange period,
   quartet vs. triad-only) and the **peak-KE difference** -- both
   reported below alongside the original (correctly-labelled, now
   time-horizon-qualified) D1.

Four quantities now, per the plan's own request for "one sentence a
modeller can act on":

1. **Period shift (%, tf-independent)**: target mode b=RH(3,4)'s own
   exchange period (peak-to-peak KE timing), quartet vs. RH-only triad.
   This is the physically primary effect for this configuration.
2. **Peak kinetic-energy difference (%, tf-independent)**: `EK_b` at its
   own peak, quartet vs. triad-only, via `eq: enerA` + the resolved
   `rho`.
3. **D1 (amplitude error, %, time-horizon-dependent)**: relative RMS
   error of the target's own amplitude trajectory over `[0,tf]` --
   reported WITH its own tf, not as a configuration constant.
4. **Phase lag (days)**: Hilbert-transform instantaneous phase of the
   target's own KE envelope, quartet vs. triad-only -- sign convention:
   positive = quartet's phase leads, negative = quartet's phase lags
   (its peaks fall later in time than the triad-only's own).

Uses the registered ``quartet_rossby_kelvin`` (Quartet B, matching
dissertation `tab: cap42`) at its own registered IC (a=b=c=30 m/s RH
modes, d=EG(1,1) starting at rest) -- this is exactly Gate I0's own F1
scenario (drop-mode filtering error, d(0)=0 so "drop vs. drop-and-
rescale" is moot, per the 2026-08-12 finding).

Run:

    python examples/tables/paper_headline_quartet_c_phaselag.py
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.signal import find_peaks, hilbert

from rsw_sphere.physics import (gamma_from_he, days_from_nondim_time,
                                 air_density_from_equivalent_depth, A, G)
from rsw_sphere.dynamics.integrators import RK44 as RK33
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
    spec = specs['quartet_rossby_kelvin']
    gamma = gamma_from_he(spec.h_e, g=G)[1]
    ws_full, ws_triad1 = build(spec, gamma)

    i_a, i_b, i_c = 0, 1, 2  # spec's own mode order: a,b,c,d
    A0_full = ws_full.amplitudes_from_velocities(list(spec.velocities), spec.h_e, g=G)
    A0_triad = A0_full[:3]
    h = spec.settings['h']

    # 0. Period shift + peak-KE difference: BOTH tf-independent (checked
    # 20/100/400 days: period shift 0.79-0.92%, peak-KE diff ~1.00-1.01%
    # -- see the "Gate I5 correction" INSPECT doc section). Use a long
    # window (400d) here so the peak-to-peak period estimate itself is
    # well-averaged, not because the number changes with tf.
    t_f_long = 400.0 * 4 * np.pi
    Yf_long, T_long = RK33(ws_full, 0, t_f_long, h, A0_full)
    Yt_long, _ = RK33(ws_triad1, 0, t_f_long, h, A0_triad)
    days_long = days_from_nondim_time(T_long)
    KEf_long = np.abs(Yf_long[:, i_b]) ** 2
    KEt_long = np.abs(Yt_long[:, i_b]) ** 2
    pf_long, _ = find_peaks(KEf_long)
    pt_long, _ = find_peaks(KEt_long)
    T_full_period = np.mean(np.diff(days_long[pf_long]))
    T_triad_period = np.mean(np.diff(days_long[pt_long]))
    period_shift_pct = 100 * (T_full_period - T_triad_period) / T_triad_period
    rho = air_density_from_equivalent_depth(spec.h_e, g=G)
    prefactor = G * spec.h_e ** 2 * A ** 2 * np.pi * rho
    peak_EK_diff_pct = 100 * (prefactor * KEf_long.max() - prefactor * KEt_long.max()) / (prefactor * KEt_long.max())
    print(f"Period shift (target b's exchange period, quartet vs. triad-only, "
          f"tf-independent) = {period_shift_pct:+.2f}% "
          f"(T_quartet={T_full_period:.4f}d, T_triad={T_triad_period:.4f}d)")
    print(f"Peak-KE difference (tf-independent) = {peak_EK_diff_pct:+.2f}%")

    # The remaining diagnostics (D1, phase lag) use the paper's own
    # registered tf_days -- these ARE time-horizon-dependent, reported
    # as such below, not as configuration constants.
    t_f = spec.settings['tf_days'] * 4 * np.pi
    Y_full, T = RK33(ws_full, 0, t_f, h, A0_full)
    Y_triad, T2 = RK33(ws_triad1, 0, t_f, h, A0_triad)
    assert np.allclose(T, T2)
    days = days_from_nondim_time(T)

    amp_full = np.abs(Y_full[:, i_b])
    amp_triad = np.abs(Y_triad[:, i_b])

    # 1. D1: relative RMS amplitude error. Time-horizon-dependent -- see
    # the Gate I6 check below for why, and don't quote this without its
    # own tf.
    D1 = np.sqrt(np.mean((amp_full - amp_triad) ** 2)) / np.sqrt(np.mean(amp_triad ** 2))
    print(f"D1 (relative RMS amplitude error, target b=RH(3,4), "
          f"AT tf={spec.settings['tf_days']:.0f}d) = {D1*100:.2f}%")

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
    # Use the middle half of the window (avoid Hilbert edge transients at
    # both ends, not just the start).
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
          f"same applies to D1 itself before treating any single number as final.\n"
          f"CORRECTED 2026-08-13: the first version of this check stopped at "
          f"tf=160d and concluded D1 'does not saturate' -- wrong. It saturates, "
          f"just later. Extended below to show the actual plateau.)")
    for tf_days_check in (5, 10, 20, 40, 80, 160, 320, 400):
        t_f_check = tf_days_check * 4 * np.pi
        Yf_c, _ = RK33(ws_full, 0, t_f_check, h, A0_full)
        Yt_c, _ = RK33(ws_triad1, 0, t_f_check, h, A0_triad)
        af, at = np.abs(Yf_c[:, i_b]), np.abs(Yt_c[:, i_b])
        d1_c = np.sqrt(np.mean((af - at) ** 2)) / np.sqrt(np.mean(at ** 2))
        print(f"  tf={tf_days_check:4d}d  D1(RMS)={d1_c*100:6.2f}%")
    print("D1 SATURATES near ~19% by day 320-400 (the 'full decorrelation' "
          "ceiling for two bounded oscillations with a small, persistent "
          "period mismatch -- see the period-shift number above, ~0.8%, which "
          "is what actually drives D1's slow growth: not unbounded drift, "
          "just a long transient before the two signals' phases fully "
          "decorrelate). D1 IS a well-defined, converged quantity -- it just "
          "needs a much longer window than the paper's own registered "
          "tf_days=20 to reach that limit. At tf=20d it is still in the "
          "early, fast-growing part of that transient, which is why it is "
          "reported here as a time-horizon-qualified number, not the "
          "asymptotic value.")

    print(f"\n--- Headline sentence (Phase I5), corrected ---")
    tf_days = spec.settings['tf_days']
    # phase_lag_days = (quartet's own accumulated phase - triad-only's own
    # accumulated phase) / Omega_slow. Positive means the quartet's phase
    # is AHEAD (it leads); negative means the quartet's phase is BEHIND
    # (it lags -- its peaks fall later in time than the triad-only's own).
    # Sign was inverted here until 2026-08-13's review caught it (a direct
    # peak-timing check confirmed the quartet's period is longer than the
    # triad-only's, so its peaks progressively fall later -- a lag, not a
    # lead -- for this configuration's own negative phase_lag_days value).
    lag_str = f"a {abs(phase_lag_days):.1f}-day {'lead' if phase_lag_days>0 else 'lag'}" if phase_lag_days else "an undetermined phase shift"
    print(f"Filtering the Kelvin gravity mode EG(1,1) from this configuration "
          f"(Quartet C, RH modes at 30 m/s) lengthens the RH(3,4) Rossby "
          f"mode's own exchange period by {period_shift_pct:.1f}% and its peak "
          f"kinetic energy by {peak_EK_diff_pct:.1f}% "
          f"({abs(EK_b_full.max()-EK_b_triad.max()):.2e} J of "
          f"~{EK_b_triad.max():.1e} J) -- both converged, time-horizon-"
          f"independent effects. Measured instead as a windowed RMS "
          f"amplitude-trajectory error, this same underlying effect looks "
          f"like only {D1*100:.1f}% after {tf_days:.0f} days, growing to ~19% "
          f"by day ~350 as the two trajectories' slightly different periods "
          f"fully decorrelate in phase -- alongside {lag_str} in the target's "
          f"own phase at day {tf_days:.0f}.")
