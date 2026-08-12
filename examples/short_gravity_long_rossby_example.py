"""Search for, and demonstrate, the user's requested "genuinely
short-period gravity wave affecting long-period Rossby waves in BOTH
amplitude and frequency" example (memory `project-short-gravity-long-
rossby-quartet-idea`, flagged 2026-08-11: existing registered quartets
use relatively long/slow Kelvin-like gravity modes and only demonstrate
amplitude modulation -- frequency shift has never been shown, and the
memory note explicitly says not to just re-pick a shorter-period EG mode
ad hoc without checking the coupling actually produces a visible effect.

**Candidate selection uses Gate I2's own extended catalogue**
(`examples/gate_i2_map_data.npy`, 26 candidates closing edge
RH(4,5)+RH(3,4)) rather than guessing: ranked by (a) short period and
(b) the two-channel law's own predicted D1, `WG(7,9)` is the clear
winner -- period 3.66 hours (vs. EG(1,1)'s already-registered 32.46
hours = 1.35 days), and the SECOND-highest predicted D1 in the entire
catalogue (only EG(7,9) itself, its eastward counterpart, predicts
higher). It is also a WESTWARD gravity mode, filling a gap
`PLAN-section-3-experiments.md` explicitly flags ("the paper is
currently EG-only with no stated reason；teruya2023wavenumber... reports
unexpected westward-gravity behaviour").

**Result: a genuine, sizeable frequency-shift effect, not previously
demonstrated.** At x=0.5 (half the total energy on the gravity mode),
the target Rossby mode's own exchange period shortens from ~4.25 days
(RH-only triad) to ~2.3-2.5 days (full quartet) -- a 41-45% frequency
shift (range reflects a real, if modest, tf-sensitivity: h-converged
cleanly, but tf=30d vs. 60d gives 45.1% vs. 40.8% -- reported as a range,
not a false-precision single number). This effect is essentially absent
at low x (x=0.1: -0.04%, noise-level) and grows smoothly with x.

**A genuinely new finding about the existing diagnostic, not just a new
example**: D1 (the amplitude-error diagnostic used everywhere else in
this investigation) stays modest for this candidate (1.3-2.9% measured,
noticeably below the two-channel law's own naive prediction of ~17% at
this x -- because D1, an RMS-of-amplitude-difference-over-a-fixed-window
metric, does not cleanly separate a "the two signals have similar
amplitude but different phase/frequency" effect from "the two signals
have different amplitude" -- a large frequency shift can leave D1
looking unremarkable even though the underlying physics is dramatic).
**This is exactly the point the original memory note anticipated**: a
genuinely fast, strongly-coupled gravity mode needs its OWN diagnostic
(frequency shift, measured directly here via peak-to-peak timing, not
inferred from D1) to be visible at all.

Run:

    python examples/short_gravity_long_rossby_example.py
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
from scipy.signal import find_peaks, savgol_filter

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, G
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet

H_E = 10000.0
A_MODE, B_MODE, C_MODE = (4, 5, 3), (3, 4, 3), (1, 2, 3)
D_MODE = (7, 9, 2)  # WG(7,9): period 3.66h, 2nd-highest predicted D1 in the catalogue


def build():
    gamma = gamma_from_he(H_E, g=G)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_full = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, D_MODE], [(0, 1, 2), (3, 0, 1)], N=10, deg=300)
        ws_triad = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    return ws_full, ws_triad


def measure(ws_full, ws_triad, A0_full, A0_triad, tf_days, h=0.01):
    """T_exchange (days) for target b, quartet vs. RH-only triad, and D1
    (relative RMS amplitude error). The quartet's own KE trace is
    smoothed (Savitzky-Golay, ~1-day window) before peak-detecting the
    SLOW envelope -- otherwise the fast gravity-mode ripple (period
    ~0.15 days) dominates the peak count and corrupts the period
    measurement.
    """
    t_f = tf_days * 4 * np.pi
    Yf, T = RK33(ws_full, 0, t_f, h, A0_full)
    Yt, T2 = RK33(ws_triad, 0, t_f, h, A0_triad)
    days = days_from_nondim_time(T)
    amp_f, amp_t = np.abs(Yf[:, 1]), np.abs(Yt[:, 1])
    KEf, KEt = amp_f ** 2, amp_t ** 2
    dt_days = days[1] - days[0]
    win = int(1.0 / dt_days)
    win = win if win % 2 == 1 else win + 1
    KEf_smooth = savgol_filter(KEf, win, 2)
    pf, _ = find_peaks(KEf_smooth)
    pt, _ = find_peaks(KEt)
    T_full = np.mean(np.diff(days[pf])) if len(pf) >= 2 else None
    T_triad = np.mean(np.diff(days[pt])) if len(pt) >= 2 else None
    D1 = np.sqrt(np.mean((amp_f - amp_t) ** 2)) / np.sqrt(np.mean(amp_t ** 2))
    return T_full, T_triad, D1


if __name__ == "__main__":
    ws_full, ws_triad = build()
    print(f"WG(7,9): omega_d={ws_full.omega[3]:.4f}, delta_2={ws_full.delta[1]:.4f}, "
          f"period={0.5/abs(ws_full.omega[3])*24:.2f} hours")

    A0_base = ws_triad.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))

    print(f"\n{'x':>5} {'T_quartet(d)':>13} {'T_triad(d)':>11} {'freq shift':>11} {'D1':>8}")
    for x in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        e_each, e_d = (1 - x) / 3 * e_tot, x * e_tot
        A0_full = np.array([np.sqrt(e_each)] * 3 + [np.sqrt(e_d)], dtype=complex)
        A0_triad = A0_full[:3]
        Tf, Tt, D1 = measure(ws_full, ws_triad, A0_full, A0_triad, tf_days=30.0)
        shift = 100 * (Tt - Tf) / Tt if Tf and Tt else float('nan')
        print(f"{x:>5.1f} {Tf:>13.4f} {Tt:>11.4f} {shift:>+10.1f}% {D1*100:>7.2f}%")

    print("\nConvergence check at x=0.5 (tf=30 vs 60 days, h=0.01 vs 0.001):")
    x = 0.5
    e_each, e_d = (1 - x) / 3 * e_tot, x * e_tot
    A0_full = np.array([np.sqrt(e_each)] * 3 + [np.sqrt(e_d)], dtype=complex)
    A0_triad = A0_full[:3]
    for tf_days, h, label in ((30, 0.01, 'base'), (60, 0.01, '2x tf'), (30, 0.001, 'fine h')):
        Tf, Tt, D1 = measure(ws_full, ws_triad, A0_full, A0_triad, tf_days, h)
        shift = 100 * (Tt - Tf) / Tt
        print(f"  {label:8s} (tf={tf_days},h={h}): shift={shift:+.2f}%")
