"""Systematic search for a genuine fast-gravity / frequency-shift effect
on the Rossby target RH(3,4), extending the 2026-08-11 user request
(short_gravity_long_rossby_example.py) beyond one hand-picked mode
(WG(7,9), whose own reported 41-45% shift was RETRACTED 2026-08-13 as a
Savitzky-Golay smoothing artifact -- the window-independent estimators
built for that retraction, FFT-with-parabolic-interpolation and
prominence-filtered peak timing, both gave <=0.1% for WG(7,9)).

This script uses ONLY those two window-independent estimators (never
Savitzky-Golay), applied SYMMETRICALLY to both the full quartet and the
RH-only sub-triad (the retraction's other root cause was an asymmetric
pipeline -- quartet trace smoothed, triad-only trace not), and searches
the FULL 26-candidate Gate I3 catalogue (edge RH(4,5)+RH(3,4)) rather
than one candidate, in two stages:

  Stage 1 (cheap): all 26 candidates, one representative x=0.3, both
    estimators, flagging any candidate where they disagree (a sign of a
    measurement artifact, not a real effect).
  Stage 2 (targeted): the top candidates from Stage 1 get a full x-sweep
    plus a tf convergence check before any effect is trusted.

Run:

    python examples/frequency_shift_catalogue_search.py
"""
import os
import sys
import time
import warnings

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from scipy.signal import find_peaks

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time
from rsw_sphere.dynamics.integrators import RK44 as RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from gate_i2_map_extension import find_catalogue, A_MODE, B_MODE, C_MODE, G, H_E
from gate_i4_scaling_law import build_full_and_sub

I_B = 1  # target mode b = RH(3,4)


def fft_period_parabolic(t_days, ke, dt_days):
    """Dominant period via real FFT of the mean-subtracted series, with
    parabolic interpolation of the peak bin against its two neighbors --
    window-independent, no tunable smoothing parameter. Returns period
    in days.
    """
    x = ke - ke.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=dt_days)
    # Ignore the DC bin and any bin below a resolvable minimum period
    # (2*dt, Nyquist) -- and ignore very-low-frequency bins (period
    # longer than a third of the window) which are unreliable.
    valid = (freqs > 1.0 / (t_days[-1] / 3)) & (freqs < 1.0 / (2 * dt_days))
    if not np.any(valid):
        return None
    idx_candidates = np.where(valid)[0]
    k = idx_candidates[np.argmax(spec[idx_candidates])]
    if k <= 0 or k >= len(spec) - 1:
        return 1.0 / freqs[k]
    y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-300 else 0.0
    delta = np.clip(delta, -0.5, 0.5)
    f_refined = freqs[k] + delta * (freqs[1] - freqs[0])
    return 1.0 / f_refined if f_refined > 0 else None


def prominence_period(t_days, ke):
    """Peak-to-peak period via prominence-filtered peaks on the RAW
    (unsmoothed) trace -- prominence threshold set relative to the
    trace's own amplitude range, not an absolute/tuned constant.
    """
    prom = 0.1 * (ke.max() - ke.min())
    peaks, _ = find_peaks(ke, prominence=prom)
    if len(peaks) < 3:
        return None
    return np.mean(np.diff(t_days[peaks]))


def measure_periods(ws_full, ws_sub, x, e_tot, tf_days, h=0.01):
    e_each = (1 - x) / 3 * e_tot
    A0_sub = np.sqrt(e_each) * np.ones(3, dtype=complex)
    A0_full = np.concatenate([A0_sub, [np.sqrt(x * e_tot)]]).astype(complex)
    t_f = tf_days * 4 * np.pi
    Yf, Tf = RK33(ws_full, 0, t_f, h, A0_full)
    Ys, Ts = RK33(ws_sub, 0, t_f, h, A0_sub)
    assert np.allclose(Tf, Ts)
    days = days_from_nondim_time(Tf)
    dt_days = days[1] - days[0]
    KEf = np.abs(Yf[:, I_B]) ** 2
    KEs = np.abs(Ys[:, I_B]) ** 2

    Tf_fft = fft_period_parabolic(days, KEf, dt_days)
    Ts_fft = fft_period_parabolic(days, KEs, dt_days)
    Tf_prom = prominence_period(days, KEf)
    Ts_prom = prominence_period(days, KEs)
    return Tf_fft, Ts_fft, Tf_prom, Ts_prom


if __name__ == "__main__":
    gamma = gamma_from_he(H_E, g=G)[1]
    print("Building catalogue...")
    catalogue = find_catalogue(gamma)
    print(f"{len(catalogue)} candidates\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_baseline = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    A0_base = ws_baseline.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))

    print("=== Stage 1: all 26 candidates, x=0.3, tf=120d ===")
    print(f"{'label':>10} {'T_fft_full':>11} {'T_fft_sub':>10} {'shift_fft':>10} "
          f"{'T_prom_full':>12} {'T_prom_sub':>11} {'shift_prom':>11} {'agree?':>7}")
    t0 = time.time()
    results = []
    for cand in catalogue:
        ws_full, ws_sub = build_full_and_sub(gamma, cand)
        Tf_fft, Ts_fft, Tf_prom, Ts_prom = measure_periods(ws_full, ws_sub, 0.3, e_tot, tf_days=120.0)
        shift_fft = 100 * (Tf_fft - Ts_fft) / Ts_fft if (Tf_fft and Ts_fft) else float('nan')
        shift_prom = 100 * (Tf_prom - Ts_prom) / Ts_prom if (Tf_prom and Ts_prom) else float('nan')
        agree = abs(shift_fft - shift_prom) < 1.0 if np.isfinite(shift_fft) and np.isfinite(shift_prom) else False
        results.append(dict(cand, shift_fft=shift_fft, shift_prom=shift_prom, agree=agree))
        print(f"{cand['label']:>10} {Tf_fft or float('nan'):>11.4f} {Ts_fft or float('nan'):>10.4f} "
              f"{shift_fft:>+9.2f}% {Tf_prom or float('nan'):>12.4f} {Ts_prom or float('nan'):>11.4f} "
              f"{shift_prom:>+10.2f}% {'YES' if agree else 'NO':>7}")
    print(f"\nStage 1 done in {time.time()-t0:.1f}s")

    np.save(os.path.join(_ROOT, "frequency_shift_stage1.npy"), results, allow_pickle=True)

    reliable = [r for r in results if r['agree']]
    reliable.sort(key=lambda r: -abs(r['shift_fft']))
    print(f"\n=== Ranked by |shift| among the {len(reliable)}/{len(results)} candidates where "
          f"both estimators agree (within 1pp) ===")
    for r in reliable[:10]:
        print(f"  {r['label']:>10}  shift_fft={r['shift_fft']:+.2f}%  omega_d={r['omega_d']:+.4f}  "
              f"delta_2={r['delta_2']:+.4f}")

    disagreeing = [r for r in results if not r['agree']]
    if disagreeing:
        print(f"\n{len(disagreeing)} candidates where estimators DISAGREE (>1pp) -- "
              f"treat these as unreliable measurements, not necessarily large effects:")
        for r in disagreeing:
            print(f"  {r['label']:>10}  shift_fft={r['shift_fft']:+.2f}%  shift_prom={r['shift_prom']:+.2f}%")
