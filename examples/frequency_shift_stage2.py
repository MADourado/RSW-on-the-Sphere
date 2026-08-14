"""Stage 2 of the fast-gravity/frequency-shift catalogue search
(`frequency_shift_catalogue_search.py`'s own Stage 1 screen, redone here
with x-sweeps and tf-convergence checks for the top-ranked candidates,
before trusting any single number -- exactly the discipline that would
have caught the WG(7,9) Savitzky-Golay artifact immediately, since a
real effect must (a) hold up across estimators, (b) hold up across tf,
and (c) vary smoothly with x, none of which the retracted claim did).

Edit CANDIDATE_LABELS below to match Stage 1's own top-ranked, agreeing
candidates before running.

Run:

    python examples/frequency_shift_stage2.py
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

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.wave_sets import WaveSet
from gate_i2_map_extension import find_catalogue, A_MODE, B_MODE, C_MODE, G, H_E
from gate_i4_scaling_law import build_full_and_sub
from frequency_shift_catalogue_search import measure_periods

CANDIDATE_LABELS = ['EG(1,1)', 'WG(1,1)']  # Stage 1's only two "real effect" candidates
X_SWEEP = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)

if __name__ == "__main__":
    if not CANDIDATE_LABELS:
        print("Set CANDIDATE_LABELS from Stage 1's ranked output first.")
        sys.exit(1)

    gamma = gamma_from_he(H_E, g=G)[1]
    catalogue = find_catalogue(gamma)
    by_label = {c['label']: c for c in catalogue}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_baseline = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    A0_base = ws_baseline.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))

    for label in CANDIDATE_LABELS:
        cand = by_label[label]
        print(f"\n=== {label} (omega_d={cand['omega_d']:+.4f}, delta_2={cand['delta_2']:+.4f}) ===")
        ws_full, ws_sub = build_full_and_sub(gamma, cand)

        print(f"{'x':>5} {'shift_fft':>10} {'shift_prom':>11} {'agree?':>7}")
        t0 = time.time()
        for x in X_SWEEP:
            Tf_fft, Ts_fft, Tf_prom, Ts_prom = measure_periods(ws_full, ws_sub, x, e_tot, tf_days=120.0)
            shift_fft = 100 * (Tf_fft - Ts_fft) / Ts_fft if (Tf_fft and Ts_fft) else float('nan')
            shift_prom = 100 * (Tf_prom - Ts_prom) / Ts_prom if (Tf_prom and Ts_prom) else float('nan')
            agree = np.isfinite(shift_fft) and np.isfinite(shift_prom) and abs(shift_fft - shift_prom) < 1.0
            print(f"{x:>5.2f} {shift_fft:>+9.2f}% {shift_prom:>+10.2f}% {'YES' if agree else 'NO':>7}")
        print(f"  ({time.time()-t0:.1f}s)")

        print(f"tf convergence check at x=0.5:")
        for tf_days in (60.0, 120.0, 240.0, 480.0):
            Tf_fft, Ts_fft, Tf_prom, Ts_prom = measure_periods(ws_full, ws_sub, 0.5, e_tot, tf_days=tf_days)
            shift_fft = 100 * (Tf_fft - Ts_fft) / Ts_fft if (Tf_fft and Ts_fft) else float('nan')
            shift_prom = 100 * (Tf_prom - Ts_prom) / Ts_prom if (Tf_prom and Ts_prom) else float('nan')
            print(f"  tf={tf_days:5.0f}d  shift_fft={shift_fft:+.3f}%  shift_prom={shift_prom:+.3f}%")
