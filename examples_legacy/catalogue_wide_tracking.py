"""Extended tracking of frequency shift + amplitude effect (Fmax) across
the FULL 26-candidate catalogue, for BOTH modes on the shared edge
(RH(3,4), the pump/target used everywhere else in §3.3, AND RH(4,5),
the shared/sum mode) -- per user request 2026-08-14: "test also with
higher frequency modes, not just EG(1,1) and WG(1,1). Even if the
effect is small, we should track it" + "change the target ... to see
if anything arises."

Not just a coarser version of `frequency_shift_catalogue_search.py`'s
Stage 1 (which only measured x=0.3 and only target RH(3,4)): this
tracks 3 energy fractions x 2 targets x 26 candidates, and gets BOTH
targets from a SINGLE integration per (candidate, x) -- RK33 already
returns every mode's own trajectory, so extracting mode a's (RH(4,5))
own kinetic energy costs nothing extra once mode b's (RH(3,4)) is
already being computed.

$\\mathcal{F}_2$ is NOT recomputed here (already exists for target
RH(3,4) at all 26 candidates x 8 x-values in
`gate_i4_scaling_law_data.npy`, \\S3.3.3) -- only for target RH(4,5),
which has no existing data, is it computed fresh (from the same
trajectory as everything else here).

Cross-checks the FFT-based frequency shift against a second,
prominence-based estimator for every point (not just the two EG(1,1)/
WG(1,1) outliers `frequency_shift_catalogue_search.py` found) and flags
disagreement -- added after a first version of this script (and of
`alternate_topology_probe.py`) reported a spurious, implausibly large
single-estimator shift for a new-topology candidate, caught only by
this cross-check.

Run:

    python examples/catalogue_wide_tracking.py
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

from rsw_sphere.physics import (gamma_from_he, days_from_nondim_time,
                                 air_density_from_equivalent_depth, A, G)
from rsw_sphere.dynamics.integrators import RK44 as RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from gate_i2_map_extension import find_catalogue, A_MODE, B_MODE, C_MODE, H_E
from gate_i4_scaling_law import build_full_and_sub
from frequency_shift_catalogue_search import fft_period_parabolic, prominence_period

X_VALUES = (0.1, 0.3, 0.5)
TARGETS = {'RH(4,5)': 0, 'RH(3,4)': 1}  # a, b in the [a,b,c,d] mode ordering


def measure_both_targets(ws_full, ws_sub, x, e_tot, tf_days, h, prefactor):
    e_each = (1 - x) / 3 * e_tot
    A0_sub = np.sqrt(e_each) * np.ones(3, dtype=complex)
    A0_full = np.concatenate([A0_sub, [np.sqrt(x * e_tot)]]).astype(complex)
    t_f = tf_days * 4 * np.pi
    Yf, Tf = RK33(ws_full, 0, t_f, h, A0_full)
    Ys, Ts = RK33(ws_sub, 0, t_f, h, A0_sub)
    assert np.allclose(Tf, Ts)
    days = days_from_nondim_time(Tf)
    dt_days = days[1] - days[0]

    out = {}
    for name, i in TARGETS.items():
        KEf = np.abs(Yf[:, i]) ** 2
        KEs = np.abs(Ys[:, i]) ** 2
        Tf_fft = fft_period_parabolic(days, KEf, dt_days)
        Ts_fft = fft_period_parabolic(days, KEs, dt_days)
        shift_fft = 100 * (Tf_fft - Ts_fft) / Ts_fft if (Tf_fft and Ts_fft) else float('nan')
        Tf_prom = prominence_period(days, KEf)
        Ts_prom = prominence_period(days, KEs)
        shift_prom = 100 * (Tf_prom - Ts_prom) / Ts_prom if (Tf_prom and Ts_prom) else float('nan')
        agree = (np.isfinite(shift_fft) and np.isfinite(shift_prom)
                 and abs(shift_fft - shift_prom) < 1.0)
        EKf_peak = prefactor * KEf.max()
        EKs_peak = prefactor * KEs.max()
        fmax = 100 * (EKf_peak - EKs_peak) / EKs_peak
        # Range-based amplitude diagnostic (Delta EK = max-min, already
        # established via eq: Pa's own denominator): robust where raw
        # peak-KE degenerates for a NET-LOSER target mode, whose peak
        # sits at t=0 by construction (identical in both full and sub
        # trajectories regardless of the gravity mode's real effect --
        # caught directly, not assumed, for RH(4,5) under several
        # EG-family candidates in this catalogue's first run).
        dEKf = prefactor * (KEf.max() - KEf.min())
        dEKs = prefactor * (KEs.max() - KEs.min())
        frange = 100 * (dEKf - dEKs) / dEKs
        out[name] = (shift_fft, shift_prom, agree, fmax, frange)
    return out


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
    rho = air_density_from_equivalent_depth(H_E, g=G)
    prefactor = G * H_E ** 2 * A ** 2 * np.pi * rho

    rows = []
    t0 = time.time()
    for i, cand in enumerate(catalogue):
        ws_full, ws_sub = build_full_and_sub(gamma, cand)
        for x in X_VALUES:
            m = measure_both_targets(ws_full, ws_sub, x, e_tot, tf_days=120.0, h=0.01, prefactor=prefactor)
            sa, sap, oka, fa, ra = m['RH(4,5)']
            sb, sbp, okb, fb, rb = m['RH(3,4)']
            rows.append(dict(label=cand['label'], role=cand['role'], omega_d=cand['omega_d'],
                              delta_2=cand['delta_2'], x=x,
                              shift_a=sa, shift_a_prom=sap, agree_a=oka, fmax_a=fa, frange_a=ra,
                              shift_b=sb, shift_b_prom=sbp, agree_b=okb, fmax_b=fb, frange_b=rb))
        print(f"[{i+1:2d}/{len(catalogue)}] {cand['label']:>10}  ({time.time()-t0:.0f}s elapsed)")

    np.save(os.path.join(_ROOT, "catalogue_wide_tracking_data.npy"), rows, allow_pickle=True)
    print(f"\nSaved {len(rows)} rows to examples/catalogue_wide_tracking_data.npy")

    print(f"\n{'label':>10} {'x':>5} | {'shift_a':>8} {'ok?':>4} {'fmax_a':>8} {'frange_a':>9} | "
          f"{'shift_b':>8} {'ok?':>4} {'fmax_b':>8} {'frange_b':>9}")
    print("  (target a=RH(4,5), the shared/sum mode -- indirect channel only)")
    print("  (target b=RH(3,4), the pump mode used everywhere else in S3.3 -- both channels)")
    print("  (fmax = peak-KE diff; frange = Delta-EK diff, robust for net-loser targets)")
    for r in sorted(rows, key=lambda r: (abs(r['omega_d']), r['x'])):
        print(f"{r['label']:>10} {r['x']:>5.2f} | {r['shift_a']:>+7.2f}% {'Y' if r['agree_a'] else 'N':>4} "
              f"{r['fmax_a']:>+7.2f}% {r['frange_a']:>+8.2f}% | {r['shift_b']:>+7.2f}% "
              f"{'Y' if r['agree_b'] else 'N':>4} {r['fmax_b']:>+7.2f}% {r['frange_b']:>+8.2f}%")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")
