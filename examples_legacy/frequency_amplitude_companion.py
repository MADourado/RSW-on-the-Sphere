"""Companion to §3.3.5's frequency-shift finding (`frequency_shift_
catalogue_search.py` / `frequency_shift_stage2.py`): for EG(1,1) and
WG(1,1) -- the only two catalogue candidates with a real frequency
effect -- reports the SAME diagnostic triple used everywhere else in
§3.3 (\\S3.1's own $\\mathcal{F}_2^a$/$\\mathcal{F}_{max}^a$) alongside the
frequency shift, at matching energy fractions $x$, per the user's
2026-08-14 request ("check the amplitude effects too, not just
frequency").

$\\mathcal{F}_2^a$ itself was already computed for the full catalogue in
`gate_i4_scaling_law_data.npy` (reused here, not recomputed);
$\\mathcal{F}_{max}^a$ (signed peak-KE difference, eq. Fmax) was not --
this script adds it, at the same $x$ values, same $t_f=20$d horizon as
$\\mathcal{F}_2^a$ elsewhere in this section.

Run:

    python examples/frequency_amplitude_companion.py
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

from rsw_sphere.physics import gamma_from_he, air_density_from_equivalent_depth, A, G
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from gate_i2_map_extension import find_catalogue, A_MODE, B_MODE, C_MODE, H_E
from gate_i4_scaling_law import build_full_and_sub

X_VALUES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
TABLE_X = (0.1, 0.3, 0.5)  # the 3 rows quoted in Table freq_amp
LABELS = ('EG(1,1)', 'WG(1,1)')


def fmax_sweep(ws_full, ws_sub, e_tot, tf_days=20.0, h=0.01):
    rho = air_density_from_equivalent_depth(H_E, g=G)
    prefactor = G * H_E ** 2 * A ** 2 * np.pi * rho
    out = {}
    for x in X_VALUES:
        e_each = (1 - x) / 3 * e_tot
        A0_sub = np.sqrt(e_each) * np.ones(3, dtype=complex)
        A0_full = np.concatenate([A0_sub, [np.sqrt(x * e_tot)]]).astype(complex)
        t_f = tf_days * 4 * np.pi
        Yf, _ = RK33(ws_full, 0, t_f, h, A0_full)
        Ys, _ = RK33(ws_sub, 0, t_f, h, A0_sub)
        EKf = prefactor * (np.abs(Yf[:, 1]) ** 2).max()
        EKs = prefactor * (np.abs(Ys[:, 1]) ** 2).max()
        out[x] = 100 * (EKf - EKs) / EKs
    return out


if __name__ == "__main__":
    gamma = gamma_from_he(H_E, g=G)[1]
    catalogue = find_catalogue(gamma)
    f2_rows = {r['label']: r for r in np.load(
        os.path.join(_ROOT, "gate_i4_scaling_law_data.npy"), allow_pickle=True)}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_baseline = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    A0_base = ws_baseline.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))

    print(f"{'mode':>8} {'x':>5} {'F2 (%)':>8} {'Fmax (%)':>9}")
    for label in LABELS:
        cand = next(c for c in catalogue if c['label'] == label)
        ws_full, ws_sub = build_full_and_sub(gamma, cand)
        fmax = fmax_sweep(ws_full, ws_sub, e_tot)
        f2_vals = dict(zip(X_VALUES, f2_rows[label]['f2_vals']))
        for x in TABLE_X:
            print(f"{label:>8} {x:>5.2f} {f2_vals[x]*100:>8.2f} {fmax[x]:>+9.2f}")
        print(f"  (full x-sweep, {label}): " +
              " ".join(f"x={x:.2f}:Fmax={fmax[x]:+.2f}%" for x in X_VALUES))
