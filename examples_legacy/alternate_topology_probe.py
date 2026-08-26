"""Alternate quartet topology probe -- per user request 2026-08-14:
"change the target and control mode structure to see if anything
arises in a different experiment of topology/energy flux."

Every quartet in §3 so far pairs the shared edge with EITHER two RH
modes (Quartet A/B, all-Rossby) OR a gravity mode on the
RH(4,5)+RH(3,4) edge (Quartet C/D, this section). Nobody has tried a
gravity mode on Quartet A's OWN edge, RH(4,5)+RH(1,2) -- a genuinely
new topology. This script builds a small probe catalogue there (not
the full n<=15 sweep of `gate_i2_map_extension.py`'s own
`find_catalogue` -- a first look, to see whether S3.3.5's "small
timescale separation -> large frequency+amplitude effect" finding is
specific to the RH(4,5)+RH(3,4) edge or a more general phenomenon).

**Design note (caught by a sanity check before the full run, not
afterward):** the first version of this script used a bare 3-mode
"edge triad" {RH(4,5), RH(1,2), gravity} and tried to build a
gravity-absent baseline by zeroing the gravity mode's OWN initial
amplitude. That is not a valid baseline for a 3-mode system: with only
one triad, `dA_gravity/dt` is still driven by `A_a*A_b` regardless of
the gravity mode's own IC, so it does not stay zero -- "zero A_d(0)"
does not mean "gravity mode absent" the way it does in a genuine
4-mode quartet with two separate triads (Gate I5's own scenario).
Fixed by using the SAME structure as `gate_i4_scaling_law.py`: a full
4-mode quartet {RH(4,5), RH(1,2), RH(3,4), gravity}, where triad1 =
{RH(4,5),RH(1,2),RH(3,4)} is EXACTLY Quartet A's own registered triad1
(`quartet_rh_preference`, sum=RH(4,5)) -- a genuine, already-verified
RH-only reference -- and triad2 = {RH(4,5),RH(1,2),gravity} is the new
one. Dropping the gravity mode now means literally not including it in
the sub-WaveSet, matching every other comparison in this paper.

Selection rule for triad2: a gravity mode d closes a triad with
{RH(4,5) (m=4), RH(1,2) (m=1)} either as MEMBER (RH(4,5) is sum,
m_d = 4-1 = 3) or as SUM (m_d = 4+1 = 5).

Run:

    python examples/alternate_topology_probe.py
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
from rsw_sphere.dynamics.dynamic_triads import TRIAD
from frequency_shift_catalogue_search import fft_period_parabolic, prominence_period

H_E = 10000.0
A_MODE, B_MODE, C_MODE = (4, 5, 3), (1, 2, 3), (3, 4, 3)  # RH(4,5), RH(1,2), RH(3,4) -- Quartet A's own triad1
X_VALUES = (0.1, 0.3, 0.5)
TARGETS = {'RH(4,5)': 0, 'RH(1,2)': 1}


def find_probe_catalogue(gamma, n_max=11):
    """Small probe: member role (m_d=3, RH(4,5) is sum) and sum role
    (m_d=5), n up to 11, EG/WG, selection-rule survivors only, checked
    against the SAME edge/member pair TRIAD sees at runtime (not the
    C_MODE=RH(3,4) triad, which is unrelated to this selection check).
    """
    candidates = []
    for role, m_d in (('member', 3), ('sum', 5)):
        for alpha_d in (1, 2):
            for n in range(1, n_max + 1):
                if n < m_d:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if role == 'member':
                            t = TRIAD(gamma, *B_MODE, *(m_d, n, alpha_d), *A_MODE, N=10, deg=300)
                        else:
                            t = TRIAD(gamma, *A_MODE, *B_MODE, *(m_d, n, alpha_d), N=10, deg=300)
                        coup_p, coup_s = abs(t.coef_ABC), abs(t.coef_CAB)
                except Exception:
                    continue
                if coup_p < 1e-6 and coup_s < 1e-6:
                    continue
                candidates.append(dict(label=f"{'EG' if alpha_d == 1 else 'WG'}({m_d},{n})",
                                        role=role, m_d=m_d, n=n, alpha_d=alpha_d))
    return candidates


def build_full_and_sub(gamma, cand):
    """4-mode quartet {RH(4,5), RH(1,2), RH(3,4), gravity} + the
    3-mode RH-only sub-triad {RH(4,5), RH(1,2), RH(3,4)} -- Quartet
    A's own triad1, reused exactly (sum=RH(4,5), members=RH(1,2)/RH(3,4)).
    """
    m_d, n, alpha_d = cand['m_d'], cand['n'], cand['alpha_d']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cand['role'] == 'member':
            ws_full = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (m_d, n, alpha_d)],
                               [(0, 1, 2), (0, 1, 3)], N=10, deg=300)
        else:
            ws_full = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (m_d, n, alpha_d)],
                               [(0, 1, 2), (3, 0, 1)], N=10, deg=300)
        ws_sub = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    return ws_full, ws_sub


def measure(ws_full, ws_sub, x, e_tot, prefactor, tf_days=120.0, h=0.01):
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
        fmax = 100 * (prefactor * KEf.max() - prefactor * KEs.max()) / (prefactor * KEs.max())
        # Range-based (Delta EK) amplitude diagnostic -- see
        # catalogue_wide_tracking.py's own note: robust where raw
        # peak-KE degenerates for a net-loser target mode (peak at
        # t=0, identical in both trajectories by construction).
        dEKf = prefactor * (KEf.max() - KEf.min())
        dEKs = prefactor * (KEs.max() - KEs.min())
        frange = 100 * (dEKf - dEKs) / dEKs
        out[name] = (shift_fft, shift_prom, agree, fmax, frange)
    return out


if __name__ == "__main__":
    gamma = gamma_from_he(H_E, g=G)[1]
    print("Building probe catalogue on edge RH(4,5)+RH(1,2) (Quartet A's own edge)...")
    catalogue = find_probe_catalogue(gamma)
    print(f"{len(catalogue)} candidates: " + ", ".join(c['label'] for c in catalogue) + "\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_baseline = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    A0_base = ws_baseline.amplitudes_from_velocities([40.0, 40.0, 0.0], H_E, g=G)  # matches quartet_rh_preference's own registered a/b velocity
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))
    rho = air_density_from_equivalent_depth(H_E, g=G)
    prefactor = G * H_E ** 2 * A ** 2 * np.pi * rho

    print(f"{'label':>10} {'role':>6} {'omega_d':>9} {'x':>5} | "
          f"{'shift_45':>9} {'ok?':>4} {'fmax_45':>9} {'frange_45':>10} | "
          f"{'shift_12':>9} {'ok?':>4} {'fmax_12':>9} {'frange_12':>10}")
    rows = []
    t0 = time.time()
    for i, cand in enumerate(catalogue):
        ws_full, ws_sub = build_full_and_sub(gamma, cand)
        omega_d = ws_full.omega[3]
        for x in X_VALUES:
            m = measure(ws_full, ws_sub, x, e_tot, prefactor)
            s45, s45p, ok45, f45, r45 = m['RH(4,5)']
            s12, s12p, ok12, f12, r12 = m['RH(1,2)']
            rows.append(dict(cand, x=x, omega_d=omega_d,
                              shift_45=s45, shift_45_prom=s45p, agree_45=ok45, fmax_45=f45, frange_45=r45,
                              shift_12=s12, shift_12_prom=s12p, agree_12=ok12, fmax_12=f12, frange_12=r12))
            print(f"{cand['label']:>10} {cand['role']:>6} {omega_d:>+9.4f} {x:>5.2f} | "
                  f"{s45:>+8.2f}% {'Y' if ok45 else 'N':>4} {f45:>+8.2f}% {r45:>+9.2f}% | "
                  f"{s12:>+8.2f}% {'Y' if ok12 else 'N':>4} {f12:>+8.2f}% {r12:>+9.2f}%")
        print(f"  [{i+1}/{len(catalogue)}] ({time.time()-t0:.0f}s elapsed)")

    np.save(os.path.join(_ROOT, "alternate_topology_probe_data.npy"), rows, allow_pickle=True)
    print(f"\n{time.time()-t0:.0f}s total. Saved to examples/alternate_topology_probe_data.npy")
