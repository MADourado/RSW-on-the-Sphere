"""Quartet-level companion to ``rh_partner_family.py``: for each even
n (RH(3,n) symmetric with the all-symmetric driving triad; odd n have
zero coupling by the equatorial-symmetry selection rule and are skipped),
builds the genuine four-wave quartet RH(4,5)+RH(1,2)+RH(3,4)+RH(3,n)
(RH(3,4) fixed at rest as the established preferred partner, RH(3,n)
swept as the second private member) and reports the P-measure (paper eq.
Pa) for target RH(3,n): how much the presence of the competing RH(3,4)
partner enhances or inhibits RH(3,n)'s own energy capture, relative to
RH(3,n) captured in isolation (rsw_sphere.utilities.pmeasure).

n=4 is excluded from the sweep (it IS the fixed partner, not a target).

Each (n, run) is fully independent -- built and integrated as its own
process via concurrent.futures.ProcessPoolExecutor, one worker per n.

P-measure is ill-conditioned whenever the isolated-triad dEK denominator
is near-zero -- n=14, 16 were confirmed (rh_partner_family.py's own
family, cross-checked bit-for-bit flat from t_f=30d to t_f=12000d, a 400x
span -- see examples/rh_partner_family_long_tf_check.py) to have
genuinely negligible isolated-triad efficiency, not an under-integration
artifact. p_measure()'s own MIN_REFERENCE_DEK gate (rsw_sphere.utilities.pmeasure)
now returns NaN for these -- re-verify this script's own n=14/16 numbers
against that gate before quoting them.

Run:

    python examples/rh_partner_quartet_family.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.utilities.pmeasure import p_measure

#: Driving pair, shared across every quartet (matches Table cap41 / rhfamily).
MODE_A = (4, 5, 3)   # RH(4,5), sum mode of both constituent triads
MODE_B = (1, 2, 3)   # RH(1,2)
MODE_C = (3, 4, 3)   # RH(3,4), fixed private partner, at rest
U_A, U_B, U_C = 40.0, 40.0, 0.0

H_E = 10000.0
#: n values swept as the second private partner (n=4 excluded: it IS mode_c).
N_VALUES = [6, 8, 10, 12, 14, 16]
#: Uniform t_f for every quartet: matches the isolated triad's own
#: confirmed-converged horizon for its slowest member (n=10, see
#: rh_partner_family.py's convergence check) rather than each quartet's
#: own default, since quartet beat periods are not guaranteed to be
#: shorter than the isolated triad's.
TF_DAYS = 240.0
H_STEP = 0.01
N_GRID = 10
DEG = 300


def _run_one(n, tf_days):
    """One quartet RH(4,5)+RH(1,2)+RH(3,4)+RH(3,n): builds its own modes/
    triads (WaveSet's index convention) and returns p_measure's result
    for target RH(3,n). Top-level function (not a closure) so it can be
    pickled across the process-pool boundary.
    """
    mode_d = (3, n, 3)
    modes = [MODE_A, MODE_B, MODE_C, mode_d]
    velocities = [U_A, U_B, U_C, 0.0]
    # Two constituent triads sharing edge (a, b): sum mode always last,
    # per WaveSet's TRIAD-slot-c convention (see wave_sets.py docstring).
    triads = [(0, 1, 2), (0, 1, 3)]  # (a,b,c) and (a,b,d), sum=a=index 0

    result = p_measure(modes, triads, velocities, h_e=H_E,
                        target_indices=[3], reference_triad=0,
                        tf_days=tf_days, h=H_STEP, N=N_GRID, deg=DEG)
    return {
        'n': n,
        'P': float(result['P'][0]),
        'dEK_full': float(result['dEK_full'][0]),
        'dEK_triad': float(result['dEK_triad'][0]),
        'drift': float(result['drift']),
        'triad_index_used': result['triad_index_used'][0],
    }


def run_family_parallel(n_values=N_VALUES, tf_days=TF_DAYS, max_workers=None):
    max_workers = max_workers or min(len(n_values), os.cpu_count() or 4)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_one, n, tf_days): n for n in n_values}
        for fut in as_completed(futures):
            n = futures[fut]
            r = fut.result()
            results[n] = r
            p_str = f"{r['P']:.2f}%" if np.isfinite(r['P']) else "n/a (dEK_triad~0)"
            print(f"n={n:2d} done: P={p_str}  dEK_full={r['dEK_full']:.3e}  "
                  f"dEK_triad={r['dEK_triad']:.3e}  drift={r['drift']:.2e}", flush=True)
    return dict(sorted(results.items()))


if __name__ == "__main__":
    print(f"Quartet RH(4,5)+RH(1,2)+RH(3,4)+RH(3,n), driving pair at "
          f"{U_A:.0f}/{U_B:.0f} m/s, RH(3,4) and RH(3,n) at rest, "
          f"t_f={TF_DAYS:.0f}d, h_e={H_E:.0f}m\n")
    results = run_family_parallel()

    print(f"\n{'n':>3} {'P (%)':>12} {'dEK_full':>12} {'dEK_triad':>12} {'drift':>10}")
    for n, r in results.items():
        p_str = f"{r['P']:.2f}" if np.isfinite(r['P']) else "n/a"
        print(f"{n:>3} {p_str:>12} {r['dEK_full']:>12.3e} {r['dEK_triad']:>12.3e} {r['drift']:>10.2e}")
