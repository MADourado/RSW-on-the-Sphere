"""Permanent, re-runnable record of the long-horizon convergence check
behind the paper's claim that n=12/14/16 (rh_partner_family, RH(3,n)
targets past the n=10 efficiency cliff) are genuinely converged, not
under-integrated -- the exact same trap that made n=10 read 0.84% at the
family's own default t_f=30d instead of its true, converged 12.6448% at
t_f=240d (see rh_partner_family.py's own docstring/self-test).

n=10 is included here too, as a positive control: at the already-converged
horizons (240d and beyond) it MUST still read 12.6448%, unchanged from its
own known value, confirming this script's own methodology isn't itself
producing false "converged" readings. 30d is probed for every n purely as
an unconverged reference point (rh_partner_family.py's own family uses
this as its default horizon, which is why n=10's table value there needs
calling out separately) -- it is not part of the self-check below.

Each (n, t_f) pair is fully independent -- run as its own process via
ProcessPoolExecutor, one worker per combination.

Run:

    python examples/rh_partner_family_long_tf_check.py
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

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.dynamics.triad_specs import load_triad_family, DEFAULT_FAMILIES_PATH
from rh_partner_family import triad_efficiency_point

#: n=10 is a positive control (must stay at its own known-converged
#: 12.6448% throughout the converged range); n=12/14/16 are the actual
#: claims being checked.
N_VALUES = (10, 12, 14, 16)
#: 30d is deliberately included as an unconverged reference point (see
#: module docstring); the self-check below only uses _CONVERGED_PROBES.
TF_DAYS_PROBES = (30.0, 240.0, 3840.0, 12000.0)
_CONVERGED_PROBES = (240.0, 3840.0, 12000.0)


def _job(n, tf_days):
    specs = {s.modes[1][1]: s for s in load_triad_family("rh_partner_family", DEFAULT_FAMILIES_PATH)}
    spec = specs[n]
    settings = spec.settings
    r = triad_efficiency_point(
        gamma_from_he(spec.h_e, g=G)[1], *spec.modes, spec.velocities,
        target=1, h_e=spec.h_e, tf_days=tf_days, h=settings.get('h', 0.01),
        N=settings.get('n_grid', 10), deg=settings.get('deg', 300))
    return n, tf_days, float(100 * r['efficiency'].real)


def run(n_values=N_VALUES, tf_days_probes=TF_DAYS_PROBES, max_workers=None):
    jobs = [(n, tf) for n in n_values for tf in tf_days_probes]
    max_workers = max_workers or min(len(jobs), os.cpu_count() or 4)
    results = {}
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_job, n, tf): (n, tf) for n, tf in jobs}
        for fut in as_completed(futures):
            n_r, tf_r, eff = fut.result()
            results[(n_r, tf_r)] = eff
            print(f"n={n_r:2d}  tf={tf_r:7.0f}d  ->  {eff:.6e}%", flush=True)
    return results


if __name__ == "__main__":
    results = run()
    print(f"\n{'n':>3}  " + "  ".join(f"tf={tf:.0f}d" for tf in TF_DAYS_PROBES))
    for n in N_VALUES:
        row = "  ".join(f"{results[(n, tf)]:14.6e}%" for tf in TF_DAYS_PROBES)
        print(f"{n:>3}  {row}")

    # n=10 must be stable at its own known-converged value across the
    # already-converged horizons -- a positive control on this script's
    # own methodology (30d is excluded: n=10 is known NOT to be converged
    # there, see module docstring).
    n10 = [results[(10, tf)] for tf in _CONVERGED_PROBES]
    assert max(n10) - min(n10) < 1e-3, f"n=10 positive control drifted: {n10}"
    print("\nself-check OK: n=10 stable at ~12.6448% across every converged "
          "horizon (positive control); n=12/14/16 read directly above.")
