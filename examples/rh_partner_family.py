"""Triad-family efficiency sweep: which member of a family (two modes
fixed, a third varying) captures the most energy from a fixed pair of
driving modes. The family itself (which modes, which velocities, which
n-values, which integration settings) is config-driven --
``examples/triad_families.yaml`` + ``rsw_sphere.dynamics.triad_specs.
load_triad_family`` -- not hardcoded here; this script is a thin runner
over whichever family key is requested (default: ``rh_partner_family``,
RH(4,5)+RH(1,2)+RH(3,n)).

The evaluator (``triad_efficiency_point``) is fully general: any three
modes, any initial velocities, any target slot. Every triad it builds is
first passed through ``WaveSet``'s validating constructor (raises on a
selection-rule violation or ``n<m``) purely as a gate -- the same
discipline every other triad/wave-set construction in this repo goes
through -- before being handed to ``TRIAD``/``Triad_dynamics`` for the
actual integration and efficiency (eq. effor).

This reproduces and corrects two earlier, uncaught bugs, both now guarded
against structurally rather than by one-off vigilance. First: an earlier
pass at the RH(3,n) family placed RH(3,n) in TRIAD's sum slot instead of
RH(4,5), silently computing a different (wrong) physical quantity, since
TRIAD does not itself validate the selection rule -- the ``WaveSet`` gate
above exists specifically so that cannot recur silently. Second: a later
pass used a fixed 30-day integration window for every family member,
which under-integrates the weakly-coupled, near-resonant members (e.g.
RH(3,10)) before their own, much slower exchange has completed even one
cycle -- the tf-convergence check in ``__main__`` below exists so that
cannot recur silently either.

Run:

    python examples/rh_partner_family.py
    python examples/rh_partner_family.py --family rh_partner_family
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.hough_harmonics.normalization import velocity_to_amplitude
from rsw_sphere.dynamics.dynamic_triads import TRIAD, Triad_dynamics
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.triad_specs import load_triad_family, DEFAULT_FAMILIES_PATH


def triad_efficiency_point(gamma, mode_a, mode_b, mode_c, velocities, target,
                            h_e=10000.0, tf_days=30.0, h=0.01, N=10, deg=300):
    """Selection-rule-gated efficiency of one triad at one initial condition.

    General building block for any "vary one member, hold the others
    fixed" family or catalogue sweep -- not specific to any one triad.

    Parameters
    ----------
    mode_a, mode_b, mode_c : (m, n, alpha) int triples
        ``TRIAD``'s own slot convention: mismatch = -freq_c+freq_b+freq_a,
        i.e. mode_c is the "sum" mode (m_c = m_a + m_b).
    velocities : sequence of 3 floats
        Initial zonal velocity (m/s) for modes a, b, c, in that order.
    target : int
        Which slot's efficiency is the headline value (0/1/2 = a/b/c);
        the other two are returned alongside it.

    Returns
    -------
    dict with ``delta`` (mismatch), ``alpha_target`` (coupling into the
    target slot, magnitude), ``omega_target`` (target's own linear
    frequency), ``efficiency`` (target slot, eq. effor -- eq: effor's
    $\\mathcal{E}$, not a separately-named quantity), ``efficiency_all``
    (all three slots, same order as the inputs).
    """
    # Selection-rule gate: raises ValueError if m_sum != m_p + m_q or n<m.
    WaveSet(gamma, [mode_a, mode_b, mode_c], [(2, 0, 1)], N=N, deg=deg)

    Triad = TRIAD(gamma, *mode_a, *mode_b, *mode_c, N, deg)
    components = [Triad.uvh_a[0], Triad.uvh_b[0], Triad.uvh_c[0]]
    A0 = np.array([velocity_to_amplitude(v, comp, h_e, g=G)
                   for v, comp in zip(velocities, components)])

    t_f = tf_days * 4 * np.pi
    ea, eb, ec, _, _, _ = Triad_dynamics(Triad, A0, 0, t_f, h)
    effs = [ea, eb, ec]
    alphas = [abs(Triad.coef_ABC), abs(Triad.coef_BAC), abs(Triad.coef_CAB)]
    omegas = [Triad.freq_a, Triad.freq_b, Triad.freq_c]

    return {'delta': Triad.mismatch, 'alpha_target': alphas[target],
            'omega_target': omegas[target],
            'efficiency': effs[target], 'efficiency_all': effs}


def run_family(family_key, yaml_path=DEFAULT_FAMILIES_PATH, tf_days=None):
    """Evaluate every member of a registered triad family.

    Parameters
    ----------
    family_key : str
        Entry name in the family registry (``examples/triad_families.yaml``).
    tf_days : float or None, optional
        Override the family's own registered ``settings.tf_days`` (used
        by the convergence check below to re-run at a longer horizon).

    Returns
    -------
    list of dict
        One ``triad_efficiency_point`` result per family member, each
        also carrying its own ``n`` (the varying mode's meridional index).
    """
    specs = load_triad_family(family_key, yaml_path)
    results = []
    for spec in specs:
        settings = spec.settings
        gamma = gamma_from_he(spec.h_e, g=G)[1]
        mode_a, mode_b, mode_c = spec.modes
        r = triad_efficiency_point(
            gamma, mode_a, mode_b, mode_c, spec.velocities, target=1,
            h_e=spec.h_e, tf_days=tf_days if tf_days is not None else settings.get('tf_days', 30.0),
            h=settings.get('h', 0.01), N=settings.get('n_grid', 10), deg=settings.get('deg', 300))
        r['n'] = mode_b[1]
        results.append(r)
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default="rh_partner_family",
                         help="entry name in examples/triad_families.yaml (default: rh_partner_family)")
    parser.add_argument("--specs", default=DEFAULT_FAMILIES_PATH,
                         help=f"path to the family registry YAML (default: {DEFAULT_FAMILIES_PATH})")
    args = parser.parse_args()

    specs = load_triad_family(args.family, args.specs)
    results = run_family(args.family, args.specs)

    from rsw_sphere.physics import linear_period_days

    print(f"{'n':>3} {'delta':>10} {'omega':>10} {'period(d)':>10} {'|alpha|':>10} {'efficiency (%)':>16}")
    for r in results:
        period = np.real(linear_period_days(r['omega_target']))
        print(f"{r['n']:>3} {r['delta'].real:>10.4f} {r['omega_target'].real:>10.5f} "
              f"{period:>10.3f} {r['alpha_target']:>10.4f} {100*np.real(r['efficiency']):>16.4f}")

    by_n = {r['n']: np.real(r['efficiency']) for r in results}

    # tf-convergence check (the discipline that caught the n=10 bug above):
    # every reported value must be stable under 8x longer integration.
    # One re-integration per family member, at that member's own base
    # tf_days -- NOT a full re-run of the family per member (an earlier
    # version of this loop called run_family() inside the loop, making it
    # O(n^2) instead of O(n); caught in review before this script was
    # trusted for anything).
    print("\n--- tf-convergence check (registered tf_days vs. 8x) ---")
    unconverged = []
    for spec in specs:
        n = spec.modes[1][1]
        base = by_n[n]
        settings = spec.settings
        base_tf = settings.get('tf_days', 30.0)
        gamma = gamma_from_he(spec.h_e, g=G)[1]
        mode_a, mode_b, mode_c = spec.modes
        long_r = triad_efficiency_point(
            gamma, mode_a, mode_b, mode_c, spec.velocities, target=1,
            h_e=spec.h_e, tf_days=8 * base_tf,
            h=settings.get('h', 0.01), N=settings.get('n_grid', 10), deg=settings.get('deg', 300))
        long_run = np.real(long_r['efficiency'])
        moved = abs(long_run - base) > max(0.02 * abs(long_run), 1e-4)
        flag = " [NOT CONVERGED]" if moved else ""
        if moved:
            unconverged.append(n)
        print(f"n={n:2d}  tf={base_tf:.0f}d: {100*base:.4f}%  tf={8*base_tf:.0f}d: {100*long_run:.4f}%{flag}")
    if unconverged:
        print(f"\n{len(unconverged)} member(s) not converged at the registered tf_days: {unconverged} "
              "-- report these with their own (longer) tf, not the family default.")

    # Regression checks against the corrected, converged result (specific
    # to the default rh_partner_family -- skipped for any other family).
    if args.family == "rh_partner_family":
        assert by_n[6] > by_n[4] > by_n[8], \
            "self-check FAILED: expected efficiency ranking n=6 > n=4 > n=8"
        for n_odd in (5, 7, 9, 11):
            assert by_n[n_odd] < 1e-6, \
                f"self-check FAILED: odd n={n_odd} should have ~0 efficiency, got {by_n[n_odd]}"
        print("\nself-check OK: n=6 > n=4 > n=8, odd n ~ 0")
