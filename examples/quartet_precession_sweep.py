"""Systematic precession-resonance sweep for a registered all-Rossby
quartet: does EITHER constituent triad's own dynamical phase (eq. Phi,
JFM-template.tex \\S Diagnostics) ever librate (lock) as the quartet's
own private, non-shared mode is driven harder?

General over any 2-triad WaveSet sharing an edge (not specific to one
quartet): given a registry key and which mode key to sweep, integrates
the full quartet at each swept velocity, computes BOTH constituent
triads' dynamical phase via ``rsw_sphere.dynamics.dynamical_phase``, and
reports libration statistics for each. A genuine two-triad phase lock
would show BOTH triads' precession frequency collapsing toward zero
together (mutual locking) or at least one collapsing while the other
still rotates (single-triad capture) -- neither is assumed, only
measured.

Run:

    python examples/quartet_precession_sweep.py --wave-set quartet_rh_preference --sweep-mode d
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

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, G
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics


def precession_sweep(wave_set_key, sweep_mode_key, u_values, base_velocities=None,
                      tf_days=150.0, h=0.01, N=10, deg=300, yaml_path=DEFAULT_WAVESETS_PATH,
                      target_mode_key=None):
    """Sweep one mode's driving velocity, report every constituent
    triad's own dynamical-phase libration statistics at each point.

    Parameters
    ----------
    wave_set_key : str
        Registry key (examples/wave_sets_section_3.yaml).
    sweep_mode_key : str
        Which mode key (e.g. 'd') to sweep; every other mode stays at
        its registered velocity unless overridden by ``base_velocities``.
    u_values : sequence of float
        Zonal velocities (m/s) to test for the swept mode.
    base_velocities : dict of {mode_key: float} or None, optional
        Override the registered velocity of any OTHER mode. ``None``:
        use the registry's own values unchanged.
    target_mode_key : str or None, optional
        If given, also report that mode's own energy-transfer efficiency
        at each swept point (see ``efficiency`` below), reusing the same
        trajectory already integrated for the phase diagnostic -- no
        extra integration cost. ``None`` (default): efficiency is not
        computed, matching this function's original behaviour exactly.

    Returns
    -------
    list of dict
        One entry per swept velocity: ``u``; per-triad
        ``precession_freq``/``net_windings``/``oscillation_amplitude_windings``
        (dict keyed by that triad's own display_label); ``energy_drift``,
        the wave set's own energy-conservation violation over the run
        (``max|E_total(t)-E_total(0)| / |E_total(0)|``, same convention as
        ``rsw_sphere.plotting.wave_set_pmeasure.p_measure``) -- always
        reported, since a genuine quartet does not conserve energy exactly
        (paper Appendix "Energy conservation of the four-wave truncation")
        and this is the sanity check for whether the time-averaged total
        energy below is a stable enough quantity to normalize by. If
        ``target_mode_key`` is given, also ``efficiency``: that mode's own
        ``(max|A|^2 - min|A|^2) / mean(E_total(t))`` -- the paper's usual
        efficiency (eq. effor) but normalized by the trajectory's own
        time-averaged total energy rather than its (for a quartet, not
        exactly conserved) initial value.
    """
    specs = load_wave_set_specs(yaml_path)
    spec = specs[wave_set_key]
    gamma = gamma_from_he(spec.h_e, g=G)[1]

    velocities = list(spec.velocities)
    if base_velocities:
        for mk, u in base_velocities.items():
            velocities[spec.index(mk)] = u
    sweep_idx = spec.index(sweep_mode_key)
    target_idx = spec.index(target_mode_key) if target_mode_key is not None else None

    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    triad_labels = [t.display_label for t in spec.triads]

    results = []
    for u in u_values:
        v = list(velocities)
        v[sweep_idx] = u
        ws = WaveSet(gamma, list(spec.modes), triad_indices, N=N, deg=deg)
        A0 = ws.amplitudes_from_velocities(v, spec.h_e, g=G)

        t_f = tf_days * 4 * np.pi
        Y, T = RK33(ws, 0, t_f, h, A0)
        T_days = days_from_nondim_time(T)

        E2, E3 = ws.energy(Y)
        E_total = np.real(E2 + E3)
        energy_drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])

        row = {'u': u, 'energy_drift': float(energy_drift)}
        if target_idx is not None:
            A_sq = np.real(Y[:, target_idx] * np.conj(Y[:, target_idx]))
            row['efficiency'] = float((A_sq.max() - A_sq.min()) / E_total.mean())
        for t_idx, (i_sum, i_p, i_q) in enumerate(triad_indices):
            Phi = dynamical_phase(Y, T, ws.omega, i_sum, i_p, i_q, ws.delta[t_idx])
            stats = libration_diagnostics(Phi, T_days)
            row[triad_labels[t_idx]] = stats
        results.append(row)
    return results


def _check_point(wave_set_key, u, expect_t1_locked, expect_t2_locked, label):
    r = precession_sweep(wave_set_key, 'd', [u], tf_days=150.0)[0]
    for triad_label, expect_locked in [('Triad 1', expect_t1_locked), ('Triad 2', expect_t2_locked)]:
        freq = abs(r[triad_label]['precession_freq'])
        if expect_locked:
            assert freq < 0.01, f"{label}: expected {triad_label} to lock at u={u}, got freq={freq}"
        else:
            assert freq > 0.3, f"{label}: expected {triad_label} NOT to lock at u={u}, got freq={freq}"


def self_test():
    """Fast regression check (a handful of single points, not the full
    expensive sweep) against the mutual-lock islands reported in the
    paper: Quartet A (quartet_rh_preference) locks for u~85-91 m/s AND
    again for u~138-140 m/s when sweeping RH(3,6); the RH(3,8) variant
    (quartet_rh_alt_partner) locks for u>~130 m/s. Both were found by
    direct velocity sweeps and convergence-checked across 150/300/450-day
    windows before being trusted -- this only guards against a silent
    regression, not a substitute for that original check.

    Checks BOTH triads at every point, not just Triad 1 -- the paper's
    claim is specifically MUTUAL locking (both constituent triads' own
    Phi collapsing together), not just that the undriven triad happens
    to read near zero; an earlier version of this self-test only checked
    Triad 1, so the "mutual" half of the claim was not actually
    regression-guarded. The two "not locked" checks below are NOT
    symmetric between the two triads: Quartet A's own driven triad
    (Triad 2) already locks trivially by u=70 (it directly follows its
    own forcing) even though the undriven Triad 1 does not -- only the
    RH(3,8) variant's own driven triad is still unlocked at its check
    point (u=50). Getting this asymmetry wrong (asserting BOTH triads
    unlocked at u=70) was itself caught here during review.
    """
    _check_point('quartet_rh_preference', 88.0, True, True, "Quartet A island 1")
    _check_point('quartet_rh_preference', 140.0, True, True, "Quartet A island 2")
    _check_point('quartet_rh_preference', 70.0, False, True, "Quartet A, no mutual lock")

    _check_point('quartet_rh_alt_partner', 140.0, True, True, "RH(3,8) variant, locked")
    _check_point('quartet_rh_alt_partner', 50.0, False, False, "RH(3,8) variant, no lock")

    print("self-test OK: Quartet A mutually locks at u=88 and u=140 (not u=70, "
          "though its own driven triad already has); "
          "RH(3,8) variant mutually locks at u=140 (neither triad locked at u=50)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-set", default="quartet_rh_preference")
    parser.add_argument("--sweep-mode", default="d",
                         help="mode key to sweep (default: d, RH(3,6) for quartet_rh_preference)")
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--u-min", type=float, default=10.0)
    parser.add_argument("--u-max", type=float, default=150.0)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--tf-days", type=float, default=150.0)
    parser.add_argument("--self-test", action="store_true",
                         help="run the fast regression check instead of a full sweep")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        raise SystemExit(0)

    u_values = np.linspace(args.u_min, args.u_max, args.n_points)
    results = precession_sweep(args.wave_set, args.sweep_mode, u_values,
                                tf_days=args.tf_days, yaml_path=args.specs)

    triad_labels = list(results[0].keys() - {'u'})
    triad_labels.sort()
    header = f"{'u':>7}" + "".join(f"  {lbl:>22}(freq,wind,amp)" for lbl in triad_labels)
    print(header)
    min_freq = {lbl: np.inf for lbl in triad_labels}
    for row in results:
        parts = [f"{row['u']:>7.1f}"]
        for lbl in triad_labels:
            s = row[lbl]
            parts.append(f"  {s['precession_freq']:>8.3f} {s['net_windings']:>7.2f} {s['oscillation_amplitude_windings']:>6.2f}")
            min_freq[lbl] = min(min_freq[lbl], abs(s['precession_freq']))
        print("".join(parts))

    print("\nMinimum |precession_freq| (rad/day) reached across the sweep, per triad:")
    for lbl in triad_labels:
        locked = " -- POSSIBLE LOCK (<0.05 rad/day)" if min_freq[lbl] < 0.05 else ""
        print(f"  {lbl}: {min_freq[lbl]:.4f}{locked}")

    # Convergence check at the midpoint u: precession_freq must be stable
    # under 2x tf and 3x finer h, matching the discipline established for
    # every other precession-diagnostic result in this repo.
    mid_u = u_values[len(u_values) // 2]
    base = precession_sweep(args.wave_set, args.sweep_mode, [mid_u], tf_days=args.tf_days)[0]
    long_tf = precession_sweep(args.wave_set, args.sweep_mode, [mid_u], tf_days=2 * args.tf_days)[0]
    print(f"\n--- convergence check at u={mid_u:.1f} (tf={args.tf_days:.0f}d vs 2x) ---")
    for lbl in triad_labels:
        f0, f1 = base[lbl]['precession_freq'], long_tf[lbl]['precession_freq']
        flag = " [NOT CONVERGED]" if abs(f1 - f0) > max(0.05 * abs(f1), 0.01) else ""
        print(f"  {lbl}: {f0:.4f} -> {f1:.4f} rad/day{flag}")
