"""Precession-frequency + efficiency (+ individual-mode phase) sweep for a
quartet/quintet ("wave set") example -- generalizes
``examples/quartet_precession_sweep.py``'s ``precession_sweep`` and
``examples/precession_sweep_figure.py``'s ``sweep``/``plot_sweep`` into
this repo's own registry-driven ``wave_set_<topic>.py`` pattern (see
``docs/wave_sets.md``'s intro table), sitting alongside
``wave_set_table.py``/``wave_set_dynamics.py``/``wave_set_periods.py``/
``wave_set_pmeasure.py`` -- per that convention, computation + sweep +
plot live together here, not split across ``dynamics``/``plotting``.

Every swept trajectory is cached via
``rsw_sphere.dynamics.trajectory_cache.run_and_cache`` under
``outputs/trajectories/<wave_set_key>/`` -- re-running the same sweep a
second time is a cache hit per point, not a re-integration.

Run from the command line (registry-driven CLI, matching the other three
``wave_set_*.py`` scripts' own ``--wave-set``/``--specs`` convention):

    python rsw_sphere/plotting/wave_set_precession.py outputs/figures/wave_sets/quartet_rh_preference_precession.png --wave-set quartet_rh_preference --sweep-mode d --target c

or import and call it from another script (e.g. ``run_sweep.py``):

    from rsw_sphere.plotting.wave_set_precession import (
        precession_frequency_efficiency, plot_dual_axis_frequency_efficiency)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, G
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics, individual_phase
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.labels import _mode_label


def precession_frequency_efficiency(spec, sweep_mode_key, u_values, base_velocities=None,
                                     target_mode_key=None, individual_mode_keys=None,
                                     tf_days=None, h=None, N=10, deg=300,
                                     cache_root="outputs/trajectories", sweep_cache_path=None):
    """Sweep one mode's driving velocity for a registered wave set,
    reporting every constituent triad's dynamical-phase libration
    statistics, plus (optionally) one target mode's energy-transfer
    efficiency and/or one or more modes' own raw ``individual_phase``
    slope -- all from the SAME cached trajectory per swept point, no
    extra integration cost between quantities.

    Parameters
    ----------
    spec : rsw_sphere.dynamics.wave_set_specs.WaveSetSpec
    sweep_mode_key : str
        Mode key (e.g. ``'d'``) whose velocity is swept; every other mode
        stays at its registered velocity unless overridden by
        ``base_velocities``.
    u_values : sequence of float
        Zonal velocities (m/s) to test for the swept mode.
    base_velocities : dict of {mode_key: float} or None, optional
        Override the registered velocity of any OTHER mode.
    target_mode_key : str or None, optional
        If given, also report that mode's own time-averaged-total-energy
        efficiency (same convention as
        ``examples/quartet_precession_sweep.py``'s ``precession_sweep``).
    individual_mode_keys : sequence of str or None, optional
        Mode keys to also report ``individual_phase``'s (Raphaldini et
        al. 2022's raw ``phi_j~``) fitted slope for, at each swept point
        (Phase 5's individual-mode-reversal diagnostic: a sign flip in
        this slope across the sweep is the reported "reversal").
    tf_days, h : float or None, optional
        Integration horizon (days) / RK33 step. Default: ``spec.settings``
        if not given, else ``150.0``/``0.01``.
    N, deg : int, optional
        Hough truncation / quadrature degree. Default ``N=10, deg=300``.
    cache_root : str, optional
        Passed to ``run_and_cache``. Default ``"outputs/trajectories"``.

    Returns
    -------
    dict
        ``u_values``, ``freq_by_triad`` (dict of triad display_label ->
        ndarray, rad/day), ``energy_drift`` (ndarray), ``efficiency``
        (ndarray or ``None``), ``individual_slope`` (dict of mode_key ->
        ndarray, or ``None``), ``triad_labels`` (list, in registration
        order).

    Other Parameters
    ----------------
    sweep_cache_path : str or None, optional
        ``.npz`` cache for this function's own SWEEP-LEVEL result (the
        summary arrays returned below), separate from -- and layered on
        top of -- ``run_and_cache``'s per-point RAW-TRAJECTORY cache:
        even with every trajectory already cached, re-deriving the
        summary arrays from 45+ ``.npz`` loads is real but avoidable
        work. If given and the file exists, loaded directly (trajectories
        are not touched). If given and missing, computed as usual and
        then saved there. Same pattern as
        ``rsw_sphere.plotting.wave_set_pmeasure.p_measure_sweep``'s own
        ``cache_path``. ``None`` (default): never cached at this level.
    """
    if sweep_cache_path and os.path.exists(sweep_cache_path):
        data = np.load(sweep_cache_path)
        triad_labels = list(data['triad_labels'])
        freq_by_triad = {lbl: data[f'freq_{i}'] for i, lbl in enumerate(triad_labels)}
        individual_mode_keys = list(data['individual_mode_keys']) if 'individual_mode_keys' in data else []
        individual_slope = ({mk: data[f'indiv_{mk}'] for mk in individual_mode_keys}
                             if individual_mode_keys else None)
        return {
            'u_values': data['u_values'],
            'freq_by_triad': freq_by_triad,
            'energy_drift': data['energy_drift'],
            'efficiency': data['efficiency'] if 'efficiency' in data else None,
            'individual_slope': individual_slope,
            'triad_labels': triad_labels,
        }

    settings = spec.settings
    tf_days = tf_days if tf_days is not None else settings.get('tf_days', 150.0)
    h = h if h is not None else settings.get('h', 0.01)
    gamma = gamma_from_he(spec.h_e, g=G)[1]

    velocities = list(spec.velocities)
    if base_velocities:
        for mk, u in base_velocities.items():
            velocities[spec.index(mk)] = u
    sweep_idx = spec.index(sweep_mode_key)
    target_idx = spec.index(target_mode_key) if target_mode_key is not None else None
    individual_mode_keys = list(individual_mode_keys) if individual_mode_keys else []

    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    triad_labels = [t.display_label for t in spec.triads]

    ws = WaveSet(gamma, list(spec.modes), triad_indices, N=N, deg=deg)
    t_f = tf_days * 4 * np.pi

    n = len(u_values)
    freq_by_triad = {lbl: np.empty(n) for lbl in triad_labels}
    energy_drift = np.empty(n)
    efficiency = np.empty(n) if target_idx is not None else None
    individual_slope = {mk: np.empty(n) for mk in individual_mode_keys}

    for k, u in enumerate(u_values):
        v = list(velocities)
        v[sweep_idx] = u
        A0 = ws.amplitudes_from_velocities(v, spec.h_e, g=G)
        run_label = f"{sweep_mode_key}{u:.2f}_tf{tf_days:.0f}_h{h}"
        Y, T, _ = run_and_cache(ws, A0, t_f, h, spec.key, run_label, output_root=cache_root)
        T_days = days_from_nondim_time(T)

        E2, E3 = ws.energy(Y)
        E_total = np.real(E2 + E3)
        energy_drift[k] = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])

        if target_idx is not None:
            A_sq = np.real(Y[:, target_idx] * np.conj(Y[:, target_idx]))
            efficiency[k] = (A_sq.max() - A_sq.min()) / E_total.mean()

        for t_idx, (i_sum, i_p, i_q) in enumerate(triad_indices):
            Phi = dynamical_phase(Y, T, ws.omega, i_sum, i_p, i_q, ws.delta[t_idx])
            freq_by_triad[triad_labels[t_idx]][k] = libration_diagnostics(Phi, T_days)['precession_freq']

        for mk in individual_mode_keys:
            phi_j = individual_phase(Y, spec.index(mk))
            individual_slope[mk][k] = np.polyfit(T_days, phi_j, 1)[0]

    result = {
        'u_values': np.asarray(u_values, dtype=float),
        'freq_by_triad': freq_by_triad,
        'energy_drift': energy_drift,
        'efficiency': efficiency,
        'individual_slope': individual_slope if individual_mode_keys else None,
        'triad_labels': triad_labels,
    }

    if sweep_cache_path:
        save_kwargs = {'u_values': result['u_values'], 'energy_drift': energy_drift,
                        'triad_labels': np.array(triad_labels)}
        save_kwargs.update({f'freq_{i}': freq_by_triad[lbl] for i, lbl in enumerate(triad_labels)})
        if efficiency is not None:
            save_kwargs['efficiency'] = efficiency
        if individual_mode_keys:
            save_kwargs['individual_mode_keys'] = np.array(individual_mode_keys)
            save_kwargs.update({f'indiv_{mk}': individual_slope[mk] for mk in individual_mode_keys})
        cache_dir = os.path.dirname(sweep_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.savez(sweep_cache_path, **save_kwargs)

    return result


def plot_dual_axis_frequency_efficiency(result, spec, plot_triad=None,
                                         xlabel='', title='', plot_u_max=None, path=None):
    """Twin-axis "precession frequency (dotted) + efficiency (solid)"
    figure -- the reusable version of
    ``examples/precession_sweep_figure.py``'s/
    ``examples/borrowed_topology_precession_figure.py``'s own already-
    approved styling, moved here so both scripts (and any future one)
    call one shared function instead of each keeping its own matplotlib.

    Parameters
    ----------
    result : dict
        Output of ``precession_frequency_efficiency``.
    spec : rsw_sphere.dynamics.wave_set_specs.WaveSetSpec
        Used only to derive each triad's "RH(a)+RH(b)+RH(c)"-style legend
        label from the registry itself (never hardcoded/duplicated).
    plot_triad : int or None, optional
        If given, draw only that one triad's frequency curve (index into
        ``spec.triads``) instead of every constituent triad's. Every
        triad's frequency is still computed by
        ``precession_frequency_efficiency`` regardless -- this only
        restricts what gets drawn.
    xlabel, title : str, optional
    plot_u_max : float or None, optional
        Crop the plotted (and auto-scaled) range to ``u_values <=
        plot_u_max`` -- ``result``'s own arrays are untouched.
    path : str or None, optional
        If given, the figure is saved (PNG, 200 dpi) and closed.
        Otherwise shown interactively.
    """
    u_values = result['u_values']
    freq_by_triad = dict(result['freq_by_triad'])
    efficiency = result['efficiency']
    labels = result['triad_labels']

    mode_str = {}
    for i, t in enumerate(spec.triads):
        i_sum, i_p, i_q = spec.triad_indices(i)
        mode_str[t.display_label] = "+".join(_mode_label(*spec.modes[j]) for j in (i_sum, i_p, i_q))

    if plot_u_max is not None:
        mask = u_values <= plot_u_max
        u_values = u_values[mask]
        freq_by_triad = {lbl: v[mask] for lbl, v in freq_by_triad.items()}
        if efficiency is not None:
            efficiency = efficiency[mask]

    apply_house_style()
    markers = ['o', 's', '^', 'v']
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, lbl in enumerate(labels):
        if plot_triad is not None and i != plot_triad:
            continue
        ax.plot(u_values, np.abs(freq_by_triad[lbl]), markers[i % len(markers)] + ':', ms=3,
                color='C0', label=f'{lbl} ({mode_str[lbl]})', alpha=1.0 if i == 0 else 0.6)
    ax.axhline(0.01, color='grey', ls=':', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$|$precession frequency$|$ (rad/day)', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_title(title)

    if efficiency is not None:
        ax2 = ax.twinx()
        ax2.plot(u_values, 100 * efficiency, 'd-', ms=3, color='C3',
                 label=r'Efficiency $\mathcal{E}_{\mathrm{avg}}$')
        ax2.set_ylabel(r'Efficiency $\mathcal{E}_{\mathrm{avg}}$ (\%)', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='best')
    else:
        ax.legend(fontsize=8)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_phase_trace(phi_list, T_days_list, labels, title='', ylabel=r'$\Phi$ (rad)', path=None):
    """One or more ``phi(t)`` vs. time traces -- works for both the
    combined dynamical phase ``Phi`` (``dynamical_phase``) and individual
    raw mode phases ``phi_j~`` (``individual_phase``), for a low-vs-high
    driving-amplitude comparison (mirrors Raphaldini et al. 2022's own
    Fig. 3 layout).

    Parameters
    ----------
    phi_list : sequence of ndarray
        One or more phase traces (already unwrapped).
    T_days_list : sequence of ndarray
        Matching time axes (days), same length as ``phi_list``.
    labels : sequence of str
        Legend label per trace.
    title, ylabel : str, optional
    path : str or None, optional
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for phi, T_days, lbl in zip(phi_list, T_days_list, labels):
        ax.plot(T_days, phi, label=lbl)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs

    parser = argparse.ArgumentParser(
        description="Compute (with trajectory caching) and plot a "
                    "precession-frequency + efficiency sweep for a quartet/"
                    "quintet example from the wave-set registry YAML.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
                         default="quartet_rh_preference")
    parser.add_argument("--sweep-mode", default="d", help="mode key to sweep")
    parser.add_argument("--target", default=None, help="mode key to report efficiency for")
    parser.add_argument("--plot-triad", type=int, default=None)
    parser.add_argument("--u-min", type=float, default=10.0)
    parser.add_argument("--u-max", type=float, default=150.0)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--tf", dest="tf_days", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[args.wave_set]

    u_values = np.linspace(args.u_min, args.u_max, args.n_points)
    result = precession_frequency_efficiency(
        spec, args.sweep_mode, u_values, target_mode_key=args.target,
        tf_days=args.tf_days, h=args.h)

    plot_dual_axis_frequency_efficiency(
        result, spec, plot_triad=args.plot_triad,
        xlabel=f"{args.sweep_mode} driving velocity (m/s)",
        title=spec.display_label or spec.key, path=args.path)

    min_freq = {lbl: np.min(np.abs(v)) for lbl, v in result['freq_by_triad'].items()}
    print(f"{args.wave_set}: min |precession_freq| per triad: {min_freq}")


if __name__ == "__main__":
    main()
