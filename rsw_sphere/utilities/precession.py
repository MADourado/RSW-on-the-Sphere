"""Precession-frequency + efficiency (+ individual-mode phase) sweep for
a wave set. Every swept trajectory is cached via
rsw_sphere.dynamics.trajectory_cache.run_and_cache.
"""
import os

import numpy as np

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, G
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics, individual_phase
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.utilities.periods import low_frequency_power


def precession_frequency_efficiency(spec, sweep_mode_key, u_values, base_velocities=None,
                                     target_mode_key=None, individual_mode_keys=None,
                                     low_freq_period_cutoff_days=None,
                                     tf_days=None, h=None, N=10, deg=300,
                                     cache_root="outputs/trajectories", sweep_cache_path=None):
    """Sweep one mode's velocity, reporting every constituent triad's
    dynamical-phase libration stats, plus optionally one target mode's
    efficiency (E_total.mean()-normalized), its low-frequency power, and
    individual-mode phase slopes -- all from one cached trajectory per point.

    Parameters
    ----------
    spec : WaveSetSpec
    sweep_mode_key : str -- mode key swept; others stay at registered velocity
        unless overridden by base_velocities.
    u_values : sequence of float (m/s)
    target_mode_key : str or None -- report efficiency for this mode.
    individual_mode_keys : sequence of str or None -- report individual_phase
        slope (sign flip = Raphaldini et al. 2022's "reversal").
    low_freq_period_cutoff_days : float or None -- if given with
        target_mode_key, also report low_frequency_power on E_target/E_total.
    tf_days, h : default spec.settings, else 150.0/0.01.
    cache_root : passed to run_and_cache.
    sweep_cache_path : .npz cache for this function's own summary arrays
        (separate from run_and_cache's per-point cache).

    Returns
    -------
    dict
        u_values, freq_by_triad (label -> ndarray, rad/day), energy_drift,
        efficiency (or None), low_freq_power (or None), individual_slope
        (dict or None), triad_labels.
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
            'low_freq_power': data['low_freq_power'] if 'low_freq_power' in data else None,
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
    compute_low_freq = target_idx is not None and low_freq_period_cutoff_days is not None
    low_freq_power = np.empty(n) if compute_low_freq else None
    individual_slope = {mk: np.empty(n) for mk in individual_mode_keys}

    for k, u in enumerate(u_values):
        v = list(velocities)
        v[sweep_idx] = u
        A0 = ws.amplitudes_from_velocities(v, spec.h_e, g=G)
        Y, T, _ = run_and_cache(ws, A0, t_f, h, velocities=v, output_root=cache_root)
        T_days = days_from_nondim_time(T)

        E2, E3 = ws.energy(Y)
        E_total = np.real(E2 + E3)
        energy_drift[k] = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])

        if target_idx is not None:
            A_sq = np.real(Y[:, target_idx] * np.conj(Y[:, target_idx]))
            efficiency[k] = (A_sq.max() - A_sq.min()) / E_total.mean()

            if compute_low_freq:
                Q_target = A_sq / E_total
                low_freq_power[k] = low_frequency_power(
                    T_days, Q_target, period_cutoff_days=low_freq_period_cutoff_days)

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
        'low_freq_power': low_freq_power,
        'individual_slope': individual_slope if individual_mode_keys else None,
        'triad_labels': triad_labels,
    }

    if sweep_cache_path:
        save_kwargs = {'u_values': result['u_values'], 'energy_drift': energy_drift,
                        'triad_labels': np.array(triad_labels)}
        save_kwargs.update({f'freq_{i}': freq_by_triad[lbl] for i, lbl in enumerate(triad_labels)})
        if efficiency is not None:
            save_kwargs['efficiency'] = efficiency
        if low_freq_power is not None:
            save_kwargs['low_freq_power'] = low_freq_power
        if individual_mode_keys:
            save_kwargs['individual_mode_keys'] = np.array(individual_mode_keys)
            save_kwargs.update({f'indiv_{mk}': individual_slope[mk] for mk in individual_mode_keys})
        cache_dir = os.path.dirname(sweep_cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        np.savez(sweep_cache_path, **save_kwargs)

    return result
