"""Functional diagnostics -- efficiency and low-frequency energy -- for a
wave set, over a 2D velocity sweep. Unlike pmeasure.py's pairwise
diagnostics, these need only the full wave set's own trajectory, no
reference-triad integration.
"""
import os

import numpy as np

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.utilities.efficiency import default_velocity_range
from rsw_sphere.utilities.efficiency import wave_set_efficiency
from rsw_sphere.utilities.periods import low_frequency_power

G = 9.8

#: Registered functional diagnostics. Add an entry here for a further
#: single-trajectory (no reference-triad) diagnostic.
_DIAGNOSTIC_ARRAY_KEYS = {"efficiency": "Efficiency", "low_frequency_energy": "LowFreqEnergy"}


def functional_diagnostics_sweep(modes, triads, h_e: float, swept_indices, fixed_velocities: dict,
                                  target_indices, diagnostics=("efficiency", "low_frequency_energy"),
                                  u1_range=None, u2_range=None,
                                  drift_max: float = 0.1, low_freq_period_cutoff_days: float = 10.0,
                                  n_grid: int = 40, tf_days: float = 10, h: float = 0.01,
                                  N: int = 10, deg: int = 300, cache_path: str = None,
                                  verbose: bool = False, progress_label: str = ""):
    """2D sweep of functional diagnostics for one or more target modes,
    one full-wave-set integration per grid point.

    Parameters as pmeasure.wave_set_diagnostics_sweep, plus:
    diagnostics : subset of _DIAGNOSTIC_ARRAY_KEYS.
    drift_max : gate for `efficiency` (rsw_sphere.utilities.efficiency).
    low_freq_period_cutoff_days : cutoff for `low_frequency_energy`.

    Returns
    -------
    dict
        U1, U2, drift, labels, plus one array per requested diagnostic.
    """
    unknown = set(diagnostics) - set(_DIAGNOSTIC_ARRAY_KEYS)
    if unknown:
        raise ValueError(f"unknown diagnostic(s) {unknown} -- must be a subset of "
                          f"{set(_DIAGNOSTIC_ARRAY_KEYS)}")

    idx1, idx2 = swept_indices
    if u1_range is None:
        u1_range = default_velocity_range(modes[idx1][2])
    if u2_range is None:
        u2_range = default_velocity_range(modes[idx2][2])

    array_keys = [_DIAGNOSTIC_ARRAY_KEYS[d] for d in diagnostics]
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        out = {'U1': data['U1'], 'U2': data['U2'], 'drift': data['drift'],
               'labels': list(data['labels'])}
        out.update({k: data[k] for k in array_keys})
        return out

    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    t_f = tf_days * 4 * np.pi

    u1 = np.linspace(u1_range[0], u1_range[1], n_grid)
    u2 = np.linspace(u2_range[0], u2_range[1], n_grid)
    U1, U2 = np.meshgrid(u1, u2)

    results = {name: np.full((n_grid, n_grid, len(target_indices)), np.nan) for name in array_keys}
    DRIFT = np.empty((n_grid, n_grid))

    if verbose:
        import time
        t_start = time.time()

    need_low_freq = "low_frequency_energy" in diagnostics

    for i in range(n_grid):
        for j in range(n_grid):
            velocities = np.empty(ws.n_modes)
            for m in range(ws.n_modes):
                if m == idx1:
                    velocities[m] = U1[i, j]
                elif m == idx2:
                    velocities[m] = U2[i, j]
                else:
                    velocities[m] = fixed_velocities[m]

            A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)
            Y, T = RK44(ws, 0, t_f, h, A0)
            E2, E3 = ws.energy(Y)
            E_total = np.real(E2 + E3)
            drift = np.max(np.abs(E_total - E_total[0])) / np.maximum(np.abs(E_total[0]), 1e-300)
            DRIFT[i, j] = drift
            T_days = days_from_nondim_time(T) if need_low_freq else None

            for k, tgt in enumerate(target_indices):
                E_tgt = np.real(Y[:, tgt] * np.conj(Y[:, tgt]))

                if "efficiency" in diagnostics:
                    results["Efficiency"][i, j, k] = wave_set_efficiency(E_tgt, E_total, drift, drift_max)
                if need_low_freq:
                    Q_tgt = E_tgt / E_total
                    results["LowFreqEnergy"][i, j, k] = low_frequency_power(
                        T_days, Q_tgt, period_cutoff_days=low_freq_period_cutoff_days)

        if verbose:
            done_rows = i + 1
            elapsed = time.time() - t_start
            eta = elapsed / done_rows * (n_grid - done_rows)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"    {prefix}row {done_rows}/{n_grid} "
                  f"({100 * done_rows / n_grid:.0f}%) "
                  f"elapsed {elapsed:.0f}s, eta {eta:.0f}s", flush=True)

    labels = [_mode_label(*modes[tgt]) for tgt in target_indices]
    out = {'U1': U1, 'U2': U2, 'drift': DRIFT, 'labels': labels, **results}
    if cache_path:
        np.savez(cache_path, **out)
    return out
