"""Diagnostic registry for run_sweep.py's 2D velocity sweeps. Dispatches
each requested diagnostic to whichever engine computes it (pairwise:
full vs. reference triad; functional: full wave set alone), and exposes
one plot function per diagnostic.

Add a new diagnostic by adding it to pmeasure.py/functional.py's own
_DIAGNOSTIC_ARRAY_KEYS and a plot function here -- no new sweep loop.
"""
import numpy as np

from rsw_sphere.utilities.pmeasure import (
    wave_set_diagnostics_sweep, _DIAGNOSTIC_ARRAY_KEYS as _PAIRWISE_KEYS)
from rsw_sphere.utilities.functional import (
    functional_diagnostics_sweep, _DIAGNOSTIC_ARRAY_KEYS as _FUNCTIONAL_KEYS)
from rsw_sphere.plotting.pmeasure_map import (
    plot_p_measure_map, plot_filtering_error_map, plot_frequency_shift_map, plot_fmax_map)
from rsw_sphere.plotting.functional_map import plot_efficiency_map, plot_low_frequency_energy_map

PAIRWISE = frozenset(_PAIRWISE_KEYS)  # {"p_measure", "filtering_error", "frequency_shift", "fmax"}
FUNCTIONAL = frozenset(_FUNCTIONAL_KEYS)  # {"efficiency", "low_frequency_energy"}
ALL_2D = PAIRWISE | FUNCTIONAL

DIAGNOSTIC_ARRAY_KEYS = {**_PAIRWISE_KEYS, **_FUNCTIONAL_KEYS}

DIAGNOSTIC_PLOT_FNS = {
    "p_measure": plot_p_measure_map,
    "filtering_error": plot_filtering_error_map,
    "frequency_shift": plot_frequency_shift_map,
    "fmax": plot_fmax_map,
    "efficiency": plot_efficiency_map,
    "low_frequency_energy": plot_low_frequency_energy_map,
}


def sweep_2d(modes, triads, h_e, swept_indices, fixed_velocities, target_indices,
             diagnostics, u1_range=None, u2_range=None,
             reference_triad: int = 0, triad_index=None,
             drift_max: float = 0.1, low_freq_period_cutoff_days: float = 10.0,
             n_grid: int = 40, tf_days: float = 10, h: float = 0.01,
             N: int = 10, deg: int = 300, cache_dir: str = None,
             verbose: bool = False, progress_label: str = ""):
    """Compute every requested 2D diagnostic, dispatching to the pairwise
    and/or functional sweep engine(s) as needed and merging their output.

    diagnostics : subset of ALL_2D.
    cache_dir : if given, each engine's own .npz is cached under here
        (one file per engine actually invoked, not per diagnostic).

    Returns
    -------
    dict
        U1, U2, drift, labels, plus one array per requested diagnostic
        (keyed by DIAGNOSTIC_ARRAY_KEYS[name]).
    """
    import os

    unknown = set(diagnostics) - ALL_2D
    if unknown:
        raise ValueError(f"unknown diagnostic(s) {unknown} -- must be a subset of {ALL_2D}")

    pairwise = [d for d in diagnostics if d in PAIRWISE]
    functional = [d for d in diagnostics if d in FUNCTIONAL]

    out = None
    if pairwise:
        cache_path = os.path.join(cache_dir, f"{progress_label}_pairwise_{'-'.join(sorted(pairwise))}.npz") \
            if cache_dir else None
        r = wave_set_diagnostics_sweep(
            modes, triads, h_e, swept_indices, fixed_velocities, target_indices,
            diagnostics=tuple(pairwise), u1_range=u1_range, u2_range=u2_range,
            reference_triad=reference_triad, triad_index=triad_index,
            n_grid=n_grid, tf_days=tf_days, h=h, N=N, deg=deg,
            cache_path=cache_path, verbose=verbose, progress_label=progress_label)
        out = dict(r)

    if functional:
        cache_path = os.path.join(cache_dir, f"{progress_label}_functional_{'-'.join(sorted(functional))}.npz") \
            if cache_dir else None
        r = functional_diagnostics_sweep(
            modes, triads, h_e, swept_indices, fixed_velocities, target_indices,
            diagnostics=tuple(functional), u1_range=u1_range, u2_range=u2_range,
            drift_max=drift_max, low_freq_period_cutoff_days=low_freq_period_cutoff_days,
            n_grid=n_grid, tf_days=tf_days, h=h, N=N, deg=deg,
            cache_path=cache_path, verbose=verbose, progress_label=progress_label)
        if out is None:
            out = dict(r)
        else:
            out.update({k: v for k, v in r.items() if k not in ('U1', 'U2', 'labels')})
            out['drift'] = np.maximum(out['drift'], r['drift'])

    return out
