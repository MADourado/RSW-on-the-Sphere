"""Efficiency-variation compute for a wave set (paper eq. `effvar`).

Compares a target mode's own trajectory in the full wave set against its
trajectory in one constituent triad alone:

    efficiency_var (%) = 100 * (dEK_full - dEK_triad) / dEK_triad

Both sides of the ratio are raw energy variations, deliberately not
standalone efficiencies: each configuration's own mean total energy
differs for reasons unrelated to the target's response (a fixed driving
velocity costs more energy at a higher wavenumber), so a ratio of
separately normalized efficiencies drifts with its denominator. Dividing
both sides by the same reference budget cancels it out and leaves the
formula above.
"""
import numpy as np

from rsw_sphere.utilities.efficiency import MIN_REFERENCE_DEK
from rsw_sphere.utilities.periods import novel_frequency_content, spectral_deviation, DEFAULT_EXCLUSION_FRAC

#: Below this, a target's own reference-triad dEK is numerically
#: indistinguishable from zero -- the ratio is left NaN rather than
#: blowing up. Shared with
#: `rsw_sphere.utilities.efficiency.efficiency_variation`'s own identical
#: guard on its efficiency denominator -- defined once in efficiency.py
#: (imported here, not the reverse, to avoid a circular import).


def _default_triad_index_for_mode(triads, reference_triad, mode_idx):
    """Denominator triad for a target mode: reference_triad if it's a
    member, else the first triad containing it."""
    if mode_idx in triads[reference_triad]:
        return reference_triad
    for t, tri in enumerate(triads):
        if mode_idx in tri:
            return t
    return None


def efficiency_variation_final(dEK_full, dEK_subs):
    """Efficiency variation using, as its reference, whichever candidate
    sub-triad gives the target its own LARGEST energy variation -- instead
    of one fixed reference triad, which for a mode shared across triads
    can be wildly inflated: the mode may happen to be a weak or
    off-resonance participant in THAT one triad even while a different
    containing triad drives it hard, so the fixed reference's own small
    dEK (not a genuinely large effect) is what drives the number.

    Broadcastable: ``dEK_full``/each ``dEK_subs`` value can be a scalar
    (single-run reporting) or an ndarray of matching shape (an entire 2D
    sweep grid at once) -- a sweep engine can call this exactly the same
    way it already calls the single-reference version
    (``pairwise_target_diagnostics``).

    Parameters
    ----------
    dEK_full : float or ndarray
    dEK_subs : dict
        label/index -> float or ndarray (same shape as ``dEK_full``), one
        entry per sub-triad containing this target.

    Returns
    -------
    dict
        efficiency_var (same shape as ``dEK_full``; NaN wherever every
        candidate's own dEK_sub is <= MIN_REFERENCE_DEK), reference (the
        winning ``dEK_subs`` key -- or an object array of them for the
        sweep case -- None/``np.nan`` where efficiency_var is NaN),
        dEK_reference (same shape as ``dEK_full``).
    """
    dEK_full_arr = np.asarray(dEK_full, dtype=float)
    scalar = dEK_full_arr.ndim == 0
    shape = (1,) if scalar else dEK_full_arr.shape
    dEK_full_flat = dEK_full_arr.reshape(shape)

    keys = list(dEK_subs)
    stacked = np.stack(
        [np.broadcast_to(np.asarray(dEK_subs[k], dtype=float), shape) for k in keys], axis=-1)

    valid = stacked > MIN_REFERENCE_DEK
    masked = np.where(valid, stacked, -np.inf)
    any_valid = np.any(valid, axis=-1)
    best_idx = np.argmax(masked, axis=-1)
    dEK_ref = np.take_along_axis(stacked, best_idx[..., None], axis=-1)[..., 0]

    with np.errstate(invalid='ignore', divide='ignore'):
        eff_var = 100 * (dEK_full_flat - dEK_ref) / dEK_ref
    eff_var = np.where(any_valid, eff_var, np.nan)
    dEK_ref = np.where(any_valid, dEK_ref, np.nan)

    key_arr = np.array(keys + [None], dtype=object)  # trailing None: "no valid candidate" sentinel
    reference = key_arr[np.where(any_valid, best_idx, len(keys))]

    if scalar:
        return {'efficiency_var': float(eff_var[0]), 'reference': reference[0],
                'dEK_reference': float(dEK_ref[0])}
    return {
        'efficiency_var': eff_var.reshape(dEK_full_arr.shape),
        'reference': reference.reshape(dEK_full_arr.shape),
        'dEK_reference': dEK_ref.reshape(dEK_full_arr.shape),
    }


def _spectral_deviation(T_days, E_full, E_total_full, E_sub, E_total_sub, dEK_sub, xmax: float = None):
    """Gated wrapper around ``periods.spectral_deviation`` -- builds each
    run's own AMPLITUDE share (``q = |A| / sqrt(E_total.mean())``, i.e.
    ``sqrt(E / E_total.mean())`` -- linear in amplitude, unlike the energy
    share ``E/E_total.mean()``, which is quadratic and so amplifies a
    given relative amplitude difference before it ever reaches the ratio)
    before comparing spectra. Still corrects for comparing systems with
    different total-energy budgets, the same fix as
    ``rsw_sphere.utilities.efficiency.wave_set_efficiency``'s own, just
    without the extra quadratic amplification. NaN if dEK_sub is too
    small (mode not really excited in the reference triad at all -- same
    gate as every other pairwise diagnostic here).
    """
    if dEK_sub <= MIN_REFERENCE_DEK:
        return np.nan
    q_full = np.sqrt(E_full / E_total_full.mean())
    q_sub = np.sqrt(E_sub / E_total_sub.mean())
    return spectral_deviation(T_days, q_full, T_days, q_sub, xmax=xmax)




def _novelty_period(T_days, amp_full, amp_sub, dEK_sub, exclusion_frac: float = DEFAULT_EXCLUSION_FRAC,
                     min_prominence: float = 0.02, xmax: float = None):
    """(dominant novel period in days, its relevance %) -- see
    ``rsw_sphere.utilities.periods.novel_frequency_content`` for the
    algorithm (excludes only the sub-triad's own
    dominant peak, not a "how much did the dominant period shift"
    comparison). NaN/0 if dEK_sub is too small, or if nothing survives
    the prominence threshold.
    """
    if dEK_sub <= MIN_REFERENCE_DEK:
        return np.nan, 0.0
    E_full, E_sub = amp_full ** 2, amp_sub ** 2
    result = novel_frequency_content(T_days, E_full, T_days, E_sub,
                                      exclusion_frac=exclusion_frac, min_prominence=min_prominence, xmax=xmax)
    if not result['novel_peaks']:
        return np.nan, 0.0
    dominant = result['novel_peaks'][0]
    return dominant['period_days'], dominant['relevance_pct']


def pairwise_target_diagnostics(T_days, amp_full, amp_sub, E_total_full, E_total_sub,
                                 novelty_exclusion_frac: float = DEFAULT_EXCLUSION_FRAC,
                                 novelty_min_prominence: float = 0.02,
                                 spectral_xmax: float = None, novelty_xmax: float = None) -> dict:
    """Every pairwise (full wave set vs. one sub-triad) diagnostic for a
    SINGLE already-integrated target-mode comparison -- the same formulas
    the now-deleted ``wave_set_diagnostics_sweep`` engine used to compute
    at each grid cell, just for one point rather than a swept grid
    (single-run reporting, e.g. ``run_dynamics.py --diagnostics``, and
    ``rsw_sphere.dynamics.diagnostics_report.compute_diagnostics_report``'s
    own per-point call).

    amp_full, amp_sub : |A_target(t)|, full wave set / one sub-triad
        alone, on the SAME time grid (same tf_days/h for both).
    E_total_full, E_total_sub : each run's own total-energy series
        (``ws.energy(Y)``'s ``E2+E3``, real part) -- needed only for
        ``spectral_deviation``'s own share normalization.
    """
    E_full, E_sub = amp_full ** 2, amp_sub ** 2
    dEK_full = E_full.max() - E_full.min()
    dEK_sub = E_sub.max() - E_sub.min()
    eff_var = 100 * (dEK_full - dEK_sub) / dEK_sub if dEK_sub > MIN_REFERENCE_DEK else np.nan
    sd = _spectral_deviation(T_days, E_full, E_total_full, E_sub, E_total_sub, dEK_sub,
                              xmax=spectral_xmax)
    novelty_period, novelty_relevance = _novelty_period(
        T_days, amp_full, amp_sub, dEK_sub,
        exclusion_frac=novelty_exclusion_frac, min_prominence=novelty_min_prominence, xmax=novelty_xmax)
    return {
        'efficiency_var': eff_var, 'spectral_deviation': sd,
        'novelty_period': novelty_period, 'novelty_relevance': novelty_relevance,
    }


def efficiency_variation_combined_for_target(results: dict, target_label: str) -> dict:
    """Single combined efficiency variation for ``target_label`` across
    ALL of its containing sub-triad units at once (see
    ``efficiency_variation_final``) -- one result per target, not one per
    (target, sub-triad) pair like the per-pair value computed inline by
    ``run_dynamics.py --diagnostics``'s own pairwise table
    (``pairwise_target_diagnostics``). Mirrors
    ``rsw_sphere.utilities.novelty_frequency.novelty_combined_for_target``'s
    own results-dict convention.

    Parameters
    ----------
    results : dict
        ``run_dynamics.run_dynamics()``'s own output (``{'full': {...},
        '<sub_unit_name>': {...}, ...}``, each with ``t``/``E``/``labels``).
    target_label : str

    Returns
    -------
    dict
        efficiency_var, reference (winning sub-unit name, or None if every
        candidate's own dEK_sub is below MIN_REFERENCE_DEK),
        dEK_reference, dEK_full, spectral_deviation (against that SAME
        winning reference, so efficiency_var/spectral_deviation always
        agree on which sub-triad a shared mode is being compared against
        -- there is only one reference-selection rule now, largest raw
        dEK; see ``rsw_sphere.dynamics.diagnostics_report.
        compute_diagnostics_report``'s own docstring). Note this "final"
        spectral_deviation is NOT the "filtering error" of a specific
        known mode removal for a shared target -- see
        ``spectral_deviation``'s own docstring and
        ``efficiency_variation_final``'s -- it reports the deviation from
        whichever single containing triad best explains the target, not
        a canonical single removal experiment (there isn't one for a
        shared mode). The PAIRWISE, single-named-triad computation
        (``pairwise_target_diagnostics``) keeps the "filtering error"
        framing valid for any mode, private or shared.
    """
    full = results["full"]
    if target_label not in full["labels"]:
        raise ValueError(f"{target_label!r} not found in the full wave set "
                          f"(available: {full['labels']})")
    j_full = full["labels"].index(target_label)
    E_full = full["E"][:, j_full]
    E_total_full = full["E_total"]
    dEK_full = E_full.max() - E_full.min()

    dEK_subs = {}
    E_subs = {}
    E_total_subs = {}
    for name, unit in results.items():
        if name == "full" or target_label not in unit["labels"]:
            continue
        j_sub = unit["labels"].index(target_label)
        E_sub = unit["E"][:, j_sub]
        dEK_subs[name] = E_sub.max() - E_sub.min()
        E_subs[name] = E_sub
        E_total_subs[name] = unit["E_total"]
    if not dEK_subs:
        raise ValueError(f"{target_label!r} not found in any sub-triad")

    result = efficiency_variation_final(dEK_full, dEK_subs)
    result['dEK_full'] = float(dEK_full)
    result['spectral_deviation'] = (
        _spectral_deviation(full["t"], E_full, E_total_full,
                             E_subs[result['reference']], E_total_subs[result['reference']],
                             result['dEK_reference'])
        if result['reference'] is not None else np.nan)
    return result


def efficiency_variation_combined_for_all_targets(results: dict) -> dict:
    """``efficiency_variation_combined_for_target`` for every mode in the full wave set."""
    return {label: efficiency_variation_combined_for_target(results, label)
            for label in results["full"]["labels"]}
