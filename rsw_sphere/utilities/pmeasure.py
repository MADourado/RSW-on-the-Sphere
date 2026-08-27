"""P-measure compute for a wave set (paper eq. Pa).

Compares a target mode's own trajectory in the full wave set against its
trajectory in one constituent triad alone:

    P (%) = 100 * (dEK_full - dEK_triad) / dEK_triad

Run as a quick self-check:

    python -m rsw_sphere.utilities.pmeasure
"""
import os

import numpy as np

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.utilities.efficiency import default_velocity_range
from rsw_sphere.utilities.periods import novel_frequency_content, spectral_deviation

G = 9.8

#: Below this, a target's own reference-triad dEK is numerically
#: indistinguishable from zero -- P and F2 are left NaN rather than a
#: blown-up ratio (paper eq. Pa).
MIN_REFERENCE_DEK = 1e-4


def _default_triad_index_for_mode(triads, reference_triad, mode_idx):
    """Denominator triad for a target mode: reference_triad if it's a
    member, else the first triad containing it."""
    if mode_idx in triads[reference_triad]:
        return reference_triad
    for t, tri in enumerate(triads):
        if mode_idx in tri:
            return t
    return None


def _all_triad_indices_for_mode(triads, mode_idx):
    """Every triad index containing mode_idx -- 1 for a private mode, 2+
    for a mode shared across triads (member or sum role alike)."""
    return [t for t, tri in enumerate(triads) if mode_idx in tri]


def final_p_measure(dEK_full, dEK_subs):
    """P-measure using, as its reference, whichever candidate sub-triad
    gives the target its own LARGEST energy variation -- instead of one
    fixed reference triad, which for a mode shared across triads can be
    wildly inflated: the mode may happen to be a weak or off-resonance
    participant in THAT one triad even while a different containing
    triad drives it hard, so the fixed reference's own small dEK (not a
    genuinely large effect) is what drives the number.

    Broadcastable: ``dEK_full``/each ``dEK_subs`` value can be a scalar
    (single-run reporting) or an ndarray of matching shape (an entire 2D
    sweep grid at once) -- a sweep engine can call this exactly the same
    way it already calls the single-reference P-measure.

    Parameters
    ----------
    dEK_full : float or ndarray
    dEK_subs : dict
        label/index -> float or ndarray (same shape as ``dEK_full``), one
        entry per sub-triad containing this target.

    Returns
    -------
    dict
        p_measure (same shape as ``dEK_full``; NaN wherever every
        candidate's own dEK_sub is <= MIN_REFERENCE_DEK), reference (the
        winning ``dEK_subs`` key -- or an object array of them for the
        sweep case -- None/``np.nan`` where p_measure is NaN),
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
        p = 100 * (dEK_full_flat - dEK_ref) / dEK_ref
    p = np.where(any_valid, p, np.nan)
    dEK_ref = np.where(any_valid, dEK_ref, np.nan)

    key_arr = np.array(keys + [None], dtype=object)  # trailing None: "no valid candidate" sentinel
    reference = key_arr[np.where(any_valid, best_idx, len(keys))]

    if scalar:
        return {'p_measure': float(p[0]), 'reference': reference[0], 'dEK_reference': float(dEK_ref[0])}
    return {
        'p_measure': p.reshape(dEK_full_arr.shape),
        'reference': reference.reshape(dEK_full_arr.shape),
        'dEK_reference': dEK_ref.reshape(dEK_full_arr.shape),
    }


def _integrate_sub_triad_amplitude(gamma, modes, triad, velocities, h_e, t0, t_f, h, N, deg, mode_idx):
    """mode_idx's own |A(t)| within constituent triad alone."""
    i_sum, i_p, i_q = triad
    sub_modes = [modes[i_p], modes[i_q], modes[i_sum]]
    sub_velocities = [velocities[i_p], velocities[i_q], velocities[i_sum]]
    local = {i_p: 0, i_q: 1, i_sum: 2}[mode_idx]

    sub_ws = WaveSet(gamma, sub_modes, [(2, 0, 1)], N=N, deg=deg)
    A0 = sub_ws.amplitudes_from_velocities(sub_velocities, h_e, g=G)
    Y, _ = RK44(sub_ws, t0, t_f, h, A0)
    return np.abs(Y[:, local])


def _dEK_for_triad(gamma, modes, triad, velocities, h_e, t0, t_f, h, N, deg, mode_idx):
    """mode_idx's kinetic-energy variation within constituent triad alone."""
    amp = _integrate_sub_triad_amplitude(gamma, modes, triad, velocities, h_e, t0, t_f, h, N, deg, mode_idx)
    E = amp ** 2
    return E.max() - E.min()


def _spectral_deviation(T_days, E_full, E_total_full, E_sub, E_total_sub, dEK_sub, xmax: float = 3.0):
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


def p_measure(modes, triads, velocities, h_e: float = 10000,
              target_indices=None, reference_triad: int = 0, triad_index=None,
              t0: float = 0, tf_days: float = 10, h: float = 0.01,
              N: int = 10, deg: int = 300):
    """P-measure (%) for one or more target modes, at one fixed IC.

    Returns
    -------
    dict
        P (percent, NaN below MIN_REFERENCE_DEK), dEK_full, dEK_triad,
        triad_index_used, drift, labels.
    """
    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)

    t_f = tf_days * 4 * np.pi
    Y, T = RK44(ws, t0, t_f, h, A0)
    E = np.real(Y * np.conj(Y))
    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)
    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])

    if target_indices is None:
        target_indices = list(range(ws.n_modes))
    triad_index = dict(triad_index or {})

    P = np.full(len(target_indices), np.nan)
    dEK_full = np.full(len(target_indices), np.nan)
    dEK_triad = np.full(len(target_indices), np.nan)
    triad_index_used = []

    for k, tgt in enumerate(target_indices):
        t_idx = triad_index.get(tgt, _default_triad_index_for_mode(triads, reference_triad, tgt))
        triad_index_used.append(t_idx)
        dEK_full[k] = E[:, tgt].max() - E[:, tgt].min()
        if t_idx is None:
            continue
        dEK_triad[k] = _dEK_for_triad(gamma, modes, triads[t_idx], velocities, h_e,
                                       t0, t_f, h, N, deg, tgt)
        if dEK_triad[k] > MIN_REFERENCE_DEK:
            P[k] = 100 * (dEK_full[k] - dEK_triad[k]) / dEK_triad[k]

    labels = [_mode_label(*modes[tgt]) for tgt in target_indices]
    return {'P': P, 'dEK_full': dEK_full, 'dEK_triad': dEK_triad,
            'triad_index_used': triad_index_used, 'drift': drift, 'labels': labels}


def p_measure_sweep(modes, triads, h_e: float, swept_indices, fixed_velocities: dict,
                     target_indices, u1_range=None, u2_range=None,
                     reference_triad: int = 0, triad_index=None,
                     n_grid: int = 40, tf_days: float = 10, h: float = 0.01,
                     N: int = 10, deg: int = 300, cache_path: str = None,
                     verbose: bool = False, progress_label: str = ""):
    """Pure-compute 2D sweep of P-measure over two modes' velocities.

    cache_path: .npz, cache-if-absent/load-if-present. Cache format is
    pinned -- do not change the saved payload shape.

    Returns
    -------
    dict
        U1, U2 (meshgrid, m/s), P (n_grid, n_grid, len(target_indices), %),
        drift (n_grid, n_grid), labels.
    """
    idx1, idx2 = swept_indices
    if u1_range is None:
        u1_range = default_velocity_range(modes[idx1][2])
    if u2_range is None:
        u2_range = default_velocity_range(modes[idx2][2])

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return {'U1': data['U1'], 'U2': data['U2'], 'P': data['P'],
                'drift': data['drift'], 'labels': list(data['labels'])}

    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    t_f = tf_days * 4 * np.pi

    triad_index = dict(triad_index or {})
    t_idx_for_target = [
        triad_index.get(tgt, _default_triad_index_for_mode(triads, reference_triad, tgt))
        for tgt in target_indices
    ]
    u1 = np.linspace(u1_range[0], u1_range[1], n_grid)
    u2 = np.linspace(u2_range[0], u2_range[1], n_grid)
    U1, U2 = np.meshgrid(u1, u2)
    # U1[i,j]=u1[j] varies across columns, U2[i,j]=u2[i] constant per row --
    # a target's denominator triad is row-cacheable only if it excludes idx1.
    axis1_in_triad = [
        (t_idx is not None and idx1 in triads[t_idx]) for t_idx in t_idx_for_target
    ]

    P = np.full((n_grid, n_grid, len(target_indices)), np.nan)
    DRIFT = np.empty((n_grid, n_grid))

    if verbose:
        import time
        t_start = time.time()

    for i in range(n_grid):
        row_cache = {}  # (triad_idx, target_idx) -> dEK
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
            Y, _ = RK44(ws, 0, t_f, h, A0)
            E = np.real(Y * np.conj(Y))
            E2, E3 = ws.energy(Y)
            E_total = np.real(E2 + E3)
            DRIFT[i, j] = np.max(np.abs(E_total - E_total[0])) / np.maximum(np.abs(E_total[0]), 1e-300)

            for k, tgt in enumerate(target_indices):
                t_idx = t_idx_for_target[k]
                dEK_full = E[:, tgt].max() - E[:, tgt].min()
                if t_idx is None:
                    continue
                cache_key = (t_idx, tgt)
                if (not axis1_in_triad[k]) and cache_key in row_cache:
                    dEK_triad = row_cache[cache_key]
                else:
                    dEK_triad = _dEK_for_triad(gamma, modes, triads[t_idx], velocities, h_e,
                                                0, t_f, h, N, deg, tgt)
                    if not axis1_in_triad[k]:
                        row_cache[cache_key] = dEK_triad
                if dEK_triad > MIN_REFERENCE_DEK:
                    P[i, j, k] = 100 * (dEK_full - dEK_triad) / dEK_triad

        if verbose:
            done_rows = i + 1
            elapsed = time.time() - t_start
            eta = elapsed / done_rows * (n_grid - done_rows)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"    {prefix}row {done_rows}/{n_grid} "
                  f"({100 * done_rows / n_grid:.0f}%) "
                  f"elapsed {elapsed:.0f}s, eta {eta:.0f}s", flush=True)

    labels = [_mode_label(*modes[tgt]) for tgt in target_indices]
    if cache_path:
        np.savez(cache_path, U1=U1, U2=U2, P=P, drift=DRIFT, labels=np.array(labels))

    return {'U1': U1, 'U2': U2, 'P': P, 'drift': DRIFT, 'labels': labels}


#: Registered diagnostics for wave_set_diagnostics_sweep. Add an entry
#: here (not a new sweep loop) for a further per-target diagnostic.
#: NOTE: spectral_deviation (rsw_sphere.utilities.periods.spectral_deviation,
#: replacing the retired time-domain "filtering error"/F2) is NOT wired in
#: here -- it needs each candidate sub-triad's own E_total series, which
#: this 2D sweep engine doesn't currently carry (only amp/dEK per triad).
#: Single-run reporting (pairwise_target_diagnostics,
#: p_measure_combined_for_target) has it; a 2D sweep does not yet.
_DIAGNOSTIC_ARRAY_KEYS = {"p_measure": "P",
                          "novelty_period": "NoveltyPeriod",
                          "p_measure_final": "PFinal"}


def _novelty_period(T_days, amp_full, amp_sub, dEK_sub, exclusion_frac: float = 0.20,
                     min_prominence: float = 0.02):
    """(dominant novel period in days, its relevance %) -- see
    ``rsw_sphere.utilities.periods.novel_frequency_content`` for the
    algorithm (2026-08-26 design: excludes only the sub-triad's own
    dominant peak, not a "how much did the dominant period shift"
    comparison). NaN/0 if dEK_sub is too small, or if nothing survives
    the prominence threshold.
    """
    if dEK_sub <= MIN_REFERENCE_DEK:
        return np.nan, 0.0
    E_full, E_sub = amp_full ** 2, amp_sub ** 2
    result = novel_frequency_content(T_days, E_full, T_days, E_sub,
                                      exclusion_frac=exclusion_frac, min_prominence=min_prominence)
    if not result['novel_peaks']:
        return np.nan, 0.0
    dominant = result['novel_peaks'][0]
    return dominant['period_days'], dominant['relevance_pct']


def pairwise_target_diagnostics(T_days, amp_full, amp_sub, E_total_full, E_total_sub,
                                 novelty_exclusion_frac: float = 0.20,
                                 novelty_min_prominence: float = 0.02,
                                 spectral_xmax: float = 3.0) -> dict:
    """Every pairwise (full wave set vs. one sub-triad) diagnostic for a
    SINGLE already-integrated target-mode comparison -- reuses the exact
    same per-grid-point formulas ``wave_set_diagnostics_sweep`` computes
    at each cell, just for one point rather than a swept grid (single-run
    reporting, e.g. ``run_dynamics.py --diagnostics``).

    amp_full, amp_sub : |A_target(t)|, full wave set / one sub-triad
        alone, on the SAME time grid (same tf_days/h for both).
    E_total_full, E_total_sub : each run's own total-energy series
        (``ws.energy(Y)``'s ``E2+E3``, real part) -- needed only for
        ``spectral_deviation``'s own share normalization.
    """
    E_full, E_sub = amp_full ** 2, amp_sub ** 2
    dEK_full = E_full.max() - E_full.min()
    dEK_sub = E_sub.max() - E_sub.min()
    p = 100 * (dEK_full - dEK_sub) / dEK_sub if dEK_sub > MIN_REFERENCE_DEK else np.nan
    sd = _spectral_deviation(T_days, E_full, E_total_full, E_sub, E_total_sub, dEK_sub,
                              xmax=spectral_xmax)
    novelty_period, novelty_relevance = _novelty_period(
        T_days, amp_full, amp_sub, dEK_sub,
        exclusion_frac=novelty_exclusion_frac, min_prominence=novelty_min_prominence)
    return {
        'p_measure': p, 'spectral_deviation': sd,
        'novelty_period': novelty_period, 'novelty_relevance': novelty_relevance,
    }


def p_measure_combined_for_target(results: dict, target_label: str) -> dict:
    """Single combined P-measure for ``target_label`` across ALL of its
    containing sub-triad units at once (see ``final_p_measure``) -- one
    result per target, not one per (target, sub-triad) pair like the
    per-pair ``p_measure`` computed inline by ``run_dynamics.py
    --diagnostics``'s own pairwise table. Mirrors
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
        p_measure, reference (winning sub-unit name, or None if every
        candidate's own dEK_sub is below MIN_REFERENCE_DEK),
        dEK_reference, dEK_full, spectral_deviation (against that SAME
        winning reference -- not an independently-chosen one, so
        p_measure/spectral_deviation/efficiency_variation all agree on
        which sub-triad a shared mode is being compared against). Note
        this "final" spectral_deviation is NOT the "filtering error" of a
        specific known mode removal for a shared target -- see
        ``spectral_deviation``'s own docstring and
        ``final_p_measure``'s -- it reports the deviation from
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

    result = final_p_measure(dEK_full, dEK_subs)
    result['dEK_full'] = float(dEK_full)
    result['spectral_deviation'] = (
        _spectral_deviation(full["t"], E_full, E_total_full,
                             E_subs[result['reference']], E_total_subs[result['reference']],
                             result['dEK_reference'])
        if result['reference'] is not None else np.nan)
    return result


def p_measure_combined_for_all_targets(results: dict) -> dict:
    """``p_measure_combined_for_target`` for every mode in the full wave set."""
    return {label: p_measure_combined_for_target(results, label)
            for label in results["full"]["labels"]}


def wave_set_diagnostics_sweep(modes, triads, h_e: float, swept_indices, fixed_velocities: dict,
                                target_indices, diagnostics=("p_measure", "novelty_period"),
                                u1_range=None, u2_range=None,
                                reference_triad: int = 0, triad_index=None,
                                n_grid: int = 40, tf_days: float = 10, h: float = 0.01,
                                N: int = 10, deg: int = 300, cache_path: str = None,
                                verbose: bool = False, progress_label: str = "",
                                novelty_exclusion_frac: float = 0.20,
                                novelty_min_prominence: float = 0.02):
    """2D sweep computing several per-target diagnostics from one shared
    pass (one full-wave-set integration + one row-cached reference-triad
    integration per grid point), instead of one pass per diagnostic.

    Parameters as p_measure_sweep, plus:
    diagnostics : subset of _DIAGNOSTIC_ARRAY_KEYS.
    novelty_exclusion_frac, novelty_min_prominence : only used if
        "novelty_period" is requested -- passed straight through to
        ``periods.novel_frequency_content``.

    Returns
    -------
    dict
        U1, U2, drift, labels, plus one array per requested diagnostic.
        If "novelty_period" is requested, also NoveltyRelevance (%, same
        shape as NoveltyPeriod) -- see _novelty_period.
        If "p_measure_final" is requested, also PFinalRefIdx (float array,
        same shape as PFinal -- the winning triad's own index into
        ``triads`` at each grid point, NaN where no candidate's own
        dEK_sub cleared MIN_REFERENCE_DEK) -- see final_p_measure. Unlike
        "p_measure" (one fixed reference_triad/triad_index per target),
        this integrates and compares EVERY containing sub-triad at each
        point, so it costs more when a target belongs to more than one.
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
    need_novelty = "novelty_period" in diagnostics
    if need_novelty:
        array_keys = array_keys + ["NoveltyRelevance"]
    need_final_p = "p_measure_final" in diagnostics
    if need_final_p:
        array_keys = array_keys + ["PFinalRefIdx"]
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        out = {'U1': data['U1'], 'U2': data['U2'], 'drift': data['drift'],
               'labels': list(data['labels'])}
        out.update({k: data[k] for k in array_keys})
        return out

    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    t_f = tf_days * 4 * np.pi

    triad_index = dict(triad_index or {})
    t_idx_for_target = [
        triad_index.get(tgt, _default_triad_index_for_mode(triads, reference_triad, tgt))
        for tgt in target_indices
    ]
    all_t_idx_for_target = (
        [_all_triad_indices_for_mode(triads, tgt) for tgt in target_indices] if need_final_p else None
    )
    u1 = np.linspace(u1_range[0], u1_range[1], n_grid)
    u2 = np.linspace(u2_range[0], u2_range[1], n_grid)
    U1, U2 = np.meshgrid(u1, u2)
    axis1_in_triad = [
        (t_idx is not None and idx1 in triads[t_idx]) for t_idx in t_idx_for_target
    ]

    results = {
        name: np.full((n_grid, n_grid, len(target_indices)), np.nan)
        for name in array_keys
    }
    DRIFT = np.empty((n_grid, n_grid))

    if verbose:
        import time
        t_start = time.time()

    for i in range(n_grid):
        row_cache = {}  # (triad_idx, target_idx) -> amp_sub
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
            DRIFT[i, j] = np.max(np.abs(E_total - E_total[0])) / np.maximum(np.abs(E_total[0]), 1e-300)
            T_days = days_from_nondim_time(T) if need_novelty else None

            for k, tgt in enumerate(target_indices):
                t_idx = t_idx_for_target[k]
                if t_idx is None:
                    continue
                amp_full = np.abs(Y[:, tgt])
                E_full = amp_full ** 2
                dEK_full = E_full.max() - E_full.min()

                cache_key = (t_idx, tgt)
                if (not axis1_in_triad[k]) and cache_key in row_cache:
                    amp_sub = row_cache[cache_key]
                else:
                    amp_sub = _integrate_sub_triad_amplitude(
                        gamma, modes, triads[t_idx], velocities, h_e, 0, t_f, h, N, deg, tgt)
                    if not axis1_in_triad[k]:
                        row_cache[cache_key] = amp_sub
                E_sub = amp_sub ** 2
                dEK_sub = E_sub.max() - E_sub.min()

                if "p_measure" in diagnostics and dEK_sub > MIN_REFERENCE_DEK:
                    results["P"][i, j, k] = 100 * (dEK_full - dEK_sub) / dEK_sub
                if need_novelty:
                    results["NoveltyPeriod"][i, j, k], results["NoveltyRelevance"][i, j, k] = \
                        _novelty_period(T_days, amp_full, amp_sub, dEK_sub,
                                        exclusion_frac=novelty_exclusion_frac,
                                        min_prominence=novelty_min_prominence)
                if need_final_p:
                    dEK_subs_k = {}
                    for t_idx2 in all_t_idx_for_target[k]:
                        axis1_in_t2 = idx1 in triads[t_idx2]
                        cache_key2 = (t_idx2, tgt)
                        if (not axis1_in_t2) and cache_key2 in row_cache:
                            amp_sub2 = row_cache[cache_key2]
                        else:
                            amp_sub2 = _integrate_sub_triad_amplitude(
                                gamma, modes, triads[t_idx2], velocities, h_e, 0, t_f, h, N, deg, tgt)
                            if not axis1_in_t2:
                                row_cache[cache_key2] = amp_sub2
                        E_sub2 = amp_sub2 ** 2
                        dEK_subs_k[t_idx2] = E_sub2.max() - E_sub2.min()
                    fp = final_p_measure(dEK_full, dEK_subs_k)
                    results["PFinal"][i, j, k] = fp['p_measure']
                    results["PFinalRefIdx"][i, j, k] = (
                        fp['reference'] if fp['reference'] is not None else np.nan)

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


if __name__ == "__main__":
    from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
    result = p_measure(spec.modes, triads, spec.velocities, h_e=spec.h_e,
                        reference_triad=spec.reference_triad, tf_days=5, h=0.02)
    assert not np.isnan(result['P']).all()
    print(f"pmeasure self-check OK: P={dict(zip(result['labels'], result['P']))}")
