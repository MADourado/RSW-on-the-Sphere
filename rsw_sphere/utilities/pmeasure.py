"""P-measure and filtering-error (F2) compute for a wave set.

Both compare a target mode's own trajectory in the full wave set against
its trajectory in one constituent triad alone (paper eq. Pa/F2).

    P (%) = 100 * (dEK_full - dEK_triad) / dEK_triad
    F2 = RMS_t(|A_full| - |A_triad|) / RMS_t(|A_triad|)

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
from rsw_sphere.utilities.periods import fft_period_parabolic, prominence_period, novel_frequency_content

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


def _f2(amp_full, amp_sub, dEK_sub):
    if dEK_sub <= MIN_REFERENCE_DEK:
        return np.nan
    rms_diff = np.sqrt(np.mean((amp_full - amp_sub) ** 2))
    rms_sub = np.sqrt(np.mean(amp_sub ** 2))
    if rms_sub <= 0:
        return np.nan
    return rms_diff / rms_sub


def _fmax(E_full, E_sub, dEK_sub):
    """% difference in PEAK kinetic energy, full wave set vs. reference
    triad -- signed, unlike F2 (paper eq. Fmax). A physical-Joules
    prefactor cancels in the ratio, so this is computed directly on the
    nondimensional |A|^2 series.
    """
    if dEK_sub <= MIN_REFERENCE_DEK:
        return np.nan
    peak_sub = E_sub.max()
    if peak_sub <= 0:
        return np.nan
    return 100 * (E_full.max() - peak_sub) / peak_sub


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
_DIAGNOSTIC_ARRAY_KEYS = {"p_measure": "P", "filtering_error": "F2", "frequency_shift": "FreqShift",
                          "fmax": "Fmax", "novelty_period": "NoveltyPeriod"}


#: Above this absolute difference (percentage points) between the two
#: period estimators' own shift %, FreqShiftAgree is False --
#: (frequency_shift_catalogue_search.py's own "agree within 1pp"
#: threshold). Disagreement is NOT treated as an error and does not null
#: FreqShift out: JFM-template.tex Sec. 3.3.5 found the two estimators
#: genuinely disagree for the catalogue's only real effect (EG(1,1)/
#: WG(1,1)), once the gravity mode's own energy share grows enough for a
#: second spectral component to appear -- prominence/peak-counting is
#: known to fail under that amplitude modulation while the FFT peak,
#: though broadened, still tracks the dominant frequency. A NaN-on-
#: disagreement gate would silently erase exactly that published result.
FREQ_SHIFT_AGREEMENT_TOL = 1.0


def _frequency_shift(T_days, amp_full, amp_sub, dEK_sub):
    """(% shift in dominant period, agree) -- full wave set vs. reference
    triad alone. The shift itself is always the FFT-with-parabolic-
    interpolation estimate (never smoothing-dependent, so already immune
    to the Savitzky-Golay artifact class that once inflated a null effect
    to a reported 41-45%); `agree` reports whether a second,
    prominence-filtered peak-timing estimator agrees within
    FREQ_SHIFT_AGREEMENT_TOL -- an advisory reliability signal, not a
    gate, since disagreement can itself be a genuine second spectral
    component (see FREQ_SHIFT_AGREEMENT_TOL's own docstring) rather than
    a measurement error. NaN/False if dEK_sub is too small or no period
    is resolvable at all.
    """
    if dEK_sub <= MIN_REFERENCE_DEK:
        return np.nan, False
    E_full, E_sub = amp_full ** 2, amp_sub ** 2

    period_full_fft = fft_period_parabolic(T_days, E_full)
    period_sub_fft = fft_period_parabolic(T_days, E_sub)
    period_full_prom = prominence_period(T_days, E_full)
    period_sub_prom = prominence_period(T_days, E_sub)

    def _shift(period_full, period_sub):
        if not period_full or not period_sub or period_sub <= 0:
            return np.nan
        return 100 * (period_full - period_sub) / period_sub

    shift_fft = _shift(period_full_fft, period_sub_fft)
    if not np.isfinite(shift_fft):
        return np.nan, False
    shift_prom = _shift(period_full_prom, period_sub_prom)
    agree = bool(np.isfinite(shift_prom) and abs(shift_fft - shift_prom) <= FREQ_SHIFT_AGREEMENT_TOL)
    return shift_fft, agree


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


def pairwise_target_diagnostics(T_days, amp_full, amp_sub,
                                 novelty_exclusion_frac: float = 0.20,
                                 novelty_min_prominence: float = 0.02) -> dict:
    """Every pairwise (full wave set vs. one sub-triad) diagnostic for a
    SINGLE already-integrated target-mode comparison -- reuses the exact
    same per-grid-point formulas ``wave_set_diagnostics_sweep`` computes
    at each cell, just for one point rather than a swept grid (single-run
    reporting, e.g. ``run_dynamics.py --diagnostics``).

    amp_full, amp_sub : |A_target(t)|, full wave set / one sub-triad
        alone, on the SAME time grid (same tf_days/h for both).
    """
    E_full, E_sub = amp_full ** 2, amp_sub ** 2
    dEK_full = E_full.max() - E_full.min()
    dEK_sub = E_sub.max() - E_sub.min()
    p = 100 * (dEK_full - dEK_sub) / dEK_sub if dEK_sub > MIN_REFERENCE_DEK else np.nan
    f2 = _f2(amp_full, amp_sub, dEK_sub)
    fmax = _fmax(E_full, E_sub, dEK_sub)
    freq_shift, freq_shift_agree = _frequency_shift(T_days, amp_full, amp_sub, dEK_sub)
    novelty_period, novelty_relevance = _novelty_period(
        T_days, amp_full, amp_sub, dEK_sub,
        exclusion_frac=novelty_exclusion_frac, min_prominence=novelty_min_prominence)
    return {
        'p_measure': p, 'filtering_error': f2, 'fmax': fmax,
        'frequency_shift': freq_shift, 'frequency_shift_agree': freq_shift_agree,
        'novelty_period': novelty_period, 'novelty_relevance': novelty_relevance,
    }


def wave_set_diagnostics_sweep(modes, triads, h_e: float, swept_indices, fixed_velocities: dict,
                                target_indices, diagnostics=("p_measure", "filtering_error"),
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
        If "frequency_shift" is requested, also FreqShiftAgree (bool array,
        same shape as FreqShift) -- advisory only, see _frequency_shift.
        If "novelty_period" is requested, also NoveltyRelevance (%, same
        shape as NoveltyPeriod) -- see _novelty_period.
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
    need_freq_shift = "frequency_shift" in diagnostics
    if need_freq_shift:
        array_keys = array_keys + ["FreqShiftAgree"]
    need_novelty = "novelty_period" in diagnostics
    if need_novelty:
        array_keys = array_keys + ["NoveltyRelevance"]
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
    u1 = np.linspace(u1_range[0], u1_range[1], n_grid)
    u2 = np.linspace(u2_range[0], u2_range[1], n_grid)
    U1, U2 = np.meshgrid(u1, u2)
    axis1_in_triad = [
        (t_idx is not None and idx1 in triads[t_idx]) for t_idx in t_idx_for_target
    ]

    results = {
        name: (np.zeros((n_grid, n_grid, len(target_indices)), dtype=bool) if name == "FreqShiftAgree"
               else np.full((n_grid, n_grid, len(target_indices)), np.nan))
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
            T_days = days_from_nondim_time(T) if (need_freq_shift or need_novelty) else None

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
                if "filtering_error" in diagnostics:
                    results["F2"][i, j, k] = _f2(amp_full, amp_sub, dEK_sub)
                if "fmax" in diagnostics:
                    results["Fmax"][i, j, k] = _fmax(E_full, E_sub, dEK_sub)
                if need_freq_shift:
                    results["FreqShift"][i, j, k], results["FreqShiftAgree"][i, j, k] = \
                        _frequency_shift(T_days, amp_full, amp_sub, dEK_sub)
                if need_novelty:
                    results["NoveltyPeriod"][i, j, k], results["NoveltyRelevance"][i, j, k] = \
                        _novelty_period(T_days, amp_full, amp_sub, dEK_sub,
                                        exclusion_frac=novelty_exclusion_frac,
                                        min_prominence=novelty_min_prominence)

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
