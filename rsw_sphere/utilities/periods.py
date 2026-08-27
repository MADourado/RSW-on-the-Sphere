"""Dominant-period (FFT) analysis and low-frequency spectral power for a
kinetic-energy time series |A_j(t)|^2.

Run as a quick self-check (synthetic two-tone signal):

    python -m rsw_sphere.utilities.periods
"""
import numpy as np
from scipy.signal import find_peaks, peak_widths


def _power_spectrum(t_days, E_j, max_period_days: float = None):
    """FFT prep shared by dominant_periods/low_frequency_power: detrend,
    rfft, drop the zero-frequency bin, exclude periods beyond
    max_period_days (default: record length).

    Returns (periods_days, power).
    """
    t_days = np.asarray(t_days)
    E_j = np.asarray(E_j)
    n = len(t_days)
    dt = t_days[1] - t_days[0]

    if max_period_days is None:
        max_period_days = t_days[-1] - t_days[0]

    E_detrended = E_j - E_j.mean()
    power_full = np.abs(np.fft.rfft(E_detrended))
    freqs_full = np.fft.rfftfreq(n, d=dt)  # cycles/day

    keep = (freqs_full > 0) & (1.0 / np.maximum(freqs_full, 1e-300) <= max_period_days)
    freqs = freqs_full[keep]
    power = power_full[keep]
    periods_days = 1.0 / freqs
    return periods_days, power


def low_frequency_power(t_days, E_j, period_cutoff_days: float = 10.0, max_period_days: float = None):
    """Integrated low-frequency spectral power (Raphaldini et al. 2022,
    eq. 37): sqrt(trapz(power[period>=cutoff]^2, freq)).
    """
    periods_days, power = _power_spectrum(t_days, E_j, max_period_days)
    if len(power) == 0:
        return 0.0
    low = periods_days >= period_cutoff_days
    if not np.any(low):
        return 0.0
    freqs_low = 1.0 / periods_days[low]
    order = np.argsort(freqs_low)
    integral = np.trapz(power[low][order] ** 2, freqs_low[order])
    return float(np.sqrt(max(integral, 0.0)))


def dominant_periods(t_days, E_j, max_period_days: float = None, min_prominence_frac: float = 0.01,
                      n_top: int = 3):
    """Dominant period(s) via FFT.

    Returns
    -------
    dict
        period_global (largest spectral peak, days), period_local_max
        (largest-period local max, or None if same as global),
        periods_days/power (full spectrum), horizon_limited (bool: within
        10% of max_period_days -- likely a finite-window artifact),
        top_peaks (up to `n_top` local maxima -- the global peak plus any
        other `find_peaks`-significant one -- sorted by descending power;
        each a dict with period_days and power_frac, that peak's power as
        a percentage of the global peak's own power, so the first entry
        is always 100%).
    """
    periods_days, power = _power_spectrum(t_days, E_j, max_period_days)
    if max_period_days is None:
        max_period_days = t_days[-1] - t_days[0]

    if len(power) == 0:
        return {'period_global': np.nan, 'period_local_max': None,
                'periods_days': periods_days, 'power': power, 'horizon_limited': False,
                'top_peaks': []}

    i_global = int(np.argmax(power))
    period_global = periods_days[i_global]

    prominence = min_prominence_frac * power.max()
    peak_idx, _ = find_peaks(power, prominence=prominence)
    if len(peak_idx) > 1:
        candidate_periods = periods_days[peak_idx]
        period_local_max = float(np.max(candidate_periods))
        if period_local_max == period_global:
            period_local_max = None
    else:
        period_local_max = None

    horizon_limited = (period_global >= 0.9 * max_period_days) or \
        (period_local_max is not None and period_local_max >= 0.9 * max_period_days)

    # top_peaks: `find_peaks` alone can miss the global maximum itself
    # (e.g. a spectrum still rising at the lowest resolvable frequency,
    # so there's no interior local max there to detect) -- union it in
    # before ranking so the reported #1 peak always matches period_global.
    all_idx = np.union1d(peak_idx, [i_global])
    order = all_idx[np.argsort(-power[all_idx])][:n_top]
    top_peaks = [{'period_days': float(periods_days[idx]),
                  'power_frac': float(100 * power[idx] / power[i_global])}
                 for idx in order]

    return {
        'period_global': period_global,
        'period_local_max': period_local_max,
        'periods_days': periods_days,
        'power': power,
        'horizon_limited': horizon_limited,
        'top_peaks': top_peaks,
    }


def novel_frequency_content_multi(t_full, E_full, subs, xmax: float = 3.0,
                                   min_prominence: float = 0.02, exclusion_frac: float = 0.20,
                                   n_grid: int = 4000):
    """"Novelty" frequency content of a full wave-set trajectory relative
    to EVERY sub-triad that contains the same target mode at once.

    Deliberately NOT "how much did the dominant period shift": a naive
    argmax of a (full - sub) difference spectrum picks up a dipole
    artifact from any small shift in a sub-triad's own tall, narrow peak
    (dwarfing genuinely new but smaller spectral content) rather than
    real novel content. Instead: at every period, compare the full
    spectrum against the BEST explanation any single sub-triad offers
    (elementwise max of every sub-triad's own peak-normalized spectrum --
    a mode already well-explained by ONE of its constituent triads isn't
    "novel" just because a DIFFERENT triad doesn't also contain it),
    excluding a +/-`exclusion_frac` window around EACH sub-triad's own
    dominant peak (never the full spectrum's own peak -- excluding that
    too breaks detection whenever the full spectrum's dominant peak IS
    the genuine novel content, since a small shift already keeps the two
    peaks within the same sub-only window when that's the artifact case).
    Local maxima of what remains are the candidate novel peaks.

    Parameters
    ----------
    t_full, E_full : full wave-set trajectory (KE = |A_target|^2).
    subs : sequence of (t_sub, E_sub) pairs, one per sub-triad containing
        the same target mode (length 1 for a private mode; 2+ for a mode
        shared across triads, member or sum role alike).
    xmax : upper period (days) considered.
    min_prominence : `scipy.signal.find_peaks` prominence threshold on the
        peak-normalized difference spectrum (0-1ish scale) -- a candidate
        novel peak below this is treated as noise, not reported.
    exclusion_frac : half-width of the excluded window around each
        sub-triad's own dominant peak, as a fraction of that peak's period.

    Returns
    -------
    dict
        periods_days, power_diff (NaN inside the excluded window),
        excluded (bool mask, union across every sub), novel_peaks (list
        of dicts: period_days, prominence, band_days (peak_widths
        half-max band), relevance_pct (that band's share of the FULL
        spectrum's own total raw power) -- sorted dominant (highest
        relevance_pct) first, empty if nothing survives min_prominence).
    """
    p_full_raw, pow_full_raw = _power_spectrum(t_full, E_full)
    peak_full = pow_full_raw.max() if len(pow_full_raw) else 1.0
    pow_full_n = pow_full_raw / peak_full if peak_full > 0 else pow_full_raw

    common_periods = np.linspace(0.01, xmax, n_grid)
    interp_full = np.interp(common_periods, p_full_raw[::-1], pow_full_n[::-1])
    interp_full_raw = np.interp(common_periods, p_full_raw[::-1], pow_full_raw[::-1])
    total_power_full = np.trapz(interp_full_raw, common_periods)

    envelope = np.zeros_like(common_periods)
    excluded = np.zeros_like(common_periods, dtype=bool)
    for t_sub, E_sub in subs:
        p_sub_raw, pow_sub_raw = _power_spectrum(t_sub, E_sub)
        peak_sub = pow_sub_raw.max() if len(pow_sub_raw) else 1.0
        pow_sub_n = pow_sub_raw / peak_sub if peak_sub > 0 else pow_sub_raw
        interp_sub = np.interp(common_periods, p_sub_raw[::-1], pow_sub_n[::-1])
        envelope = np.maximum(envelope, interp_sub)

        sub_dominant_period = dominant_periods(t_sub, E_sub)['period_global']
        if np.isfinite(sub_dominant_period):
            lo = sub_dominant_period * (1 - exclusion_frac)
            hi = sub_dominant_period * (1 + exclusion_frac)
            excluded |= (common_periods >= lo) & (common_periods <= hi)

    diff = interp_full - envelope

    search = diff.copy()
    search[excluded] = 0.0  # zeroed (not NaN) so find_peaks sees a valid, flat region there
    peak_idx, props = find_peaks(search, prominence=min_prominence)

    novel_peaks = []
    for idx, prom in zip(peak_idx, props['prominences']):
        period = common_periods[idx]
        widths, width_heights, left_ips, right_ips = peak_widths(search, [idx], rel_height=0.5)
        lo_i = np.interp(left_ips[0], np.arange(len(common_periods)), common_periods)
        hi_i = np.interp(right_ips[0], np.arange(len(common_periods)), common_periods)
        band_mask = (common_periods >= lo_i) & (common_periods <= hi_i)
        band_power = np.trapz(interp_full_raw[band_mask], common_periods[band_mask]) \
            if band_mask.sum() > 1 else 0.0
        relevance_pct = 100 * band_power / total_power_full if total_power_full > 0 else 0.0
        novel_peaks.append({
            'period_days': float(period), 'prominence': float(prom),
            'band_days': (float(lo_i), float(hi_i)), 'relevance_pct': float(relevance_pct),
        })
    novel_peaks.sort(key=lambda d: -d['relevance_pct'])

    diff_masked = diff.copy()
    diff_masked[excluded] = np.nan
    return {
        'periods_days': common_periods, 'power_diff': diff_masked, 'excluded': excluded,
        'novel_peaks': novel_peaks,
    }


def novel_frequency_content(t_full, E_full, t_sub, E_sub, **kwargs):
    """Single-sub-triad case of ``novel_frequency_content_multi`` (exactly
    equivalent -- an envelope of one spectrum is just that spectrum, and
    the excluded window is just that one sub-triad's own). Kept as its
    own name since most callers only ever have one sub-triad in hand
    (e.g. ``pmeasure.pairwise_target_diagnostics``'s own per-target-point
    diagnostic, which compares against one reference triad per target,
    not every containing triad).
    """
    return novel_frequency_content_multi(t_full, E_full, [(t_sub, E_sub)], **kwargs)


def spectral_deviation(t_full, Q_full, t_sub, Q_sub, xmax: float = 3.0, n_grid: int = 4000):
    """Spectral, share-normalized replacement for the retired time-domain
    "filtering error" (RMS trajectory-tracking error) -- built on the
    same primitive (``_power_spectrum``) as ``novel_frequency_content``:
    compare power spectra, not raw point-by-point time series.

    Two fixes over the retired version: (1) the caller passes each
    series' own amplitude *share*, ``q = |A_target|/sqrt(E_total.mean())``,
    not raw amplitude -- otherwise a sub-triad with fewer active modes
    (and therefore less total energy) reads as "different" just for
    having a smaller budget (mirrors ``wave_set_efficiency``'s own fix;
    amplitude share rather than energy share also avoids quadratically
    amplifying a given relative difference). (2) comparing spectra rather
    than raw time series avoids the old metric's dependence on wherever
    the two trajectories happen to be out of phase at the integration's
    arbitrary cutoff ``t_f`` -- a periodogram integrates coherently over
    the whole window instead.

    Parameters
    ----------
    t_full, Q_full : full wave set's own trajectory -- pass a share-
        normalized series (typically amplitude share ``q``), NOT raw
        amplitude or raw energy.
    t_sub, Q_sub : one sub-triad's own trajectory, same convention.
    xmax : upper period (days) considered (matches
        ``novel_frequency_content``'s own default range).
    n_grid : common period-grid resolution.

    Returns
    -------
    float
        Relative L2 distance between the two (peak-unnormalized, share-
        normalized) power spectra on a common period grid, as a
        percentage (matching P-measure/efficiency_variation's own
        convention) --
        ``100 * sqrt(sum((Qhat_full-Qhat_sub)^2)) / max(norm_full, norm_sub)``.
        Normalizing by whichever side has the LARGER spectral power
        (not the reference alone) avoids the same "weak fixed reference"
        inflation ``final_p_measure`` guards against for P/efficiency --
        a reference that barely excites the target has almost no power
        to normalize by. 0 if the spectra are identical, NaN if neither
        side has any resolvable spectral power.
    """
    p_full_raw, pow_full_raw = _power_spectrum(t_full, Q_full)
    p_sub_raw, pow_sub_raw = _power_spectrum(t_sub, Q_sub)

    common_periods = np.linspace(0.01, xmax, n_grid)
    interp_full = np.interp(common_periods, p_full_raw[::-1], pow_full_raw[::-1])
    interp_sub = np.interp(common_periods, p_sub_raw[::-1], pow_sub_raw[::-1])

    norm_full = np.sqrt(np.sum(interp_full ** 2))
    norm_sub = np.sqrt(np.sum(interp_sub ** 2))
    denom = max(norm_full, norm_sub)
    if denom <= 0:
        return float("nan")
    return float(100 * np.sqrt(np.sum((interp_full - interp_sub) ** 2)) / denom)


if __name__ == "__main__":
    # 4-day + 12-day tone: expect period_global~4, period_local_max~12.
    t = np.linspace(0, 60, 4000)
    E = 1.0 + 0.02 * np.cos(2 * np.pi * t / 4.0) + 0.008 * np.cos(2 * np.pi * t / 12.0)
    E = E ** 2
    r = dominant_periods(t, E)
    assert abs(r['period_global'] - 4.0) < 0.3, "self-check FAILED: period_global"
    assert r['period_local_max'] is not None and abs(r['period_local_max'] - 12.0) < 1.0, \
        "self-check FAILED: period_local_max"

    E_low = (1.0 + 0.02 * np.cos(2 * np.pi * t / 20.0)) ** 2
    E_high = (1.0 + 0.02 * np.cos(2 * np.pi * t / 2.0)) ** 2
    p_low = low_frequency_power(t, E_low, period_cutoff_days=10.0)
    p_high = low_frequency_power(t, E_high, period_cutoff_days=10.0)
    assert p_low > 10 * p_high, "self-check FAILED: low_frequency_power did not discriminate"

    # novel_frequency_content: E_sub a clean 4-day tone; E_full the SAME
    # tone plus a smaller, genuinely new 12-day tone -- expect it detected
    # with a sane relevance %, and nothing when the two signals are equal.
    t2 = np.linspace(0, 60, 4000)
    A_sub = 1.0 + 0.02 * np.cos(2 * np.pi * t2 / 4.0)
    E_sub2 = A_sub ** 2
    A_full = A_sub + 0.008 * np.cos(2 * np.pi * t2 / 12.0)
    E_full2 = A_full ** 2

    r_novel = novel_frequency_content(t2, E_full2, t2, E_sub2, xmax=20.0)
    assert r_novel['novel_peaks'], "self-check FAILED: novel_frequency_content found nothing"
    dominant = r_novel['novel_peaks'][0]
    assert abs(dominant['period_days'] - 12.0) < 1.0, \
        f"self-check FAILED: novelty period {dominant['period_days']} != ~12d"
    assert 0 < dominant['relevance_pct'] < 100, \
        f"self-check FAILED: relevance_pct {dominant['relevance_pct']} out of range"

    r_null = novel_frequency_content(t2, E_sub2, t2, E_sub2, xmax=20.0)
    assert not r_null['novel_peaks'], "self-check FAILED: identical signals reported a novel peak"

    print("periods self-check OK")
