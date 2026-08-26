"""Dominant-period (FFT) analysis and low-frequency spectral power for a
kinetic-energy time series |A_j(t)|^2.

Run as a quick self-check (synthetic two-tone signal):

    python -m rsw_sphere.utilities.periods
"""
import numpy as np
from scipy.signal import find_peaks


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


def fft_period_parabolic(t_days, E_j):
    """Dominant period via real FFT of the mean-subtracted series, with
    parabolic interpolation of the peak bin against its two neighbors --
    window-independent, no tunable smoothing parameter (unlike a
    Savitzky-Golay-smoothed peak search, which can silently report a
    period that is a monotone function of the smoothing window --
    caught this way once already, see `prominence_period`'s own
    docstring). Returns period in days, or None if no resolvable peak.
    """
    t_days = np.asarray(t_days)
    E_j = np.asarray(E_j)
    dt_days = t_days[1] - t_days[0]
    x = E_j - E_j.mean()
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=dt_days)
    # Ignore the DC bin, anything below a resolvable minimum period
    # (2*dt, Nyquist), and anything longer than a third of the window
    # (unreliable).
    valid = (freqs > 1.0 / (t_days[-1] / 3)) & (freqs < 1.0 / (2 * dt_days))
    if not np.any(valid):
        return None
    idx_candidates = np.where(valid)[0]
    k = idx_candidates[np.argmax(spec[idx_candidates])]
    if k <= 0 or k >= len(spec) - 1:
        return 1.0 / freqs[k]
    y0, y1, y2 = spec[k - 1], spec[k], spec[k + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-300 else 0.0
    delta = np.clip(delta, -0.5, 0.5)
    f_refined = freqs[k] + delta * (freqs[1] - freqs[0])
    return 1.0 / f_refined if f_refined > 0 else None


def prominence_period(t_days, E_j):
    """Peak-to-peak period via prominence-filtered peaks on the RAW
    (unsmoothed) trace -- prominence threshold set relative to the
    trace's own amplitude range, not an absolute/tuned constant. Cross-
    checking this against `fft_period_parabolic` is what caught a
    Savitzky-Golay smoothing artifact that had inflated a reported
    frequency shift to 41-45% when the true, window-independent effect
    was <=0.1% (2026-08-13) -- a single estimator would not have. Returns
    period in days, or None if fewer than 3 peaks are found.
    """
    t_days = np.asarray(t_days)
    E_j = np.asarray(E_j)
    prom = 0.1 * (E_j.max() - E_j.min())
    peaks, _ = find_peaks(E_j, prominence=prom)
    if len(peaks) < 3:
        return None
    return float(np.mean(np.diff(t_days[peaks])))


def dominant_periods(t_days, E_j, max_period_days: float = None, min_prominence_frac: float = 0.01):
    """Dominant period(s) via FFT.

    Returns
    -------
    dict
        period_global (largest spectral peak, days), period_local_max
        (largest-period local max, or None if same as global),
        periods_days/power (full spectrum), horizon_limited (bool: within
        10% of max_period_days -- likely a finite-window artifact).
    """
    periods_days, power = _power_spectrum(t_days, E_j, max_period_days)
    if max_period_days is None:
        max_period_days = t_days[-1] - t_days[0]

    if len(power) == 0:
        return {'period_global': np.nan, 'period_local_max': None,
                'periods_days': periods_days, 'power': power, 'horizon_limited': False}

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

    return {
        'period_global': period_global,
        'period_local_max': period_local_max,
        'periods_days': periods_days,
        'power': power,
        'horizon_limited': horizon_limited,
    }


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
    print("periods self-check OK")
