"""Dominant-period (Fourier) analysis for quartet/quintet ("wave set")
kinetic-energy time series -- §3/§4's ``fig: domper``/``fig: power*``
figures, modernized.

Two differences from the legacy ``four_waves_2.py``/``five_waves.py``
implementations (both now deleted, replaced by ``WaveSet`` -- both used
the same copy-pasted FFT block):

1. **Input is the kinetic-energy series** ``|A_j(t)|^2`` **, not the raw
   complex amplitude.** The paper's own captions describe "the Fourier
   periods ... obtained via FFT of the time series representing the
   kinetic energy field", but the legacy code actually FFT'd the raw
   amplitude (``Y4a``) -- an inconsistency between prose and code, resolved
   here in the prose's favor since kinetic energy is the physically
   meaningful, real-valued quantity whose periodicity the paper discusses.
2. **Peak-finding uses ``scipy.signal.find_peaks``** on the power
   spectrum, not the legacy's float-equality scan (``fft_A[i] > fft_A[i+1]
   and fft_A[i] > fft_A[i-1]``, then indexing back into the array with
   ``fft_A == max_fft[-1]`` -- fragile under ties/noise and returns
   whichever local max happened to be found *last* while scanning
   left-to-right, not the most meaningful one).

``dominant_periods`` returns **two distinct scalars the legacy conflates
under one "dominant period" label**: the period of the single largest
spectral peak (``period_global``, matching ``fig: domper``), and the
*largest-period* local maximum (``period_local_max``, matching the
">50 days" criterion in ``fig: power 4``'s discussion -- a real,
lower-power periodicity that the global max alone would hide).
``max_period_days`` defaults to the record length (``t_days[-1] -
t_days[0]``): a period longer than the observed horizon is unresolvable,
so any period this function reports near that ceiling should be read as
horizon-limited, not necessarily a genuine ">50 day" periodicity -- flag
this explicitly in any caption that cites such a value.

Run as a quick self-check against a synthetic two-tone signal:

    python -m rsw_sphere.plotting.wave_set_periods
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style, mode_color
from rsw_sphere.plotting.labels import _mode_label


def _power_spectrum(t_days, E_j, max_period_days: float = None):
    """Shared FFT prep for ``dominant_periods``/``low_frequency_power``:
    detrend (subtract the mean), ``rfft``, drop the zero-frequency bin, and
    exclude periods beyond ``max_period_days`` (default: the record
    length -- a period longer than what was integrated cannot be
    trusted).

    Returns
    -------
    periods_days, power : ndarray
        Periods (days, descending... actually ascending frequency order,
        i.e. descending period) and spectral magnitude ``|rfft|`` at each.
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
    """Integrated low-frequency spectral power of a kinetic-energy time
    series -- Raphaldini et al. (2022)'s eq. 37 diagnostic
    (``P(omega_tilde) = int_0^omega_tilde |A_hat_j(omega)|^2 domega``,
    restricted there to periods longer than 10 days), used in their own
    Figs. 2(c)/5(c)/8(c)/11(c) to show that the low-frequency power in a
    target mode's kinetic energy tracks the precession-resonance
    efficiency peak -- a claim distinct from (and not implied by) their
    Fig. 3 individual-mode phase-reversal claim already checked in
    ``examples/individual_mode_reversal_investigation.py``.

    Same FFT convention as ``dominant_periods`` (kinetic-energy input,
    mean removed, zero-frequency bin dropped, periods beyond the
    resolvable horizon excluded) -- built on the same ``_power_spectrum``
    helper so the two never drift apart.

    Parameters
    ----------
    t_days : ndarray, shape (n,)
        Uniformly-sampled time, in days.
    E_j : ndarray, shape (n,)
        Kinetic energy series ``|A_j(t)|^2`` for one mode.
    period_cutoff_days : float, optional
        Only periods >= this count as "low frequency" (matching
        Raphaldini et al.'s own 10-day threshold, their eq. 37 caption).
        Default ``10.0``.
    max_period_days : float or None, optional
        See ``_power_spectrum``. Default: the record length.

    Returns
    -------
    float
        ``sqrt(trapz(power[low-freq band]^2, freqs[low-freq band]))`` --
        matches the quantity plotted (as ``sqrt(PSD)``) in Raphaldini et
        al.'s own Figs. 2(c)/5(c)/8(c)/11(c); comparisons across different
        driving amplitudes/scales within one experiment are meaningful
        even though the absolute normalization is not calibrated to match
        their own units exactly (a relative, not absolute, reproduction --
        same caveat as ``examples/reproduce_raphaldini2022_fig2.py``'s own
        "scale, not literal alpha" convention).
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


def dominant_periods(t_days, E_j, max_period_days: float = None, min_prominence_frac: float = 0.01):
    """Dominant period(s) of a kinetic-energy time series via FFT.

    Parameters
    ----------
    t_days : ndarray, shape (n,)
        Uniformly-sampled time, in days.
    E_j : ndarray, shape (n,)
        Kinetic energy series ``|A_j(t)|^2`` for one mode (real, >= 0).
    max_period_days : float or None, optional
        Longest period considered resolvable; periods beyond this are
        excluded before peak-finding. Default: the record length
        ``t_days[-1] - t_days[0]`` (a period longer than what was
        integrated cannot be trusted).
    min_prominence_frac : float, optional
        ``scipy.signal.find_peaks``'s ``prominence``, as a fraction of the
        spectrum's peak power (filters numerical-noise "peaks" of
        negligible height). Default ``0.01``.

    Returns
    -------
    dict
        ``period_global`` (period, days, of the single largest spectral
        peak, excluding the zero-frequency/mean component),
        ``period_local_max`` (period of the local maximum with the
        *largest* period among all detected peaks, or ``None`` if only one
        peak is found -- i.e. the global max itself has no distinct
        "longer, weaker" companion), ``periods_days``/``power`` (full
        spectrum, for plotting), ``horizon_limited`` (bool: whether
        ``period_global`` or ``period_local_max`` is within 10% of
        ``max_period_days``, i.e. likely an artifact of the finite
        integration window rather than a resolved periodicity).
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
            # the largest-period peak *is* the global max -- no distinct
            # second periodicity to report.
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


def wave_set_period_panel(t_days, E, mode_labels, mode_mnalpha, max_period_days: float = None,
                           path: str = None, ax=None):
    """Power-spectrum figure with the dominant/local-max periods marked,
    for every mode in a wave set (one line per mode, matching the energy
    panel's color/legend conventions).

    Parameters
    ----------
    t_days : ndarray, shape (n,)
    E : ndarray, shape (n, n_modes)
        Raw kinetic energy ``|A_j|^2`` per mode -- e.g.
        ``wave_set_dynamics.wave_set_energy_evolution(...)['E']``.
    mode_labels : sequence of str
    mode_mnalpha : sequence of (m, n, alpha)
    max_period_days : float or None, optional
        See ``dominant_periods``.
    path : str or None, optional
    ax : matplotlib.axes.Axes or None, optional

    Returns
    -------
    list of dict
        One ``dominant_periods`` result per mode, in the same order as
        ``mode_labels``.
    """
    results = []
    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    for j in range(E.shape[1]):
        r = dominant_periods(t_days, E[:, j], max_period_days=max_period_days)
        results.append(r)
        m, n, alpha = mode_mnalpha[j]
        color = mode_color(m, n, alpha)
        ax.plot(r['periods_days'], r['power'], label=mode_labels[j], color=color)
        ax.axvline(r['period_global'], color=color, ls='--', lw=1, alpha=0.6)

    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Spectral power (a.u.)')
    ax.set_xlim(left=0)
    ax.legend(loc='upper right', fontsize=7)

    if not own_fig:
        return results

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return results


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
    from rsw_sphere.plotting.wave_set_dynamics import wave_set_energy_evolution_from_spec

    parser = argparse.ArgumentParser(
        description="Plot the power-spectrum (period) figure for a "
                    "quartet/quintet example loaded from the wave-set "
                    "registry YAML (rsw_sphere.dynamics.wave_set_specs."
                    "load_wave_set_specs).")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output image path. If omitted, shown interactively.")
    parser.add_argument(
        "--specs", default=DEFAULT_WAVESETS_PATH,
        help=f"path to the wave-set registry YAML (default: {DEFAULT_WAVESETS_PATH}).")
    parser.add_argument(
        "--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
        default="quartet_rh_preference",
        help="which registered wave set (role key) to integrate and analyze.")
    parser.add_argument(
        "--tf", dest="tf_days", type=float, default=None,
        help="final integration time, in days (default: from registry settings).")
    parser.add_argument(
        "--h", type=float, default=None,
        help="RK33 step size, nondimensional time (default: from registry settings).")
    parser.add_argument(
        "--max-period", dest="max_period_days", type=float, default=None,
        help="longest period considered resolvable, days (default: the "
             "integration horizon tf_days -- a period longer than what "
             "was integrated cannot be trusted, see dominant_periods()'s "
             "docstring).")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[args.wave_set]

    # Throwaway axes: wave_set_energy_evolution_from_spec is a plot
    # function that also returns the integrated trajectory -- called here
    # only for the data, so pass ax= to suppress its own plt.show()/save
    # (calling it with path=ax=None triggered an unwanted interactive
    # window, caught while writing this CLI).
    _, _throwaway_ax = plt.subplots()
    r = wave_set_energy_evolution_from_spec(spec, tf_days=args.tf_days, h=args.h, ax=_throwaway_ax)
    plt.close(_throwaway_ax.figure)

    results = wave_set_period_panel(
        r['t'], r['E'], r['labels'], list(spec.modes),
        max_period_days=args.max_period_days, path=args.path)

    for label, pr in zip(r['labels'], results):
        flag = " [HORIZON-LIMITED]" if pr['horizon_limited'] else ""
        print(f"{label}: period_global={pr['period_global']:.3g}d, "
              f"period_local_max={pr['period_local_max']}{flag}")


if __name__ == "__main__":
    # Synthetic two-tone self-check: a 4-day and a 12-day period signal
    # (plus noise) should be recovered as period_global=4, period_local_max=12
    # (or vice versa depending on relative amplitude -- global is whichever
    # has more power).
    t = np.linspace(0, 60, 4000)
    E = 1.0 + 0.02 * np.cos(2 * np.pi * t / 4.0) + 0.008 * np.cos(2 * np.pi * t / 12.0)
    E = E ** 2  # kinetic-energy-like (always positive, dominated by the mean)
    r = dominant_periods(t, E)
    print(f"period_global      = {r['period_global']:.3f} days (expect ~4)")
    print(f"period_local_max   = {r['period_local_max']}  (expect ~12)")
    print(f"horizon_limited    = {r['horizon_limited']}")
    assert abs(r['period_global'] - 4.0) < 0.3, "self-check FAILED: period_global"
    assert r['period_local_max'] is not None and abs(r['period_local_max'] - 12.0) < 1.0, \
        "self-check FAILED: period_local_max"
    print("self-check OK")

    # low_frequency_power: a signal with only long-period content should
    # score higher than a pure short-period signal of the same amplitude.
    E_low = (1.0 + 0.02 * np.cos(2 * np.pi * t / 20.0)) ** 2
    E_high = (1.0 + 0.02 * np.cos(2 * np.pi * t / 2.0)) ** 2
    p_low = low_frequency_power(t, E_low, period_cutoff_days=10.0)
    p_high = low_frequency_power(t, E_high, period_cutoff_days=10.0)
    print(f"\nlow_frequency_power: 20-day signal = {p_low:.4f}, 2-day signal = {p_high:.4f} (expect low >> high)")
    assert p_low > 10 * p_high, "self-check FAILED: low_frequency_power did not discriminate"
    print("low_frequency_power self-check OK")
