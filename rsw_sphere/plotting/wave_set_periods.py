"""Dominant-period (Fourier) analysis for quartet/quintet ("wave set")
kinetic-energy time series -- §3/§4's ``fig: domper``/``fig: power*``
figures, modernized.

Two differences from the legacy ``four_waves_2.py``/``five_waves.py``
implementations (both used the same copy-pasted block, reproduced in
``paper-nonlinear-interactions-SWE-sphere/.claude/HARVEST-section-3.md``):

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
    t_days = np.asarray(t_days)
    E_j = np.asarray(E_j)
    n = len(t_days)
    dt = t_days[1] - t_days[0]

    if max_period_days is None:
        max_period_days = t_days[-1] - t_days[0]

    E_detrended = E_j - E_j.mean()
    power_full = np.abs(np.fft.rfft(E_detrended))
    freqs_full = np.fft.rfftfreq(n, d=dt)  # cycles/day

    # Drop the zero-frequency bin (mean already removed, but rfftfreq[0]==0
    # bin can still carry residual power from windowing) and anything whose
    # period exceeds the resolvable horizon.
    keep = (freqs_full > 0) & (1.0 / np.maximum(freqs_full, 1e-300) <= max_period_days)
    freqs = freqs_full[keep]
    power = power_full[keep]
    periods_days = 1.0 / freqs

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
