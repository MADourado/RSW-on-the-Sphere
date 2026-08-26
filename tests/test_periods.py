"""dominant_periods/low_frequency_power/novel_frequency_content on
synthetic signals with known periods."""
import numpy as np

from rsw_sphere.utilities.periods import dominant_periods, low_frequency_power, novel_frequency_content


def test_dominant_periods_recovers_known_tones():
    t = np.linspace(0, 60, 4000)
    E = (1.0 + 0.02 * np.cos(2 * np.pi * t / 4.0) + 0.008 * np.cos(2 * np.pi * t / 12.0)) ** 2
    r = dominant_periods(t, E)
    assert abs(r['period_global'] - 4.0) < 0.3
    assert r['period_local_max'] is not None
    assert abs(r['period_local_max'] - 12.0) < 1.0


def test_low_frequency_power_discriminates_long_vs_short_period():
    t = np.linspace(0, 60, 4000)
    E_low = (1.0 + 0.02 * np.cos(2 * np.pi * t / 20.0)) ** 2
    E_high = (1.0 + 0.02 * np.cos(2 * np.pi * t / 2.0)) ** 2
    p_low = low_frequency_power(t, E_low, period_cutoff_days=10.0)
    p_high = low_frequency_power(t, E_high, period_cutoff_days=10.0)
    assert p_low > 10 * p_high


def test_novel_frequency_content_recovers_injected_tone_and_reports_nothing_when_identical():
    t = np.linspace(0, 60, 4000)
    A_sub = 1.0 + 0.02 * np.cos(2 * np.pi * t / 4.0)
    E_sub = A_sub ** 2
    A_full = A_sub + 0.008 * np.cos(2 * np.pi * t / 12.0)
    E_full = A_full ** 2

    r_novel = novel_frequency_content(t, E_full, t, E_sub, xmax=20.0)
    assert r_novel['novel_peaks']
    dominant = r_novel['novel_peaks'][0]
    assert abs(dominant['period_days'] - 12.0) < 1.0
    assert 0 < dominant['relevance_pct'] < 100

    r_null = novel_frequency_content(t, E_sub, t, E_sub, xmax=20.0)
    assert not r_null['novel_peaks']
