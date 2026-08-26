"""dominant_periods/low_frequency_power on synthetic signals with known periods."""
import numpy as np

from rsw_sphere.utilities.periods import dominant_periods, low_frequency_power


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
