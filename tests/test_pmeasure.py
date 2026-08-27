"""p_measure's MIN_REFERENCE_DEK gate, and final_p_measure/
p_measure_combined_for_target's own reference-selection logic."""
import math

import numpy as np
import pytest

from rsw_sphere.utilities.pmeasure import (
    p_measure, MIN_REFERENCE_DEK,
    final_p_measure, p_measure_combined_for_target)

pytestmark = pytest.mark.slow

N, DEG = 6, 300  # low N for speed; deg MUST stay 300 -- see tests/test_wave_sets.py

MODES = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]  # sum=0, members=1,2
TRIADS = [(0, 1, 2)]


def test_zero_denominator_gives_nan():
    velocities = [0.0, 0.0, 0.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert result['dEK_triad'][0] == 0.0
    assert math.isnan(result['P'][0])


def test_small_nonzero_denominator_below_threshold_gives_nan():
    velocities = [1.0, 1.0, 1.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert 0.0 < result['dEK_triad'][0] < MIN_REFERENCE_DEK
    assert math.isnan(result['P'][0])


def test_denominator_above_threshold_gives_finite_value():
    velocities = [30.0, 30.0, 30.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert result['dEK_triad'][0] > MIN_REFERENCE_DEK
    assert not math.isnan(result['P'][0])


def test_final_p_measure_scalar_and_array():
    r = final_p_measure(0.03, {"A": 0.0009, "B": 0.031})
    assert r['reference'] == 'B'  # B has the larger own dEK -> picked as reference
    assert math.isclose(r['p_measure'], 100 * (0.03 - 0.031) / 0.031)

    dEK_full = np.array([0.03, 0.01])
    r2 = final_p_measure(dEK_full, {"A": np.array([0.02, 1e-6]), "B": np.array([0.001, 1e-6])})
    assert r2['reference'][0] == 'A'
    assert math.isnan(r2['p_measure'][1])  # both candidates below MIN_REFERENCE_DEK
    assert r2['reference'][1] is None


def test_p_measure_combined_for_target_picks_largest_dEK_reference():
    """Regression for the small-denominator inflation found 2026-08-27 in
    quartet_rossby_gravity_influence: a mode shared across triads must be
    scored against whichever containing triad gives it the larger own
    energy variation, not an arbitrary/default one."""
    t = np.linspace(0, 60, 2000)
    E_full = (0.02 + 0.01 * np.cos(2 * np.pi * t / 4.0)) ** 2
    E_triadA = (0.0169 + 0.0004 * np.cos(2 * np.pi * t / 4.0)) ** 2  # small dEK
    E_triadB = (0.032 + 0.016 * np.cos(2 * np.pi * t / 4.0)) ** 2    # large dEK
    ones = np.ones_like(t)
    results = {
        'full': {'labels': ['X'], 'E': E_full[:, None], 'E_total': ones, 't': t},
        'triadA': {'labels': ['X'], 'E': E_triadA[:, None], 'E_total': ones, 't': t},
        'triadB': {'labels': ['X'], 'E': E_triadB[:, None], 'E_total': ones, 't': t},
    }
    r = p_measure_combined_for_target(results, 'X')
    assert r['reference'] == 'triadB'
    assert math.isclose(r['dEK_reference'], E_triadB.max() - E_triadB.min())
    assert np.isfinite(r['spectral_deviation'])
