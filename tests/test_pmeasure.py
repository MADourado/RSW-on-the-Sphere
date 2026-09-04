"""efficiency_variation_final / efficiency_variation_combined_for_target
reference-selection logic."""
import math

import numpy as np

from rsw_sphere.utilities.pmeasure import (
    efficiency_variation_final, efficiency_variation_combined_for_target)


def test_efficiency_variation_final_scalar_and_array():
    r = efficiency_variation_final(0.03, {"A": 0.0009, "B": 0.031})
    assert r['reference'] == 'B'  # B has the larger own dEK -> picked as reference
    assert math.isclose(r['efficiency_var'], 100 * (0.03 - 0.031) / 0.031)

    dEK_full = np.array([0.03, 0.01])
    r2 = efficiency_variation_final(dEK_full, {"A": np.array([0.02, 1e-6]), "B": np.array([0.001, 1e-6])})
    assert r2['reference'][0] == 'A'
    assert math.isnan(r2['efficiency_var'][1])  # both candidates below MIN_REFERENCE_DEK
    assert r2['reference'][1] is None


def test_efficiency_variation_combined_for_target_picks_largest_dEK_reference():
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
    r = efficiency_variation_combined_for_target(results, 'X')
    assert r['reference'] == 'triadB'
    assert math.isclose(r['dEK_reference'], E_triadB.max() - E_triadB.min())
    assert np.isfinite(r['spectral_deviation'])
