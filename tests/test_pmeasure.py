"""p_measure's MIN_REFERENCE_DEK gate, and wave_set_diagnostics_sweep's
combined P/F2 sweep against p_measure_sweep."""
import math

import numpy as np
import pytest

from rsw_sphere.utilities.pmeasure import (
    p_measure, p_measure_sweep, wave_set_diagnostics_sweep, MIN_REFERENCE_DEK,
    final_p_measure, p_measure_combined_for_target)

pytestmark = pytest.mark.slow

N, DEG = 6, 300  # low N for speed; deg MUST stay 300 -- see tests/test_wave_sets.py

MODES = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]  # sum=0, members=1,2
TRIADS = [(0, 1, 2)]

# a=RH(4,5), b=RH(3,4), c=RH(1,2), d=EG(1,1); triads {a,b,c} and {a,b,d}.
QUARTET_MODES = [(4, 5, 3), (3, 4, 3), (1, 2, 3), (1, 1, 1)]
QUARTET_TRIADS = [(0, 1, 2), (0, 1, 3)]


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


def test_diagnostics_sweep_p_measure_matches_p_measure_sweep():
    """P from wave_set_diagnostics_sweep and p_measure_sweep must agree
    (both share _integrate_sub_triad_amplitude)."""
    kwargs = dict(swept_indices=(2, 3), fixed_velocities={0: 30.0, 1: 30.0},
                  target_indices=[2, 3], u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
                  reference_triad=0, n_grid=2, tf_days=1, h=0.05, N=N, deg=DEG)
    p_only = p_measure_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0, **kwargs)
    combined = wave_set_diagnostics_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0,
                                           diagnostics=("p_measure",), **kwargs)
    np.testing.assert_allclose(p_only['P'], combined['P'], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_diagnostics_sweep_shared_gate_matches_own_private_mode_velocity():
    """target c's own triad {a,b,c} excludes d, so P for c is NaN only
    where c=0; symmetric for target d."""
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        u1_range=(0.0, 30.0), u2_range=(0.0, 30.0), reference_triad=0,
        n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)

    values = result['P']
    # grid: index 0 -> velocity 0.0, index 1 -> velocity 30.0
    assert math.isnan(values[0, 0, 0])
    assert math.isnan(values[1, 0, 0])
    assert not math.isnan(values[0, 1, 0])
    assert math.isnan(values[0, 0, 1])
    assert math.isnan(values[0, 1, 1])
    assert not math.isnan(values[1, 0, 1])


def test_diagnostics_sweep_subset_only_computes_requested_arrays():
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("novelty_period",), u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        reference_triad=0, n_grid=2, tf_days=1, h=0.05, N=N, deg=DEG)
    assert "NoveltyPeriod" in result
    assert "P" not in result


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


def test_wave_set_diagnostics_sweep_p_measure_final_matches_forced_single_reference():
    """PFinal must, at every grid point, equal exactly the single-reference
    P a caller would get by forcing that SAME triad via triad_index --
    it picks among the existing per-triad numbers, it doesn't invent one."""
    kwargs = dict(swept_indices=(2, 3), fixed_velocities={0: 30.0, 1: 30.0},
                  u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
                  reference_triad=0, n_grid=2, tf_days=1, h=0.05, N=N, deg=DEG)
    target_indices = [0, 1]  # a=RH(4,5), b=RH(3,4): shared members of BOTH triads

    p0 = wave_set_diagnostics_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0,
                                     target_indices=target_indices, diagnostics=("p_measure",),
                                     triad_index={0: 0, 1: 0}, **kwargs)['P']
    p1 = wave_set_diagnostics_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0,
                                     target_indices=target_indices, diagnostics=("p_measure",),
                                     triad_index={0: 1, 1: 1}, **kwargs)['P']
    final = wave_set_diagnostics_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0,
                                        target_indices=target_indices,
                                        diagnostics=("p_measure_final",), **kwargs)
    pf, ref = final['PFinal'], final['PFinalRefIdx']

    matches_either = np.isclose(pf, p0, equal_nan=True) | np.isclose(pf, p1, equal_nan=True)
    assert matches_either.all()
    finite_ref = ref[np.isfinite(ref)]
    assert finite_ref.size > 0
    assert set(np.unique(finite_ref).astype(int)).issubset({0, 1})


