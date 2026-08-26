"""p_measure's MIN_REFERENCE_DEK gate, and wave_set_diagnostics_sweep's
combined P/F2 sweep against p_measure_sweep."""
import math

import numpy as np

from rsw_sphere.utilities.pmeasure import (
    p_measure, p_measure_sweep, wave_set_diagnostics_sweep, MIN_REFERENCE_DEK)

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
                  reference_triad=0, n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    p_only = p_measure_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0, **kwargs)
    combined = wave_set_diagnostics_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0,
                                           diagnostics=("p_measure",), **kwargs)
    np.testing.assert_allclose(p_only['P'], combined['P'], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_diagnostics_sweep_shared_gate_matches_own_private_mode_velocity():
    """target c's own triad {a,b,c} excludes d, so P/F2 for c are NaN only
    where c=0; symmetric for target d."""
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        u1_range=(0.0, 30.0), u2_range=(0.0, 30.0), reference_triad=0,
        n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)

    for array_key in ('P', 'F2'):
        values = result[array_key]
        # grid: index 0 -> velocity 0.0, index 1 -> velocity 30.0
        assert math.isnan(values[0, 0, 0])
        assert math.isnan(values[1, 0, 0])
        assert not math.isnan(values[0, 1, 0])
        assert math.isnan(values[0, 0, 1])
        assert math.isnan(values[0, 1, 1])
        assert not math.isnan(values[1, 0, 1])

    finite_f2 = result['F2'][~np.isnan(result['F2'])]
    assert finite_f2.size > 0
    assert (finite_f2 >= 0).all()


def test_diagnostics_sweep_subset_only_computes_requested_arrays():
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("filtering_error",), u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        reference_triad=0, n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    assert "F2" in result
    assert "P" not in result


def test_frequency_shift_matches_shared_gate():
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("frequency_shift",), u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        reference_triad=0, n_grid=2, tf_days=5, h=0.02, N=N, deg=DEG)
    values = result['FreqShift']
    assert math.isnan(values[0, 0, 0])  # c=0 -> target-c undefined
    assert math.isnan(values[0, 0, 1])  # d=0 -> target-d undefined
    finite = values[~np.isnan(values)]
    assert finite.size > 0
