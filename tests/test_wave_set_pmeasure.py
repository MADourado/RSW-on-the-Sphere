"""``p_measure``'s NaN guard: fires on an exact-zero denominator dEK AND
on the wider near-zero band below ``MIN_REFERENCE_DEK`` -- widened
2026-08-25 from an exact-zero-only check to a threshold one, matching the
paper's own eq. Pa statement that P is undefined below $10^{-4}$
(nondimensional $\\|A\\|^2$), not just at exactly 0 (see
rsw_sphere.plotting.wave_set_pmeasure's own module docstring, point 2).

Also covers ``wave_set_diagnostics_sweep`` (the combined P-measure +
filtering-error sweep added the same day, replacing a separate
``wave_set_filtering_error.py`` module that duplicated the grid loop):
its own P-measure output must match ``p_measure_sweep``'s exactly (the
refactor that introduced ``_integrate_sub_triad_amplitude`` must not
change any P value), and its shared MIN_REFERENCE_DEK gate must behave
the same way for F2 as it does for P.
"""
import math

import numpy as np

from rsw_sphere.plotting.wave_set_pmeasure import (
    p_measure, p_measure_sweep, wave_set_diagnostics_sweep, MIN_REFERENCE_DEK)

N, DEG = 6, 300  # low N for speed; deg MUST stay 300 -- see tests/test_wave_sets.py

MODES = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]  # sum=0, members=1,2
TRIADS = [(0, 1, 2)]

# Quartet C's own mode shape (m_sum=m_p+m_q for both constituent triads):
# a=RH(4,5), b=RH(3,4), c=RH(1,2), d=EG(1,1); triads {a,b,c} and {a,b,d}.
QUARTET_MODES = [(4, 5, 3), (3, 4, 3), (1, 2, 3), (1, 1, 1)]
QUARTET_TRIADS = [(0, 1, 2), (0, 1, 3)]


def test_zero_denominator_gives_nan():
    """Every mode in the (only) reference triad starts at rest -> its own
    dEK_triad is exactly 0 -> P must be NaN, not a division-by-zero error
    or a spurious finite value."""
    velocities = [0.0, 0.0, 0.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert result['dEK_triad'][0] == 0.0
    assert math.isnan(result['P'][0])


def test_small_nonzero_denominator_below_threshold_gives_nan():
    """A driving velocity small enough that dEK_triad is nonzero but still
    under MIN_REFERENCE_DEK -> P must be NaN (the ill-conditioned regime
    the threshold exists to guard, not just the exact-zero corner)."""
    velocities = [1.0, 1.0, 1.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert 0.0 < result['dEK_triad'][0] < MIN_REFERENCE_DEK
    assert math.isnan(result['P'][0])


def test_denominator_above_threshold_gives_finite_value():
    """A driving velocity large enough that dEK_triad clears
    MIN_REFERENCE_DEK -> P must be a finite number, not NaN."""
    velocities = [30.0, 30.0, 30.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert result['dEK_triad'][0] > MIN_REFERENCE_DEK
    assert not math.isnan(result['P'][0])


def test_diagnostics_sweep_p_measure_matches_p_measure_sweep():
    """wave_set_diagnostics_sweep's own 'P' array must agree with
    p_measure_sweep's to floating-point precision -- the refactor that
    factored both onto _integrate_sub_triad_amplitude must not change any
    P value beyond the ULP-level difference between real(z*conj(z)) and
    abs(z)**2 (mathematically identical, not bit-identical)."""
    kwargs = dict(swept_indices=(2, 3), fixed_velocities={0: 30.0, 1: 30.0},
                  target_indices=[2, 3], u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
                  reference_triad=0, n_grid=3, tf_days=2, h=0.02, N=N, deg=DEG)
    p_only = p_measure_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0, **kwargs)
    combined = wave_set_diagnostics_sweep(QUARTET_MODES, QUARTET_TRIADS, 10000.0,
                                           diagnostics=("p_measure",), **kwargs)
    np.testing.assert_allclose(p_only['P'], combined['P'], rtol=1e-9, atol=1e-9, equal_nan=True)


def test_diagnostics_sweep_shared_gate_matches_own_private_mode_velocity():
    """Sweeping c=index2, d=index3 from 0 to 30 m/s with a,b fixed at 30:
    target c's own reference triad {a,b,c} doesn't depend on d, so both P
    and F2 for target c are NaN only in the row where c=0, regardless of
    d; symmetric for target d."""
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        u1_range=(0.0, 30.0), u2_range=(0.0, 30.0), reference_triad=0,
        n_grid=2, tf_days=2, h=0.02, N=N, deg=DEG)

    for array_key in ('P', 'F2'):
        values = result[array_key]
        # grid: index 0 -> velocity 0.0, index 1 -> velocity 30.0
        assert math.isnan(values[0, 0, 0])  # c=0 -> target-c undefined
        assert math.isnan(values[1, 0, 0])  # c=0 (any d) -> target-c undefined
        assert not math.isnan(values[0, 1, 0])  # c=30 -> target-c defined
        assert math.isnan(values[0, 0, 1])  # d=0 -> target-d undefined
        assert math.isnan(values[0, 1, 1])  # d=0 (any c) -> target-d undefined
        assert not math.isnan(values[1, 0, 1])  # d=30 -> target-d defined

    finite_f2 = result['F2'][~np.isnan(result['F2'])]
    assert finite_f2.size > 0
    assert (finite_f2 >= 0).all()  # F2 is an RMS relative error -- must be >= 0


def test_diagnostics_sweep_subset_only_computes_requested_arrays():
    """diagnostics=('filtering_error',) must not return a 'P' key -- the
    switch actually selects what gets computed, not just what gets
    plotted."""
    result = wave_set_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("filtering_error",), u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        reference_triad=0, n_grid=2, tf_days=2, h=0.02, N=N, deg=DEG)
    assert "F2" in result
    assert "P" not in result
