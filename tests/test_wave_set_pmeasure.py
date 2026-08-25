"""``p_measure``'s NaN guard: fires exactly on an EXACT-zero denominator
ΔEK, not on a merely-small one (rsw_sphere.plotting.wave_set_pmeasure's
own module docstring, point 2)."""
import math

from rsw_sphere.plotting.wave_set_pmeasure import p_measure

N, DEG = 6, 300  # low N for speed; deg MUST stay 300 -- see tests/test_wave_sets.py

MODES = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]  # sum=0, members=1,2
TRIADS = [(0, 1, 2)]


def test_zero_denominator_gives_nan():
    """Every mode in the (only) reference triad starts at rest -> its own
    dEK_triad is exactly 0 -> P must be NaN, not a division-by-zero error
    or a spurious finite value."""
    velocities = [0.0, 0.0, 0.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert result['dEK_triad'][0] == 0.0
    assert math.isnan(result['P'][0])


def test_small_nonzero_denominator_gives_finite_value():
    """A merely small (not exactly zero) driving velocity gives a small
    but nonzero dEK_triad -> P must be a finite number, not NaN."""
    velocities = [1.0, 1.0, 1.0]
    result = p_measure(MODES, TRIADS, velocities, target_indices=[0],
                        tf_days=2, h=0.02, N=N, deg=DEG)
    assert result['dEK_triad'][0] > 0.0
    assert not math.isnan(result['P'][0])
