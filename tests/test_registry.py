"""registry.sweep_2d: dispatches to pairwise/functional engines and merges results."""
import numpy as np

from rsw_sphere.utilities.registry import sweep_2d, PAIRWISE, FUNCTIONAL, ALL_2D, DIAGNOSTIC_ARRAY_KEYS

N, DEG = 6, 300
QUARTET_MODES = [(4, 5, 3), (3, 4, 3), (1, 2, 3), (1, 1, 1)]
QUARTET_TRIADS = [(0, 1, 2), (0, 1, 3)]


def test_diagnostic_partition_is_disjoint_and_covers_all():
    assert PAIRWISE & FUNCTIONAL == set()
    assert PAIRWISE | FUNCTIONAL == ALL_2D
    assert set(DIAGNOSTIC_ARRAY_KEYS) == ALL_2D


def test_sweep_2d_merges_pairwise_and_functional():
    result = sweep_2d(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("p_measure", "efficiency"),
        u1_range=(0.0, 30.0), u2_range=(0.0, 30.0), n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    assert "P" in result
    assert "Efficiency" in result
    assert "F2" not in result
    assert result["P"].shape == (2, 2, 2)
    assert result["Efficiency"].shape == (2, 2, 2)
    np.testing.assert_array_equal(result["U1"].shape, (2, 2))


def test_sweep_2d_pairwise_only():
    result = sweep_2d(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("filtering_error",),
        u1_range=(0.0, 30.0), u2_range=(0.0, 30.0), n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    assert "F2" in result
    assert "Efficiency" not in result
