"""functional_diagnostics_sweep: efficiency + low_frequency_energy over a
2D velocity grid, full-wave-set trajectory only."""
import math

import numpy as np
import pytest

from rsw_sphere.utilities.functional import functional_diagnostics_sweep
from rsw_sphere.utilities.efficiency import efficiency_variation

pytestmark = pytest.mark.slow

N, DEG = 6, 300

# a=RH(4,5), b=RH(3,4), c=RH(1,2), d=EG(1,1); triads {a,b,c} and {a,b,d}.
QUARTET_MODES = [(4, 5, 3), (3, 4, 3), (1, 2, 3), (1, 1, 1)]
QUARTET_TRIADS = [(0, 1, 2), (0, 1, 3)]


def test_efficiency_finite_and_bounded_when_drift_small():
    result = functional_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("efficiency",), u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    eff = result['Efficiency']
    assert np.isfinite(eff).all()
    assert (eff >= 0).all()


def test_efficiency_gated_by_drift_max():
    result = functional_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("efficiency",), drift_max=0.0, u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    assert np.isnan(result['Efficiency']).all()


def test_subset_only_computes_requested_arrays():
    result = functional_diagnostics_sweep(
        QUARTET_MODES, QUARTET_TRIADS, 10000.0, (2, 3), {0: 30.0, 1: 30.0}, [2, 3],
        diagnostics=("efficiency",), u1_range=(0.0, 30.0), u2_range=(0.0, 30.0),
        n_grid=2, tf_days=1, h=0.02, N=N, deg=DEG)
    assert "Efficiency" in result
    assert "LowFreqEnergy" not in result


def test_efficiency_variation_matches_p_measure_formula():
    assert math.isclose(efficiency_variation(0.05, 0.04), 100 * (0.05 - 0.04) / 0.04)


def test_efficiency_variation_nan_when_either_input_nan():
    assert math.isnan(efficiency_variation(float("nan"), 0.04))
    assert math.isnan(efficiency_variation(0.05, float("nan")))
    assert math.isnan(efficiency_variation(0.05, 0.0))
