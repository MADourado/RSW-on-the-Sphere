"""precession_frequency_efficiency lock-island regression: locks
Quartet A's own published boundary -- u~83-92/u~140 lock both constituent
triads, u=70 doesn't."""
import pytest

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.utilities.precession import precession_frequency_efficiency

pytestmark = pytest.mark.slow

_LOCK_TOL = 1e-3


def _mutual_lock(spec, u):
    result = precession_frequency_efficiency(spec, "d", [u], tf_days=150.0, h=0.01)
    freqs = {lbl: float(v[0]) for lbl, v in result["freq_by_triad"].items()}
    return all(abs(f) < _LOCK_TOL for f in freqs.values())


def test_quartet_a_locks_at_u88():
    spec = load_wave_set_specs()["quartet_rh_preference"]
    assert _mutual_lock(spec, 88.0)


def test_quartet_a_does_not_lock_at_u70():
    spec = load_wave_set_specs()["quartet_rh_preference"]
    assert not _mutual_lock(spec, 70.0)
