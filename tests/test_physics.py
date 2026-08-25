"""Round-trip/sanity checks on ``rsw_sphere.physics``'s nondimensionalization
helpers."""
import numpy as np
import pytest

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, linear_period_days


def test_gamma_from_he_roundtrip():
    """gamma = 1/sqrt(eps), eps = 4 a^2 Omega^2 / (g h_e) -- eps*gamma^2 == 1."""
    eps, gamma = gamma_from_he(10000.0)
    assert eps > 0
    assert gamma == pytest.approx(1.0 / np.sqrt(eps))
    assert eps * gamma ** 2 == pytest.approx(1.0)


def test_gamma_from_he_monotonic_in_h_e():
    """Larger equivalent height -> smaller eps -> larger gamma."""
    eps_small, gamma_small = gamma_from_he(1000.0)
    eps_large, gamma_large = gamma_from_he(50000.0)
    assert eps_large < eps_small
    assert gamma_large > gamma_small


def test_days_from_nondim_time():
    """A 'day' is t = 4*pi nondimensional time units (docs/code_guide.md)."""
    assert days_from_nondim_time(4 * np.pi) == pytest.approx(1.0)
    assert days_from_nondim_time(np.array([0.0, 4 * np.pi, 8 * np.pi])) == pytest.approx([0.0, 1.0, 2.0])


def test_linear_period_days():
    """T = 0.5/omega days; halving the frequency doubles the period."""
    assert linear_period_days(0.5) == pytest.approx(1.0)
    assert linear_period_days(0.25) == pytest.approx(2.0 * linear_period_days(0.5))
