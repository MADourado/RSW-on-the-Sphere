"""RK44 convergence-order test. Despite the name, RK44 is the classical
4-stage Runge-Kutta method (4th-order accurate, not 3rd) -- confirmed
empirically here, not assumed from the name."""
import numpy as np

from rsw_sphere.dynamics.integrators import RK44


class _Decay:
    """dA/dt = -A, exact solution A(t) = A0*exp(-t)."""
    def f(self, A):
        return -A


def test_rk44_is_fourth_order():
    exact = np.exp(-1.0)
    hs = [0.1, 0.05, 0.025]
    errors = [abs(RK44(_Decay(), 0, 1, h, np.array([1.0]))[0][-1][0] - exact) for h in hs]

    orders = [np.log2(errors[i - 1] / errors[i]) for i in range(1, len(errors))]
    assert all(3.8 < o < 4.2 for o in orders), orders
