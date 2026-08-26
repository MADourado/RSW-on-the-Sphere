"""Shared time integrator for amplitude-equation systems (triads, quartets,
quintets, ...).

``RK44`` was previously copy-pasted, byte-identical apart from the parameter
name, into ``dynamic_triads.py`` and all six legacy four/five-wave scripts.
It only ever calls ``X.f(state)``, so it is arity-agnostic: any object with
an ``f(AMP) -> dAMP/dt`` method works, whether ``AMP`` has 3, 4 or 5
components (or a batch of them, if ``f`` is written to broadcast).

Run as a quick sanity check:

    python -m rsw_sphere.dynamics.integrators
"""
import numpy as np


def RK44(system, t_0, t_f, h, A_0):
    """Fixed-step, classical 4-stage, 4th-order-accurate Runge-Kutta integration.

    Parameters
    ----------
    system : object with an ``f(AMP) -> dAMP/dt`` method
        E.g. a ``TRIAD`` or ``WaveSet`` instance. Not restricted to any
        particular number of components -- ``system.f`` is called with
        whatever shape ``A_0`` has and must return the same shape.
    t_0, t_f : float
        Nondimensional start/end time.
    h : float
        Step size (nondimensional time).
    A_0 : ndarray
        Initial state, any shape ``system.f`` accepts.

    Returns
    -------
    Y : ndarray, shape (n+1,) + A_0.shape
        State at each of the ``n+1`` time points.
    T : ndarray, shape (n+1,)
        Nondimensional time points, ``linspace(t_0, t_f, n+1)``.

    Examples
    --------
    >>> class Decay:
    ...     def f(self, A): return -A
    >>> Y, T = RK44(Decay(), 0, 1, 0.1, np.array([1.0]))
    >>> round(float(Y[-1][0]), 4)
    0.3679
    """
    n = (t_f - t_0) / h
    n = int(n)

    y_0 = A_0

    Y = [y_0]

    for k in range(n):
        k1 = system.f(Y[-1])
        k2 = system.f(Y[-1] + h / 2 * k1)
        k3 = system.f(Y[-1] + h / 2 * k2)
        k4 = system.f(Y[-1] + h * k3)

        Y += [Y[-1] + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)]

    Y = np.array(Y)
    T = np.linspace(t_0, t_f, n + 1)

    return Y, T


if __name__ == "__main__":
    import doctest
    failures, _ = doctest.testmod()
    if failures == 0:
        print("RK44 doctest OK")
