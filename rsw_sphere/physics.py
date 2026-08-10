"""Shared physical constants and nondimensionalization helpers for the RSW
equations on the sphere.

These are the ``eps``/``gamma`` (Lamb's parameter) computations that were
previously copy-pasted across ``dynamic_three_waves.py``, ``hough_spatial_ev.py``,
``hough_and_derivatives.py``, ``dispersion_relation_fancy.py`` and
``normalization.py``, plus the nondimensional-time <-> days conversions used
throughout the triad-dynamics code.

Run as a quick sanity check:

    python -m rsw_sphere.physics
"""
import numpy as np

# Default physical constants (Earth-like), shared by every module in this
# package unless a caller overrides them.
G = 9.8              # gravitational acceleration, m/s^2
A = 6.38e+06          # planetary radius, m
OMEGA = 2 * np.pi / 86400   # planetary rotation rate, rad/s


def gamma_from_he(h_e: float, g: float = G, a: float = A, Omega: float = OMEGA):
    """Lamb's parameter ``gamma = 1/sqrt(eps)`` for a given equivalent height.

    Parameters
    ----------
    h_e : float
        Equivalent height (equivalent depth), in metres.
    g : float, optional
        Gravitational acceleration, m/s^2. Default ``9.8``.
    a : float, optional
        Planetary radius, m. Default ``6.38e6`` (Earth).
    Omega : float, optional
        Planetary rotation rate, rad/s. Default ``2*pi/86400`` (Earth).

    Returns
    -------
    eps : float
        Lamb's number ``4 a^2 Omega^2 / (g h_e)``.
    gamma : float
        ``1/sqrt(eps)``.

    Examples
    --------
    >>> eps, gamma = gamma_from_he(10000)
    >>> round(eps, 6), round(gamma, 8)
    (8.78633, 0.33736206)
    """
    eps = (4 * a * a * Omega * Omega) / (g * h_e)
    gamma = 1 / np.sqrt(eps)
    return eps, gamma


def days_from_nondim_time(t):
    """Convert nondimensional triad-dynamics time to days.

    The triad ODEs (``TRIAD.f`` / ``RK33`` in
    ``rsw_sphere.dynamics.dynamic_triads``) are integrated in units of
    ``1/(2*Omega)``; this is the convention already used in
    ``Triad_dynamics`` (``t = np.linspace(0, t_f/(4*np.pi), len(T))``).

    Parameters
    ----------
    t : array_like or float
        Nondimensional time.

    Returns
    -------
    array_like or float
        Time in days.
    """
    return t / (4 * np.pi)


def linear_period_days(omega):
    """Linear wave period, in days, from a nondimensional frequency.

    Derivation: ``omega_dim = omega * 2*Omega``, ``2*Omega = 4*pi/86400``
    rad/s, so ``T = 2*pi/omega_dim = 43200/omega`` s ``= 0.5/omega`` days.

    Parameters
    ----------
    omega : array_like or float
        Nondimensional frequency (as returned by ``norm_Hough``/``TRIAD``).

    Returns
    -------
    array_like or float
        Period in days.

    Examples
    --------
    >>> round(linear_period_days(-0.099437), 2)   # RH(1,2)
    5.03
    >>> round(linear_period_days(0.369690), 2)    # EG(1,1) Kelvin
    1.35
    """
    return 0.5 / np.abs(omega)


if __name__ == "__main__":
    eps, gamma = gamma_from_he(10000)
    print(f"h_e=10000 -> eps={eps:.6f}, gamma={gamma:.8f}")
    print(f"RH(1,2)  T = {linear_period_days(-0.099437):.2f} d (expect 5.03)")
    print(f"EG(1,1)  T = {linear_period_days(0.369690):.2f} d (expect 1.35)")
