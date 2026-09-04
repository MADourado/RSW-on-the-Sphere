"""Shared physical constants and nondimensionalization helpers for the RSW
equations on the sphere.

The ``eps``/``gamma`` (Lamb's parameter) computations and the
nondimensional-time <-> days conversions, defined once here and shared by
every module that needs them.

Run as a quick sanity check:

    python -m rsw_sphere.physics
"""
import numpy as np

# Default physical constants (Earth-like), shared by every module in this
# package unless a caller overrides them.
G = 9.8              # gravitational acceleration, m/s^2
A = 6.38e+06          # planetary radius, m
OMEGA = 2 * np.pi / 86400   # planetary rotation rate, rad/s
P_S = 101325.0        # standard sea-level atmospheric pressure, Pa (ICAO standard atmosphere)


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


def air_density_from_equivalent_depth(h_e: float, g: float = G, p_s: float = P_S):
    """Reference air density for the ``eq: enerA`` Joules conversion
    (``JFM-template.tex``, "One way to avoid division by the total
    energy...") -- ``rho`` in ``EK_a = (g h_0^2 a^2 pi) * rho * ||A_a||^2``.

    ``h_e`` (the RSW layer's equivalent depth) represents one vertical
    normal mode of the real, stratified atmosphere, not a literal fluid
    layer (paper Introduction, citing Majda 2003) -- so ``rho`` should not
    be read off as a literal near-surface air density. Instead, following
    the convention used for atmospheric vertical normal-mode
    decompositions -- where a mode's own column mass per unit area is
    expressed through the mean surface pressure ``p_s`` rather than an
    assumed density profile (Kasahara & Puri 1981; Marques, Marta-Almeida
    & Castanheira 2020, *Geosci. Model Dev.* 13, 2763) -- this sets the
    equivalent layer's column mass ``rho * h_e`` equal to the real
    atmosphere's own column mass per unit area, ``p_s / g``:

        rho = p_s / (g * h_e)

    Parameters
    ----------
    h_e : float
        Equivalent depth, m.
    g : float, optional
        Gravitational acceleration, m/s^2. Default ``9.8``.
    p_s : float, optional
        Mean surface pressure, Pa. Default ``P_S`` (standard sea-level
        pressure, 101325 Pa).

    Returns
    -------
    float
        Reference density, kg/m^3.

    Examples
    --------
    >>> round(air_density_from_equivalent_depth(10000), 2)
    1.03
    """
    return p_s / (g * h_e)


def total_energy_joules(e_nondim, h_e: float, a: float = A, g: float = G, p_s: float = P_S):
    """Convert a nondimensional wave-set energy quantity to Joules, via the
    same ``eq: enerA`` prefactor used for a single mode's own ``||A_a||^2``
    (``EK_a = (g h_0^2 a^2 pi) * rho * ||A_a||^2``, ``rho`` from
    ``air_density_from_equivalent_depth``).

    Valid for ``WaveSet.energy()``'s own ``E_total = E2 + E3`` (or any mean
    of it over time), not just a single mode's ``E_a``: ``E2 = sum_a
    ||A_a||^2`` is a plain sum of exactly that same per-mode quantity, and
    ``E3`` (the cubic correction) is built from the same per-mode-normalized
    amplitudes throughout -- so the prefactor that turns one mode's
    ``||A_a||^2`` into Joules turns the whole sum into Joules too, with no
    separate rescaling needed.

    Parameters
    ----------
    e_nondim : float or array_like
        Nondimensional energy (``E_total``, or ``E_total.mean()`` over a
        trajectory).
    h_e : float
        Equivalent depth, m (``h_0`` in ``eq: enerA``).
    a : float, optional
        Planetary radius, m. Default ``A`` (Earth).
    g : float, optional
        Gravitational acceleration, m/s^2. Default ``G``.
    p_s : float, optional
        Mean surface pressure, Pa. Default ``P_S``.

    Returns
    -------
    float or array_like
        Energy in Joules.
    """
    rho = air_density_from_equivalent_depth(h_e, g=g, p_s=p_s)
    return g * h_e ** 2 * a ** 2 * np.pi * rho * e_nondim


def days_from_nondim_time(t):
    """Convert nondimensional triad-dynamics time to days.

    The triad ODEs (``TRIAD.f`` / ``RK44`` in
    ``rsw_sphere.dynamics.dynamic_triads``) are integrated in units of
    ``1/(2*Omega)``; this is the convention already used in
    the amplitude-equation drivers (``t = np.linspace(0, t_f/(4*np.pi), len(T))``).

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
