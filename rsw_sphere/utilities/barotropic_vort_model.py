"""Barotropic non-divergent vorticity equation, four-wave truncation,
transcribed from Raphaldini, Peixoto, Teruya, Raupp & Bustamante (2022),
"Precession resonance of Rossby wave triads and the generation of low
frequency atmospheric oscillations", Physics of Fluids 34, Section III.A
(their eq. 31/35), modes {(n,m)} = {(3,1),(7,3),(5,4),(9,2)}.

Deliberately independent of ``rsw_sphere``'s Hough-harmonic machinery: the
paper's own model is the barotropic vorticity equation (quadratic energy,
exactly conserved for any spectral truncation -- eq. 11/Fjortoft's
theorem), a genuinely different governing equation from this repository's
shallow-water/RSW system (cubic energy, not conserved under truncation,
see ``rsw_sphere.dynamics.wave_sets``'s own module docstring). Used by
``examples/raphaldini2022_compare/`` to compare against the RSW
``WaveSet`` system on the identical four-wave topology.

Every constant below (``OMEGA1-4``, ``C*_**``, ``KAPPA1-4``) is the
paper's own quoted value, used verbatim rather than re-derived from the
coupling-coefficient formula (their eq. 9) -- re-deriving risks a
normalization-convention mismatch that the paper's own text does not give
enough detail to rule out.

The paper's initial condition has an unresolved "/a" (planetary radius?)
factor not recoverable from the published text, so this module sweeps
the overall amplitude *scale* directly (see ``integrate``) rather than a
literal "alpha" matching the paper's own units -- ``scale=1.8e-3``
reproduces their reported peak efficiency (~9-10%, "centered at about
alpha=15") almost exactly.
"""
import numpy as np

# Section III.A's own quoted values, verbatim (not re-derived).
OMEGA1, OMEGA2, OMEGA3, OMEGA4 = -0.083, -0.053, -0.133, -0.022
DELTA1 = OMEGA1 + OMEGA2 - OMEGA3  # Triad {1,2,3}: paper's own -0.003
DELTA2 = OMEGA1 + OMEGA4 - OMEGA2  # Triad {1,2,4}: paper's own -0.051

C1_23, C2_13, C3_12 = -0.773, -2.497, -3.270  # Triad {1,2,3}
C1_24, C4_12, C2_14 = -0.569, -5.527, -6.096  # Triad {1,2,4}

# kappa_j = n_j(n_j+1) for modes (n,m) = (3,1),(7,3),(5,4),(9,2).
KAPPA1, KAPPA2, KAPPA3, KAPPA4 = 3 * 4, 7 * 8, 5 * 6, 9 * 10
KAPPA = np.array([KAPPA1, KAPPA2, KAPPA3, KAPPA4])

#: Default initial-condition direction: three comparable-amplitude modes
#: (the constituent triad {1,2,3}) plus a near-zero fourth mode (4) whose
#: own growth "efficiency" is measured. Scaled by ``scale`` in ``integrate``.
IC_DIRECTION = np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)


def rhs(t, A):
    """Eq. (31) of the paper, verbatim."""
    A1, A2, A3, A4 = A
    dA1 = 1j * C1_23 * np.conj(A2) * A3 * np.exp(1j * DELTA1 * t) \
        + 1j * C1_24 * np.conj(A4) * A2 * np.exp(1j * DELTA2 * t)
    dA2 = 1j * C2_13 * np.conj(A1) * A3 * np.exp(1j * DELTA1 * t) \
        + 1j * C2_14 * A4 * A1 * np.exp(-1j * DELTA2 * t)
    dA3 = 1j * C3_12 * A1 * A2 * np.exp(-1j * DELTA1 * t)
    dA4 = 1j * C4_12 * np.conj(A1) * A2 * np.exp(1j * DELTA2 * t)
    return np.array([dA1, dA2, dA3, dA4])


def _rk4_steps(f, t0, tf, h, A0):
    n = int((tf - t0) / h)
    A, t = A0.copy(), t0
    for _ in range(n):
        k1 = f(t, A)
        k2 = f(t + h / 2, A + h / 2 * k1)
        k3 = f(t + h / 2, A + h / 2 * k2)
        k4 = f(t + h, A + h * k3)
        A = A + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
        yield t, A


def default_step(scale):
    """RK4 step scaled to resolve the nonlinear rate (``~scale^2 *
    max|C|``) at the given amplitude -- larger scale needs a finer step.
    """
    return min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))


def integrate(scale, t_f=1500.0, h=None):
    """Integrate IC ``scale*(1,1,1,1e-3)`` from ``t=0`` to ``t_f``
    (nondimensional time, units of ``1/(2*Omega)``, matching ``omega``'s
    own normalization -- see ``rsw_sphere.physics.days_from_nondim_time``).

    Parameters
    ----------
    scale : float
        Overall amplitude scale (see module docstring for why this is
        not literally the paper's own "alpha").
    t_f : float, optional
        Integration horizon. Default 1500. Must be long enough to sample
        the bounded quasi-periodic orbit's actual maximum -- too short
        underestimates it; too long (several thousand) lets the orbit
        explore its full accessible phase space regardless of resonance
        and washes out the alpha-dependent structure entirely.
    h : float or None, optional
        RK4 step. Default: ``default_step(scale)``.

    Returns
    -------
    Y : ndarray, shape (n_times, 4), complex
        ``A_j(t)``, already in the interaction picture (the paper's own
        ``rhs`` is written that way) -- no extra ``exp(i*omega_j*t)``
        factor needed, unlike a raw RSW ``WaveSet``/``RK44`` trajectory.
    T : ndarray, shape (n_times,)
    """
    if h is None:
        h = default_step(scale)
    A0 = scale * IC_DIRECTION
    As, ts = [A0], [0.0]
    for t, A in _rk4_steps(rhs, 0, t_f, h, A0):
        As.append(A.copy())
        ts.append(t)
    return np.array(As), np.array(ts)


def efficiency_from_trajectory(Y):
    """Peak inter-triad energy-transfer efficiency (paper's eq. 35),
    ``max_t |A4|^2*kappa4 / sum_j(|Aj|^2*kappaj)``, from an
    already-integrated trajectory (``integrate``'s ``Y``).
    """
    E = np.abs(Y) ** 2 * KAPPA
    return float((E[:, 3] / E.sum(axis=1)).max())


def efficiency(scale, t_f=1500.0, h=None):
    """``efficiency_from_trajectory(integrate(scale, t_f, h)[0])``."""
    Y, _ = integrate(scale, t_f, h)
    return efficiency_from_trajectory(Y)


def dynamical_phases(Y, T):
    """Dynamical phase :math:`\\Phi(t)` (eq. \\ref{eq: Phi} in the paper
    this repository backs) of both constituent triads, from an
    already-in-interaction-picture trajectory (``integrate``'s ``Y``, ``T``).

    Triad-role convention (sum mode first in each mismatch definition,
    ``DELTA1 = omega1+omega2-omega3``, ``DELTA2 = omega1+omega4-omega2``):
    triad 1 sum=mode 3, members=(mode 1, mode 2); triad 2 sum=mode 2,
    members=(mode 1, mode 4). Cross-checked against ``dA3/dt`` and the
    ``A2`` term ``+i*C2_14*A4*A1*exp(-i*DELTA2*t)`` in ``rhs`` (the
    ``exp(-i*delta*t)`` sign is the "sum" role).

    Unlike a raw RSW ``WaveSet``/``RK44`` trajectory, no
    ``exp(i*omega_j*t)`` correction is needed first: ``rhs`` is already
    written in the interaction picture.

    Returns
    -------
    Phi1, Phi2 : ndarray, shape (n_times,)
        Unwrapped dynamical phase of triads {1,2,3} and {1,2,4).
    """
    A1, A2, A3, A4 = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3]
    raw1 = np.angle(A3) - np.angle(A1) - np.angle(A2)
    Phi1 = np.unwrap(raw1) + DELTA1 * T
    raw2 = np.angle(A2) - np.angle(A1) - np.angle(A4)
    Phi2 = np.unwrap(raw2) + DELTA2 * T
    return Phi1, Phi2


def individual_phase_tilde(Y, j, omega_j, T):
    """Raphaldini et al. (2022)'s own "corrected" individual phase
    ``phi_j~(t) = arg(A_j(t)) - omega_j*t`` (their Section III.A / Fig. 3),
    for mode ``j`` (0-indexed) of an already-in-interaction-picture
    trajectory. The correction undoes ``rhs``'s own interaction-picture
    convention, recovering the lab-frame phase whose slope reversal is
    the paper's reported finding -- see
    ``rsw_sphere.dynamics.dynamical_phase.individual_phase``'s own
    docstring for the opposite (raw ``WaveSet``/``RK44``) convention,
    where no correction is needed.
    """
    return np.unwrap(np.angle(Y[:, j])) - omega_j * T
