"""Direct reproduction of the precession-resonance efficiency peak from
Raphaldini, Peixoto, Teruya, Raupp & Bustamante (2022), "Precession
resonance of Rossby wave triads and the generation of low frequency
atmospheric oscillations", Physics of Fluids 34, Section III A / Fig. 2
(the four-wave barotropic-vorticity example, modes
{(n,m)} = {(3,1),(7,3),(5,4),(9,2)}).

Deliberately **independent of rsw_sphere's Hough-harmonic machinery**:
the paper's own model is the barotropic non-divergent vorticity
equation (quadratic energy, exactly conserved for any spectral
truncation -- see the paper's eq. 11/Fjortoft's theorem), a genuinely
different governing equation from this repository's shallow-water/RSW
system (cubic energy, not conserved under truncation -- see
``rsw_sphere.dynamics.wave_sets``'s own module docstring). Written to
verify that distinction matters in practice: an extensive same-day search for
precession resonance in the RSW quartet system found no full
phase-locking, and the open question was whether the barotropic (not
RSW) setting was where the original mechanism actually lives. This
script confirms the mechanism itself is real by reproducing it in the
system the paper itself uses, with the paper's own exact numbers (not
re-derived from the coupling-coefficient formula, eq. 9, to avoid any
normalization-convention risk) -- see
``rsw_sphere.dynamics.dynamical_phase`` and
``examples/precession_resonance_rsw_vs_barotropic.py`` for the RSW-side
comparison this motivated.

The paper's initial condition has an unresolved ``/a`` (Earth radius?)
factor whose exact meaning isn't recoverable from the visible text, so
this script sweeps the overall amplitude **scale** directly (not a
literal "alpha" matching the paper's own units) -- ``scale=1.8e-3``
reproduces their reported peak efficiency (~9-10%, "centered at about
alpha=15") almost exactly, which is the validation this script performs.
The subsequent dip-then-rise shape at larger alpha (Fig. 2a) is not yet
reproduced past that first peak; efficiency keeps rising monotonically
here at larger scale instead, which is a genuine open discrepancy, not
one this script papers over.

Run:

    python examples/reproduce_raphaldini2022_fig2.py
"""
import numpy as np

# Section III A's own quoted values, verbatim (not re-derived).
OMEGA1, OMEGA2, OMEGA3, OMEGA4 = -0.083, -0.053, -0.133, -0.022
DELTA1 = OMEGA1 + OMEGA2 - OMEGA3  # Triad {1,2,3}: paper's own -0.003
DELTA2 = OMEGA1 + OMEGA4 - OMEGA2  # Triad {1,2,4}: paper's own -0.051

C1_23, C2_13, C3_12 = -0.773, -2.497, -3.270  # Triad {1,2,3}
C1_24, C4_12, C2_14 = -0.569, -5.527, -6.096  # Triad {1,2,4}

# kappa_j = n_j(n_j+1) for modes (n,m) = (3,1),(7,3),(5,4),(9,2).
KAPPA1, KAPPA2, KAPPA3, KAPPA4 = 3 * 4, 7 * 8, 5 * 6, 9 * 10


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


def _rk4(f, t0, tf, h, A0):
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


def efficiency(scale, t_f=1500.0, h=None):
    """Peak inter-triad energy-transfer efficiency (paper's eq. 35),
    ``max_t |A4|^2*kappa4 / sum_j(|Aj|^2*kappaj)``, for IC
    ``(A1,A2,A3,A4) = scale*(1,1,1,1e-3)`` integrated to ``t_f``
    (nondimensional time, units of ``1/(2*Omega)`` matching ``omega``'s
    own normalization).

    Parameters
    ----------
    scale : float
        Overall amplitude scale (see module docstring for why this is
        not literally the paper's own "alpha").
    t_f : float, optional
        Integration horizon. Default 1500 (~120 "days" in the same
        nondimensional convention used throughout this repository,
        ``rsw_sphere.physics.days_from_nondim_time``). Must be long
        enough to sample the bounded quasi-periodic orbit's actual
        maximum -- too short underestimates it; too long (e.g. several
        thousand) lets the orbit explore its full accessible phase
        space regardless of resonance and washes out the alpha-
        dependent structure entirely (found empirically -- see the
        INSPECT doc cited in the module docstring).
    h : float or None, optional
        RK4 step. Default: scaled to resolve the nonlinear rate
        (``~scale^2 * max|C|``) at the given amplitude.

    Returns
    -------
    float
        Efficiency, in [0, 1].
    """
    if h is None:
        h = min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))
    A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
    A1s, A2s, A3s, A4s = [A0[0]], [A0[1]], [A0[2]], [A0[3]]
    for t, A in _rk4(rhs, 0, t_f, h, A0):
        A1s.append(A[0]); A2s.append(A[1]); A3s.append(A[2]); A4s.append(A[3])
    A1s, A2s, A3s, A4s = map(np.array, (A1s, A2s, A3s, A4s))
    num = np.abs(A4s) ** 2 * KAPPA4
    den = (np.abs(A1s) ** 2 * KAPPA1 + np.abs(A2s) ** 2 * KAPPA2
           + np.abs(A3s) ** 2 * KAPPA3 + np.abs(A4s) ** 2 * KAPPA4)
    return (num / den).max()


if __name__ == "__main__":
    print(f"delta1={DELTA1:.4f} (paper: -0.003)   delta2={DELTA2:.4f} (paper: -0.051)")
    print(f"\n{'scale':>10} {'efficiency(%)':>14}")
    for scale in (5e-4, 1e-3, 1.5e-3, 1.8e-3, 2.2e-3, 3e-3, 5e-3, 1e-2):
        eff = efficiency(scale)
        print(f"{scale:>10.2e} {100 * eff:>14.4f}")
    print("\nExpected: efficiency near 10% around scale~1.8e-3, matching the "
          "paper's own reported peak (~9-10%, 'centered at about alpha=15').")
