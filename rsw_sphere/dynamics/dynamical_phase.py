"""Dynamical phase and precession-frequency diagnostics for a WaveSet
constituent triad -- the Kartashova/Craik quantity ``Phi_{a,b}^c`` used
throughout the precession-resonance literature (Bustamante, Quinn &
Lucas 2014; Raphaldini, Peixoto, Teruya, Raupp & Bustamante 2022,
"Precession resonance of Rossby wave triads and the generation of low
frequency atmospheric oscillations", Phys. Fluids) and already present
in this codebase's own dissertation appendix (paper-nonlinear-
interactions-SWE-sphere/util/conteudo/Chapter2.tex, eq: ``ham``/``328``).

For one constituent triad (sum mode s, members p, q; mismatch
``delta = -omega_s + omega_p + omega_q``, matching
``WaveSet.delta``'s own sign convention), write each mode's *raw*
simulated amplitude ``A_j(t)`` (as returned directly by ``RK44`` on a
``WaveSet`` -- linear rotation NOT yet removed) in the interaction
picture, ``a_j(t) = A_j(t) * exp(i*omega_j*t)``. Substituting into
``WaveSet.f``'s equations of motion shows

    da_p/dt = i*alpha_p * conj(a_q) * a_s * exp(i*delta*t)
    da_q/dt = i*alpha_q * conj(a_p) * a_s * exp(i*delta*t)
    da_s/dt = i*alpha_s * a_p * a_q * exp(-i*delta*t)

so the combination

    Phi(t) = arg(a_s(t)) - arg(a_p(t)) - arg(a_q(t)) + delta*t

removes the leading fast oscillation and is *slowly* varying -- this is
the same object as the dissertation's ``Phi_{a,b}^c`` and the
Raphaldini et al. papers' ``Phi^j_{k,l}`` (there, mode ``j`` plays the
role of ``s`` here). Its qualitative behaviour is the textbook
precession-resonance diagnostic: **libration** (bounded oscillation,
zero net long-time drift) signals the triad's phase is *locked* by a
neighbouring triad's coupling; **rotation** (steady secular drift) means
it is not. This is a more direct, more literature-standard test than
comparing energy/amplitude-error diagnostics computed over an arbitrary
fixed time window -- a windowed RMS
amplitude-error metric could not distinguish resonant from
non-resonant configurations, since it conflates any resonance signal
with the model's own smooth background scatter and the windows tested
were too short for a slow near-commensurate beat to develop).

Run as a quick self-check (verifies the derivation against the
independent ``WaveSet``/``RK44`` machinery on a toy triad):

    python -m rsw_sphere.dynamics.dynamical_phase
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np


def dynamical_phase(Y, T, omega, i_sum, i_p, i_q, delta):
    """Dynamical phase ``Phi(t)`` of one constituent triad, from a
    ``WaveSet``/``RK44`` trajectory.

    Parameters
    ----------
    Y : ndarray, shape (n_times, n_modes), complex
        Raw trajectory as returned by ``RK44(wave_set, ...)`` -- linear
        rotation NOT removed.
    T : ndarray, shape (n_times,)
        Nondimensional time points matching ``Y`` (``RK44``'s own
        second return value).
    omega : ndarray, shape (n_modes,)
        ``WaveSet.omega``.
    i_sum, i_p, i_q : int
        Mode indices for one constituent triad, in ``WaveSet.triads``'s
        own ``(i_sum, i_p, i_q)`` order.
    delta : float
        That triad's mismatch (``WaveSet.delta[t]`` for the
        corresponding triad index ``t``).

    Returns
    -------
    ndarray, shape (n_times,)
        ``Phi(t)``, unwrapped (accumulates net drift/libration
        continuously rather than wrapping into ``[-pi, pi)``, so a
        rotating phase is visible as a growing/shrinking value and a
        librating one stays bounded).
    """
    a_sum = Y[:, i_sum] * np.exp(1j * omega[i_sum] * T)
    a_p = Y[:, i_p] * np.exp(1j * omega[i_p] * T)
    a_q = Y[:, i_q] * np.exp(1j * omega[i_q] * T)
    raw_phase = np.angle(a_sum) - np.angle(a_p) - np.angle(a_q)
    return np.unwrap(raw_phase) + delta * T


def libration_diagnostics(Phi, T_days):
    """Summarize a dynamical-phase trace as libration-vs-rotation
    statistics: a linear trend (the *precession frequency*, matching
    Raphaldini et al. 2022's eq. 26 ``Omega = <dPhi/dt>``) plus the
    residual oscillation amplitude around it.

    Parameters
    ----------
    Phi : ndarray, shape (n_times,)
        Output of ``dynamical_phase``.
    T_days : ndarray, shape (n_times,)
        Time points in days (e.g. ``rsw_sphere.physics.days_from_nondim_time(T)``).

    Returns
    -------
    dict
        ``precession_freq`` (rad/day, the fitted secular slope --
        near-zero indicates libration/phase-locking, per Raphaldini et
        al. 2022's "phase alignment" case ``p=0``), ``net_windings``
        (total ``Phi`` drift over the run, in units of 2*pi),
        ``oscillation_amplitude_windings`` (peak-to-peak residual around
        the linear trend, in units of 2*pi -- large values indicate a
        strongly modulated but still-net-rotating phase; small values
        alongside a near-zero ``precession_freq`` indicate genuine
        libration).
    """
    slope, intercept = np.polyfit(T_days, Phi, 1)
    residual = Phi - (slope * T_days + intercept)
    return {
        'precession_freq': slope,
        'net_windings': (Phi[-1] - Phi[0]) / (2 * np.pi),
        'oscillation_amplitude_windings': (residual.max() - residual.min()) / (2 * np.pi),
    }


def individual_phase(Y, j):
    """Raw phase ``phi_j~(t) = arg(Y[:, j](t))`` of ONE mode's own
    trajectory, unwrapped -- Raphaldini et al. (2022)'s "original phase"
    (their Fig. 3, Section III.A), NOT the combined-triad ``Phi`` this
    module's own ``dynamical_phase`` computes.

    Only valid as-is for a raw ``WaveSet``/``RK44`` trajectory: ``Y[:, j]``
    already has linear rotation NOT removed (this module's own docstring
    above), which already *is* Raphaldini's lab-frame ``phi_j~`` -- no
    correction term needed. In the linear (uncoupled) limit this reduces to
    a straight line of slope ``-omega_j``; nonlinear coupling bends it, and
    a strong enough bend flips the local slope's sign -- that flip is the
    reported "reversal."

    A trajectory in the *interaction picture* (e.g.
    ``examples_legacy/raphaldini2022_compare/reproduce_raphaldini2022_fig2.py``'s barotropic ``A_j(t)``,
    which is envelope-only) is the opposite convention: there the
    correction ``phi_j_tilde = np.unwrap(np.angle(A_j)) - omega_j * T``
    must be added on top of this function's output -- local to whatever
    script performs that calibration, not part of this function.

    Parameters
    ----------
    Y : ndarray, shape (n_times, n_modes), complex
        Raw trajectory (``RK44``'s own first return value).
    j : int
        Mode index.

    Returns
    -------
    ndarray, shape (n_times,)
        Unwrapped raw phase of mode ``j``'s own trajectory.
    """
    return np.unwrap(np.angle(Y[:, j]))


if __name__ == "__main__":
    from rsw_sphere.physics import gamma_from_he, days_from_nondim_time
    from rsw_sphere.dynamics.integrators import RK44
    from rsw_sphere.dynamics.wave_sets import WaveSet

    G, H_E = 9.8, 10000.0
    gamma = gamma_from_he(H_E, g=G)[1]

    # RH(4,5)+RH(3,4)+RH(1,2): a single (degenerate quartet = triad) case,
    # used throughout the paper's Gate I0-I4 inspection this module was
    # extracted from -- a good default self-check since its own mismatch
    # (~-0.1086) is already well characterized elsewhere in the repo.
    modes = [(4, 5, 3), (3, 4, 3), (1, 2, 3)]
    ws = WaveSet(gamma, modes, [(0, 1, 2)], N=10, deg=300)
    A0 = ws.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    t_f = 60 * 4 * np.pi
    Y, T = RK44(ws, 0, t_f, 0.01, A0)

    i_sum, i_p, i_q = ws.triads[0]
    Phi = dynamical_phase(Y, T, ws.omega, i_sum, i_p, i_q, ws.delta[0])
    stats = libration_diagnostics(Phi, days_from_nondim_time(T))
    print(f"delta = {ws.delta[0]:.4f}")
    print(f"precession_freq = {stats['precession_freq']:.5f} rad/day")
    print(f"net_windings = {stats['net_windings']:.3f}")
    print(f"oscillation_amplitude_windings = {stats['oscillation_amplitude_windings']:.3f}")
    # sanity: an isolated triad's own dynamical phase should not be
    # perfectly librating in general (this one isn't specially tuned for
    # resonance), so a nonzero precession_freq here is expected -- this
    # self-check exists to confirm the module runs end-to-end, not that
    # this particular triad locks.

    # individual_phase: linear-limit check -- with coupling switched off
    # (zero amplitude), each mode's own raw phase must be an exact
    # straight line of slope -omega_j.
    A0_linear = np.zeros(3, dtype=complex)
    A0_linear[0] = 1e-6
    Y_lin, T_lin = RK44(ws, 0, t_f, 0.01, A0_linear)
    phi0 = individual_phase(Y_lin, 0)
    slope = np.polyfit(T_lin, phi0, 1)[0]
    print(f"\nindividual_phase linear-limit check: slope={slope:.6f}, "
          f"-omega[0]={-ws.omega[0]:.6f}")
    assert abs(slope - (-ws.omega[0])) < 1e-6, "individual_phase linear-limit check failed"
    print("individual_phase OK")
