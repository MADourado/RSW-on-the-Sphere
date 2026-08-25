"""Apply the dynamical-phase libration/rotation diagnostic
(``rsw_sphere.dynamics.dynamical_phase``) to the two precession-resonance
efficiency sweeps built earlier (``examples/reproduce_raphaldini2022_fig2.py``
for the barotropic reproduction, ``examples/precession_resonance_rsw_vs_
barotropic.py`` for the RSW comparison), at their own respective peak/dip
scales. Only the energy-efficiency metric had been checked so far;
this tests whether the efficiency peak actually corresponds to genuine
phase-locking (bounded Phi, near-zero precession frequency), mirroring
the paper's own Fig. 3, rather than just an amplitude bump.

Two systems, two different amplitude conventions:

- Barotropic (``reproduce_raphaldini2022_fig2``): ``rhs`` is already
  written in the interaction picture (eq. 31 of the paper) -- its own
  ``A_j(t)`` trajectory IS ``a_j(t)``, no extra ``exp(i*omega_j*t)``
  factor needed. ``Phi = arg(A_sum) - arg(A_p) - arg(A_q) + delta*t``
  applied directly to the raw trajectory.
- RSW (``precession_resonance_rsw_vs_barotropic`` / ``WaveSet``/``RK33``):
  the raw trajectory is NOT in the interaction picture (``RK33`` solves
  ``dA/dt = -i*omega*A + ...``), so ``dynamical_phase`` performs that
  transform itself -- see that module's docstring.

Triad-role convention for both scripts (sum mode first in each pair, per
the mismatch definitions ``DELTA1 = omega1+omega2-omega3``,
``DELTA2 = omega1+omega4-omega2``): triad1 sum=mode3, members=(mode1,
mode2); triad2 sum=mode2, members=(mode1,mode4). Cross-checked against
``dA3/dt`` and the ``A2`` term ``+i*C2_14*A4*A1*exp(-i*DELTA2*t)`` (the
``exp(-i*delta*t)`` sign is the "sum" role in ``WaveSet.f``'s own
convention).

Run:

    python examples/precession_resonance_phase_diagnostic.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.physics import days_from_nondim_time
from rsw_sphere.plotting.wave_set_periods import low_frequency_power

import reproduce_raphaldini2022_fig2 as baro
import precession_resonance_rsw_vs_barotropic as rsw_comp


def barotropic_phases(scale, t_f=1500.0, h=None):
    if h is None:
        h = min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))
    A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
    A1s, A2s, A3s, A4s, ts = [A0[0]], [A0[1]], [A0[2]], [A0[3]], [0.0]
    for t, A in baro._rk4(baro.rhs, 0, t_f, h, A0):
        A1s.append(A[0]); A2s.append(A[1]); A3s.append(A[2]); A4s.append(A[3])
        ts.append(t)
    A1s, A2s, A3s, A4s = map(np.array, (A1s, A2s, A3s, A4s))
    T = np.array(ts)

    raw1 = np.angle(A3s) - np.angle(A1s) - np.angle(A2s)
    Phi1 = np.unwrap(raw1) + baro.DELTA1 * T
    raw2 = np.angle(A2s) - np.angle(A1s) - np.angle(A4s)
    Phi2 = np.unwrap(raw2) + baro.DELTA2 * T
    return Phi1, Phi2, T


def rsw_phases(ws, scale, t_f=1500.0, h=None):
    if h is None:
        h = min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))
    A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
    Y, T = RK33(ws, 0, t_f, h, A0)
    i_sum1, i_p1, i_q1 = rsw_comp.TRIADS[0]
    i_sum2, i_p2, i_q2 = rsw_comp.TRIADS[1]
    Phi1 = dynamical_phase(Y, T, ws.omega, i_sum1, i_p1, i_q1, ws.delta[0])
    Phi2 = dynamical_phase(Y, T, ws.omega, i_sum2, i_p2, i_q2, ws.delta[1])
    return Phi1, Phi2, T


def rsw_phases_and_efficiency(ws, scale, t_f=1500.0, h=None, low_freq_period_cutoff_days=None):
    """Same as ``rsw_phases``, plus the target mode's (RH(2,9), index 3
    -- see ``precession_resonance_rsw_vs_barotropic.efficiency``'s own
    docstring) energy-transfer efficiency and the run's own energy
    drift, computed from the SAME trajectory (no extra integration): the
    time-averaged-total-energy analogue of ``rsw_comp.efficiency``'s
    instantaneous-max-fraction (see ``quartet_precession_sweep.
    precession_sweep``'s own ``efficiency``/``energy_drift``, same
    convention, applied here to the standalone (non-registry) Quartet B
    RSW build). The trajectory itself is cached via
    ``rsw_sphere.dynamics.trajectory_cache.run_and_cache`` under
    ``outputs/trajectories/quartets/`` -- re-running at the same
    ``(scale, t_f, h)`` is a cache hit.

    Parameters
    ----------
    low_freq_period_cutoff_days : float or None, optional
        If given, also compute the target mode's own integrated
        low-frequency spectral power
        (``rsw_sphere.plotting.wave_set_periods.low_frequency_power``,
        Raphaldini et al. 2022's eq. 37 diagnostic -- distinct from their
        Fig. 3 individual-mode-reversal claim), on the SAME normalized
        energy-fraction series used for ``efficiency`` (raw ``|A|^2``
        inflates trivially with ``scale^2`` and is not used -- see
        ``examples/low_frequency_precession_check.py``'s own docstring).
        ``None`` (default): not computed.

    Returns
    -------
    Phi1, Phi2, T, efficiency, energy_drift, low_freq_power
        ``low_freq_power`` is ``None`` unless
        ``low_freq_period_cutoff_days`` is given.
    """
    if h is None:
        h = min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))
    A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
    # No `velocities` here: A0 is driven directly by `scale`, not built via
    # amplitudes_from_velocities -- an explicit label (same formula as
    # every other scale-based call site in this repo) keeps this and
    # individual_mode_reversal_investigation.py/low_frequency_precession_check.py's
    # step2 sharing one cache namespace, matching a set-driven run's own
    # natural identifier (scale/tf/h) rather than a per-mode IC label.
    label = f"scale{scale:.6g}_tf{t_f:.0f}_h{h:.5f}"
    Y, T, _ = run_and_cache(ws, A0, t_f, h, label=label)
    i_sum1, i_p1, i_q1 = rsw_comp.TRIADS[0]
    i_sum2, i_p2, i_q2 = rsw_comp.TRIADS[1]
    Phi1 = dynamical_phase(Y, T, ws.omega, i_sum1, i_p1, i_q1, ws.delta[0])
    Phi2 = dynamical_phase(Y, T, ws.omega, i_sum2, i_p2, i_q2, ws.delta[1])

    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)
    energy_drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    A_sq = np.real(Y[:, 3] * np.conj(Y[:, 3]))
    efficiency = (A_sq.max() - A_sq.min()) / E_total.mean()

    low_freq_power = None
    if low_freq_period_cutoff_days is not None:
        Q_target = A_sq / E_total
        T_days = days_from_nondim_time(T)
        low_freq_power = low_frequency_power(T_days, Q_target,
                                              period_cutoff_days=low_freq_period_cutoff_days)

    return Phi1, Phi2, T, float(efficiency), float(energy_drift), low_freq_power


def report(label, Phi1, Phi2, T):
    days = days_from_nondim_time(T)
    s1 = libration_diagnostics(Phi1, days)
    s2 = libration_diagnostics(Phi2, days)
    print(f"\n{label}")
    for name, s in (("Phi1 (triad1, sum=mode3)", s1), ("Phi2 (triad2, sum=mode2)", s2)):
        print(f"  {name}: precession_freq={s['precession_freq']:+.5f} rad/day  "
              f"net_windings={s['net_windings']:+.3f}  "
              f"osc_amp_windings={s['oscillation_amplitude_windings']:.3f}")


if __name__ == "__main__":
    print("=== Barotropic reproduction (Raphaldini et al. 2022, Fig. 2/3) ===")
    print("Peak scale (efficiency ~10%, matching paper's reported peak):")
    Phi1, Phi2, T = barotropic_phases(scale=1.8e-3)
    report("barotropic, scale=1.8e-3 (efficiency peak)", Phi1, Phi2, T)

    print("\nOff-peak control (scale=2e-4, well below the peak):")
    Phi1c, Phi2c, Tc = barotropic_phases(scale=2e-4)
    report("barotropic, scale=2e-4 (control)", Phi1c, Phi2c, Tc)

    print("\n=== RSW comparison (same topology, h_e=10000) ===")
    ws = rsw_comp.build()

    print("Peak scale (efficiency ~19.9%, from the earlier sweep):")
    Phi1, Phi2, T = rsw_phases(ws, scale=3e-2)
    report("RSW, scale=3e-2 (efficiency peak)", Phi1, Phi2, T)

    print("\nDip scale (efficiency ~17.2%):")
    Phi1, Phi2, T = rsw_phases(ws, scale=0.1)
    report("RSW, scale=0.1 (efficiency dip)", Phi1, Phi2, T)

    print("\nOff-peak control (scale=3e-3, ~10x below the peak -- matched "
          "log-ratio to the barotropic control above, NOT the ultra-linear "
          "1e-4 regime where any weakly-driven mode trivially phase-tracks "
          "its forcing regardless of resonance):")
    Phi1c, Phi2c, Tc = rsw_phases(ws, scale=3e-3)
    report("RSW, scale=3e-3 (control)", Phi1c, Phi2c, Tc)

    print("\nInterpretation: near-zero precession_freq + small oscillation "
          "amplitude at the peak/dip vs. large secular drift in the control "
          "would indicate genuine phase-locking (matching the paper's own "
          "Fig. 3 at its peak); a similar (nonzero) precession_freq at both "
          "peak and control would mean the efficiency bump is not "
          "accompanied by real phase-locking.")
