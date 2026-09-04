"""Does the precession-resonance regime generate low-frequency
oscillations, the way Raphaldini et al. (2022) report (their Figs.
2(c)/5(c)/8(c)/11(c): integrated low-frequency spectral power in a target
mode's kinetic energy tracks the efficiency peak)? A distinct claim from
their Fig. 3 individual-mode phase reversal already checked in
``individual_mode_reversal_investigation.py`` -- this script checks the
OTHER finding. Backs the live claim at JFM-template.tex's ``sec:
quartet_rh_precession`` ("efficiency peaks in both quartets also
coincide with elevated low-frequency spectral power ...").

Three checks, reusing already-cached trajectories where available:

1. Barotropic calibration -- direct reproduction, same scale range as
   ``rsw_sphere.utilities.barotropic_vort_model``'s own module docstring.
2. Quartet B, RSW -- reuses the trajectories cached under
   ``outputs/trajectories/quartets/`` by
   ``individual_mode_reversal_investigation.step2_quartet_b_rsw`` and
   ``precession_comparison.rsw_phases_and_efficiency`` (same explicit
   scale-based label, both share this cache entry).
3. Quartet A, RSW -- reuses the trajectories cached under
   ``outputs/trajectories/quartets/`` by
   ``rsw_sphere.utilities.precession`` (the confirmed lock at
   u~83-92) -- automatically, since ``run_and_cache``'s own label is now
   built from every mode's own initial condition
   (``rsw_sphere.dynamics.trajectory_cache.ic_label``), not a
   caller-supplied tag, so any two call sites integrating the same
   modes at the same velocities land in the same cache entry.

Run:

    python examples/raphaldini2022_compare/low_frequency_precession_check.py
"""
import _bootstrap  # noqa: F401 -- repo root on sys.path

import numpy as np

from rsw_sphere.physics import days_from_nondim_time
from rsw_sphere.utilities.periods import low_frequency_power
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
import rsw_sphere.utilities.barotropic_vort_model as baro

import precession_comparison as comp


def step1_barotropic(scales=None):
    """Target mode 4 (RH(2,9) analogue, index 3) -- same range as
    ``rsw_sphere.utilities.barotropic_vort_model``'s own established range.

    Uses the NORMALIZED kinetic-energy fraction ``Q4(t) =
    |A4|^2*kappa4 / sum_j(|Aj|^2*kappaj)`` (Raphaldini et al.'s own eq. 35,
    also their own "% energy" convention for any mode-energy-vs-time plot,
    e.g. Fig. 9's caption) as input to the FFT, NOT the raw ``|A4(t)|^2``
    -- the raw series inflates trivially with ``scale^2`` (total energy is
    exactly conserved in this truncation and scales as ``scale^2`` with
    the IC), which would swamp any genuine low-frequency signal and is
    not what their own bounded, non-monotonic Fig. 2(c) shows.
    """
    if scales is None:
        scales = (5e-4, 1e-3, 1.5e-3, 1.8e-3, 2.2e-3, 3e-3, 5e-3, 1e-2)

    print("=== Step 1: barotropic calibration (target mode 4, normalized energy fraction) ===")
    print(f"{'scale':>10} {'efficiency(%)':>14} {'low_freq_power':>16}")
    for s in scales:
        Y, T = baro.integrate(s, t_f=1500.0)
        E = np.abs(Y) ** 2 * baro.KAPPA
        Q4 = E[:, 3] / E.sum(axis=1)
        # T is nondimensional time normalized by 2*Omega, same convention as
        # the rest of this repo -- days_from_nondim_time applies here too.
        days = days_from_nondim_time(T)
        p = low_frequency_power(days, Q4, period_cutoff_days=10.0)
        eff = baro.efficiency_from_trajectory(Y)
        print(f"{s:>10.2e} {100 * eff:>14.4f} {p:>16.6e}")


def step2_quartet_b_rsw(scales=None, t_f=1500.0):
    """Target mode RH4 (RH(2,9), index 3) -- reuses the trajectories
    already cached under outputs/trajectories/quartets/ by
    individual_mode_reversal_investigation.step2_quartet_b_rsw (same
    scales/t_f/h -- a cache hit, no new integration).

    Uses the normalized instantaneous energy fraction ``E4(t)/E_total(t)``
    as FFT input, not raw ``|A4(t)|^2`` -- see step1_barotropic's own
    docstring for why (raw energy inflates trivially with ``scale^2``).
    RSW's E_total is not exactly conserved (wave_sets.py's own module
    docstring), so this uses the actual instantaneous total, not a
    constant -- consistent with every other efficiency/fraction
    computation elsewhere in this repository.
    """
    if scales is None:
        scales = np.array([1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0])

    ws, spec = comp.build_rsw_waveset()
    target_idx = spec.index(comp.TARGET_MODE_KEY)
    print("\n=== Step 2: Quartet B, RSW (target mode RH4=RH(2,9), normalized energy fraction) ===")
    print(f"{'scale':>10} {'efficiency(%)':>14} {'low_freq_power':>16}")
    for s in scales:
        Y, T = comp.rsw_trajectory(ws, s, t_f=t_f)
        days = days_from_nondim_time(T)
        E = np.real(Y * np.conj(Y))
        E_total_t = np.sum(E, axis=1)
        Q4 = E[:, target_idx] / E_total_t
        p = low_frequency_power(days, Q4, period_cutoff_days=10.0)

        eff = (E[:, target_idx].max() - E[:, target_idx].min()) / E_total_t.mean()
        print(f"{s:>10.2e} {100 * eff:>14.4f} {p:>16.6e}")


def step3_quartet_a_rsw(u_values=None, tf_days=150.0):
    """Target mode c (RH(3,4)) -- reuses the trajectories already cached
    under outputs/trajectories/quartets/ by the run_sweep.py
    verification run (same tf_days/h -- a cache hit for u values already
    swept, e.g. the 45-point examples/sweep_quartet_a_rh36.yaml grid)."""
    if u_values is None:
        u_values = np.linspace(10.0, 150.0, 45)  # matches sweep_quartet_a_rh36.yaml

    specs = load_wave_set_specs(DEFAULT_WAVESETS_PATH)
    spec = specs["quartet_rh_preference"]
    from rsw_sphere.physics import gamma_from_he, G
    from rsw_sphere.dynamics.wave_sets import WaveSet

    gamma = gamma_from_he(spec.h_e, g=G)[1]
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    ws = WaveSet(gamma, list(spec.modes), triad_indices, N=10, deg=300)
    velocities = list(spec.velocities)
    sweep_idx = spec.index('d')
    target_idx = spec.index('c')
    t_f = tf_days * 4 * np.pi
    h = 0.01

    print("\n=== Step 3: Quartet A, RSW (target mode c=RH(3,4), normalized energy fraction) ===")
    print(f"{'u':>7} {'efficiency(%)':>14} {'low_freq_power':>16}  note")
    for u in u_values:
        v = list(velocities)
        v[sweep_idx] = u
        A0 = ws.amplitudes_from_velocities(v, spec.h_e, g=G)
        Y, T, _ = run_and_cache(ws, A0, t_f, h, velocities=v)
        days = days_from_nondim_time(T)
        E_c = np.real(Y[:, target_idx] * np.conj(Y[:, target_idx]))
        E_total = np.real(sum(ws.energy(Y)))
        Q_c = E_c / E_total
        p = low_frequency_power(days, Q_c, period_cutoff_days=10.0)

        eff = (E_c.max() - E_c.min()) / E_total.mean()
        note = " <-- LOCK REGION" if 83.0 <= u <= 92.0 else ""
        print(f"{u:>7.2f} {100 * eff:>14.4f} {p:>16.6e}{note}")


if __name__ == "__main__":
    step1_barotropic()
    step2_quartet_b_rsw()
    step3_quartet_a_rsw()
