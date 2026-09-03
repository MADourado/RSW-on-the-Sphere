"""Table ``precession_comparison`` and Figure ``borrowed_topology_precession``
(JFM-template.tex, ``sec: quartet_rh_precession``): efficiency and
dynamical-phase precession frequency, Raphaldini et al. (2022)'s own
barotropic vorticity equation vs. this paper's RSW system, on the
identical four-wave topology (their (n,m) modes {(3,1),(7,3),(5,4),(9,2)},
here "Quartet B" -- registered as ``quartet_rh_borrowed_topology`` in
``wave_sets_default.yaml``: a=RH(4,5), b=RH(1,3), c=RH(3,7), d=RH(2,9),
triad 1 sum=a members=(b,c), triad 2 sum=c members=(b,d)).

One driver combining the barotropic model
(``rsw_sphere.utilities.barotropic_vort_model``) + the RSW ``WaveSet``
(built from the registry entry above) + the dynamical-phase diagnostic
(``rsw_sphere.dynamics.dynamical_phase``) + the sweep/plot for the
figure. The compute/plot split is preserved (``sweep`` -> ``.npz`` cache
-> ``plot_sweep``), matching every other expensive figure in this
repository.

Both quartets' IC is ``scale*(1,1,1,1e-3)`` in each system's own mode
order (three comparable-amplitude modes forming one constituent triad,
plus a near-zero fourth mode -- RH(2,9) on the RSW side -- whose own
growth "efficiency" is measured). RSW's efficiency is the
time-averaged-total-energy fraction (``rsw_sphere.utilities.precession``'s
own convention, robust to RSW's non-conserved energy); the barotropic
model's own energy is exactly conserved, so its simpler instantaneous-max
fraction (eq. 35) is adequate as-is -- both formulas agree to within
0.1pp at the scales used here (checked directly).

Run:

    python examples/raphaldini2022_compare/precession_comparison.py table
    python examples/raphaldini2022_compare/precession_comparison.py figure \\
        outputs/figures/wave_sets/quartet_rh_borrowed_topology/precession_cache.npz \\
        outputs/figures/wave_sets/quartet_rh_borrowed_topology/borrowed_topology_precession.png
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, G
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.plotting.style import apply_house_style, add_outward_twin_axis
from rsw_sphere.utilities.periods import low_frequency_power
import rsw_sphere.utilities.barotropic_vort_model as baro

WAVE_SET_KEY = "quartet_rh_borrowed_topology"
TARGET_MODE_KEY = "d"  # RH(2,9): near-zero IC, growth measures efficiency.
IC_DIRECTION = np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)  # in spec.mode_keys order (a,b,c,d)

# Each system's own control/peak amplitude scale (Table precession_comparison).
BARO_CONTROL_SCALE, BARO_PEAK_SCALE = 2e-4, 1.8e-3
RSW_CONTROL_SCALE, RSW_PEAK_SCALE = 3e-3, 3e-2

# Sweep ranges for Figure borrowed_topology_precession -- capped
# independently per system (checked directly, not assumed): barotropic to
# 0.05 (oscillation amplitude still drifting beyond this, even where the
# frequency slope alone looks stable), RSW to 0.25 (its own frequency
# flips sign entirely by scale=1.09 between a 1500- and 3000-day window).
SCALES_BARO = np.geomspace(2e-4, 0.05, 20)
SCALES_RSW = np.geomspace(3e-3, 0.25, 20)


def build_rsw_waveset(N=12, deg=400):
    """Build Quartet B's RSW ``WaveSet`` from its registry entry.

    Returns
    -------
    ws : WaveSet
    spec : WaveSetSpec
    """
    specs = load_wave_set_specs(DEFAULT_WAVESETS_PATH)
    spec = specs[WAVE_SET_KEY]
    gamma = gamma_from_he(spec.h_e, g=G)[1]
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws = WaveSet(gamma, list(spec.modes), triad_indices, N=N, deg=deg)
    return ws, spec


def _rsw_step(scale):
    return min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))


def rsw_trajectory(ws, scale, t_f=1500.0, h=None):
    """Integrate Quartet B's RSW IC ``scale*(1,1,1,1e-3)``, cached via
    ``run_and_cache``.
    """
    if h is None:
        h = _rsw_step(scale)
    A0 = scale * IC_DIRECTION
    label = f"scale{scale:.6g}_tf{t_f:.0f}_h{h:.5f}"
    Y, T, _ = run_and_cache(ws, A0, t_f, h, label=label)
    return Y, T


def rsw_phases_and_efficiency(ws, spec, scale, t_f=1500.0, h=None,
                               low_freq_period_cutoff_days=None):
    """Both constituent triads' dynamical phase, the target mode's
    (RH(2,9)) time-averaged-total-energy efficiency and energy drift, and
    (optionally) its integrated low-frequency spectral power -- all from
    one cached RSW trajectory.

    Returns
    -------
    Phi1, Phi2, T, efficiency, energy_drift, low_freq_power
        ``low_freq_power`` is ``None`` unless
        ``low_freq_period_cutoff_days`` is given.
    """
    Y, T = rsw_trajectory(ws, scale, t_f, h)
    i_sum1, i_p1, i_q1 = spec.triad_indices(0)
    i_sum2, i_p2, i_q2 = spec.triad_indices(1)
    Phi1 = dynamical_phase(Y, T, ws.omega, i_sum1, i_p1, i_q1, ws.delta[0])
    Phi2 = dynamical_phase(Y, T, ws.omega, i_sum2, i_p2, i_q2, ws.delta[1])

    target_idx = spec.index(TARGET_MODE_KEY)
    E = np.real(Y * np.conj(Y))
    E_total = np.sum(E, axis=1)
    energy_drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    A_sq = E[:, target_idx]
    efficiency = (A_sq.max() - A_sq.min()) / E_total.mean()

    low_freq_power = None
    if low_freq_period_cutoff_days is not None:
        Q_target = A_sq / E_total
        T_days = days_from_nondim_time(T)
        low_freq_power = low_frequency_power(T_days, Q_target,
                                              period_cutoff_days=low_freq_period_cutoff_days)

    return Phi1, Phi2, T, float(efficiency), float(energy_drift), low_freq_power


def barotropic_phases(scale, t_f=1500.0, h=None):
    """Both constituent triads' dynamical phase, barotropic side."""
    Y, T = baro.integrate(scale, t_f, h)
    Phi1, Phi2 = baro.dynamical_phases(Y, T)
    return Phi1, Phi2, T


def _report(label, Phi1, Phi2, T):
    days = days_from_nondim_time(T)
    s1 = libration_diagnostics(Phi1, days)
    s2 = libration_diagnostics(Phi2, days)
    print(f"\n{label}")
    for name, s in (("Phi1 (triad1, sum=RH(4,5)/mode3)", s1),
                    ("Phi2 (triad2, sum=RH(3,7)/mode2)", s2)):
        print(f"  {name}: precession_freq={s['precession_freq']:+.5f} rad/day  "
              f"net_windings={s['net_windings']:+.3f}  "
              f"osc_amp_windings={s['oscillation_amplitude_windings']:.3f}")


def print_table():
    """Print Table ``precession_comparison``'s own numbers (barotropic
    vs. RSW: delta1, efficiency peak, precession freq control/peak) --
    the same quantities hand-typed into JFM-template.tex's ``tab:
    precession_comparison``.
    """
    ws, spec = build_rsw_waveset()

    baro_eff_peak = baro.efficiency(BARO_PEAK_SCALE)
    baro_Phi1_c, baro_Phi2_c, baro_T_c = barotropic_phases(BARO_CONTROL_SCALE)
    baro_Phi1_p, baro_Phi2_p, baro_T_p = barotropic_phases(BARO_PEAK_SCALE)
    baro_freq_control = libration_diagnostics(baro_Phi1_c, days_from_nondim_time(baro_T_c))['precession_freq']
    baro_freq_peak = libration_diagnostics(baro_Phi1_p, days_from_nondim_time(baro_T_p))['precession_freq']

    rsw_Phi1_c, rsw_Phi2_c, rsw_T_c, _, _, _ = rsw_phases_and_efficiency(ws, spec, RSW_CONTROL_SCALE)
    rsw_Phi1_p, rsw_Phi2_p, rsw_T_p, rsw_eff_peak, _, _ = rsw_phases_and_efficiency(ws, spec, RSW_PEAK_SCALE)
    rsw_freq_control = libration_diagnostics(rsw_Phi1_c, days_from_nondim_time(rsw_T_c))['precession_freq']
    rsw_freq_peak = libration_diagnostics(rsw_Phi1_p, days_from_nondim_time(rsw_T_p))['precession_freq']

    print(f"{'':30}{'Barotropic':>14}{'RSW':>14}   (paper's Table precession_comparison)")
    print(f"{'delta1':30}{baro.DELTA1:>14.4f}{ws.delta[0]:>14.4f}   (-0.003 / +0.015)")
    print(f"{'Efficiency peak (%)':30}{100 * baro_eff_peak:>14.1f}{100 * rsw_eff_peak:>14.1f}   (10.0 / 19.9)")
    print(f"{'Precession freq., control':30}{baro_freq_control:>14.4f}{rsw_freq_control:>14.4f}   (-0.041 / +0.192)")
    print(f"{'Precession freq., peak':30}{baro_freq_peak:>14.4f}{rsw_freq_peak:>14.4f}   (+0.0001 / +0.363)")

    print("\nFull phase-libration diagnostics:")
    _report(f"Barotropic, control (scale={BARO_CONTROL_SCALE:g})", baro_Phi1_c, baro_Phi2_c, baro_T_c)
    _report(f"Barotropic, peak (scale={BARO_PEAK_SCALE:g})", baro_Phi1_p, baro_Phi2_p, baro_T_p)
    _report(f"RSW, control (scale={RSW_CONTROL_SCALE:g})", rsw_Phi1_c, rsw_Phi2_c, rsw_T_c)
    _report(f"RSW, peak (scale={RSW_PEAK_SCALE:g})", rsw_Phi1_p, rsw_Phi2_p, rsw_T_p)


def sweep(cache_path, scales_baro=SCALES_BARO, scales_rsw=SCALES_RSW,
          t_f_baro=1500.0, t_f_rsw=1500.0, low_freq_period_cutoff_days=None):
    """Pure compute: run both curves (precession frequency vs. amplitude
    scale, plus RSW's own target-mode efficiency/energy drift) and write
    them to ``cache_path`` (.npz). Re-running with an already-existing
    ``cache_path`` is a no-op -- delete the file first to force a
    recompute.

    Parameters
    ----------
    low_freq_period_cutoff_days : float or None, optional
        If given, also compute the RSW target mode's integrated
        low-frequency spectral power (Raphaldini et al. 2022's eq. 37
        diagnostic) from the same RSW trajectory. No barotropic-side
        equivalent is computed (matching ``plot_sweep``'s own
        two-axis convention). ``None`` (default): not computed.

    Returns
    -------
    scales_baro, scales_rsw, f_baro, f_rsw, efficiency_rsw, energy_drift_rsw, low_freq_power_rsw
    """
    if os.path.exists(cache_path):
        d = np.load(cache_path)
        return (d['scales_baro'], d['scales_rsw'], d['f_baro'], d['f_rsw'],
                d['efficiency_rsw'], d['energy_drift_rsw'],
                d['low_freq_power_rsw'] if 'low_freq_power_rsw' in d else None)

    f_baro = []
    for s in scales_baro:
        Phi1, _, T = barotropic_phases(s, t_f=t_f_baro)
        days = days_from_nondim_time(T)
        f_baro.append(libration_diagnostics(Phi1, days)['precession_freq'])

    ws, spec = build_rsw_waveset()
    f_rsw, efficiency_rsw, energy_drift_rsw = [], [], []
    low_freq_power_rsw = [] if low_freq_period_cutoff_days is not None else None
    for s in scales_rsw:
        Phi1, _, T, eff, drift, lf = rsw_phases_and_efficiency(
            ws, spec, s, t_f=t_f_rsw, low_freq_period_cutoff_days=low_freq_period_cutoff_days)
        days = days_from_nondim_time(T)
        f_rsw.append(libration_diagnostics(Phi1, days)['precession_freq'])
        efficiency_rsw.append(eff)
        energy_drift_rsw.append(drift)
        if low_freq_power_rsw is not None:
            low_freq_power_rsw.append(lf)

    f_baro, f_rsw = np.array(f_baro), np.array(f_rsw)
    efficiency_rsw, energy_drift_rsw = np.array(efficiency_rsw), np.array(energy_drift_rsw)
    save_kwargs = dict(scales_baro=scales_baro, scales_rsw=scales_rsw, f_baro=f_baro, f_rsw=f_rsw,
                        efficiency_rsw=efficiency_rsw, energy_drift_rsw=energy_drift_rsw)
    if low_freq_power_rsw is not None:
        low_freq_power_rsw = np.array(low_freq_power_rsw)
        save_kwargs['low_freq_power_rsw'] = low_freq_power_rsw
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    np.savez(cache_path, **save_kwargs)
    return (scales_baro, scales_rsw, f_baro, f_rsw, efficiency_rsw, energy_drift_rsw,
            low_freq_power_rsw)


def plot_sweep(cache_path, path=None, plot_scale_min=2e-3):
    """Pure plotting: read ``cache_path`` (must already exist -- see
    ``sweep``) and draw Figure ``borrowed_topology_precession``.
    Efficiency is plotted on a twin y-axis, RSW only -- the barotropic
    system's energy is exactly conserved, so the standard efficiency
    (eq. 35) already applies to it without RSW's time-averaging
    correction.

    ``plot_scale_min`` crops both curves' own low-amplitude tail below
    this scale from the plot only -- ``sweep``'s own cache keeps the
    full range regardless.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"{cache_path} does not exist -- run sweep({cache_path!r}) first.")
    d = np.load(cache_path)
    scales_baro, scales_rsw, f_baro, f_rsw = d['scales_baro'], d['scales_rsw'], d['f_baro'], d['f_rsw']
    efficiency_rsw = d['efficiency_rsw']
    low_freq_power_rsw = d['low_freq_power_rsw'] if 'low_freq_power_rsw' in d else None

    if plot_scale_min is not None:
        mask_baro = scales_baro >= plot_scale_min
        scales_baro, f_baro = scales_baro[mask_baro], f_baro[mask_baro]
        mask_rsw = scales_rsw >= plot_scale_min
        scales_rsw, f_rsw = scales_rsw[mask_rsw], f_rsw[mask_rsw]
        efficiency_rsw = efficiency_rsw[mask_rsw]
        if low_freq_power_rsw is not None:
            low_freq_power_rsw = low_freq_power_rsw[mask_rsw]

    apply_house_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(scales_baro, np.abs(f_baro), 'o:', ms=3, color='C0',
            label='Barotropic')
    ax.plot(scales_rsw, np.abs(f_rsw), 's:', ms=3, color='C1',
            label='RSW')
    ax.axhline(0.01, color='grey', ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('Amplitude scale')
    ax.set_ylabel(r'$|$precession frequency$|$ (rad/day)')
    ax.set_title('Quartet B: Barotropic vs. RSW')

    ax2 = ax.twinx()
    ax2.plot(scales_rsw, 100 * efficiency_rsw, '^-', ms=3, color='C3',
             label='RSW Effic. RH(2,9)')
    ax2.set_ylabel('RSW Effic. RH(2,9) (%)', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    all_lines, all_labels = lines1 + lines2, labels1 + labels2

    if low_freq_power_rsw is not None:
        ax3 = add_outward_twin_axis(ax, scales_rsw, low_freq_power_rsw, marker_style='v-',
                                     color='C2', ylabel='RSW low-freq. power (target mode)',
                                     label='RSW low-freq. power')
        lines3, labels3 = ax3.get_legend_handles_labels()
        all_lines += lines3
        all_labels += labels3

    ax.legend(all_lines, all_labels, loc='upper left')

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("table", help="print Table precession_comparison's own numbers")

    p_fig = sub.add_parser("figure", help="compute/plot Figure borrowed_topology_precession")
    p_fig.add_argument("cache", help="path to the .npz cache (computed if missing, else reused)")
    p_fig.add_argument("path", nargs="?", default=None, help="output PNG path")
    p_fig.add_argument("--plot-only", action="store_true",
                        help="skip sweep() entirely and error if the cache is missing.")

    args = parser.parse_args()
    if args.cmd == "table":
        print_table()
    elif args.cmd == "figure":
        if not args.plot_only:
            sweep(args.cache)
        plot_sweep(args.cache, args.path)
        print(f"Saved to {args.path}" if args.path else "Shown interactively")
