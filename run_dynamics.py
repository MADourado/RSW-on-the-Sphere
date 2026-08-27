"""Integrate a wave set's full dynamics, plus every constituent triad
separately

 Each integration is cached via trajectory_cache.run_and_cache and
 each integration is plotted separately (unless --no-plot) 
 under a path mirroring its own trajectory's cache path.

Run:

    python run_dynamics.py --wave-set quartet_rossby_kelvin

or import and call it from another script (e.g. run_sweep.py):

    from run_dynamics import run_dynamics
    from rsw_sphere.dynamics.run_config import RunConfig
    run_dynamics(RunConfig.from_wave_set(spec))
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time, G
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.trajectory_cache import run_and_cache, _mode_slug
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.energy_evolution import plot_energy_evolution
from rsw_sphere.utilities.periods import dominant_periods
from rsw_sphere.utilities.tables import dynamics_summary_rows, write_csv


def _build_units(spec):
    """(name, modes, triads, velocities, title, triad_labels) per topology
    unit: the full wave set, plus each constituent triad if
    spec.has_subtriads().

    A sub-triad's own name is built from its two MEMBER modes' own
    filesystem-safe slugs (e.g. ``triad_rh34_rh45``) -- not a generic
    ``triad0``/``triad1`` index -- so it stays meaningful on its own (as
    a dict key, a figure filename, a table row) without needing the
    registry's own display_label for context. The shared sum mode is
    left out of the SLUG since it's often common to every constituent
    triad in a quartet/quintet and wouldn't help distinguish them --
    but every human-facing LABEL below still names it explicitly (all
    three modes, not just the two members): a triad is only fully
    identified once its sum mode is named too.

    triad_labels : one display label per entry in this unit's own
    ``triads`` (the full wave set carries every constituent triad's own
    label; a sub-triad unit carries just its own, matching its title) --
    used to key each unit's own dynamical-phase precession frequency."""
    def _labeled(t, i, sum_mode):
        return f"{t.display_label or f'Triad {i + 1}'} → {_mode_label(*sum_mode)}"

    full_triad_labels = [_labeled(t, i, spec.modes[spec.triad_indices(i)[0]])
                          for i, t in enumerate(spec.triads)]
    units = [("full", spec.modes, [spec.triad_indices(i) for i in range(spec.n_triads())],
              spec.velocities, spec.display_label or spec.key, full_triad_labels)]
    if spec.has_subtriads():
        for i, t in enumerate(spec.triads):
            member_p, member_q, sum_mode = spec.sub_triad_modes(i)
            name = f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}"
            label = _labeled(t, i, sum_mode)
            units.append((name, spec.sub_triad_modes(i), [(2, 0, 1)],
                          spec.sub_triad_velocities(i), label, [label]))
    return units


def _integrate_and_plot_unit(args):
    """Worker (module-level so it's picklable for ProcessPoolExecutor)."""
    (name, modes, triads, velocities, title, triad_labels,
     h_e, tf_days, h, output_root, plot, wave_set_key) = args

    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, list(modes), triads, N=10, deg=300)
    A0 = ws.amplitudes_from_velocities(list(velocities), h_e, g=G)
    t_f = tf_days * 4 * np.pi

    traj_root = os.path.join(output_root, "trajectories")
    Y, T, traj_path = run_and_cache(ws, A0, t_f, h, velocities=list(velocities), output_root=traj_root)

    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)
    E = np.real(Y * np.conj(Y))
    t_days = days_from_nondim_time(T)
    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    dEK = E.max(axis=0) - E.min(axis=0)
    labels = [_mode_label(*m) for m in modes]

    precession_freq = {}
    for t_idx, (i_sum, i_p, i_q) in enumerate(triads):
        Phi = dynamical_phase(Y, T, ws.omega, i_sum, i_p, i_q, ws.delta[t_idx])
        precession_freq[triad_labels[t_idx]] = float(
            libration_diagnostics(Phi, t_days)['precession_freq'])

    fig_path = None
    if plot:
        import matplotlib.pyplot as plt
        from rsw_sphere.plotting.style import apply_house_style
        # "full"'s own filename lists every one of its modes (no shared
        # sum to omit the way a sub-triad's own name does -- there's
        # only one "full" unit, so no ambiguity to resolve by omitting
        # anything); a sub-triad's own name (already "triad_<m1>_<m2>")
        # is used as-is.
        fig_name = f"full_{'_'.join(_mode_slug(*m) for m in modes)}.png" if name == "full" else f"{name}.png"
        fig_path = os.path.join(output_root, "figures", "dynamics", wave_set_key, fig_name)
        os.makedirs(os.path.dirname(fig_path), exist_ok=True)

        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        plot_energy_evolution(t_days, E, E_total, labels, modes, ax=ax)
        ax.set_title(title)
        fig.savefig(fig_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

    return {
        'name': name, 'title': title, 't': t_days, 'E': E, 'E_total': E_total,
        'labels': labels, 'drift': float(drift), 'dEK': dEK,
        'omega': ws.omega, 'precession_freq': precession_freq,
        'trajectory_path': traj_path, 'figure_path': fig_path,
    }


def run_dynamics(config: RunConfig, write_table: bool = True) -> dict:
    """Integrate + plot every topology unit of config.wave_set_spec.

    write_table : write a CSV summary to <output_root>/tables/<spec.key>.csv.
        False when called once per grid point (e.g. run_sweep.py's own
        per-point pass).

    Returns dict: unit name ('full', 'triad_<member1>_<member2>', ...) -> result
    dict (t, E, E_total, labels, drift, dEK, omega, precession_freq,
    trajectory_path, figure_path, title).
    """
    spec = config.wave_set_spec
    units = _build_units(spec)
    args = [(name, modes, triads, velocities, title, triad_labels, spec.h_e, config.tf_days, config.h,
             config.output_root, config.plot, spec.key)
            for name, modes, triads, velocities, title, triad_labels in units]

    if config.parallel and len(units) > 1:
        max_workers = config.max_workers or max(1, (os.cpu_count() or 2) // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_integrate_and_plot_unit, args))
    else:
        results = [_integrate_and_plot_unit(a) for a in args]

    out = {r['name']: r for r in results}
    if write_table:
        table_path = os.path.join(config.output_root, "tables", f"{spec.key}.csv")
        write_csv(dynamics_summary_rows(out, spec), table_path)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-set", required=True,
                         help="registry role key (rsw_sphere.dynamics.wave_set_specs)")
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--tf-days", type=float, default=None,
                         help="override the wave set's own registered tf_days")
    parser.add_argument("--h", type=float, default=None,
                         help="override the wave set's own registered step size")
    parser.add_argument("--no-plot", action="store_true", help="compute/cache only, skip figures")
    parser.add_argument("--no-parallel", action="store_true", help="force serial execution")
    parser.add_argument("--diagnostics", action="store_true",
                         help="also compute/print efficiency (per mode/unit), dynamical phase "
                              "(precession frequency per triad, full vs. alone, with its own "
                              "percent variation), and every pairwise diagnostic (p_measure, "
                              "efficiency_var, spectral_deviation, novelty_period) for every target "
                              "mode against every sub-triad that contains it, plus a 'final' version "
                              "of each across every containing sub-triad at once, and write the "
                              "novelty-frequency spectrum figures (rsw_sphere.plotting.novelty_frequency_panel)")
    parser.add_argument("--novelty-exclusion-frac", type=float, default=0.20,
                         help="novelty_period: +/- period window excluded around each "
                              "sub-triad's own dominant peak")
    parser.add_argument("--novelty-min-prominence", type=float, default=0.02,
                         help="novelty_period: minimum find_peaks prominence to report a novel peak")
    parser.add_argument("--efficiency-drift-max", type=float, default=0.1,
                         help="efficiency/efficiency_var: leave a unit's own efficiency undefined "
                              "if its total-energy drift exceeds this fraction of its own mean "
                              "(rsw_sphere.utilities.efficiency.wave_set_efficiency)")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    if args.wave_set not in specs:
        parser.error(f"--wave-set {args.wave_set!r} not found in {args.specs!r} "
                     f"(available: {list(specs)})")
    config = RunConfig.from_wave_set(specs[args.wave_set], tf_days=args.tf_days, h=args.h,
                                      output_root=args.output_root,
                                      plot=not args.no_plot, parallel=not args.no_parallel)

    results = run_dynamics(config)
    for name, r in results.items():
        print(f"[{name}] {r['title']}: Energy drift={r['drift']:.3e}, "
              f"dEK={dict(zip(r['labels'], r['dEK']))}")
        # Linear period = 1/(2*|omega|) days -- omega is rad per nondim time
        # unit, and 1 day = 4*pi nondim time units throughout this codebase
        # (rsw_sphere.physics.days_from_nondim_time), so period_nondim =
        # 2*pi/|omega| converts to period_days = period_nondim/(4*pi).
        linear_periods = {lbl: round(1.0 / (2 * abs(w)), 3) for lbl, w in zip(r['labels'], r['omega'])}
        print(f"  linear periods (days, own mode frequency): {linear_periods}")
        periods = {lbl: round(float(dominant_periods(r['t'], r['E'][:, j])['period_global']), 3)
                   for j, lbl in enumerate(r['labels'])}
        print(f"  periods (days, dominant FFT peak): {periods}")
        prec = {lbl: round(v, 5) for lbl, v in r['precession_freq'].items()}
        print(f"  precession_freq (rad/day): {prec}")
        print(f"  trajectory -> {r['trajectory_path']}")
        if r['figure_path']:
            print(f"  figure -> {r['figure_path']}")

    if args.diagnostics:
        import numpy as np
        from rsw_sphere.utilities.pmeasure import pairwise_target_diagnostics, p_measure_combined_for_all_targets
        from rsw_sphere.utilities.novelty_frequency import novelty_combined_for_all_targets
        from rsw_sphere.utilities.efficiency import wave_set_efficiency, efficiency_variation
        from rsw_sphere.plotting.novelty_frequency_panel import novelty_frequency_figures

        spec = specs[args.wave_set]
        full = results["full"]

        def _efficiency(r, j_local):
            return wave_set_efficiency(r["E"][:, j_local], r["E_total"], r["drift"],
                                        drift_max=args.efficiency_drift_max)

        # Efficiency (rsw_sphere.utilities.efficiency.wave_set_efficiency): one
        # value per (mode, integrated unit) -- unlike p_measure/spectral_deviation
        # this isn't a full-vs-sub comparison, just a property of ONE
        # already-integrated unit (mean-total-energy-normalized, NaN if that
        # unit's own energy drift exceeds --efficiency-drift-max), so it gets
        # its own table rather than a column in the pairwise one below.
        print("\n=== efficiency (per mode, per integrated unit) ===")
        eff_rows = []
        eff_cache = {}
        for name, r in results.items():
            for j, label in enumerate(r["labels"]):
                eff = _efficiency(r, j)
                eff_cache[(name, label)] = eff
                eff_rows.append([label, name, f"{eff:.4f}" if np.isfinite(eff) else "n/a (drift)"])
        eff_headers = ["mode", "unit", "efficiency"]
        eff_widths = [max(len(h), *(len(r[i]) for r in eff_rows)) if eff_rows else len(h)
                      for i, h in enumerate(eff_headers)]
        eff_fmt = "  ".join(f"{{:<{w}}}" for w in eff_widths)
        print(eff_fmt.format(*eff_headers))
        print(eff_fmt.format(*["-" * w for w in eff_widths]))
        for r in eff_rows:
            print(eff_fmt.format(*r))

        # Dynamical phase (rsw_sphere.dynamics.dynamical_phase): a property
        # of one constituent triad, not a mode -- no shared-triad ambiguity
        # to resolve (a triad is never shared the way a mode can be), so a
        # single full-vs-alone comparison is already well-defined, mirroring
        # the same "vs." structure as the pairwise table below.
        print("\n=== dynamical phase (precession frequency, rad/day) ===")
        phase_rows = []
        for triad_label, prec_full in full["precession_freq"].items():
            prec_alone = None
            for name, r in results.items():
                if name != "full" and triad_label in r["precession_freq"]:
                    prec_alone = r["precession_freq"][triad_label]
                    break
            if prec_alone is not None and abs(prec_alone) > 1e-12:
                var_str = f"{100 * (abs(prec_full) - abs(prec_alone)) / abs(prec_alone):.2f}%"
            else:
                var_str = "n/a"
            alone_str = f"{prec_alone:.5f}" if prec_alone is not None else "n/a"
            phase_rows.append([triad_label, f"{prec_full:.5f}", alone_str, var_str])
        phase_headers = ["triad", "precession_freq_full", "precession_freq_alone", "phase_variation"]
        phase_widths = [max(len(h), *(len(r[i]) for r in phase_rows)) if phase_rows else len(h)
                        for i, h in enumerate(phase_headers)]
        phase_fmt = "  ".join(f"{{:<{w}}}" for w in phase_widths)
        print(phase_fmt.format(*phase_headers))
        print(phase_fmt.format(*["-" * w for w in phase_widths]))
        for r in phase_rows:
            print(phase_fmt.format(*r))

        print("\n=== pairwise diagnostics (full wave set vs. each containing sub-triad) ===")
        rows = []
        for j, label in enumerate(full["labels"]):
            amp_full = np.sqrt(full["E"][:, j])
            eff_full = eff_cache[("full", label)]
            for name, r in results.items():
                if name == "full" or label not in r["labels"]:
                    continue
                j_sub = r["labels"].index(label)
                amp_sub = np.sqrt(r["E"][:, j_sub])
                d = pairwise_target_diagnostics(
                    full["t"], amp_full, amp_sub, full["E_total"], r["E_total"],
                    novelty_exclusion_frac=args.novelty_exclusion_frac,
                    novelty_min_prominence=args.novelty_min_prominence)
                eff_var = efficiency_variation(eff_full, eff_cache[(name, label)])
                novelty_str = (f"{d['novelty_period']:.4f}d ({d['novelty_relevance']:.2f}%)"
                               if np.isfinite(d['novelty_period']) else "none detected")
                rows.append([
                    label, name, f"{d['p_measure']:.2f}%",
                    f"{eff_var:.2f}%" if np.isfinite(eff_var) else "n/a",
                    f"{d['spectral_deviation']:.2f}%" if np.isfinite(d['spectral_deviation']) else "n/a",
                    novelty_str,
                ])

        headers = ["mode", "vs.", "p_measure", "efficiency_var", "spectral_dev", "novelty_period (%)"]
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                  for i, h in enumerate(headers)]
        row_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(row_fmt.format(*headers))
        print(row_fmt.format(*["-" * w for w in widths]))
        for r in rows:
            print(row_fmt.format(*r))

        # "Final" diagnostics: one row per target mode, considering every
        # containing sub-triad at once instead of one pairwise comparison
        # at a time -- avoids the small-denominator inflation a mode
        # shared across triads can otherwise show against whichever
        # sub-triad happens to leave it weakly excited (see
        # final_p_measure's own docstring). efficiency_var_final reuses
        # the SAME winning sub-triad p_measure_final already picked
        # (rather than an independent selection), so the two stay
        # directly comparable -- they should read similarly whenever the
        # full wave set's own total energy stays close to constant, and
        # diverge only when it doesn't (see efficiency_variation's own
        # docstring).
        print("\n=== final diagnostics (per target, across all containing sub-triads) ===")
        pfinal = p_measure_combined_for_all_targets(results)
        novelty_final = novelty_combined_for_all_targets(
            results, min_prominence=args.novelty_min_prominence,
            exclusion_frac=args.novelty_exclusion_frac)
        final_rows = []
        for label in full["labels"]:
            pf = pfinal[label]
            p_str = f"{pf['p_measure']:.2f}%" if np.isfinite(pf['p_measure']) else "n/a"
            ref_str = pf['reference'] or "none"
            eff_var_final = (efficiency_variation(eff_cache[("full", label)], eff_cache[(ref_str, label)])
                              if pf['reference'] else np.nan)
            eff_var_str = f"{eff_var_final:.2f}%" if np.isfinite(eff_var_final) else "n/a"
            sd_str = f"{pf['spectral_deviation']:.2f}%" if np.isfinite(pf['spectral_deviation']) else "n/a"
            peaks = novelty_final[label]['novel_peaks']
            novelty_str = (f"{peaks[0]['period_days']:.4f}d ({peaks[0]['relevance_pct']:.2f}%)"
                           if peaks else "none detected")
            final_rows.append([label, p_str, eff_var_str, sd_str, ref_str, novelty_str])

        final_headers = ["mode", "p_measure_final", "efficiency_var_final", "spectral_dev_final",
                          "vs.", "novelty_period_final (%)"]
        final_widths = [max(len(h), *(len(r[i]) for r in final_rows)) if final_rows else len(h)
                        for i, h in enumerate(final_headers)]
        final_fmt = "  ".join(f"{{:<{w}}}" for w in final_widths)
        print(final_fmt.format(*final_headers))
        print(final_fmt.format(*["-" * w for w in final_widths]))
        for r in final_rows:
            print(final_fmt.format(*r))

        novelty_dir = os.path.join(config.output_root, "figures", "dynamics", spec.key)
        novelty_paths = novelty_frequency_figures(
            results, novelty_dir, min_prominence=args.novelty_min_prominence,
            exclusion_frac=args.novelty_exclusion_frac)
        print(f"\n=== novelty-frequency spectrum figures ({len(novelty_paths)}) ===")
        for p in novelty_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
