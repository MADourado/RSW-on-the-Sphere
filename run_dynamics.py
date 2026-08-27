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
from rsw_sphere.dynamics.trajectory_cache import run_and_cache, _mode_slug, ic_label, topology_folder
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics
from rsw_sphere.plotting.labels import _mode_label, mode_fs_label
from rsw_sphere.plotting.energy_evolution import plot_energy_evolution
from rsw_sphere.utilities.periods import dominant_periods
from rsw_sphere.utilities.tables import write_csv
from rsw_sphere.utilities.efficiency import wave_set_efficiency


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
    triad in a quartet/quintet and wouldn't help distinguish them -- but
    every human-facing LABEL below names all three modes plainly (e.g.
    ``Triad 1 (RH(3,4)+RH(4,5)+WG(7,9))``), built directly from the
    modes themselves rather than the registry's own free-text
    display_label: once every mode is named, a "(RH-only)"/"(with
    WG(1,1))"-style annotation is redundant, and listing all three reads
    the same way regardless of which registry entry it came from.

    triad_labels : one display label per entry in this unit's own
    ``triads`` (the full wave set carries every constituent triad's own
    label; a sub-triad unit carries just its own, matching its title) --
    used to key each unit's own dynamical-phase precession frequency."""
    def _triad_label(i, mode_p, mode_q, mode_sum):
        return (f"Triad {i + 1} ({_mode_label(*mode_p)}+{_mode_label(*mode_q)}"
                f"+{_mode_label(*mode_sum)})")

    full_triad_labels = []
    for i in range(spec.n_triads()):
        i_sum, i_p, i_q = spec.triad_indices(i)
        full_triad_labels.append(_triad_label(i, spec.modes[i_p], spec.modes[i_q], spec.modes[i_sum]))
    units = [("full", spec.modes, [spec.triad_indices(i) for i in range(spec.n_triads())],
              spec.velocities, spec.display_label or spec.key, full_triad_labels)]
    if spec.has_subtriads():
        for i in range(spec.n_triads()):
            member_p, member_q, sum_mode = spec.sub_triad_modes(i)
            name = f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}"
            label = _triad_label(i, member_p, member_q, sum_mode)
            units.append((name, spec.sub_triad_modes(i), [(2, 0, 1)],
                          spec.sub_triad_velocities(i), label, [label]))
    return units


def _integrate_and_plot_unit(args):
    """Worker (module-level so it's picklable for ProcessPoolExecutor)."""
    (name, modes, triads, velocities, title, triad_labels,
     h_e, tf_days, h, output_root, plot, run_dir) = args

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
        # Filename carries the initial conditions too (run_dir's own
        # basename, e.g. "rh34_50.00-...-wg79_50.00_tf20_h0.01") since these
        # figures get pulled into the paper individually -- the run folder
        # alone wouldn't survive that, but a self-contained filename does.
        # Every mode is listed (sum mode included, not just a sub-triad's
        # own two members), sorted the same way trajectory_cache.ic_label
        # sorts them (by mode tuple) -- matches this run's own trajectory
        # filename(s) so the two stay recognizably paired.
        run_label = os.path.basename(run_dir)
        mode_tag = "_".join(mode_fs_label(*m) for m in sorted(modes))
        if name == "full":
            topology = topology_folder(len(modes)).rstrip('s')
            fig_name = f"evol_{topology}_{mode_tag}_{run_label}.png"
        else:
            fig_name = f"evol_triad_{mode_tag}_{run_label}.png"
        fig_path = os.path.join(run_dir, fig_name)
        os.makedirs(run_dir, exist_ok=True)

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
        'trajectory_path': traj_path, 'figure_path': fig_path, 'run_dir': run_dir,
    }


def run_dynamics(config: RunConfig) -> dict:
    """Integrate + plot every topology unit of config.wave_set_spec.

    Returns dict: unit name ('full', 'triad_<member1>_<member2>', ...) -> result
    dict (t, E, E_total, labels, drift, dEK, omega, precession_freq,
    trajectory_path, figure_path, title).
    """
    spec = config.wave_set_spec
    units = _build_units(spec)
    # run_dir: outputs/dynamics/<wave_set_key>/<run_label>/, shared by every
    # unit (full + every sub-triad) of this one invocation -- run_label
    # reuses trajectory_cache's own ic_label/tf/h convention (the same
    # readable part of a trajectory's own cache filename, minus its hash
    # suffix, which guards against a numerics-only difference that doesn't
    # matter for a figures/tables folder name).
    run_label = f"{ic_label(spec.modes, spec.velocities)}_tf{config.tf_days:.0f}_h{config.h:g}"
    run_dir = os.path.join(config.output_root, "dynamics", spec.key, run_label)
    args = [(name, modes, triads, velocities, title, triad_labels, spec.h_e, config.tf_days, config.h,
             config.output_root, config.plot, run_dir)
            for name, modes, triads, velocities, title, triad_labels in units]

    if config.parallel and len(units) > 1:
        max_workers = config.max_workers or max(1, (os.cpu_count() or 2) // 2)
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_integrate_and_plot_unit, args))
    else:
        results = [_integrate_and_plot_unit(a) for a in args]

    return {r['name']: r for r in results}


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
    spec = specs[args.wave_set]
    config = RunConfig.from_wave_set(spec, tf_days=args.tf_days, h=args.h,
                                      output_root=args.output_root,
                                      plot=not args.no_plot, parallel=not args.no_parallel)

    results = run_dynamics(config)
    # freq_rows_by_unit feeds the single consolidated frequency table printed
    # after this loop (frequency units are kept consistent throughout --
    # cycles/day, not rad/day -- so linear/observed/precession frequencies
    # are all directly comparable at a glance). eff_cache is built here too
    # (one value per mode per unit, same shape as the frequency table) so it
    # can be folded into that same table instead of getting its own --
    # reused as-is later by the --diagnostics pairwise/final tables below.
    freq_rows_by_unit = {}
    eff_cache = {}
    diag_evol_rows = []
    run_dir = results["full"]["run_dir"]
    # run_label tags every diag_*.csv/diag_freq_novel_*.png filename below,
    # same convention as the evol_*.png figures (and, underneath that,
    # trajectory_cache.ic_label) -- one consistent naming scheme across
    # everything this run writes.
    run_label = os.path.basename(run_dir)
    for idx, (name, r) in enumerate(results.items()):
        if idx > 0:
            print()
        print(f"[{name}] {r['title']}: Energy drift={r['drift']:.3e}, "
              f"dEK={dict(zip(r['labels'], r['dEK']))}")
        # Linear period = 1/(2*|omega|) days -- omega is rad per nondim time
        # unit, and 1 day = 4*pi nondim time units throughout this codebase
        # (rsw_sphere.physics.days_from_nondim_time), so period_nondim =
        # 2*pi/|omega| converts to period_days = period_nondim/(4*pi);
        # linear frequency (cycles/day) is just its reciprocal, 2*|omega|.
        period_results = [dominant_periods(r['t'], r['E'][:, j]) for j in range(len(r['labels']))]
        rows = []
        for j, (lbl, w, pr, dEK) in enumerate(zip(r['labels'], r['omega'], period_results, r['dEK'])):
            lin_period = 1.0 / (2 * abs(w))
            lin_freq = 2 * abs(w)
            peaks_str = "; ".join(
                f"{p['period_days']:.3f} ({1.0 / p['period_days']:.4f}, {p['power_frac']:.0f}%)"
                for p in pr['top_peaks']) or "n/a"
            eff = wave_set_efficiency(r["E"][:, j], r["E_total"], r["drift"],
                                       drift_max=args.efficiency_drift_max)
            eff_cache[(name, lbl)] = eff
            eff_str = f"{eff:.4f}" if np.isfinite(eff) else "n/a (drift)"
            rows.append([lbl, name, f"{dEK:.4f}", eff_str, f"{lin_period:.3f}", f"{lin_freq:.4f}", peaks_str])

            # diag_evol.csv: same data as the printed table above, but one
            # column per value (peak1_period_days, peak1_freq_cpd, ...)
            # instead of a packed "period (freq, pct); period (freq, pct)"
            # string -- easier to load into a dataframe/spreadsheet.
            evol_row = {
                'wave_set': spec.key, 'unit': name, 'mode': lbl,
                'dEK': float(dEK), 'efficiency': eff,
                'linear_period_days': lin_period, 'linear_freq_cpd': lin_freq,
            }
            for k in range(3):
                if k < len(pr['top_peaks']):
                    p = pr['top_peaks'][k]
                    evol_row[f'peak{k + 1}_period_days'] = p['period_days']
                    evol_row[f'peak{k + 1}_freq_cpd'] = 1.0 / p['period_days']
                    evol_row[f'peak{k + 1}_power_pct'] = p['power_frac']
                else:
                    evol_row[f'peak{k + 1}_period_days'] = ''
                    evol_row[f'peak{k + 1}_freq_cpd'] = ''
                    evol_row[f'peak{k + 1}_power_pct'] = ''
            diag_evol_rows.append(evol_row)
        freq_rows_by_unit[name] = rows

        prec = {lbl: round(v / (2 * np.pi), 5) for lbl, v in r['precession_freq'].items()}
        print(f"  precession_freq (cycles/day): {prec}")

        # Heuristic t_f-adequacy gate -- NOT a rigorous test, just a cheap
        # warning using numbers already computed above: a mode whose own
        # top 2 FFT peaks (by power) don't each complete at least 4 full
        # cycles within t_f has too few cycles resolved to trust that
        # period estimate. (Deliberately NOT gating on precession_freq
        # anymore -- a genuinely locked/near-locked triad has a precession
        # frequency near zero *by construction*, no matter how long t_f
        # is, so "hasn't completed one revolution yet" was flagging the
        # expected, physically correct case just as often as a real
        # under-resolved run.) Can't prove a longer run is unnecessary,
        # only flag when this one clearly isn't enough.
        t_f_days = float(r['t'][-1] - r['t'][0])
        insufficient_cycles = [lbl for lbl, pr in zip(r['labels'], period_results)
                                if any(p['period_days'] * 4 > t_f_days for p in pr['top_peaks'][:2])]
        if insufficient_cycles:
            print(f"  WARNING: t_f={t_f_days:.1f}d may not resolve enough cycles (heuristic, not "
                  f"conclusive -- consider a longer run before trusting these numbers)")
            print(f"    fewer than 4 cycles of a top-2 FFT peak resolved for: {insufficient_cycles}")

        print(f"  trajectory -> {r['trajectory_path']}")
        if r['figure_path']:
            print(f"  figure -> {r['figure_path']}")

    # Consolidated frequency table: one row per (mode, unit) -- same
    # mode/unit shape as the efficiency table below, kept as its own table
    # (not merged into the pairwise diagnostics table, which compares only
    # full-vs-sub-triad pairs and would be awkward to fold this into).
    print("\n=== linear vs. observed (FFT) frequency, per mode per unit ===")
    freq_headers = ["mode", "unit", "dEK", "efficiency", "linear_period (d)", "linear_freq (cpd)",
                     "observed peaks (period d, freq cpd, % of dominant peak)"]
    all_freq_rows = [row for rows in freq_rows_by_unit.values() for row in rows]
    freq_widths = [max(len(h), *(len(row[i]) for row in all_freq_rows)) if all_freq_rows else len(h)
                   for i, h in enumerate(freq_headers)]
    freq_fmt = "  ".join(f"{{:<{w}}}" for w in freq_widths)
    print(freq_fmt.format(*freq_headers))
    print(freq_fmt.format(*["-" * w for w in freq_widths]))
    for idx, rows in enumerate(freq_rows_by_unit.values()):
        if idx > 0:
            print()
        for row in rows:
            print(freq_fmt.format(*row))
    diag_evol_path = os.path.join(run_dir, f"diag_evol_{run_label}.csv")
    write_csv(diag_evol_rows, diag_evol_path)
    print(f"\n  table -> {diag_evol_path}")

    if args.diagnostics:
        from rsw_sphere.utilities.pmeasure import pairwise_target_diagnostics, p_measure_combined_for_all_targets
        from rsw_sphere.utilities.novelty_frequency import novelty_combined_for_all_targets
        from rsw_sphere.utilities.efficiency import efficiency_variation
        from rsw_sphere.plotting.novelty_frequency_panel import novelty_frequency_figures

        full = results["full"]
        # eff_cache (mode, unit) -> efficiency was already computed in the
        # main loop above and folded into the consolidated frequency table
        # -- reused here as-is rather than a separate efficiency table.

        # Dynamical phase (rsw_sphere.dynamics.dynamical_phase): a property
        # of one constituent triad, not a mode -- no shared-triad ambiguity
        # to resolve (a triad is never shared the way a mode can be), so a
        # single full-vs-alone comparison is already well-defined, mirroring
        # the same "vs." structure as the pairwise table below. Displayed in
        # cycles/day (not rad/day) to match the frequency table above --
        # phase_variation is a ratio of two same-unit values, so it's
        # unaffected by the choice.
        print("\n=== dynamical phase (precession frequency, cycles/day) ===")
        phase_rows = []
        diag_prec_freq_rows = []
        for triad_label, prec_full in full["precession_freq"].items():
            prec_alone = None
            for name, r in results.items():
                if name != "full" and triad_label in r["precession_freq"]:
                    prec_alone = r["precession_freq"][triad_label]
                    break
            if prec_alone is not None and abs(prec_alone) > 1e-12:
                phase_variation = 100 * (abs(prec_full) - abs(prec_alone)) / abs(prec_alone)
                var_str = f"{phase_variation:.2f}%"
            else:
                phase_variation = float("nan")
                var_str = "n/a"
            alone_str = f"{prec_alone / (2 * np.pi):.5f}" if prec_alone is not None else "n/a"
            phase_rows.append([triad_label, f"{prec_full / (2 * np.pi):.5f}", alone_str, var_str])
            diag_prec_freq_rows.append({
                'triad': triad_label, 'precession_freq_full_cpd': prec_full / (2 * np.pi),
                'precession_freq_alone_cpd': prec_alone / (2 * np.pi) if prec_alone is not None else '',
                'phase_variation_pct': phase_variation,
            })
        phase_headers = ["triad", "precession_freq_full", "precession_freq_alone", "phase_variation"]
        phase_widths = [max(len(h), *(len(r[i]) for r in phase_rows)) if phase_rows else len(h)
                        for i, h in enumerate(phase_headers)]
        phase_fmt = "  ".join(f"{{:<{w}}}" for w in phase_widths)
        print(phase_fmt.format(*phase_headers))
        print(phase_fmt.format(*["-" * w for w in phase_widths]))
        for r in phase_rows:
            print(phase_fmt.format(*r))
        diag_prec_freq_path = os.path.join(run_dir, f"diag_prec_freq_{run_label}.csv")
        write_csv(diag_prec_freq_rows, diag_prec_freq_path)
        print(f"  table -> {diag_prec_freq_path}")

        print("\n=== pairwise diagnostics (full wave set vs. each containing sub-triad) ===")
        rows = []
        diag_pairwise_rows = []
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
                diag_pairwise_rows.append({
                    'mode': label, 'vs': name, 'p_measure_pct': d['p_measure'],
                    'efficiency_var_pct': eff_var, 'spectral_dev_pct': d['spectral_deviation'],
                    'novelty_period_days': d['novelty_period'] if np.isfinite(d['novelty_period']) else '',
                    'novelty_relevance_pct': d['novelty_relevance'] if np.isfinite(d['novelty_period']) else '',
                })

        headers = ["mode", "vs.", "p_measure", "efficiency_var", "spectral_dev", "novelty_period (%)"]
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                  for i, h in enumerate(headers)]
        row_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(row_fmt.format(*headers))
        print(row_fmt.format(*["-" * w for w in widths]))
        for r in rows:
            print(row_fmt.format(*r))
        diag_pairwise_path = os.path.join(run_dir, f"diag_pairwise_{run_label}.csv")
        write_csv(diag_pairwise_rows, diag_pairwise_path)
        print(f"  table -> {diag_pairwise_path}")

        # "Final" diagnostics: one row per target mode, considering every
        # containing sub-triad at once instead of one pairwise comparison
        # at a time -- avoids the small-denominator inflation a mode
        # shared across triads can otherwise show against whichever
        # sub-triad happens to leave it weakly excited (see
        # final_p_measure's own docstring). efficiency_var_final reuses
        # the SAME winning sub-triad p_measure_final already picked
        # (rather than an independent selection), so both are always
        # read against the same reference -- their signs can still
        # differ, though (see efficiency_variation's own docstring).
        print("\n=== final diagnostics (per target, across all containing sub-triads) ===")
        pfinal = p_measure_combined_for_all_targets(results)
        novelty_final = novelty_combined_for_all_targets(
            results, min_prominence=args.novelty_min_prominence,
            exclusion_frac=args.novelty_exclusion_frac)
        final_rows = []
        diag_final_rows = []
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
            diag_final_rows.append({
                'mode': label, 'p_measure_final_pct': pf['p_measure'],
                'efficiency_var_final_pct': eff_var_final, 'spectral_dev_final_pct': pf['spectral_deviation'],
                'vs': ref_str,
                'novelty_period_final_days': peaks[0]['period_days'] if peaks else '',
                'novelty_relevance_final_pct': peaks[0]['relevance_pct'] if peaks else '',
            })

        final_headers = ["mode", "p_measure_final", "efficiency_var_final", "spectral_dev_final",
                          "vs.", "novelty_period_final (%)"]
        final_widths = [max(len(h), *(len(r[i]) for r in final_rows)) if final_rows else len(h)
                        for i, h in enumerate(final_headers)]
        final_fmt = "  ".join(f"{{:<{w}}}" for w in final_widths)
        print(final_fmt.format(*final_headers))
        print(final_fmt.format(*["-" * w for w in final_widths]))
        for r in final_rows:
            print(final_fmt.format(*r))
        diag_final_path = os.path.join(run_dir, f"diag_final_{run_label}.csv")
        write_csv(diag_final_rows, diag_final_path)
        print(f"  table -> {diag_final_path}")

        novelty_paths = novelty_frequency_figures(
            results, run_dir, filename_suffix=run_label, min_prominence=args.novelty_min_prominence,
            exclusion_frac=args.novelty_exclusion_frac)
        print(f"\n=== novelty-frequency spectrum figures ({len(novelty_paths)}) ===")
        for p in novelty_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
