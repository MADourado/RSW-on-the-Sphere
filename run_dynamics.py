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
from rsw_sphere.dynamics.run_config import RunConfig, default_max_workers
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.dynamics.dynamical_phase import dynamical_phase, libration_diagnostics
from rsw_sphere.plotting.labels import _mode_label, mode_fs_label
from rsw_sphere.plotting.energy_evolution import plot_energy_evolution
from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report, write_diagnostics_files
from rsw_sphere.utilities.periods import DEFAULT_EXCLUSION_FRAC


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
        max_workers = config.max_workers or default_max_workers()
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
    parser.add_argument("--novelty-exclusion-frac", type=float, default=DEFAULT_EXCLUSION_FRAC,
                         help="novelty_period: +/- period window excluded around each "
                              "sub-triad's own dominant peak")
    parser.add_argument("--novelty-min-prominence", type=float, default=0.02,
                         help="novelty_period: minimum find_peaks prominence to report a novel peak")
    parser.add_argument("--novelty-xmax", type=float, default=None,
                         help="novelty spectrum: upper period (days) shown/searched. Default: "
                              "round(sqrt(tf_days / 2)), so a long run's own window scales with it")
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
    run_dir = results["full"]["run_dir"]
    # run_label tags every diag_*.csv/diag_freq_novel_*.png filename below,
    # same convention as the evol_*.png figures (and, underneath that,
    # trajectory_cache.ic_label) -- one consistent naming scheme across
    # everything this run writes.
    run_label = os.path.basename(run_dir)

    # compute_diagnostics_report always computes everything (cheap -- FFT
    # and a handful of numpy calls, no integration) -- shared with
    # run_sweep.py's own per-grid-point diagnostics so both source from the
    # exact same engine. Printing/table selection below still follows
    # args.diagnostics, same as before this was factored out.
    report = compute_diagnostics_report(
        results, spec, novelty_exclusion_frac=args.novelty_exclusion_frac,
        novelty_min_prominence=args.novelty_min_prominence,
        efficiency_drift_max=args.efficiency_drift_max)

    # freq_rows_by_unit feeds the single consolidated frequency table printed
    # after this loop (frequency units are kept consistent throughout --
    # cycles/day, not rad/day -- so linear/observed/precession frequencies
    # are all directly comparable at a glance).
    freq_rows_by_unit = {}
    for idx, (name, r) in enumerate(results.items()):
        if idx > 0:
            print()
        print(f"[{name}] {r['title']}: Energy drift={r['drift']:.3e}, "
              f"dEK={dict(zip(r['labels'], r['dEK']))}")

        rows = []
        insufficient_cycles = []
        for lbl in r['labels']:
            m = report['per_mode_unit'][name][lbl]
            peaks_str = "; ".join(
                f"{p['period_days']:.3f} ({1.0 / p['period_days']:.4f}, {p['power_frac']:.0f}%)"
                for p in m['top_peaks']) or "n/a"
            eff_str = f"{m['efficiency']:.4f}" if np.isfinite(m['efficiency']) else "n/a (drift)"
            rows.append([lbl, name, f"{m['dEK']:.4f}", eff_str,
                         f"{m['linear_period_days']:.3f}", f"{m['linear_freq_cpd']:.4f}", peaks_str])
            if m['insufficient_cycles']:
                insufficient_cycles.append(lbl)
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
        if insufficient_cycles:
            t_f_days = float(r['t'][-1] - r['t'][0])
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

    paths = write_diagnostics_files(
        results, report, run_dir, run_label, spec, diagnostics=args.diagnostics,
        novelty_exclusion_frac=args.novelty_exclusion_frac,
        novelty_min_prominence=args.novelty_min_prominence, novelty_xmax=args.novelty_xmax)
    print(f"\n  table -> {paths['diag_evol']}")

    if args.diagnostics:
        # Dynamical phase (rsw_sphere.dynamics.dynamical_phase): a property
        # of one constituent triad, not a mode -- no shared-triad ambiguity
        # to resolve (a triad is never shared the way a mode can be), so a
        # single full-vs-alone comparison is already well-defined, mirroring
        # the same "vs." structure as the pairwise table below. Displayed in
        # cycles/day (not rad/day) to match the frequency table above --
        # phase_variation is a ratio of two same-unit values, so it's
        # unaffected by the choice.
        print("\n=== dynamical phase (precession frequency, cycles/day) ===")
        phase_rows = [
            [triad_label, f"{p['freq_full_cpd']:.5f}",
             f"{p['freq_alone_cpd']:.5f}" if p['freq_alone_cpd'] is not None else "n/a",
             f"{p['phase_variation_pct']:.2f}%" if np.isfinite(p['phase_variation_pct']) else "n/a"]
            for triad_label, p in report['precession'].items()
        ]
        phase_headers = ["triad", "precession_freq_full", "precession_freq_alone", "phase_variation"]
        phase_widths = [max(len(h), *(len(r[i]) for r in phase_rows)) if phase_rows else len(h)
                        for i, h in enumerate(phase_headers)]
        phase_fmt = "  ".join(f"{{:<{w}}}" for w in phase_widths)
        print(phase_fmt.format(*phase_headers))
        print(phase_fmt.format(*["-" * w for w in phase_widths]))
        for r in phase_rows:
            print(phase_fmt.format(*r))
        if 'diag_prec_freq' in paths:
            print(f"  table -> {paths['diag_prec_freq']}")

        print("\n=== pairwise diagnostics (full wave set vs. each containing sub-triad) ===")
        rows = [
            [d['mode'], d['vs'], f"{d['p_measure_pct']:.2f}%",
             f"{d['efficiency_var_pct']:.2f}%" if np.isfinite(d['efficiency_var_pct']) else "n/a",
             f"{d['spectral_dev_pct']:.2f}%" if np.isfinite(d['spectral_dev_pct']) else "n/a",
             (f"{d['novelty_period_days']:.4f}d ({d['novelty_relevance_pct']:.2f}%)"
              if np.isfinite(d['novelty_period_days']) else "none detected")]
            for d in report['pairwise']
        ]
        headers = ["mode", "vs.", "p_measure", "efficiency_var", "spectral_dev", "novelty_period (%)"]
        widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
                  for i, h in enumerate(headers)]
        row_fmt = "  ".join(f"{{:<{w}}}" for w in widths)
        print(row_fmt.format(*headers))
        print(row_fmt.format(*["-" * w for w in widths]))
        for r in rows:
            print(row_fmt.format(*r))
        if 'diag_pairwise' in paths:
            print(f"  table -> {paths['diag_pairwise']}")

        # "Final" diagnostics: one row per target mode, considering every
        # containing sub-triad at once instead of one pairwise comparison
        # at a time -- avoids the small-denominator inflation a mode
        # shared across triads can otherwise show against whichever
        # sub-triad happens to leave it weakly excited (see
        # final_p_measure's own docstring). p_measure_final/spectral_dev_final
        # share one reference (largest raw dEK, "vs." column);
        # efficiency_var_final picks its OWN reference independently
        # (largest |efficiency|, "eff. vs." column) since efficiency
        # normalizes dEK by each sub-triad's own different mean total
        # energy -- reusing the dEK-based reference here used to let a
        # shared mode's efficiency denominator pass through zero even
        # while efficiency_full itself varied smoothly, producing spurious
        # spikes (found 2026-08-28, quartet_rossby_gravity_influence).
        print("\n=== final diagnostics (per target, across all containing sub-triads) ===")
        final_rows = [
            [d['mode'],
             f"{d['p_measure_final_pct']:.2f}%" if np.isfinite(d['p_measure_final_pct']) else "n/a",
             f"{d['efficiency_var_final_pct']:.2f}%" if np.isfinite(d['efficiency_var_final_pct']) else "n/a",
             f"{d['spectral_dev_final_pct']:.2f}%" if np.isfinite(d['spectral_dev_final_pct']) else "n/a",
             d['vs'] or "none", d['efficiency_var_vs'] or "none",
             (f"{d['novelty_period_final_days']:.4f}d ({d['novelty_relevance_final_pct']:.2f}%)"
              if np.isfinite(d['novelty_period_final_days']) else "none detected")]
            for d in report['final']
        ]
        final_headers = ["mode", "p_measure_final", "efficiency_var_final", "spectral_dev_final",
                          "vs.", "eff. vs.", "novelty_period_final (%)"]
        final_widths = [max(len(h), *(len(r[i]) for r in final_rows)) if final_rows else len(h)
                        for i, h in enumerate(final_headers)]
        final_fmt = "  ".join(f"{{:<{w}}}" for w in final_widths)
        print(final_fmt.format(*final_headers))
        print(final_fmt.format(*["-" * w for w in final_widths]))
        for r in final_rows:
            print(final_fmt.format(*r))
        if 'diag_final' in paths:
            print(f"  table -> {paths['diag_final']}")

        novelty_paths = paths.get('novelty', [])
        print(f"\n=== novelty-frequency spectrum figures ({len(novelty_paths)}) ===")
        for p in novelty_paths:
            print(f"  {p}")


if __name__ == "__main__":
    main()
