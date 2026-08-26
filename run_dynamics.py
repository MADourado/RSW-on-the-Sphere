"""Integrate a wave set's full dynamics, plus every constituent triad
separately

 Each integration is cached via trajectory_cache.run_and_cache and
 each integration is plotted separately (unless --no-plot) 
 under a path mirroring its own trajectory's cache path.

Run:

    python run_dynamics.py --wave-set quartet_gravity_kelvin

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
from rsw_sphere.dynamics.trajectory_cache import run_and_cache
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.energy_evolution import plot_energy_evolution
from rsw_sphere.utilities.tables import dynamics_summary_rows, write_csv


def _build_units(spec):
    """(name, modes, triads, velocities, title) per topology unit: the
    full wave set, plus each constituent triad if spec.has_subtriads()."""
    units = [("full", spec.modes, [spec.triad_indices(i) for i in range(spec.n_triads())],
              spec.velocities, spec.display_label or spec.key)]
    if spec.has_subtriads():
        for i, t in enumerate(spec.triads):
            units.append((f"triad{i}", spec.sub_triad_modes(i), [(2, 0, 1)],
                          spec.sub_triad_velocities(i), t.display_label or f"Triad {i + 1}"))
    return units


def _integrate_and_plot_unit(args):
    """Worker (module-level so it's picklable for ProcessPoolExecutor)."""
    name, modes, triads, velocities, title, h_e, tf_days, h, output_root, plot = args

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

    fig_path = None
    if plot:
        import matplotlib.pyplot as plt
        from rsw_sphere.plotting.style import apply_house_style
        rel = os.path.relpath(traj_path, traj_root)
        fig_path = os.path.join(output_root, "figures", "dynamics",
                                 os.path.splitext(rel)[0] + ".png")
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
        'trajectory_path': traj_path, 'figure_path': fig_path,
    }


def run_dynamics(config: RunConfig, write_table: bool = True) -> dict:
    """Integrate + plot every topology unit of config.wave_set_spec.

    write_table : write a CSV summary to <output_root>/tables/<spec.key>.csv.
        False when called once per grid point (e.g. run_sweep.py's own
        per-point pass).

    Returns dict: unit name ('full', 'triad0', 'triad1', ...) -> result
    dict (t, E, E_total, labels, drift, dEK, trajectory_path, figure_path, title).
    """
    spec = config.wave_set_spec
    units = _build_units(spec)
    args = [(name, modes, triads, velocities, title, spec.h_e, config.tf_days, config.h,
             config.output_root, config.plot)
            for name, modes, triads, velocities, title in units]

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
        print(f"  trajectory -> {r['trajectory_path']}")
        if r['figure_path']:
            print(f"  figure -> {r['figure_path']}")


if __name__ == "__main__":
    main()
