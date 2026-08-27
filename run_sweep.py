"""General sweep driver: given a RunConfig (shared with run_dynamics.py)
with a `sweep:` block, sweeps 1 or 2 modes' initial velocities.

For each grid point, calls run_dynamics.run_dynamics (caches the
trajectory, saves a per-point evolution figure unless
sweep.save_point_figures is False). Separately computes and plots every
requested diagnostic as its own figure:

- 1 swept mode (1D): `precession` only, via
  rsw_sphere.utilities.precession.precession_frequency_efficiency
  (already natively 1D). Other diagnostics are 2D-sweep-only for now.
- 2 swept modes (2D): p_measure, p_measure_final, novelty_period,
  efficiency, low_frequency_energy, via rsw_sphere.utilities.registry.sweep_2d.

Run, if the wave set's own registry entry already carries its own
sweep:/tf_days/h/plot/output/target_mode/plot_triad keys (see
wave_sets_default.yaml's quartet_rossby_kelvin/quartet_rh_preference
entries), no separate config file is needed:

    python run_sweep.py --wave-set quartet_rossby_kelvin

or point at a standalone RunConfig YAML for an ad-hoc/one-off sweep not
worth adding to the registry:

    python run_sweep.py --config path/to/sweep.yaml
"""
import argparse
import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib.pyplot as plt
import yaml

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.precession_plot import plot_dual_axis_frequency_efficiency
from rsw_sphere.utilities.precession import precession_frequency_efficiency
from rsw_sphere.utilities.registry import sweep_2d, DIAGNOSTIC_ARRAY_KEYS, DIAGNOSTIC_PLOT_FNS

from run_dynamics import run_dynamics


def _grid_point_config(config: RunConfig, axes, u_values):
    spec = config.wave_set_spec
    velocities = list(spec.velocities)
    for axis, u in zip(axes, u_values):
        velocities[spec.index(axis.mode)] = u
    point_spec = dataclasses.replace(spec, velocities=tuple(velocities))
    # parallel=False: this config is itself dispatched to a worker process
    # by the outer grid-level ProcessPoolExecutor below -- a worker
    # spawning its own child pool would raise (daemonic processes can't
    # have children).
    return RunConfig.from_wave_set(
        point_spec, tf_days=config.tf_days, h=config.h, output_root=config.output_root,
        plot=config.sweep.save_point_figures, parallel=False)


def _run_dynamics_for_grid(config: RunConfig):
    """Calls run_dynamics.run_dynamics once per grid point (cache-or-compute,
    per-point figures per sweep.save_point_figures)."""
    spec = config.wave_set_spec
    axes = config.sweep.axes
    n_grid = config.sweep.n_grid
    grids = [np.linspace(a.min, a.max, n_grid) for a in axes]

    if len(axes) == 1:
        points = [_grid_point_config(config, axes, (u,)) for u in grids[0]]
    else:
        points = [_grid_point_config(config, axes, (u1, u2))
                  for u2 in grids[1] for u1 in grids[0]]

    with ProcessPoolExecutor(max_workers=config.max_workers or max(1, (os.cpu_count() or 2) // 2)) as ex:
        list(ex.map(run_dynamics, points))


def _run_sweep_1d(config: RunConfig, output: str, plot_cfg: dict,
                   target_mode: str = None, plot_triad: int = None):
    spec = config.wave_set_spec
    axis = config.sweep.axes[0]
    u_values = np.linspace(axis.min, axis.max, config.sweep.n_grid)
    diagnostics = config.sweep.diagnostics or ("precession",)
    if set(diagnostics) - {"precession"}:
        raise ValueError("1D sweeps only support diagnostic 'precession' for now "
                          f"(got {diagnostics})")

    result = precession_frequency_efficiency(
        spec, axis.mode, u_values, target_mode_key=target_mode,
        tf_days=config.tf_days, h=config.h,
        cache_root=os.path.join(config.output_root, "trajectories"))

    plot_dual_axis_frequency_efficiency(
        result, spec, plot_triad=plot_triad,
        xlabel=plot_cfg.get("xlabel", f"{axis.mode} driving velocity (m/s)"),
        title=plot_cfg.get("title", spec.display_label or spec.key),
        plot_u_max=plot_cfg.get("plot_u_max"), path=output)

    min_freq = {lbl: float(np.min(np.abs(v))) for lbl, v in result["freq_by_triad"].items()}
    print(f"  min |precession_freq| per triad: {min_freq}")
    return result


def _run_sweep_2d(config: RunConfig, output: str, plot_cfg: dict):
    spec = config.wave_set_spec
    axes = config.sweep.axes
    swept_indices = tuple(spec.index(a.mode) for a in axes)
    u_ranges = [(a.min, a.max) for a in axes]
    fixed_velocities = {i: spec.velocities[i] for i in range(spec.n_modes()) if i not in swept_indices}
    target_indices = list(swept_indices)  # private/swept modes are the natural per-target diagnostics
    diagnostics = config.sweep.diagnostics or ("p_measure",)

    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    cache_dir = os.path.join(config.output_root, "figures", "wave_sets", spec.key)
    os.makedirs(cache_dir, exist_ok=True)

    result = sweep_2d(
        spec.modes, triad_indices, spec.h_e, swept_indices, fixed_velocities, target_indices,
        diagnostics=diagnostics, u1_range=u_ranges[0], u2_range=u_ranges[1],
        reference_triad=spec.reference_triad, n_grid=config.sweep.n_grid,
        tf_days=config.tf_days, h=config.h, cache_dir=cache_dir,
        verbose=True, progress_label=spec.key)

    apply_house_style()
    n_targets = len(target_indices)
    fig, axes_grid = plt.subplots(len(diagnostics), n_targets,
                                   figsize=(5.5 * n_targets, 4.6 * len(diagnostics)))
    axes_grid = np.atleast_2d(axes_grid)
    label1 = _mode_label(*spec.modes[swept_indices[0]])
    label2 = _mode_label(*spec.modes[swept_indices[1]])

    for row, diag in enumerate(diagnostics):
        array_key = DIAGNOSTIC_ARRAY_KEYS[diag]
        plot_fn = DIAGNOSTIC_PLOT_FNS[diag]
        values = result[array_key]
        for col, tgt in enumerate(target_indices):
            tgt_label = _mode_label(*spec.modes[tgt])
            plot_fn(result["U1"], result["U2"], values[..., col],
                    xlabel=plot_cfg.get("xlabel", f"{label1} u (m/s)"),
                    ylabel=plot_cfg.get("ylabel", f"{label2} u (m/s)"),
                    title=f"{diag}: {tgt_label}", ax=axes_grid[row, col])
        finite = values[np.isfinite(values)]
        if finite.size:
            print(f"  [{diag}] range over finite grid points: "
                  f"[{finite.min():.4g}, {finite.max():.4g}] "
                  f"({finite.size}/{values.size} finite)")

    fig.suptitle(plot_cfg.get("title", spec.display_label or spec.key), fontsize=12)
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


def run_sweep(config: RunConfig, output: str, plot_cfg: dict = None, run_per_point: bool = True,
              target_mode: str = None, plot_triad: int = None):
    """Sweep config.sweep.axes (1 or 2 modes' velocities). Also calls
    run_dynamics per grid point (caches trajectories, per-point figures
    per sweep.save_point_figures) unless run_per_point=False.

    target_mode, plot_triad : 1D sweeps only (precession diagnostic).

    NOTE: the per-point run_dynamics pass and the diagnostic sweep below
    currently integrate the same trajectories through two separate code
    paths (run_dynamics uses trajectory_cache, the diagnostic engines in
    rsw_sphere.utilities don't yet) -- real duplicated compute, a known
    follow-up to unify.
    """
    plot_cfg = plot_cfg or {}
    if run_per_point:
        _run_dynamics_for_grid(config)

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    if len(config.sweep.axes) == 1:
        return _run_sweep_1d(config, output, plot_cfg, target_mode=target_mode, plot_triad=plot_triad)
    elif len(config.sweep.axes) == 2:
        return _run_sweep_2d(config, output, plot_cfg)
    else:
        raise ValueError(f"sweep needs 1 or 2 axes, got {len(config.sweep.axes)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="path to a RunConfig YAML with a sweep: block")
    parser.add_argument("--wave-set", default=None,
                         help="registry role key -- reads sweep/tf_days/h/plot/output/target_mode/"
                              "plot_triad straight from that wave_sets_default.yaml entry, "
                              "no separate config file needed.")
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH, help="used with --wave-set")
    parser.add_argument("--output", default=None, help="override the config's own output")
    parser.add_argument("--output-root", default="outputs",
                         help="override the config's own output_root (e.g. for an isolated test run)")
    parser.add_argument("--no-per-point", action="store_true",
                         help="skip the per-grid-point run_dynamics pass (diagnostics only)")
    args = parser.parse_args()

    if bool(args.config) == bool(args.wave_set):
        parser.error("exactly one of --config or --wave-set is required")

    if args.wave_set:
        specs = load_wave_set_specs(args.specs)
        if args.wave_set not in specs:
            parser.error(f"--wave-set {args.wave_set!r} not found in {args.specs!r} "
                         f"(available: {list(specs)})")
        with open(args.specs) as f:
            raw = yaml.safe_load(f)[args.wave_set]
        config = RunConfig.from_registry_entry(args.wave_set, args.specs)
        default_output = f"{args.output_root}/figures/wave_sets/{args.wave_set}/sweep.png"
    else:
        with open(args.config) as f:
            raw = yaml.safe_load(f)
        config = RunConfig.from_yaml(args.config)
        default_output = f"{args.output_root}/figures/wave_sets/sweep.png"
    config = dataclasses.replace(config, output_root=args.output_root)

    output = args.output or raw.get("output") or default_output
    plot_cfg = raw.get("plot", {})

    print(f"Running sweep for wave set {config.wave_set_spec.key!r}...")
    run_sweep(config, output, plot_cfg, run_per_point=not args.no_per_point,
              target_mode=raw.get("target_mode"), plot_triad=raw.get("plot_triad"))
    print(f"wrote {os.path.abspath(output)}")


if __name__ == "__main__":
    main()
