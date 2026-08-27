"""General sweep driver: given a RunConfig (shared with run_dynamics.py)
with a `sweep:` block, sweeps 1 or 2 modes' initial velocities and
computes/plots every requested diagnostic -- one `run_dynamics()` call per
grid point (via `rsw_sphere.dynamics.diagnostics_report.compute_diagnostics_report`),
parallelized across grid points, with no separate/duplicated integration
path for either dimensionality.

One diagnostic vocabulary, both dimensionalities:

- `efficiency`, `dominant_freq`, `dominant_period`, `low_frequency_energy`
  -- 1D: one line per (mode, unit). 2D: one heatmap per mode (`full`
  unit's own value only -- a heatmap grid doesn't have a line plot's
  spare room for a per-unit breakdown).
- `dynamical_phase` -- 1D: one line per triad. 2D: one heatmap per triad.
- `p_measure` (alias `energy_var`), `efficiency_var`, `spectral_dev_var`,
  `novel_freq`/`novelty_freq`, `novel_period`/`novelty_period` -- one
  line/heatmap per mode, combining every containing sub-triad the same
  way `run_dynamics.py --diagnostics`'s own "final diagnostics" table
  does (same meaning in 1D and 2D). Undefined for a plain triad with no
  sub-triad to compare against -- warned and skipped there.

`diagnostics: [all]` expands to every name above. Output:
`outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png` +
a matching long/tidy `.csv`, one pair per requested diagnostic; each
write prints its own `figure ->`/`table ->` line.

Reads the wave set's own registry entry, which carries its own
sweep:/tf_days/h/plot keys (see wave_sets_default.yaml's
quartet_rossby_kelvin/quartet_rh_preference entries) -- no separate
config file needed. A wave set not yet worth adding to the default
registry can be swept via a standalone `--specs` file instead (any YAML
in the same registry schema, e.g. examples/wave_sets_custom.yaml).

    python run_sweep.py --wave-set quartet_rossby_kelvin

A composite figure combining two or more of these diagnostics (e.g. the
paper's own precession-frequency-and-efficiency figure) is a separate,
paper-specific script's job -- see
examples/figures/paper_figure006_quartet_a_precession.py, which calls
run_sweep() for `[dynamical_phase, efficiency]` (one shared computation)
and composes its own figure from the two results.
"""
import argparse
import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import yaml

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report, write_diagnostics_files
from rsw_sphere.plotting.labels import _mode_label, mode_fs_label
from rsw_sphere.plotting.sweep_diagnostics import (
    plot_mode_unit_sweep, plot_triad_sweep, plot_mode_scalar_sweep,
    plot_mode_heatmap_sweep, plot_triad_heatmap_sweep)
from rsw_sphere.utilities.tables import write_csv

from run_dynamics import run_dynamics


_MODE_UNIT_DIAGNOSTICS = ("efficiency", "dominant_freq", "dominant_period", "low_frequency_energy")
_MODE_UNIT_KEYS = {'efficiency': 'efficiency', 'dominant_freq': 'freq_global_cpd',
                    'dominant_period': 'period_global', 'low_frequency_energy': 'low_freq_power'}
_TRIAD_DIAGNOSTICS = ("dynamical_phase",)
_SCALAR_DIAGNOSTICS = ("p_measure", "efficiency_var", "spectral_dev_var", "novel_freq", "novel_period")
#: A signed % change (inhibition vs. enhancement) needs a diverging,
#: zero-centered colormap in 2D -- spectral_dev_var/novel_freq/novel_period
#: are all non-negative by construction (L2 distance / period / frequency).
_DIVERGING_2D = frozenset({"p_measure", "efficiency_var"})
_SCALAR_ALIASES = {
    "energy_var": "p_measure",
    "novelty_period": "novel_period",
    "novelty_freq": "novel_freq",
}
_ALL_DIAGNOSTICS = _MODE_UNIT_DIAGNOSTICS + _TRIAD_DIAGNOSTICS + _SCALAR_DIAGNOSTICS
_KNOWN = frozenset(_ALL_DIAGNOSTICS)

_DIAG_META = {
    'efficiency': ('Efficiency (%)', 'efficiency'),
    'dominant_freq': ('Dominant frequency (cpd)', 'dominant frequency'),
    'dominant_period': ('Dominant period (d)', 'dominant period'),
    'low_frequency_energy': ('Low-frequency power', 'low-frequency energy'),
    'dynamical_phase': (r'$|$precession frequency$|$ (rad/day)', 'dynamical phase'),
    'p_measure': ('P-measure (final, %)', 'P-measure'),
    'efficiency_var': ('Efficiency variation (final, %)', 'efficiency variation'),
    'spectral_dev_var': ('Spectral deviation (final, %)', 'spectral deviation'),
    'novel_freq': ('Novel frequency (final, cpd)', 'novel frequency'),
    'novel_period': ('Novel period (final, d)', 'novel period'),
}


def _normalize_diagnostics(requested, spec) -> list:
    """Alias-normalize, expand 'all', warn-and-skip a scalar diagnostic
    against a sub-triad-less plain triad (undefined without a "with vs.
    without" comparison). A genuinely unknown name raises.

    Returns an ordered, de-duplicated list.
    """
    expanded = []
    for name in requested:
        if name == "all":
            expanded.extend(_ALL_DIAGNOSTICS)
        else:
            expanded.append(name)

    result = []
    for name in expanded:
        canonical = _SCALAR_ALIASES.get(name, name)
        if canonical not in _KNOWN:
            raise ValueError(f"unknown sweep diagnostic {name!r} -- must be one of "
                              f"{sorted(_KNOWN | {'all'})}")
        if canonical in _SCALAR_DIAGNOSTICS and not spec.has_subtriads():
            print(f"WARNING: diagnostic {canonical!r} needs a sub-triad to compare against "
                  f"(a 'with vs. without' comparison) -- {spec.key!r} is a plain triad, skipping.")
            continue
        if canonical not in result:
            result.append(canonical)
    return result


def _sweep_point_config(config: RunConfig, axes, u_values, plot_bundle: bool) -> RunConfig:
    spec = config.wave_set_spec
    velocities = list(spec.velocities)
    for axis, u in zip(axes, u_values):
        velocities[spec.index(axis.mode)] = u
    point_spec = dataclasses.replace(spec, velocities=tuple(velocities))
    # parallel=False: this config is itself dispatched to a worker process
    # by the sweep-level pool below -- a worker spawning its own child
    # pool would raise (daemonic processes can't have children).
    return RunConfig.from_wave_set(
        point_spec, tf_days=config.tf_days, h=config.h, output_root=config.output_root,
        plot=plot_bundle, parallel=False)


def _run_sweep_point(args) -> dict:
    """Module-level (picklable) worker: integrate + cache one grid point
    (run_dynamics, no duplicated integration) and compute every
    diagnostic from the SAME results dict run_dynamics.py --diagnostics
    uses. Returns a small, no-trajectory summary for the sweep's own
    aggregation; optionally also writes that point's own full
    --diagnostics-equivalent file bundle (evolution figure + diag_*.csv +
    novelty figures) under its own outputs/dynamics/<key>/<run_label>/.

    Used for both 1D and 2D sweeps -- a grid point is a grid point,
    dimensionality only affects how many velocities `config` varies.
    """
    config, plot_bundle = args
    spec = config.wave_set_spec
    results = run_dynamics(config)
    report = compute_diagnostics_report(results, spec)

    if plot_bundle:
        run_dir = results['full']['run_dir']
        run_label = os.path.basename(run_dir)
        write_diagnostics_files(results, report, run_dir, run_label, spec, diagnostics=True)

    per_mode_unit = {
        unit: {lbl: {'efficiency': 100 * m['efficiency'], 'period_global': m['period_global'],
                      'freq_global_cpd': m['freq_global_cpd'], 'low_freq_power': m['low_freq_power']}
               for lbl, m in per_mode.items()}
        for unit, per_mode in report['per_mode_unit'].items()
    }
    precession_freq = dict(results['full']['precession_freq'])
    final = {
        d['mode']: {
            'p_measure': d['p_measure_final_pct'], 'efficiency_var': d['efficiency_var_final_pct'],
            'spectral_dev_var': d['spectral_dev_final_pct'],
            'novel_period': d['novelty_period_final_days'], 'novel_freq': d['novelty_freq_final_cpd'],
        }
        for d in report['final']
    }
    return {'per_mode_unit': per_mode_unit, 'precession_freq': precession_freq, 'final': final}


def _write_sweep_csv(path: str, u_values, series: dict):
    """Long/tidy 1D CSV: one row per (u, series key)."""
    rows = [{'u': float(u), 'series': str(key), 'value': float(v) if np.isfinite(v) else ''}
            for key, values in series.items() for u, v in zip(u_values, values)]
    write_csv(rows, path)


def _write_sweep_csv_2d(path: str, U1, U2, series: dict):
    """Long/tidy 2D CSV: one row per (u1, u2, series key)."""
    rows = [
        {'u1': float(U1[i, j]), 'u2': float(U2[i, j]), 'series': str(key),
         'value': float(v) if np.isfinite(v) else ''}
        for key, values in series.items() for (i, j), v in np.ndenumerate(values)
    ]
    write_csv(rows, path)


def _run_sweep_1d_diagnostics(config: RunConfig, requested: list, plot_per_point: bool,
                               plot_cfg: dict = None) -> dict:
    """Sweep config.sweep.axes[0], computing/plotting every diagnostic in
    `requested` via one run_dynamics() call per grid point. Writes
    outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png/.csv.
    """
    plot_cfg = plot_cfg or {}
    spec = config.wave_set_spec
    axis = config.sweep.axes[0]
    n_grid = config.sweep.n_grid
    u_values = np.linspace(axis.min, axis.max, n_grid)

    points = [(_sweep_point_config(config, (axis,), (u,), plot_per_point), plot_per_point) for u in u_values]
    with ProcessPoolExecutor(max_workers=config.max_workers or max(1, (os.cpu_count() or 2) // 2)) as ex:
        point_results = list(ex.map(_run_sweep_point, points))

    swept_mode_label = _mode_label(*spec.modes[spec.index(axis.mode)])
    swept_label = mode_fs_label(*spec.modes[spec.index(axis.mode)])
    sweep_label = f"{swept_label}_{axis.min:g}-{axis.max:g}_n{n_grid}_tf{config.tf_days:g}_h{config.h:g}"
    out_dir = os.path.join(config.output_root, "sweep", spec.key)
    os.makedirs(out_dir, exist_ok=True)
    title = plot_cfg.get("title", spec.display_label or spec.key)
    xlabel = plot_cfg.get("xlabel", f"{swept_mode_label} initial velocity (m/s)")

    results = {}
    for name in requested:
        ylabel, short_title = _DIAG_META[name]
        png_path = os.path.join(out_dir, f"sweep_diag_{name}_{sweep_label}.png")
        csv_path = os.path.join(out_dir, f"sweep_diag_{name}_{sweep_label}.csv")

        if name in _MODE_UNIT_DIAGNOSTICS:
            key = _MODE_UNIT_KEYS[name]
            units = list(point_results[0]['per_mode_unit'])
            series = {
                (lbl, unit): np.array([p['per_mode_unit'][unit][lbl][key] for p in point_results])
                for unit in units for lbl in point_results[0]['per_mode_unit'][unit]
            }
            plot_mode_unit_sweep(u_values, series, xlabel, ylabel, f"{title}: {short_title}", png_path)
        elif name in _TRIAD_DIAGNOSTICS:
            triad_labels = list(point_results[0]['precession_freq'])
            series = {
                lbl: np.abs(np.array([p['precession_freq'][lbl] for p in point_results]))
                for lbl in triad_labels
            }
            plot_triad_sweep(u_values, series, xlabel, ylabel, f"{title}: {short_title}", png_path)
        else:  # _SCALAR_DIAGNOSTICS
            mode_labels = list(point_results[0]['final'])
            series = {
                lbl: np.array([p['final'][lbl][name] for p in point_results])
                for lbl in mode_labels
            }
            plot_mode_scalar_sweep(u_values, series, xlabel, ylabel, f"{title}: {short_title}", png_path)

        _write_sweep_csv(csv_path, u_values, series)
        print(f"  figure -> {png_path}")
        print(f"  table -> {csv_path}")
        results[name] = {'u_values': u_values, 'series': series, 'path': png_path, 'csv_path': csv_path}

    return results


def _run_sweep_2d_diagnostics(config: RunConfig, requested: list, plot_per_point: bool,
                               plot_cfg: dict = None) -> dict:
    """Sweep config.sweep.axes (2 entries), computing/plotting every
    diagnostic in `requested` via one run_dynamics() call per grid point
    -- same worker as the 1D sweep, just called over a 2D grid. Writes
    outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.png/.csv,
    one heatmap panel per mode/triad per diagnostic.
    """
    plot_cfg = plot_cfg or {}
    spec = config.wave_set_spec
    axes = config.sweep.axes
    n_grid = config.sweep.n_grid
    u1 = np.linspace(axes[0].min, axes[0].max, n_grid)
    u2 = np.linspace(axes[1].min, axes[1].max, n_grid)
    U1, U2 = np.meshgrid(u1, u2)

    points = [(_sweep_point_config(config, axes, (U1[i, j], U2[i, j]), plot_per_point), plot_per_point)
              for i in range(n_grid) for j in range(n_grid)]
    with ProcessPoolExecutor(max_workers=config.max_workers or max(1, (os.cpu_count() or 2) // 2)) as ex:
        flat_results = list(ex.map(_run_sweep_point, points))
    grid_results = np.empty((n_grid, n_grid), dtype=object)
    for idx, r in enumerate(flat_results):
        grid_results[idx // n_grid, idx % n_grid] = r

    mode1_label = _mode_label(*spec.modes[spec.index(axes[0].mode)])
    mode2_label = _mode_label(*spec.modes[spec.index(axes[1].mode)])
    mode1_fs = mode_fs_label(*spec.modes[spec.index(axes[0].mode)])
    mode2_fs = mode_fs_label(*spec.modes[spec.index(axes[1].mode)])
    sweep_label = (f"{mode1_fs}_{axes[0].min:g}-{axes[0].max:g}_"
                   f"{mode2_fs}_{axes[1].min:g}-{axes[1].max:g}_n{n_grid}_tf{config.tf_days:g}_h{config.h:g}")
    out_dir = os.path.join(config.output_root, "sweep", spec.key)
    os.makedirs(out_dir, exist_ok=True)
    title = plot_cfg.get("title", spec.display_label or spec.key)
    xlabel = plot_cfg.get("xlabel", f"{mode1_label} initial velocity (m/s)")
    ylabel_axis = plot_cfg.get("ylabel", f"{mode2_label} initial velocity (m/s)")

    def _grid_of(extract):
        return np.array([[extract(grid_results[i, j]) for j in range(n_grid)] for i in range(n_grid)])

    results = {}
    for name in requested:
        cbar_label, short_title = _DIAG_META[name]
        png_path = os.path.join(out_dir, f"sweep_diag_{name}_{sweep_label}.png")
        csv_path = os.path.join(out_dir, f"sweep_diag_{name}_{sweep_label}.csv")

        if name in _MODE_UNIT_DIAGNOSTICS:
            key = _MODE_UNIT_KEYS[name]
            mode_labels = list(grid_results[0, 0]['per_mode_unit']['full'])
            series = {lbl: _grid_of(lambda p, lbl=lbl: p['per_mode_unit']['full'][lbl][key])
                      for lbl in mode_labels}
            plot_mode_heatmap_sweep(U1, U2, series, xlabel, ylabel_axis, f"{title}: {short_title}", png_path,
                                     cbar_label=cbar_label)
        elif name in _TRIAD_DIAGNOSTICS:
            triad_labels = list(grid_results[0, 0]['precession_freq'])
            series = {lbl: np.abs(_grid_of(lambda p, lbl=lbl: p['precession_freq'][lbl]))
                      for lbl in triad_labels}
            plot_triad_heatmap_sweep(U1, U2, series, xlabel, ylabel_axis, f"{title}: {short_title}", png_path,
                                      cbar_label=cbar_label)
        else:  # _SCALAR_DIAGNOSTICS
            mode_labels = list(grid_results[0, 0]['final'])
            series = {lbl: _grid_of(lambda p, lbl=lbl: p['final'][lbl][name]) for lbl in mode_labels}
            plot_mode_heatmap_sweep(U1, U2, series, xlabel, ylabel_axis, f"{title}: {short_title}", png_path,
                                     diverging=name in _DIVERGING_2D, cbar_label=cbar_label)

        _write_sweep_csv_2d(csv_path, U1, U2, series)
        print(f"  figure -> {png_path}")
        print(f"  table -> {csv_path}")
        results[name] = {'U1': U1, 'U2': U2, 'series': series, 'path': png_path, 'csv_path': csv_path}

    return results


def run_sweep(config: RunConfig, plot_cfg: dict = None, plot_per_point: bool = True) -> dict:
    """Sweep config.sweep.axes (1 or 2 modes' velocities), computing every
    diagnostic in config.sweep.diagnostics (default: `("efficiency",)` for
    1D, `("p_measure",)` for 2D). Every diagnostic manages its own output
    path (outputs/sweep/<wave_set_key>/sweep_diag_<name>_<sweep_label>.*)
    -- there is no single "the" output figure for a sweep to override; a
    composite figure combining several diagnostics is a separate script's
    job (see examples/figures/paper_figure006_quartet_a_precession.py).

    plot_per_point : whether each grid point also writes its own full
        run_dynamics.py --diagnostics-equivalent file bundle (evolution
        figure + diag_*.csv + novelty figures). False skips all of it --
        only the sweep-level outputs are written.

    Returns a dict keyed by diagnostic name, each an inner dict with
    `u_values`/`series` (1D) or `U1`/`U2`/`series` (2D), plus `path`/`csv_path`.
    """
    plot_cfg = plot_cfg or {}
    spec = config.wave_set_spec
    axes = config.sweep.axes
    if len(axes) not in (1, 2):
        raise ValueError(f"sweep needs 1 or 2 axes, got {len(axes)}")

    default = ("efficiency",) if len(axes) == 1 else ("p_measure",)
    requested = _normalize_diagnostics(config.sweep.diagnostics or default, spec)

    if len(axes) == 1:
        return _run_sweep_1d_diagnostics(config, requested, plot_per_point, plot_cfg)
    return _run_sweep_2d_diagnostics(config, requested, plot_per_point, plot_cfg)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-set", required=True,
                         help="registry role key -- reads sweep/tf_days/h/plot straight from that "
                              "wave_sets_default.yaml entry, no separate config file needed.")
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH, help="used with --wave-set")
    parser.add_argument("--output-root", default="outputs",
                         help="override the config's own output_root (e.g. for an isolated test run)")
    parser.add_argument("--no-plot-per-point", action="store_true",
                         help="skip every per-grid-point file output (evolution figure + full "
                              "--diagnostics-equivalent bundle) -- only the sweep-level outputs "
                              "are written")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    if args.wave_set not in specs:
        parser.error(f"--wave-set {args.wave_set!r} not found in {args.specs!r} "
                     f"(available: {list(specs)})")
    with open(args.specs) as f:
        raw = yaml.safe_load(f)[args.wave_set]
    config = RunConfig.from_registry_entry(args.wave_set, args.specs)
    config = dataclasses.replace(config, output_root=args.output_root)
    plot_cfg = raw.get("plot", {})

    print(f"Running sweep for wave set {config.wave_set_spec.key!r}...")
    run_sweep(config, plot_cfg, plot_per_point=not args.no_plot_per_point)


if __name__ == "__main__":
    main()
