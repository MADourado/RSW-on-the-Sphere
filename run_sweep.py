"""General sweep driver: read a YAML describing which wave set, which
diagnostic, and which parameter(s) to sweep over what range, and produce a
cached ``.npz`` + a figure -- replacing the need for a new bespoke
``examples/*.py`` script every time someone wants a new sweep combination.

Dispatches over EXISTING sweep functions rather than reimplementing their
math: ``p_measure_sweep``/``plot_p_measure_map``,
``wave_set_diagnostics_sweep``/``plot_p_measure_map``/
``plot_filtering_error_map`` (``rsw_sphere.plotting.wave_set_pmeasure``),
``efficiency_sweep``/``plot_efficiency_map``
(``rsw_sphere.plotting.triad_efficiency``), and
``precession_frequency_efficiency``/``plot_dual_axis_frequency_efficiency``
(``rsw_sphere.plotting.wave_set_precession``).

Config schema (``diagnostic: precession`` -- a 1D sweep of one mode's
velocity):

    wave_set: quartet_rh_preference     # registry key
    specs_path: examples/wave_sets_section_3.yaml   # optional
    diagnostic: precession
    sweep:
      mode: d                           # mode key to sweep
      values: {min: 10.0, max: 150.0, n: 45}
    target_mode: c                      # optional, efficiency panel
    plot_triad: 0                       # optional, which triad to draw
    tf_days: 150.0                      # optional, overrides registry settings
    h: 0.01                             # optional
    plot:
      xlabel: "RH(3,6) driving velocity $u$ (m/s)"
      title: "Quartet A (this paper)"
      plot_u_max: 120.0                 # optional plot-only crop
    output: outputs/figures/quartet_a_rh36_precession.png

``diagnostic: p_measure`` / ``diagnostic: efficiency`` are 2D sweeps (two
swept mode velocities over a grid); their own schema instead uses
``sweep.swept: [key1, key2]`` and ``sweep.n_grid`` -- see
``run_p_measure``/``run_efficiency`` below and ``docs/wave_sets.md`` §5/
``docs/triads.md`` for the underlying functions' own conventions.

``diagnostic: quartet_diagnostics`` is also a 2D sweep, but computes
*several* per-target diagnostics from one shared pass (the "switch" is
``sweep.diagnostics: [p_measure, filtering_error]`` -- any subset), one
row per diagnostic and one column per target in the output panel.
``sweep.swept``/``target_mode`` default to the wave set's own "private"
modes (``WaveSetSpec.shared_and_private_modes()`` -- a mode private to
exactly one constituent triad), so an ordinary quartet needs no config
beyond ``wave_set`` + ``output``:

    wave_set: quartet_gravity_kelvin
    diagnostic: quartet_diagnostics
    sweep:
      diagnostics: [p_measure, filtering_error]   # optional, this is the default
      n_grid: 10
    output: outputs/figures/wave_sets/quartet_gravity_kelvin_diagnostics.png

See ``run_quartet_diagnostics`` below.

Run:

    python run_sweep.py --config examples/sweep_quartet_a_rh36.yaml
"""
import argparse
import os

import numpy as np
import yaml

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.plotting.sweeps import wave_set_cache_key_hash, cache_key_hash
from rsw_sphere.plotting.wave_set_pmeasure import (
    p_measure_sweep, plot_p_measure_map, wave_set_diagnostics_sweep, plot_filtering_error_map)
from rsw_sphere.plotting.triad_efficiency import efficiency_sweep, plot_efficiency_map, default_velocity_range
from rsw_sphere.plotting.wave_set_precession import (
    precession_frequency_efficiency, plot_dual_axis_frequency_efficiency)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_precession(spec, cfg):
    sweep = cfg["sweep"]
    values = sweep["values"]
    u_values = np.linspace(values["min"], values["max"], int(values["n"]))
    target_mode = cfg.get("target_mode")
    individual_modes = cfg.get("individual_modes")

    sweep_cache_path = cfg.get("cache") or (
        os.path.splitext(cfg["output"])[0] + "_sweep.npz")

    result = precession_frequency_efficiency(
        spec, sweep["mode"], u_values, target_mode_key=target_mode,
        individual_mode_keys=individual_modes,
        tf_days=cfg.get("tf_days"), h=cfg.get("h"), sweep_cache_path=sweep_cache_path)

    plot_cfg = cfg.get("plot", {})
    plot_dual_axis_frequency_efficiency(
        result, spec, plot_triad=cfg.get("plot_triad"),
        xlabel=plot_cfg.get("xlabel", ""), title=plot_cfg.get("title", ""),
        plot_u_max=plot_cfg.get("plot_u_max"), path=cfg["output"])

    min_freq = {lbl: float(np.min(np.abs(v))) for lbl, v in result["freq_by_triad"].items()}
    print(f"  min |precession_freq| per triad: {min_freq}")
    return result


def run_p_measure(spec, cfg):
    sweep = cfg["sweep"]
    swept_keys = sweep["swept"]
    swept_indices = tuple(spec.index(k) for k in swept_keys)
    target_keys = cfg["target_mode"] if isinstance(cfg.get("target_mode"), list) else [cfg["target_mode"]]
    target_indices = [spec.index(k) for k in target_keys]
    n_grid = int(sweep.get("n_grid", 40))
    tf_days = cfg.get("tf_days", spec.settings.get("tf_days", 10))
    h = cfg.get("h", spec.settings.get("h", 0.01))
    fixed_u = cfg.get("fixed", 30.0)

    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    fixed_velocities = {i: fixed_u for i in range(spec.n_modes())
                         if i not in swept_indices}

    cache_hash = wave_set_cache_key_hash(
        spec.modes, triad_indices, spec.h_e, swept_indices, fixed_velocities,
        target_indices, spec.reference_triad, n_grid, tf_days * 4 * np.pi, h)
    cache_path = cfg.get("cache") or f"outputs/figures/wave_sets/{spec.key}_pmeasure_{cache_hash}.npz"

    result = p_measure_sweep(
        spec.modes, triad_indices, spec.h_e, swept_indices, fixed_velocities,
        target_indices, reference_triad=spec.reference_triad, n_grid=n_grid,
        tf_days=tf_days, h=h, cache_path=cache_path, verbose=True, progress_label=spec.key)

    plot_cfg = cfg.get("plot", {})
    n_targets = len(target_indices)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    axes = [axes] if n_targets == 1 else list(axes)
    for k, ax in enumerate(axes):
        plot_p_measure_map(result["U1"], result["U2"], result["P"][..., k],
                            xlabel=plot_cfg.get("xlabel"), ylabel=plot_cfg.get("ylabel"),
                            title=f"P: {result['labels'][k]}", ax=ax)
    fig.tight_layout()
    fig.savefig(cfg["output"], dpi=200, bbox_inches="tight")
    plt.close(fig)
    return result


_DIAGNOSTIC_PLOT_FNS = {"p_measure": ("P", plot_p_measure_map), "filtering_error": ("F2", plot_filtering_error_map)}


def run_quartet_diagnostics(spec, cfg):
    sweep = cfg.get("sweep", {})
    diagnostics = tuple(sweep.get("diagnostics", ("p_measure", "filtering_error")))
    shared, private = spec.shared_and_private_modes()

    if sweep.get("swept"):
        swept_indices = tuple(spec.index(k) for k in sweep["swept"])
    else:
        if len(private) != 2:
            raise ValueError(
                f"{spec.key!r} has {len(private)} private mode(s) -- "
                "quartet_diagnostics only supports the 2-private-mode "
                "(ordinary quartet) case without an explicit sweep.swept; "
                "a 3D sweep for quintets is not yet implemented.")
        swept_indices = tuple(private)

    if cfg.get("target_mode"):
        target_keys = cfg["target_mode"] if isinstance(cfg["target_mode"], list) else [cfg["target_mode"]]
        target_indices = [spec.index(k) for k in target_keys]
    else:
        target_indices = list(private) if private else list(swept_indices)

    n_grid = int(sweep.get("n_grid", spec.settings.get("n_grid", 10)))
    tf_days = cfg.get("tf_days", spec.settings.get("tf_days", 10))
    h = cfg.get("h", spec.settings.get("h", 0.01))
    t_f = tf_days * 4 * np.pi

    if "fixed" in cfg:
        fixed_velocities = {i: cfg["fixed"] for i in range(spec.n_modes()) if i not in swept_indices}
    else:
        fixed_velocities = {i: spec.velocities[i] for i in range(spec.n_modes())
                             if i not in swept_indices}

    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    cache_hash = wave_set_cache_key_hash(
        spec.modes, triad_indices, spec.h_e, swept_indices, fixed_velocities,
        target_indices, spec.reference_triad, n_grid, t_f, h)
    diag_tag = "-".join(sorted(diagnostics))
    cache_path = cfg.get("cache") or (
        f"outputs/figures/wave_sets/{spec.key}_diagnostics_{diag_tag}_{cache_hash}.npz")

    result = wave_set_diagnostics_sweep(
        spec.modes, triad_indices, spec.h_e, swept_indices, fixed_velocities,
        target_indices, diagnostics=diagnostics, reference_triad=spec.reference_triad,
        n_grid=n_grid, tf_days=tf_days, h=h, cache_path=cache_path, verbose=True,
        progress_label=spec.key)

    plot_cfg = cfg.get("plot", {})
    n_targets = len(target_indices)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(diagnostics), n_targets,
                              figsize=(5.5 * n_targets, 4.6 * len(diagnostics)))
    axes = np.atleast_2d(axes)
    label1 = _mode_label_for(spec, swept_indices[0])
    label2 = _mode_label_for(spec, swept_indices[1])
    for row, diag in enumerate(diagnostics):
        array_key, plot_fn = _DIAGNOSTIC_PLOT_FNS[diag]
        values = result[array_key]
        for col, tgt_label in enumerate(result["labels"]):
            plot_fn(result["U1"], result["U2"], values[..., col],
                    xlabel=plot_cfg.get("xlabel", f"{label1} u (m/s)"),
                    ylabel=plot_cfg.get("ylabel", f"{label2} u (m/s)"),
                    title=f"{diag}: {tgt_label}", ax=axes[row, col])
    fig.suptitle(plot_cfg.get("title", spec.display_label), fontsize=12)
    fig.tight_layout()
    fig.savefig(cfg["output"], dpi=200, bbox_inches="tight")
    plt.close(fig)

    for diag in diagnostics:
        array_key, _ = _DIAGNOSTIC_PLOT_FNS[diag]
        finite = result[array_key][np.isfinite(result[array_key])]
        if finite.size:
            print(f"  [{diag}] range over finite grid points: "
                  f"[{finite.min():.4g}, {finite.max():.4g}] "
                  f"({finite.size}/{result[array_key].size} finite)")
    return result


def _mode_label_for(spec, idx):
    from rsw_sphere.plotting.labels import _mode_label
    return _mode_label(*spec.modes[idx])


def run_efficiency(spec, cfg):
    if spec.n_modes() != 3:
        raise ValueError("diagnostic 'efficiency' wraps triad_efficiency.efficiency_sweep, "
                          f"which requires exactly 3 modes -- {spec.key!r} has {spec.n_modes()}")
    sweep = cfg["sweep"]
    target_key = cfg["target_mode"]
    target_idx = spec.index(target_key)
    n_grid = int(sweep.get("n_grid", 40))
    tf_days = cfg.get("tf_days", spec.settings.get("tf_days", 100 / (4 * np.pi)))
    h = cfg.get("h", spec.settings.get("h", 0.001))
    t_f = tf_days * 4 * np.pi

    swept_indices = [i for i in range(3) if i != target_idx]
    u1_range = default_velocity_range(spec.modes[swept_indices[0]][2])
    u2_range = default_velocity_range(spec.modes[swept_indices[1]][2])
    cache_hash = cache_key_hash(spec.modes, spec.h_e, target_idx, target_idx, 0.0,
                                 u1_range, u2_range, n_grid, t_f, h)
    cache_path = cfg.get("cache") or f"outputs/figures/triads/{spec.key}_target{target_idx}_{cache_hash}_sweep.npz"

    U1, U2, EFF = efficiency_sweep(
        list(spec.modes), h_e=spec.h_e, target=target_idx, n_grid=n_grid,
        t_f=t_f, h=h, cache_path=cache_path, verbose=True, progress_label=spec.key)

    plot_cfg = cfg.get("plot", {})
    plot_efficiency_map(U1, U2, EFF, modes=list(spec.modes), target=target_idx,
                         display_label=spec.display_label, title=plot_cfg.get("title"),
                         path=cfg["output"])
    print(f"  efficiency max: {100 * EFF.max():.2f}%")
    return {"U1": U1, "U2": U2, "EFF": EFF}


DIAGNOSTICS = {
    "precession": run_precession,
    "p_measure": run_p_measure,
    "quartet_diagnostics": run_quartet_diagnostics,
    "efficiency": run_efficiency,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="path to a sweep YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    specs_path = cfg.get("specs_path", DEFAULT_WAVESETS_PATH)
    specs = load_wave_set_specs(specs_path)
    spec = specs[cfg["wave_set"]]

    diagnostic = cfg["diagnostic"]
    if diagnostic not in DIAGNOSTICS:
        parser.error(f"unknown diagnostic {diagnostic!r} -- must be one of {list(DIAGNOSTICS)}")

    output_dir = os.path.dirname(cfg["output"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Running {diagnostic!r} sweep for wave set {cfg['wave_set']!r}...")
    DIAGNOSTICS[diagnostic](spec, cfg)
    print(f"wrote {os.path.abspath(cfg['output'])}")


if __name__ == "__main__":
    main()
