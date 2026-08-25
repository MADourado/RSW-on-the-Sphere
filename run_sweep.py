"""General sweep driver: read a YAML describing which wave set, which
diagnostic, and which parameter(s) to sweep over what range, and produce a
cached ``.npz`` + a figure -- replacing the need for a new bespoke
``examples/*.py`` script every time someone wants a new sweep combination.

Dispatches over EXISTING sweep functions rather than reimplementing their
math: ``p_measure_sweep``/``plot_p_measure_map``
(``rsw_sphere.plotting.wave_set_pmeasure``), ``efficiency_sweep``/
``plot_efficiency_map`` (``rsw_sphere.plotting.triad_efficiency``), and
``precession_frequency_efficiency``/``plot_dual_axis_frequency_efficiency``
(``rsw_sphere.plotting.wave_set_precession``, new -- see that module).

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

Run:

    python run_sweep.py --config examples/sweep_quartet_a_rh36.yaml
"""
import argparse
import os

import numpy as np
import yaml

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from rsw_sphere.plotting.sweeps import wave_set_cache_key_hash, cache_key_hash
from rsw_sphere.plotting.wave_set_pmeasure import p_measure_sweep, plot_p_measure_map
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
