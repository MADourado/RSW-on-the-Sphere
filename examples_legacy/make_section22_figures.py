"""Generate the composite figures for paper Sec. 2.2 ("Resonant Triads"):
a 2x2 Rossby-only panel and a 3x2 combined Rossby-gravity panel (three
rows: Triad C at target=0, Triad C at target=1, Triad D), each row an
(efficiency map, energy integration) pair.

Built on the unified `WaveSet`/registry system (`wave_sets_default.yaml`):
each §2.2 triad is registered there as a 1-triad wave set (a triad is the
degenerate case), and this script wraps
`rsw_sphere.utilities.functional.functional_diagnostics_sweep`
(`efficiency` diagnostic) + `rsw_sphere.plotting.energy_evolution.wave_set_energy_evolution`
-- the same functions `run_sweep.py`/`run_dynamics.py` call -- rather than
the retired `triad_table.py`/`triad_dynamics.py`/`triad_efficiency.py`
toolchain (see `docs/triads.md`).

Output convention (see docs/triads.md): everything is written under
`outputs/figures/triads/` in this (the code) repository. Copying the
finished PNGs into `paper-nonlinear-interactions-SWE-sphere/Figures/` is a
separate, explicit step -- this script prints the `cp` commands at the
end; it never performs the copy itself.

QUICK USAGE
-----------
Default run (both composite panels, all four registered triads, current
resolution settings in TRIAD_SETTINGS below):

    python examples_legacy/make_section22_figures.py

Fast, coarse look (useful while iterating on wording/layout -- a few
minutes instead of tens of minutes):

    python examples_legacy/make_section22_figures.py --n-grid 6 --tf-scale 0.5

Regenerate only one triad's pair (efficiency map + energy integration),
e.g. after editing that triad's velocities in the registry YAML:

    python examples_legacy/make_section22_figures.py --triad triad_gravity_with_rossby_catalyst

Regenerate one triad's pair for a *specific* target mode (default target
is always mode index 0 -- see TARGET-MODE CONVENTION below):

    python examples_legacy/make_section22_figures.py --triad triad_kelvin_rossby_flow --target 1

TARGET-MODE CONVENTION
-----------------------
The **target mode** is the mode whose efficiency is measured; per the
dissertation's own methodology, it is held at rest while the other two
modes are swept. `spec.modes` index 0 is always the constituent triad's
SUM mode (`WaveSetSpec`'s own convention -- `m_a = m_b + m_c`), which is
*not* generally the same mode the retired §2.2 toolchain used to number
"mode a": for `triad_kelvin_rossby_flow` the sum mode is RH(4,5) (index
0), so EG(1,1)/RH(3,4) -- this triad's two headline targets -- are
indices 1/2, not 0/1. PANELS below uses each triad's own correct index
(verified directly against `load_wave_set_specs()` -- do not assume
index 0 is the intended target without checking). `triad_kelvin_rossby_flow`
(Triad C) appears as two adjacent rows in the combined panel, target=1
(EG(1,1), RH-to-gravity direction) and target=2 (RH(3,4), its pump mode,
gravity-to-Rossby direction), to illustrate that reassigning the target
on the *same* triad reverses the net transfer direction.

ADDING OR EDITING A TRIAD
--------------------------
1. Add/edit the entry in `wave_sets_default.yaml` (modes, velocities,
   h_e) as a 1-triad wave set -- see docs/wave_sets.md for the schema. No
   Python changes needed for this step.
2. If the triad's default settings below (grid resolution, integration
   horizon) don't suit it, add an entry to TRIAD_SETTINGS keyed by the
   same role-key used in the YAML. If you skip this, DEFAULT_SETTINGS is
   used and a warning is printed -- fine for a first look, but the
   integration horizon in particular should be tuned to comfortably
   exceed the triad's own nonlinear exchange period (see "CALIBRATION
   NOTES" below), or the efficiency map will silently under-report.
3. Regenerate with `--triad <your_new_key>` first to sanity-check it in
   isolation before adding it to one of the two composite panels in
   `main()`.

CALIBRATION NOTES (read before trusting a new/edited triad's numbers)
-----------------------------------------------------------------------
- `tf_days` must comfortably exceed the triad's own nonlinear exchange
  period, or the efficiency map silently under-reports (too short a
  horizon truncates the sweep before a mode completes even one exchange
  cycle). `triad_rossby_only_near_resonant` is the cautionary example:
  its exchange period exceeds 160 days.
- Sweep velocity ranges are derived automatically from each swept mode's
  own family (RH: 0-100 m/s, EG/WG: 0-50 m/s,
  `rsw_sphere.utilities.efficiency.default_velocity_range`). The target
  mode is held at rest (`fixed_velocities={target: 0.0}`), matching the
  dissertation's methodology.
- Efficiency-sweep caches are keyed by a hash of every parameter that
  changes the result (`rsw_sphere.plotting.sweeps.wave_set_cache_key_hash`),
  not by triad name alone -- changing `tf_days`/`n_grid`/`target`/etc. for
  a triad automatically computes a fresh cache file. `--clear-cache`
  deletes the specific cache file(s) this run would otherwise reuse.
"""
import argparse
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.energy_evolution import wave_set_energy_evolution
from rsw_sphere.plotting.functional_map import plot_efficiency_map
from rsw_sphere.plotting.sweeps import wave_set_cache_key_hash
from rsw_sphere.utilities.functional import functional_diagnostics_sweep
from rsw_sphere.utilities.efficiency import default_velocity_range

OUT_DIR = os.path.join(_ROOT, "outputs", "figures", "triads")

#: Per-triad sweep/integration settings, keyed by the same role-key used in
#: the registry YAML. `tf_days` is chosen to comfortably exceed each
#: triad's own nonlinear exchange period (see CALIBRATION NOTES above);
#: `n_grid`/`h` trade resolution for runtime (see --n-grid/--tf-scale/--h
#: CLI overrides for quick coarse passes without editing this dict).
TRIAD_SETTINGS = {
    "triad_rossby_only_near_resonant": dict(tf_days=200, n_grid=15, h=0.01),
    "triad_rossby_only_non_resonant":  dict(tf_days=15,  n_grid=15, h=0.01),
    "triad_kelvin_rossby_flow":        dict(tf_days=20,  n_grid=15, h=0.01),
    "triad_gravity_with_rossby_catalyst": dict(tf_days=60, n_grid=15, h=0.01),
}

#: Fallback used for any registry triad with no entry above (e.g. a
#: newly-added one) -- deliberately conservative/coarse; tune per-triad
#: settings before trusting the result (see CALIBRATION NOTES).
DEFAULT_SETTINGS = dict(tf_days=30, n_grid=8, h=0.01)

#: The two composite panels this script builds by default: each entry is
#: out_name -> list of (role_key, target_index) rows (efficiency map +
#: energy integration per row). Extend this dict (or pass --triad for a
#: one-off) rather than hand-editing main() for new panels. Target indices
#: are into `spec.modes` (index 0 = sum mode) -- see TARGET-MODE CONVENTION
#: above; verified against `load_wave_set_specs()` directly, not assumed.
PANELS = {
    "triad_rossby_only_panel.png": [
        ("triad_rossby_only_near_resonant", 1),   # RH(3,10)
        ("triad_rossby_only_non_resonant", 1),    # RH(3,4)
    ],
    "triad_combined_panel.png": [
        ("triad_kelvin_rossby_flow", 1),   # EG(1,1)
        ("triad_kelvin_rossby_flow", 2),   # RH(3,4)
        # Triad D: index 0=EG(7,9), 1=EG(6,9) (target), 2=RH(1,7). With
        # the target forced to rest, only RH(1,7)'s registered 20 m/s
        # would otherwise carry energy, and its coupling is too weak to
        # move it -- explicit override puts the energy on EG(7,9) instead
        # so the row reproduces the near-total EG<->EG exchange its
        # caption describes.
        ("triad_gravity_with_rossby_catalyst", 1, (50.0, 0.0, 20.0)),
    ],
}


def _mode_tag(m, n, alpha):
    """Filesystem-safe short tag for a mode, e.g. (3, 4, 3) -> 'RH_3_4'."""
    family = {1: 'EG', 2: 'WG', 3: 'RH'}[alpha]
    return f"{family}_{m}_{n}"


def get_settings(key, n_grid=None, h=None, tf_scale=None):
    """Look up TRIAD_SETTINGS[key] (or DEFAULT_SETTINGS, with a warning),
    then apply CLI overrides on top: `n_grid`/`h` replace the tuned value
    outright, `tf_scale` multiplies `tf_days`.
    """
    if key in TRIAD_SETTINGS:
        settings = dict(TRIAD_SETTINGS[key])
    else:
        print(f"  WARNING: no tuned settings for '{key}' in TRIAD_SETTINGS; "
              f"using DEFAULT_SETTINGS ({DEFAULT_SETTINGS}). See the "
              f"CALIBRATION NOTES in this script's module docstring before "
              f"trusting the result -- tf_days in particular needs tuning "
              f"per-triad.")
        settings = dict(DEFAULT_SETTINGS)
    if n_grid is not None:
        settings["n_grid"] = n_grid
    if h is not None:
        settings["h"] = h
    if tf_scale is not None:
        settings["tf_days"] = settings["tf_days"] * tf_scale
    return settings


def _panel(key, spec, settings, ax_eff, ax_energy, target=0, clear_cache=False,
           energy_velocities=None):
    """Fill one (efficiency, energy) row of a composite figure for triad
    `key`, target mode index `target` (default 0, held at rest -- see
    TARGET-MODE CONVENTION in the module docstring).

    `energy_velocities`: if given, overrides the (target-zeroed) default
    entirely for the energy-integration panel only.
    """
    triads = [spec.triad_indices(0)]
    swept = [i for i in range(3) if i != target]
    idx1, idx2 = swept
    u1_range = default_velocity_range(spec.modes[idx1][2])
    u2_range = default_velocity_range(spec.modes[idx2][2])

    cache_hash = wave_set_cache_key_hash(
        spec.modes, triads, spec.h_e, (idx1, idx2), {target: 0.0}, [target],
        0, settings["n_grid"], settings["tf_days"] * 4 * math.pi, settings["h"])
    cache_path = os.path.join(OUT_DIR, f"{key}_target{target}_{cache_hash}_sweep.npz")
    if clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    result = functional_diagnostics_sweep(
        spec.modes, triads, spec.h_e, (idx1, idx2), {target: 0.0}, [target],
        diagnostics=("efficiency",), u1_range=u1_range, u2_range=u2_range,
        n_grid=settings["n_grid"], tf_days=settings["tf_days"], h=settings["h"],
        cache_path=cache_path, verbose=True, progress_label=key,
    )
    label1 = _mode_label(*spec.modes[idx1])
    label2 = _mode_label(*spec.modes[idx2])
    target_label = _mode_label(*spec.modes[target])
    plot_efficiency_map(
        result['U1'], result['U2'], result['Efficiency'][..., 0],
        xlabel=f'{label1} - zonal velocity (m/s)',
        ylabel=f'{label2} - zonal velocity (m/s)',
        title=f'{spec.display_label}: target {target_label} -- efficiency',
        ax=ax_eff)

    if energy_velocities is not None:
        velocities = energy_velocities
    else:
        velocities = list(spec.velocities)
        velocities[target] = 0.0
    energy_result = wave_set_energy_evolution(
        spec.modes, triads, velocities, h_e=spec.h_e,
        tf_days=settings["tf_days"], h=settings["h"],
        highlight=target, ax=ax_energy,
    )
    ax_energy.set_title(f"{spec.display_label}: {spec.label} -- energy integration")
    return result, energy_result


def make_panel(rows_spec, specs, out_name, n_grid=None, h=None, tf_scale=None,
                clear_cache=False):
    """Build one 2x(len(rows_spec)) composite figure (efficiency + energy
    columns, one row per (role_key, target) in `rows_spec`) and save it
    under OUT_DIR.
    """
    apply_house_style()
    fig, axes = plt.subplots(len(rows_spec), 2, figsize=(12, 4.5 * len(rows_spec)),
                              squeeze=False)
    for row, row_spec in zip(axes, rows_spec):
        key, target = row_spec[0], row_spec[1]
        energy_velocities = row_spec[2] if len(row_spec) > 2 else None
        print(f"  computing {key} (target={target}) ...")
        settings = get_settings(key, n_grid=n_grid, h=h, tf_scale=tf_scale)
        _panel(key, specs[key], settings, row[0], row[1], target=target,
               clear_cache=clear_cache, energy_velocities=energy_velocities)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def make_single_pair(key, spec, out_name, target=0, n_grid=None, h=None,
                      tf_scale=None, clear_cache=False):
    """Build a standalone 1x2 (efficiency, energy) figure for one triad at
    one target-mode index and save it under OUT_DIR.
    """
    apply_house_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    settings = get_settings(key, n_grid=n_grid, h=h, tf_scale=tf_scale)
    _panel(key, spec, settings, axes[0][0], axes[0][1], target=target,
           clear_cache=clear_cache)
    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH,
                         help="path to the wave-set registry YAML "
                              f"(default: {DEFAULT_WAVESETS_PATH})")
    parser.add_argument("--triad", default=None,
                         help="if given, only regenerate this one triad's "
                              "standalone (efficiency, energy) pair "
                              "(<key>_target<N>_pair.png), instead of the "
                              "two full composite panels.")
    parser.add_argument("--target", type=int, default=0, choices=[0, 1, 2],
                         help="target-mode index (0/1/2 = a/b/c) for the "
                              "single-triad --triad run; ignored otherwise "
                              "(composite panels always use target=0 for "
                              "every triad -- see TARGET-MODE CONVENTION "
                              "in the module docstring). Default: 0.")
    parser.add_argument("--n-grid", type=int, default=None,
                         help="override n_grid for every triad processed "
                              "this run (resolution of the efficiency sweep "
                              "along each axis).")
    parser.add_argument("--tf-scale", type=float, default=None,
                         help="multiply every triad's tf_days by this "
                              "factor this run (e.g. 0.5 for a quick, "
                              "half-horizon look).")
    parser.add_argument("--h", type=float, default=None,
                         help="override the RK44 step size (nondimensional "
                              "time) for every triad processed this run.")
    parser.add_argument("--clear-cache", action="store_true",
                         help="delete each processed triad/target's .npz "
                              "sweep cache before running, forcing a fresh "
                              "computation. Rarely needed now that caches "
                              "are keyed by a hash of every sweep "
                              "parameter (see CALIBRATION NOTES) -- kept "
                              "as an escape hatch.")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    specs = load_wave_set_specs(args.specs)

    if args.triad:
        if args.triad not in specs:
            parser.error(f"'{args.triad}' not found in {args.specs}. "
                          f"Available keys: {', '.join(specs)}")
        spec = specs[args.triad]
        mode_tag = _mode_tag(*spec.modes[args.target])
        out_name = f"{args.triad}_target_{mode_tag}_pair.png"
        print(f"Single triad: {args.triad} (target={args.target}, {mode_tag})")
        p = make_single_pair(args.triad, spec, out_name, target=args.target,
                              n_grid=args.n_grid, h=args.h, tf_scale=args.tf_scale,
                              clear_cache=args.clear_cache)
        print(f"\nDone. To copy into the paper repo (manual, explicit step):")
        print(f"  cp {p} <paper-repo>/Figures/")
        return

    outputs = []
    for out_name, rows_spec in PANELS.items():
        row_desc = ', '.join(f"{r[0]} (target={r[1]})" for r in rows_spec)
        print(f"Panel {out_name} ({row_desc}):")
        outputs.append(
            make_panel(rows_spec, specs, out_name,
                       n_grid=args.n_grid, h=args.h, tf_scale=args.tf_scale,
                       clear_cache=args.clear_cache))

    paper_figs = os.path.join(
        _ROOT, "..", "paper-nonlinear-interactions-SWE-sphere", "Figures")
    print("\nDone. To copy into the paper repo (manual, explicit step):")
    for p in outputs:
        print(f"  cp {p} {paper_figs}/")


if __name__ == "__main__":
    main()
