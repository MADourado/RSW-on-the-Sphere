"""Generate the composite figures for paper Sec. 2.2 ("Resonant Triads"):
a 2x2 Rossby-only panel and a 2x2 combined Rossby-gravity panel, each
(efficiency map, energy integration) x (two triads).

Replaces the eight separate legacy panels (`Figures/381.png`-`388.png`)
with two multi-panel figures built from the current triad registry
(`examples/triads_section_2_2.yaml`), so every pixel traces back to
`rsw_sphere.dynamics.dynamic_triads.TRIAD` rather than a hand-run,
undocumented one-off script.

Output convention (see docs/triads.md): everything is written under
`outputs/figures/triads/` in this (the code) repository. Copying the
finished PNGs into `paper-nonlinear-interactions-SWE-sphere/Figures/` is a
separate, explicit step -- this script prints the `cp` commands at the
end; it never performs the copy itself.

QUICK USAGE
-----------
Default run (both composite panels, all four registered triads, current
resolution settings in TRIAD_SETTINGS below):

    python examples/make_section22_figures.py

Fast, coarse look (useful while iterating on wording/layout -- a few
minutes instead of tens of minutes):

    python examples/make_section22_figures.py --n-grid 6 --tf-scale 0.5

Regenerate only one triad's pair (efficiency map + energy integration),
e.g. after editing that triad's velocities in the registry YAML:

    python examples/make_section22_figures.py --triad gravity_catalyst

Full command-line reference: `python examples/make_section22_figures.py --help`.

ADDING OR EDITING A TRIAD
--------------------------
1. Add/edit the entry in `examples/triads_section_2_2.yaml` (modes,
   velocities, h_e) -- see docs/triads.md for the schema. No Python
   changes needed for this step.
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
- `t_f` (integration horizon) must be given in **nondimensional** time,
  not days -- this script converts `tf_days` internally via
  `tf_days * 4 * pi` (`days_from_nondim_time`'s inverse). Getting this
  wrong is easy and silent: too short a horizon truncates the efficiency
  map before a mode has completed even one nonlinear exchange cycle,
  quietly reporting a much lower efficiency than the true maximum. The
  `rossby_near_resonant` triad is the cautionary example: its exchange
  period exceeds 160 days, so the naive default `t_f=100` (nondimensional
  -- under 8 days!) inherited from the dissertation's original
  `Triad_Precession` massively underestimates its efficiency. Always set
  `tf_days` from the triad's own linear/nonlinear period, not a shared
  default across triads of very different timescales.
- `efficiency_sweep`'s `fixed_velocity` (the third, unswept mode's
  initial zonal velocity) defaults to a generic 10 m/s inside
  `rsw_sphere.plotting.triad_efficiency`. This script instead passes
  `spec.velocities[2]` -- the swept triad's OWN registered mode-c
  velocity from the YAML -- because the dissertation's worked examples
  (e.g. Table 2.3's RH->EG flow) were computed at each triad's specific
  fixed velocity (there, 100 m/s), not a generic placeholder. Using the
  wrong fixed velocity is a silent under/over-estimate, not an error --
  this was caught once already (kelvin_rh_flow reported ~6.5% instead of
  the dissertation's ~9.5% until this was fixed). If you add a triad
  whose "fixed" mode should be something other than mode c, adjust the
  `fixed_index=2` default passed to `efficiency_sweep` in `_panel()`
  below (and its `target` if the efficiency of a different mode than
  mode a should be reported).
- Efficiency sweeps are cached to `.npz` **keyed only by triad name**,
  not by any of the parameters above -- if you change `tf_days`,
  `n_grid`, `fixed_velocity`, or anything else for a triad, delete its
  stale cache first (`rm outputs/figures/triads/<key>_sweep.npz`) or the
  old numbers will be silently reused. `--clear-cache` does this for you.
"""
import argparse
import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.triad_specs import DEFAULT_SPECS_PATH, load_triad_specs
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.triad_dynamics import triad_energy_evolution
from rsw_sphere.plotting.triad_efficiency import efficiency_sweep, plot_efficiency_map

OUT_DIR = os.path.join(_ROOT, "outputs", "figures", "triads")

#: Per-triad sweep/integration settings, keyed by the same role-key used in
#: the registry YAML. `tf_days` is chosen to comfortably exceed each
#: triad's own nonlinear exchange period (see CALIBRATION NOTES above);
#: `n_grid`/`h` trade resolution for runtime (see --n-grid/--tf-scale/--h
#: CLI overrides for quick coarse passes without editing this dict).
TRIAD_SETTINGS = {
    "rossby_near_resonant": dict(tf_days=200, n_grid=15, h=0.01,
                                  u1_range=(0, 100), u2_range=(0, 100)),
    "rossby_pump":          dict(tf_days=15,  n_grid=15, h=0.01,
                                  u1_range=(0, 100), u2_range=(0, 100)),
    "kelvin_rh_flow":       dict(tf_days=20,  n_grid=15, h=0.01,
                                  u1_range=(0, 100), u2_range=(0, 100)),
    "gravity_catalyst":     dict(tf_days=60,  n_grid=15, h=0.01,
                                  u1_range=(0, 250), u2_range=(0, 100)),
}

#: Fallback used for any registry triad with no entry above (e.g. a
#: newly-added one) -- deliberately conservative/coarse; tune per-triad
#: settings before trusting the result (see CALIBRATION NOTES).
DEFAULT_SETTINGS = dict(tf_days=30, n_grid=8, h=0.01,
                         u1_range=(0, 100), u2_range=(0, 100))

#: The two composite panels this script builds by default, and which
#: triads (in row order) go into each. Extend this dict (or pass
#: --triad for a one-off) rather than hand-editing main() for new panels.
PANELS = {
    "triad_rossby_only_panel.png": ["rossby_near_resonant", "rossby_pump"],
    "triad_combined_panel.png": ["kelvin_rh_flow", "gravity_catalyst"],
}


def get_settings(key, n_grid=None, h=None, tf_scale=None):
    """Look up TRIAD_SETTINGS[key] (or DEFAULT_SETTINGS, with a warning),
    then apply CLI overrides on top: `n_grid`/`h` replace the tuned value
    outright, `tf_scale` multiplies `tf_days` (rather than replacing it,
    so relative per-triad horizons -- e.g. rossby_near_resonant staying
    much longer than rossby_pump -- are preserved under a quick global
    speedup/slowdown).
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


def _panel(key, spec, settings, ax_eff, ax_energy, clear_cache=False):
    """Fill one (efficiency, energy) row of a composite figure for triad `key`."""
    cache_path = os.path.join(OUT_DIR, f"{key}_sweep.npz")
    if clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    tf_nondim = settings["tf_days"] * 4 * math.pi
    U1, U2, EFF = efficiency_sweep(
        spec.modes, h_e=spec.h_e,
        u1_range=settings["u1_range"], u2_range=settings["u2_range"],
        n_grid=settings["n_grid"], t_f=tf_nondim, h=settings["h"],
        fixed_velocity=spec.velocities[2],  # mode c's own registered velocity,
                                             # NOT efficiency_sweep's generic
                                             # 10 m/s default -- see
                                             # CALIBRATION NOTES above.
        cache_path=cache_path,
        verbose=True, progress_label=key,
    )
    plot_efficiency_map(U1, U2, EFF, title=f"{spec.label} -- efficiency (%)",
                         ax=ax_eff)

    triad_energy_evolution(
        spec.modes, spec.velocities, spec.h_e,
        t0=0, tf_days=settings["tf_days"], h=settings["h"],
        ax=ax_energy,
    )
    ax_energy.set_title(f"{spec.label} -- energy integration")


def make_panel(keys, specs, out_name, n_grid=None, h=None, tf_scale=None,
                clear_cache=False):
    """Build one 2x(len(keys)) composite figure (efficiency + energy
    columns, one row per triad in `keys`) and save it under OUT_DIR.

    `n_grid`/`h`/`tf_scale`: CLI-style overrides forwarded to
    `get_settings` for every triad in `keys` (see its docstring).
    """
    apply_house_style()
    fig, axes = plt.subplots(len(keys), 2, figsize=(12, 4.5 * len(keys)),
                              squeeze=False)
    for row, key in zip(axes, keys):
        print(f"  computing {key} ...")
        settings = get_settings(key, n_grid=n_grid, h=h, tf_scale=tf_scale)
        _panel(key, specs[key], settings, row[0], row[1], clear_cache=clear_cache)
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
    parser.add_argument("--specs", default=DEFAULT_SPECS_PATH,
                         help="path to the triad-registry YAML "
                              f"(default: {DEFAULT_SPECS_PATH})")
    parser.add_argument("--triad", default=None,
                         help="if given, only regenerate this one triad's "
                              "standalone (efficiency, energy) pair "
                              "(<key>_pair.png), instead of the two full "
                              "composite panels.")
    parser.add_argument("--n-grid", type=int, default=None,
                         help="override n_grid for every triad processed "
                              "this run (resolution of the efficiency sweep "
                              "along each axis).")
    parser.add_argument("--tf-scale", type=float, default=None,
                         help="multiply every triad's tf_days by this "
                              "factor this run (e.g. 0.5 for a quick, "
                              "half-horizon look).")
    parser.add_argument("--h", type=float, default=None,
                         help="override the RK33 step size (nondimensional "
                              "time) for every triad processed this run.")
    parser.add_argument("--clear-cache", action="store_true",
                         help="delete each processed triad's .npz sweep "
                              "cache before running, forcing a fresh "
                              "computation (see CALIBRATION NOTES: caches "
                              "are keyed by triad name only, not by "
                              "parameters, so this is required after "
                              "changing tf_days/n_grid/fixed_velocity/etc. "
                              "for an already-cached triad).")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    specs = load_triad_specs(args.specs)

    if args.triad:
        if args.triad not in specs:
            parser.error(f"'{args.triad}' not found in {args.specs}. "
                          f"Available keys: {', '.join(specs)}")
        print(f"Single triad: {args.triad}")
        p = make_panel([args.triad], specs, f"{args.triad}_pair.png",
                        n_grid=args.n_grid, h=args.h, tf_scale=args.tf_scale,
                        clear_cache=args.clear_cache)
        print(f"\nDone. To copy into the paper repo (manual, explicit step):")
        print(f"  cp {p} <paper-repo>/Figures/")
        return

    outputs = []
    for out_name, keys in PANELS.items():
        print(f"Panel {out_name} ({', '.join(keys)}):")
        outputs.append(
            make_panel(keys, specs, out_name,
                       n_grid=args.n_grid, h=args.h, tf_scale=args.tf_scale,
                       clear_cache=args.clear_cache))

    paper_figs = os.path.join(
        _ROOT, "..", "paper-nonlinear-interactions-SWE-sphere", "Figures")
    print("\nDone. To copy into the paper repo (manual, explicit step):")
    for p in outputs:
        print(f"  cp {p} {paper_figs}/")


if __name__ == "__main__":
    main()
