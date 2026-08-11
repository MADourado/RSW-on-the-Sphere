"""Generate the composite figures for paper Sec. 2.2 ("Resonant Triads"):
a 2x2 Rossby-only panel and a 3x2 combined Rossby-gravity panel (three
rows: Triad C at target=0, Triad C at target=1, Triad D), each row an
(efficiency map, energy integration) pair.

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

    python examples/make_section22_figures.py --triad triad_gravity_with_rossby_catalyst

Regenerate one triad's pair for a *specific* target mode (default target
is always mode index 0 -- see TARGET-MODE CONVENTION below):

    python examples/make_section22_figures.py --triad triad_kelvin_rossby_flow --target 1

TARGET-MODE CONVENTION
-----------------------
The **target mode** is the mode whose efficiency is measured; per the
dissertation's own methodology (and `triad_efficiency.efficiency_sweep`'s
defaults), it is held at rest while the other two modes are swept. Most
rows in PANELS below use target=mode index 0 ("mode a" in the registry
YAML), which lines up with each registered triad's pump mode or headline
mode already (e.g. Triad A's RH(3,10), Triad D's EG(6,9)).
`triad_kelvin_rossby_flow` (Triad C) is the exception: it appears as two
adjacent rows in the combined panel, target=0 (EG(1,1), RH-to-gravity
direction) and target=1 (RH(3,4), its pump mode, gravity-to-Rossby
direction), to illustrate that reassigning the target on the *same* triad
reverses the net transfer direction -- see the registry YAML's role note
and paper-review item 9.

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
  `triad_rossby_only_near_resonant` triad is the cautionary example: its
  exchange period exceeds 160 days, so the naive default `t_f=100`
  (nondimensional -- under 8 days!) inherited from the dissertation's
  original `Triad_Precession` massively underestimates its efficiency.
  Always set `tf_days` from the triad's own linear/nonlinear period, not a
  shared default across triads of very different timescales.
- Sweep velocity ranges are no longer set per-triad here -- `efficiency_
  sweep` derives them automatically from each swept mode's own family
  (RH: 0-100 m/s, EG/WG: 0-50 m/s; paper-review item 1), so this script no
  longer overrides `u1_range`/`u2_range` at all. The target mode is held
  at rest (`fixed_velocity=0`, matching the dissertation's methodology --
  see `efficiency_sweep`'s docstring and paper-review item 6/known-issue
  3), superseding the older, since-removed convention of holding "mode c"
  fixed at its own registered velocity.
- Efficiency-sweep caches are keyed by a hash of every parameter that
  changes the result (`triad_efficiency.cache_key_hash`), not by triad
  name alone -- so changing `tf_days`/`n_grid`/`target`/etc. for a triad
  automatically computes a fresh cache file rather than silently reusing a
  stale one (this fixes the stale-NaN bug logged in the plan's "Known
  issues" section). `--clear-cache` still works (deletes the specific
  cache file(s) this run would otherwise reuse) but is rarely needed now.
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
from rsw_sphere.plotting.triad_efficiency import (
    efficiency_sweep, plot_efficiency_map, cache_key_hash,
    default_velocity_range,
)

OUT_DIR = os.path.join(_ROOT, "outputs", "figures", "triads")

#: Per-triad sweep/integration settings, keyed by the same role-key used in
#: the registry YAML. `tf_days` is chosen to comfortably exceed each
#: triad's own nonlinear exchange period (see CALIBRATION NOTES above);
#: `n_grid`/`h` trade resolution for runtime (see --n-grid/--tf-scale/--h
#: CLI overrides for quick coarse passes without editing this dict).
#: Velocity ranges are NOT set here -- efficiency_sweep derives them from
#: each swept mode's family (see CALIBRATION NOTES).
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
#: one-off) rather than hand-editing main() for new panels.
#:
#: The combined panel is 3 rows, not 2: `triad_kelvin_rossby_flow` appears
#: twice, at target=0 (EG(1,1), RH-to-gravity direction) and target=1
#: (RH(3,4), its pump mode, gravity-to-Rossby direction) -- placing both
#: directions of the SAME triad as adjacent rows makes the "only the
#: target changed, and the direction reverses" point directly comparable
#: (paper-review item 9 / follow-up, 2026-08-11: originally a separate
#: standalone figure, folded in here since it tells a clearer story next
#: to its target=0 counterpart and avoids a near-empty page).
PANELS = {
    "triad_rossby_only_panel.png": [
        ("triad_rossby_only_near_resonant", 0),
        ("triad_rossby_only_non_resonant", 0),
    ],
    "triad_combined_panel.png": [
        ("triad_kelvin_rossby_flow", 0),
        # target=1 (RH(3,4)) row: with the target now always zeroed by
        # default in _panel, this row's trajectory automatically differs
        # from target=0's row (RH(3,4)=0, EG(1,1)/RH(4,5) at their
        # registered values) rather than replaying it verbatim.
        ("triad_kelvin_rossby_flow", 1),
        # Triad D: the registry loads the *target* EG(6,9) with 50 m/s and
        # leaves the other gravity mode EG(7,9) at 0 -- fine for the
        # target=EG(6,9)-not-necessarily-at-rest story this triad used to
        # tell, but once the target is forced to rest (see _panel), that
        # leaves only RH(1,7) with any energy, and RH(1,7)'s coupling is
        # two orders of magnitude too weak to move it -- a flat, dead
        # trajectory (caught in review, 2026-08-11). Explicit override:
        # keep the target at rest, but put the energy on EG(7,9) instead
        # of RH(1,7), so it actually has somewhere to flow from into the
        # target, reproducing the near-total EG<->EG exchange the row's
        # caption describes.
        ("triad_gravity_with_rossby_catalyst", 0, (0.0, 20.0, 50.0)),
    ],
}


def _mode_tag(m, n, alpha):
    """Filesystem-safe short tag for a mode, e.g. (3, 4, 3) -> 'RH_3_4'."""
    family = {1: 'EG', 2: 'WG', 3: 'RH'}[alpha]
    return f"{family}_{m}_{n}"


def get_settings(key, n_grid=None, h=None, tf_scale=None):
    """Look up TRIAD_SETTINGS[key] (or DEFAULT_SETTINGS, with a warning),
    then apply CLI overrides on top: `n_grid`/`h` replace the tuned value
    outright, `tf_scale` multiplies `tf_days` (rather than replacing it,
    so relative per-triad horizons -- e.g. rossby_only_near_resonant
    staying much longer than rossby_only_non_resonant -- are preserved
    under a quick global speedup/slowdown).
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

    The energy-integration (right) panel's target mode is, by default,
    forced to start at rest too -- **not** just the registered velocity
    from the YAML -- so "target mode" means the same thing (held at rest)
    in both panels of every row. Before this, the registry's own `u` for
    the target mode (nonzero for several triads, e.g. Triad C's EG(1,1) at
    50 m/s, Triad D's EG(6,9) at 50 m/s) silently overrode the trajectory
    panel's target-at-rest starting point while the map next to it still
    enforced it, an inconsistency caught in paper review (2026-08-11).

    `energy_velocities`: if given, overrides the (target-zeroed) default
    entirely for the energy-integration panel only -- e.g. to also pick a
    specific point in the swept plane for the other two modes' ICs, rather
    than their registered values.
    """
    tf_nondim = settings["tf_days"] * 4 * math.pi
    fixed_index = target
    swept = [i for i in range(3) if i != fixed_index]
    u1_range = default_velocity_range(spec.modes[swept[0]][2])
    u2_range = default_velocity_range(spec.modes[swept[1]][2])

    cache_hash = cache_key_hash(
        spec.modes, spec.h_e, target, fixed_index, 0.0,
        u1_range, u2_range, settings["n_grid"], tf_nondim, settings["h"])
    cache_path = os.path.join(OUT_DIR, f"{key}_target{target}_{cache_hash}_sweep.npz")
    if clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    U1, U2, EFF = efficiency_sweep(
        spec.modes, h_e=spec.h_e, target=target, fixed_index=fixed_index,
        fixed_velocity=0.0, u1_range=u1_range, u2_range=u2_range,
        n_grid=settings["n_grid"], t_f=tf_nondim, h=settings["h"],
        cache_path=cache_path,
        verbose=True, progress_label=key,
    )
    plot_efficiency_map(U1, U2, EFF, modes=spec.modes, target=target,
                         display_label=spec.display_label, ax=ax_eff)

    if energy_velocities is not None:
        velocities = energy_velocities
    else:
        velocities = list(spec.velocities)
        velocities[target] = 0.0
    triad_energy_evolution(
        spec.modes, velocities, spec.h_e,
        t0=0, tf_days=settings["tf_days"], h=settings["h"],
        target=target,
        ax=ax_energy,
    )
    ax_energy.set_title(f"{spec.display_label}: {spec.label} -- energy integration")


def make_panel(rows_spec, specs, out_name, n_grid=None, h=None, tf_scale=None,
                clear_cache=False):
    """Build one 2x(len(rows_spec)) composite figure (efficiency + energy
    columns, one row per (role_key, target) in `rows_spec`) and save it
    under OUT_DIR.

    `n_grid`/`h`/`tf_scale`: CLI-style overrides forwarded to
    `get_settings` for every triad in `rows_spec` (see its docstring).
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
    parser.add_argument("--specs", default=DEFAULT_SPECS_PATH,
                         help="path to the triad-registry YAML "
                              f"(default: {DEFAULT_SPECS_PATH})")
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
                         help="override the RK33 step size (nondimensional "
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
    specs = load_triad_specs(args.specs)

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
