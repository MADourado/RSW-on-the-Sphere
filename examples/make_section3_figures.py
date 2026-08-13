"""Generate the composite figures for the paper's merged §Coupled Triads
section (§3 "Multiple Triads" + §4 "Inertia-Gravity Waves" + §5 "Five-Wave
model"): a comparison panel + a period-spectrum panel per registered wave
set, plus a P-measure sweep for the two wave sets whose story is about a
gravity mode's influence on an RH-only triad.

Mirrors `examples/make_section22_figures.py`'s structure (`*_SETTINGS`
dict + `--n-grid`/`--tf-scale`/`--h`/`--clear-cache` CLI overrides), but
for `rsw_sphere.dynamics.wave_sets.WaveSet` / the
`examples/wave_sets_section_3.yaml` registry instead of `TRIAD`.

Output convention: everything is written under `outputs/figures/wave_sets/`
in this repository; copying finished PNGs into
`paper-nonlinear-interactions-SWE-sphere/Figures/` is a separate, explicit
step -- this script prints the `cp` commands at the end.

QUICK USAGE
-----------
Default run (all four registered wave sets, current WAVESET_SETTINGS):

    python examples/make_section3_figures.py

Fast, coarse look (useful while iterating on layout):

    python examples/make_section3_figures.py --n-grid 5 --tf-scale 0.5

Regenerate only one wave set:

    python examples/make_section3_figures.py --wave-set quartet_gravity_kelvin

CALIBRATION NOTES
-----------------
- P-measure sweeps are expensive: a single (swept-axis-1, swept-axis-2)
  sweep costs roughly `n_grid^2` full-wave-set integrations plus, for each
  target mode, up to `n_grid^2` (or `n_grid`, once row-caching applies --
  see `wave_set_pmeasure.p_measure_sweep`'s docstring) sub-triad
  integrations. At the dissertation's own 50x50 resolution this is tens of
  minutes per wave set even before accounting for `WaveSet` vs. `TRIAD`'s
  relative cost. **Default `n_grid` here is deliberately coarse (8)** --
  matching §2.2's own "coarse-first, hi-res is a final separate pass"
  precedent (`PLAN-section-2.2.md`) -- tune up via `--n-grid` once the
  qualitative story for each wave set is confirmed.
- `tf_days` per wave set comes from `examples/wave_sets_section_3.yaml`'s
  own `settings` block (the registry, not this script, is the source of
  truth) -- `WAVESET_SETTINGS` below only adds the P-measure-sweep-specific
  `n_grid`.
"""
import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.wave_set_dynamics import wave_set_comparison_panel_from_spec
from rsw_sphere.plotting.wave_set_periods import wave_set_period_panel
from rsw_sphere.plotting.wave_set_pmeasure import p_measure_sweep, plot_p_measure_map
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.sweeps import wave_set_cache_key_hash

OUT_DIR = os.path.join(_ROOT, "outputs", "figures", "wave_sets")

#: Which wave sets get a P-measure sweep figure, and which two modes are
#: swept (by mode key, per examples/wave_sets_section_3.yaml) -- the
#: gravity-quartet/quintet story is specifically about how a catalyst mode
#: (RH(1,2)) and a gravity mode change the RH-only triad's exchange, so the
#: swept axes are those two, matching the dissertation's own fig: 4ef.
#: `quartet_rh_preference` (all-Rossby) has no gravity mode and is
#: excluded; its own story (RH(3,4) vs RH(3,6) preference) is fully told by
#: the comparison panel alone.
PMEASURE_WAVE_SETS = {
    "quartet_gravity_kelvin": dict(swept=("c", "d"), targets=("a", "b"), n_grid=8),
    "quartet_gravity_79":     dict(swept=("c", "d"), targets=("a", "b"), n_grid=8),
    "quintet_gravity_star":   dict(swept=("c", "d"), targets=("a", "b"), n_grid=6),
}

#: Comparison-panel highlight (solid target / dashed everything else),
#: matching §2.2's own convention (rsw_sphere.plotting.triad_dynamics) --
#: by mode key, per examples/wave_sets_section_3.yaml, resolved to an
#: index via spec.index(). Two forms: a single key (one shared target,
#: e.g. the gravity mode driving an otherwise RH-only quartet -- applied
#: to every sub-panel and the full panel alike), or a dict of
#: {triad_display_label: key_or_None} for wave sets built around
#: comparing several constituent triads' own *different* private members
#: (see wave_set_dynamics.wave_set_comparison_panel's own highlight
#: docstring) -- resolved per-triad by triad_labels below, full panel
#: left all-solid (no single target once sub-panels differ).
HIGHLIGHT_WAVE_SETS = {
    # Triad 1 = {sum:a, members:[b,c]} -- contains c=RH(3,4), not d.
    # Triad 2 = {sum:a, members:[b,d]} -- contains d=RH(3,6), not c.
    # (Confirmed against wave_set_specs.WaveSetSpec.triad_indices(0)/(1)
    # directly -- an earlier version of this dict had these swapped,
    # which silently fell back to all-solid on BOTH panels rather than
    # erroring, since wave_set_comparison_panel treats "highlighted mode
    # absent from this sub-triad" as "no highlight" rather than a
    # mistake -- caught only by looking at the rendered figure.)
    "quartet_rh_preference": {"Triad 1": "c", "Triad 2": "d"},
    "quartet_gravity_kelvin": "d",
    "quartet_gravity_79": "d",
    "quintet_gravity_star": {"Triad 1 (RH-only)": None, "Triad 2 (with EG(1,1))": "d",
                              "Triad 3 (with EG(7,9))": "e"},
}


def resolve_highlight(key, spec):
    """``HIGHLIGHT_WAVE_SETS[key]`` -> ``(highlight, highlight_full)`` args
    for ``wave_set_comparison_panel_from_spec``, or ``(None, None)`` if
    ``key`` isn't registered there.
    """
    cfg = HIGHLIGHT_WAVE_SETS.get(key)
    if cfg is None:
        return None, None
    if isinstance(cfg, str):
        idx = spec.index(cfg)
        return idx, idx
    triad_labels = [t.display_label for t in spec.triads]
    per_triad = [spec.index(cfg[label]) if cfg.get(label) is not None else None
                 for label in triad_labels]
    return per_triad, None


def get_settings(spec, n_grid=None, h=None, tf_scale=None):
    settings = dict(spec.settings) if spec.settings else dict(tf_days=10, h=0.01, n_grid=10)
    if n_grid is not None:
        settings["n_grid"] = n_grid
    if h is not None:
        settings["h"] = h
    if tf_scale is not None:
        settings["tf_days"] = settings.get("tf_days", 10) * tf_scale
    return settings


def make_comparison_and_period_panels(key, spec, settings, clear_cache=False):
    energy_path = os.path.join(OUT_DIR, f"{key}_panel.png")
    print(f"  [{key}] comparison panel -> {energy_path}")
    highlight, highlight_full = resolve_highlight(key, spec)
    results = wave_set_comparison_panel_from_spec(
        spec, tf_days=settings["tf_days"], h=settings["h"],
        highlight=highlight, highlight_full=highlight_full, path=energy_path)
    r_full = results[-1]
    print(f"    drift={r_full['drift']:.3e}, "
          f"dEK={ {l: round(float(d), 6) for l, d in zip(r_full['labels'], r_full['dEK'])} }")

    period_path = os.path.join(OUT_DIR, f"{key}_periods.png")
    print(f"  [{key}] period panel -> {period_path}")
    apply_house_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    period_results = wave_set_period_panel(
        r_full['t'], r_full['E'], r_full['labels'], list(spec.modes), ax=ax)
    ax.set_title(spec.display_label)
    fig.savefig(period_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    for label, pr in zip(r_full['labels'], period_results):
        flag = " [HORIZON-LIMITED]" if pr['horizon_limited'] else ""
        print(f"    {label}: period_global={pr['period_global']:.3g}d, "
              f"period_local_max={pr['period_local_max']}{flag}")

    return results, period_results


def make_pmeasure_panel(key, spec, settings, pconfig, clear_cache=False):
    triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
    swept_indices = tuple(spec.index(k) for k in pconfig["swept"])
    target_indices = [spec.index(k) for k in pconfig["targets"]]
    n_grid = pconfig.get("n_grid", 8)
    tf_days = settings["tf_days"]
    h = settings["h"]
    t_f_nondim = tf_days * 4 * 3.141592653589793

    # Every mode not swept and not targeted needs a fixed velocity -- for
    # every registered gravity wave set, that's exactly the reference
    # triad's sum mode 'a', held at its registered velocity (matches the
    # dissertation's fig: 4ef, which fixes RH(4,5)/RH(3,4) at 30 m/s and
    # sweeps the other two).
    fixed_velocities = {
        i: spec.velocities[i] for i in range(spec.n_modes())
        if i not in swept_indices and i not in target_indices
    }
    for i in target_indices:
        if i not in swept_indices:
            fixed_velocities.setdefault(i, spec.velocities[i])

    cache_hash = wave_set_cache_key_hash(
        spec.modes, triads, spec.h_e, swept_indices, fixed_velocities,
        target_indices, spec.reference_triad, n_grid, t_f_nondim, h)
    cache_path = os.path.join(OUT_DIR, f"{key}_pmeasure_{cache_hash}.npz")
    if clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    print(f"  [{key}] P-measure sweep (n_grid={n_grid}) -> cache {os.path.basename(cache_path)}")
    result = p_measure_sweep(
        spec.modes, triads, spec.h_e, swept_indices, fixed_velocities,
        target_indices, reference_triad=spec.reference_triad,
        n_grid=n_grid, tf_days=tf_days, h=h, cache_path=cache_path,
        verbose=True, progress_label=key)

    label1 = _mode_label(*spec.modes[swept_indices[0]])
    label2 = _mode_label(*spec.modes[swept_indices[1]])

    apply_house_style()
    fig, axes = plt.subplots(1, len(target_indices), figsize=(6 * len(target_indices), 5))
    if len(target_indices) == 1:
        axes = [axes]
    for k, ax in enumerate(axes):
        cs, n_clipped = plot_p_measure_map(
            result['U1'], result['U2'], result['P'][..., k],
            xlabel=f'{label1} - zonal velocity (m/s)',
            ylabel=f'{label2} - zonal velocity (m/s)',
            title=f"P: {result['labels'][k]}", ax=ax)
        if n_clipped:
            print(f"    {result['labels'][k]}: {n_clipped} grid point(s) clipped to +-100%")
    fig.tight_layout()
    pmeasure_path = os.path.join(OUT_DIR, f"{key}_pmeasure.png")
    fig.savefig(pmeasure_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"    drift range: [{result['drift'].min():.3e}, {result['drift'].max():.3e}]")
    return result, pmeasure_path


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", default=None,
                         help="only this one registered wave set (default: all).")
    parser.add_argument("--n-grid", type=int, default=None,
                         help="override P-measure sweep grid resolution for every wave set.")
    parser.add_argument("--h", type=float, default=None,
                         help="override RK33 step size for every wave set.")
    parser.add_argument("--tf-scale", type=float, default=None,
                         help="multiply every wave set's registered tf_days.")
    parser.add_argument("--skip-pmeasure", action="store_true",
                         help="skip the (expensive) P-measure sweeps.")
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    specs = load_wave_set_specs(args.specs)
    if args.wave_set:
        specs = {args.wave_set: specs[args.wave_set]}

    written = []
    for key, spec in specs.items():
        print(f"=== {key} ({spec.display_label}) ===")
        settings = get_settings(spec, n_grid=None, h=args.h, tf_scale=args.tf_scale)
        make_comparison_and_period_panels(key, spec, settings, clear_cache=args.clear_cache)
        written += [f"{key}_panel.png", f"{key}_periods.png"]

        if not args.skip_pmeasure and key in PMEASURE_WAVE_SETS:
            pconfig = dict(PMEASURE_WAVE_SETS[key])
            if args.n_grid is not None:
                pconfig["n_grid"] = args.n_grid
            make_pmeasure_panel(key, spec, settings, pconfig, clear_cache=args.clear_cache)
            written.append(f"{key}_pmeasure.png")
        print()

    print("Done. To copy into the paper repo:")
    for name in written:
        print(f"  cp {os.path.join(OUT_DIR, name)} "
              f"paper-nonlinear-interactions-SWE-sphere/Figures/{name}")


if __name__ == "__main__":
    main()
