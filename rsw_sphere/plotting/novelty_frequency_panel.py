"""Figures for ``rsw_sphere.utilities.novelty_frequency``'s novelty-
frequency detection: ONE figure per target mode (2026-08-26 request),
combining every containing sub-triad into the same plot rather than a
separate file per (target, sub-triad) pair -- a private mode's figure
has one sub-triad curve, a mode shared across N triads has N.

Each figure shows, on one axis (all curves on the same peak-normalized
0-1ish scale):
- the full wave set's own spectrum (black, solid, strong)
- each containing sub-triad's own spectrum (colored, a distinct dash
  style per sub-triad)
- the excluded region (shaded, union of every sub-triad's own dominant
  peak window; legended last since it's a visual aid, not a data curve)

The difference spectrum itself (full vs. the BEST explanation any single
sub-triad offers -- see ``periods.novel_frequency_content_multi``) and
detected novel peaks are NOT drawn on this figure (dropped 2026-08-28):
the exclusion band's edges can produce a spurious "peak" right at the
boundary (a real, still-declining difference clamped flat inside the
excluded window creates an artificial step there), so read the
underlying value off ``diag_final_*.csv``'s ``novelty_period_final_days``
and cross-check it against this same figure by eye instead.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.utilities.novelty_frequency import novelty_combined_for_all_targets
from rsw_sphere.utilities.periods import _power_spectrum

#: One dash style per sub-triad curve (cycled if a mode is ever shared by
#: more triads than this has entries -- a quintet's own 3-triad case
#: still fits).
_SUB_DASH_STYLES = ["--", (0, (4, 1, 1, 1)), "-.", (0, (1, 1))]
_SUB_COLORS = ["tab:blue", "tab:green", "tab:purple", "tab:cyan"]

#: y-axis floor (log scale) -- peak-normalized power realistically bottoms
#: out around 1e-4 for these spectra (checked against real data); a log
#: axis can't show exact 0 anyway, and this keeps weak-but-real content
#: (e.g. the ~10%-relevance case that's invisible on a linear 0-1 axis)
#: visible instead of flattened against the bottom.
_Y_FLOOR = 1e-4


def _filesystem_safe(label: str) -> str:
    return label.replace("(", "").replace(")", "").replace(",", "_")


def novelty_frequency_figure(results: dict, target_label: str, result: dict, path: str = None,
                              xmax: float = None, ax=None):
    """One combined figure for ``target_label``, drawing every sub-triad
    in ``result['sub_names']`` (from ``novelty_combined_for_target``)
    alongside the full wave set and the novelty-detection result.

    ax : if given, draw into it (no new figure, no save) -- mirrors
    ``energy_evolution.plot_energy_evolution``'s own own-fig-vs-provided-ax
    pattern, so this panel can be composed into a multi-panel figure.
    """
    full = results["full"]
    j_full = full["labels"].index(target_label)
    p_full, pow_full = _power_spectrum(full["t"], full["E"][:, j_full])
    pow_full_n = pow_full / pow_full.max() if pow_full.max() > 0 else pow_full

    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.figure

    periods = result['periods_days']
    if xmax is None:
        xmax = periods[-1]  # matches whatever range `result` was actually computed over
    excluded = result['excluded']
    ax.fill_between(periods, _Y_FLOOR, 1, where=excluded, color="lightgray", alpha=0.5,
                     label="excluded (sub-triads' own dominant peaks)")

    for i, sub_name in enumerate(result['sub_names']):
        sub = results[sub_name]
        j_sub = sub["labels"].index(target_label)
        p_sub, pow_sub = _power_spectrum(sub["t"], sub["E"][:, j_sub])
        pow_sub_n = pow_sub / pow_sub.max() if pow_sub.max() > 0 else pow_sub
        ax.plot(p_sub, np.maximum(pow_sub_n, _Y_FLOOR), color=_SUB_COLORS[i % len(_SUB_COLORS)],
                ls=_SUB_DASH_STYLES[i % len(_SUB_DASH_STYLES)], lw=1.3,
                label=f"mode energy in {sub['title']}")

    ax.plot(p_full, np.maximum(pow_full_n, _Y_FLOOR), color="black", lw=2.2,
            label="mode energy in quartet")

    ax.set_xlim(0, xmax)
    ax.set_yscale("log")
    ax.set_ylim(_Y_FLOOR, 1.5)
    ax.set_xlabel("Period (days)")
    ax.set_ylabel("Peak-normalized power (FFT)")
    ax.set_title(f"Energy Spectra for mode: {target_label}")

    # "excluded" is plotted first (so its shading sits behind the curves)
    # but legended last -- it's a visual aid, not a data series, and
    # shouldn't lead the list of what's actually being compared.
    handles, labels = ax.get_legend_handles_labels()
    order = [i for i, lbl in enumerate(labels) if not lbl.startswith("excluded")] + \
            [i for i, lbl in enumerate(labels) if lbl.startswith("excluded")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=7, loc="upper right")

    if own_fig:
        fig.tight_layout()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def novelty_frequency_figures(results: dict, output_dir: str, xmax: float = None,
                               filename_suffix: str = "", **kwargs) -> list:
    """One combined figure per target mode for a full wave-set
    ``results`` dict (``run_dynamics.run_dynamics()``'s own return
    shape). kwargs (min_prominence, exclusion_frac) pass through to
    ``novel_frequency_content_multi`` via ``novelty_combined_for_all_targets``.

    xmax : upper period (days) shown/searched. Default (None):
        `rsw_sphere.utilities.periods._default_xmax` derives it from the
        full run's own tf_days (``round(sqrt(tf_days / 2))``), so a long
        run automatically gets a wide enough window instead of staying
        pinned at a short-run default.

    filename_suffix : appended (with a leading "_") to every filename --
        e.g. the run's own ic_label/tf/h stamp, to match run_dynamics.py's
        other diag_*/evol_* filenames.

    Returns the list of paths written.
    """
    all_results = novelty_combined_for_all_targets(results, xmax=xmax, **kwargs)
    suffix = f"_{filename_suffix}" if filename_suffix else ""

    paths = []
    for target_label, result in all_results.items():
        fname = f"diag_freq_novel_{_filesystem_safe(target_label)}{suffix}.png"
        path = os.path.join(output_dir, fname)
        novelty_frequency_figure(results, target_label, result, path, xmax=xmax)
        paths.append(path)
    return paths
