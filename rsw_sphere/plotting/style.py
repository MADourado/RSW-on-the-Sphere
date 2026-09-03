"""Shared matplotlib house style, factored out of
``dispersion_relation_fancy.py`` so other plotting modules (the triad
tools, and future ones) can match it without copy-pasting the rcParams
block.
"""
import matplotlib.pyplot as plt

BLUE = '#1a5fa8'   # gravity modes
GREEN = '#2ca02c'  # third/auxiliary series
RED = '#c0392b'    # Rossby-Haurwitz modes
GREY = '0.55'


# ---------------------------------------------------------------------------
# Per-mode global color identity (paper §2.2 review item 5): a given mode
# must be drawn in the same color in every energy-evolution figure, not
# re-assigned per-panel by triad-local index a/b/c. Warm hues (yellow ->
# orange -> red -> dark red/maroon) for Rossby-Haurwitz (RH) modes, cool
# hues (light blue -> blue -> indigo/violet) for inertia-gravity (EG/WG)
# modes, grey reserved for non-modal references (e.g. a total-energy line).
#
# Keyed by the raw ``(m, n, alpha)`` triple (alpha: 1=EIG/EG, 2=WIG/WG,
# 3=RH) rather than a formatted label string, since ``dynamic_triads.label()``
# emits "RH (m,n)" with a space while this module's own ``_mode_label``
# emits "RH(m,n)" without -- the tuple is the one thing both can produce
# identically from what they already hold.
#
# Hue assigned by each mode's *linear* period (fixed, not recomputed at
# plot time): within a family, the fastest (shortest-period) mode gets the
# lightest hue and the slowest gets the darkest, so darker == slower reads
# consistently across figures. EG (eastward-gravity) and WG (westward-
# gravity) get their OWN cool sub-ranges (blue/indigo vs. teal/turquoise)
# rather than sharing one -- they're a different family in every wave set
# that uses both (e.g. quartet_rossby_gravity_influence's WG(3,9)/WG(7,9)),
# not just an eastward/westward relabeling of the same hue. Periods used to
# fix this ordering (computed via ``wave_set_table.wave_set_table``,
# h_e=10000 m, 2026-08-11; WG periods added 2026-08-27):
#   RH(3,4)=3.72d, RH(4,5)=3.99d, RH(1,2)=5.03d, RH(3,6)=7.59d,
#   RH(3,10)=19.0d, RH(1,7)=30.3d
#   EG(6,9)=0.156d, EG(7,9)=0.156d, EG(1,1)=1.35d
#   WG(3,9)=0.152d, WG(7,9)=0.153d, WG(1,1)=0.551d
# Every mode currently used across §2.2/§3/§4 has a fixed entry below; a
# mode not in this dict falls back to GREY (see ``mode_color``) rather than
# erroring, so new triads/wave sets don't break plotting -- extend this
# dict when new modes are added.
MODE_COLORS = {
    (3, 4, 3):  '#f1c40f',   # RH(3,4)  -- fastest RH -> yellow
    (4, 5, 3):  '#e67e22',   # RH(4,5)
    (1, 2, 3):  '#d35400',   # RH(1,2)
    (3, 6, 3):  '#c94615',   # RH(3,6)  -- §3 quartet A's alternative pump mode
    (3, 10, 3): '#c0392b',   # RH(3,10)
    (1, 7, 3):  '#7f1d1d',   # RH(1,7)  -- slowest RH -> dark red/maroon
    (6, 9, 1):  '#5dade2',   # EG(6,9)  -- fastest EG -> light blue
    (7, 9, 1):  '#2e86c1',   # EG(7,9)
    (1, 1, 1):  '#4b3f8f',   # EG(1,1)  -- slowest of the three -> indigo
    (3, 9, 2):  '#76d7c4',   # WG(3,9)  -- fastest WG -> light turquoise
    (7, 9, 2):  '#17a589',   # WG(7,9)  -- nearly the same period as WG(3,9)
                             # (0.152d vs 0.153d) so hue alone can't rely on
                             # that gap -- given a clearly different shade
                             # on purpose
    (1, 1, 2):  '#0b5345',   # WG(1,1)  -- slowest WG -> dark teal
}

#: Color for non-modal reference lines (e.g. total energy).
TOTAL_ENERGY_COLOR = GREY


def mode_color(m, n, alpha, default=GREY):
    """Look up the persistent color for mode ``(m, n, alpha)`` in
    ``MODE_COLORS``, falling back to ``default`` (grey) for any mode not
    yet registered there.
    """
    return MODE_COLORS.get((int(m), int(n), int(alpha)), default)


def add_outward_twin_axis(ax, x, y, marker_style='v-', color=GREEN, ylabel='', label=None, outward=60):
    """Add a third y-axis to a figure that already has ``ax`` and one
    ``ax.twinx()`` -- spine pushed outward so it doesn't overlap the
    second axis' own ticks/label -- and plot one curve on it.

    Factored out after the identical "third twin axis, spine pushed
    outward, matching-color label" block was copy-pasted between
    ``precession_plot.plot_dual_axis_frequency_efficiency`` and
    ``examples/raphaldini2022_compare/precession_comparison.py``'s ``plot_sweep``
    and had already drifted (one copy set a log y-scale, the other
    didn't) -- see paper-nonlinear-interactions-SWE-sphere's own code
    review, 2026-08-25.

    **Deliberately always linear scale** -- do not set a log scale on the
    returned axis for a quantity that can be exactly zero (e.g.
    ``rsw_sphere.plotting.wave_set_periods.low_frequency_power``, which
    returns exactly ``0.0`` when there is no low-frequency content):
    matplotlib silently drops non-positive points on a log axis, which is
    exactly the bug this refactor fixes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The figure's primary axis (already holding one ``twinx()``).
    x, y : array_like
        Data to plot on the new axis.
    marker_style : str, optional
        Matplotlib format string. Default ``'v-'``.
    color : str, optional
        Line/label/tick color. Default ``GREEN``.
    ylabel : str, optional
    label : str or None, optional
        Legend label for this curve.
    outward : float, optional
        Spine offset in points. Default ``60``.

    Returns
    -------
    matplotlib.axes.Axes
        The new third axis, already plotted/labeled/styled -- call
        ``ax3.get_legend_handles_labels()`` to fold its legend entry into
        the figure's combined legend.
    """
    ax3 = ax.twinx()
    ax3.spines['right'].set_position(('outward', outward))
    ax3.plot(x, y, marker_style, ms=3, color=color, label=label)
    ax3.set_ylabel(ylabel, color=color)
    ax3.tick_params(axis='y', labelcolor=color)
    return ax3


def save_or_show(fig, path):
    """Save fig to path (dpi=200, tight bbox) and close it, or show() if path is None."""
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def apply_house_style(base_size: int = 13):
    """Apply the repo's shared figure style (serif, thin inward ticks).

    Matches ``dispersion_relation_fancy.py``'s style block. Call once
    before creating a figure; affects subsequent ``plt`` calls via
    ``rcParams`` (global, like any ``rcParams.update``).

    base_size : int, optional
        Tick-label size; axis labels, panel/sub-plot titles, AND the
        legend are all set EQUAL to each other, one step above it --
        legend text reading noticeably smaller than the axis labels
        right next to it was flagged repeatedly (2026-09-04 sizing pass,
        round 2: an even smaller *relative* legend size on top of an
        already-small base compounded the problem in the busiest
        panels), so there is now exactly one non-tick text size per
        figure, not two. Any per-call ``fontsize=`` override (a
        ``ax.legend(fontsize=...)`` or ``ax.set_title(..., fontsize=...)``)
        should be removed rather than added -- it silently drifts out of
        sync with this hierarchy the next time base_size changes here.
        Default 13 (bumped from 11, 2026-09-03: a multi-panel figure gets
        shrunk further once embedded at fixed `\\linewidth` in the paper,
        so 11pt read too small there -- pass a still-larger value (e.g.
        15-16) for a busy 2x2/2x3 panel grid, or for a figure combining
        several small subfigures, specifically.
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': base_size,
        'axes.titlesize': base_size + 1,
        'axes.labelsize': base_size + 1,
        'xtick.labelsize': base_size,
        'ytick.labelsize': base_size,
        'legend.fontsize': base_size + 1,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': False,
    })
