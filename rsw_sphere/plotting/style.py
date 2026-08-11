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
# modes, grey reserved for non-modal references (e.g. the total-energy
# line in triad_dynamics.py).
#
# Keyed by the raw ``(m, n, alpha)`` triple (alpha: 1=EIG/EG, 2=WIG/WG,
# 3=RH) rather than a formatted label string, since the two call sites
# (triad_dynamics.py, triad_efficiency.py) format mode labels slightly
# differently (``dynamic_triads.label()`` emits "RH (m,n)" with a space,
# ``triad_table._mode_label`` emits "RH(m,n)" without) -- the tuple is the
# one thing both can produce identically from what they already hold.
#
# Hue assigned by each mode's *linear* period (fixed, not recomputed at
# plot time): within a family, the fastest (shortest-period) mode gets the
# lightest hue and the slowest gets the darkest, so darker == slower reads
# consistently across figures. Periods used to fix this ordering (computed
# via ``triad_table.triad_properties``/``wave_set_table.wave_set_table``,
# h_e=10000 m, 2026-08-11):
#   RH(3,4)=3.72d, RH(4,5)=3.99d, RH(1,2)=5.03d, RH(3,6)=7.59d,
#   RH(3,10)=19.0d, RH(1,7)=30.3d
#   EG(6,9)=0.156d, EG(7,9)=0.156d, EG(1,1)=1.35d
# Every mode currently used across §2.2/§3 (6 RH, 3 EG, no WG) has a fixed
# entry below; a mode not in this dict falls back to GREY (see
# ``mode_color``) rather than erroring, so new triads/wave sets don't break
# plotting -- extend this dict when new modes are added.
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
}

#: Color for non-modal reference lines (e.g. total energy).
TOTAL_ENERGY_COLOR = GREY


def mode_color(m, n, alpha, default=GREY):
    """Look up the persistent color for mode ``(m, n, alpha)`` in
    ``MODE_COLORS``, falling back to ``default`` (grey) for any mode not
    yet registered there.
    """
    return MODE_COLORS.get((int(m), int(n), int(alpha)), default)


def apply_house_style():
    """Apply the repo's shared figure style (serif 11pt, thin inward ticks).

    Matches ``dispersion_relation_fancy.py``'s style block. Call once
    before creating a figure; affects subsequent ``plt`` calls via
    ``rcParams`` (global, like any ``rcParams.update``).
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': False,
    })
