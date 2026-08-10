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
