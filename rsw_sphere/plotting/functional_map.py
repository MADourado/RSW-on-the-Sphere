"""Efficiency and low-frequency-energy map rendering.

Compute lives in rsw_sphere.utilities.functional; this module only draws.
"""
import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style, save_or_show


def plot_efficiency_map(U1, U2, efficiency, xlabel: str = None, ylabel: str = None,
                         title: str = None, vmax: float = None,
                         path: str = None, ax=None):
    """Sequential-colormap efficiency map. Blank cells: drift_max gate left NaN.

    Returns (fig, ax, cs).
    """
    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    finite = np.isfinite(efficiency)
    if vmax is None:
        vmax = float(np.nanmax(efficiency)) if np.any(finite) else 1.0

    cs = ax.contourf(U1, U2, np.ma.masked_invalid(100 * efficiency),
                      levels=np.linspace(0, 100 * vmax, 101), cmap='viridis')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=r'Efficiency $\mathcal{E}_{\mathrm{avg}}$ (%)')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs


def plot_low_frequency_energy_map(U1, U2, power, xlabel: str = None, ylabel: str = None,
                                   title: str = None, vmax: float = None,
                                   path: str = None, ax=None):
    """Sequential-colormap low-frequency-power map.

    Returns (fig, ax, cs).
    """
    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    finite = np.isfinite(power)
    if vmax is None:
        vmax = float(np.nanmax(power)) if np.any(finite) else 1.0
        vmax = vmax if vmax > 0 else 1.0

    cs = ax.contourf(U1, U2, np.ma.masked_invalid(power), levels=np.linspace(0, vmax, 101), cmap='viridis')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label='Low-frequency power')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs
