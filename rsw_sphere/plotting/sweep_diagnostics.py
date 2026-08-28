"""Rendering for run_sweep.py's unified 1D/2D diagnostics
(rsw_sphere.dynamics.diagnostics_report): 1D draws one line plot per
diagnostic (swept velocity on the x-axis), 2D draws one heatmap-panel
figure per diagnostic (one panel per mode/triad). Compute lives in
run_sweep.py's own per-grid-point loop; this module only draws.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rsw_sphere.plotting.style import apply_house_style, save_or_show

#: Cycled per series (not just color) -- two series can land on nearly
#: identical values (e.g. a mode shared across triads referencing the
#: same winning sub-triad as a private mode of that triad), and matplotlib's
#: default color-only cycle makes one line invisible directly under the
#: other in that case. Markers cycle too so two adjacent colors stay
#: distinguishable even in greyscale.
_LINESTYLES = ['-', '--', '-.', ':']
_MARKERS = ['o', 's', '^', 'v', 'D', 'P', 'X', '*']


def _plot_lines(u_values, series: dict, xlabel: str, ylabel: str, title: str, path: str):
    """Shared line-plot body: one line per series key, NaN-gapped."""
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, (label, values) in enumerate(series.items()):
        ax.plot(u_values, values, marker=_MARKERS[i % len(_MARKERS)],
                ls=_LINESTYLES[i % len(_LINESTYLES)], ms=3, alpha=0.85, label=str(label))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc='best', ncol=2 if len(series) > 6 else 1)
    fig.tight_layout()
    save_or_show(fig, path)
    return fig, ax


def plot_mode_unit_sweep(u_values, series: dict, xlabel: str, ylabel: str, title: str, path: str = None):
    """``efficiency``/``dominant_freq``/``dominant_period``/
    ``low_frequency_energy``: one line per (mode_label, unit_name) pair,
    all on one figure.

    series : {(mode_label, unit_name): array}
    """
    labeled = {f"{mode_label} [{unit_name}]": v for (mode_label, unit_name), v in series.items()}
    return _plot_lines(u_values, labeled, xlabel, ylabel, title, path)


def plot_triad_sweep(u_values, series: dict, xlabel: str, ylabel: str, title: str, path: str = None):
    """``dynamical_phase``: one line per constituent triad.

    series : {triad_label: array}
    """
    return _plot_lines(u_values, series, xlabel, ylabel, title, path)


def plot_mode_scalar_sweep(u_values, series: dict, xlabel: str, ylabel: str, title: str, path: str = None):
    """The 5 scalar "final" diagnostics (p_measure, efficiency_var,
    spectral_dev_var, novel_freq, novel_period): one line per mode in the
    full wave set (private and shared alike).

    series : {mode_label: array}
    """
    return _plot_lines(u_values, series, xlabel, ylabel, title, path)


def plot_candidate_scalar(labels, values, ylabel: str, title: str, path: str = None):
    """One point per candidate mode -- no connecting line, since candidates
    (which mode fills a slot) have no natural ordering, unlike a swept
    continuous velocity. Used by run_sweep_sets.py's own candidate-mode
    screening (rsw_sphere.dynamics.diagnostics_report's "final" scalar
    diagnostics, one point per candidate instead of one line per grid point).

    labels : sequence of str -- candidate mode labels (x-axis tick labels)
    values : sequence of float, same length as labels
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(labels)), 4.5))
    x = np.arange(len(labels))
    ax.scatter(x, values, s=40, zorder=3)
    ax.axhline(0, color='grey', lw=0.8, ls='--', zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    save_or_show(fig, path)
    return fig, ax


def _heatmap_grid(U1, U2, series: dict, xlabel: str, ylabel: str, title: str, path: str,
                   diverging: bool = False, cbar_label: str = ''):
    """Shared heatmap-panel-grid body: one contourf panel per series key
    (mode or triad label), up to 3 columns, independent per-panel
    colorbar/scale. Sequential (viridis) by default; diverging (RdBu_r,
    symmetric around 0) for a diagnostic that can be negative (e.g. a
    signed % change -- p_measure, efficiency_var).
    """
    apply_house_style()
    n = len(series)
    ncols = min(n, 3)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, (label, values) in enumerate(series.items()):
        ax = axes_flat[i]
        finite = values[np.isfinite(values)]
        if diverging:
            vlim = float(np.max(np.abs(finite))) if finite.size else 1.0
            vlim = vlim if vlim > 0 else 1.0
            norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
            cs = ax.contourf(U1, U2, np.clip(values, -vlim, vlim),
                              levels=np.linspace(-vlim, vlim, 101), cmap='RdBu_r', norm=norm)
        else:
            vmax = float(np.nanmax(finite)) if finite.size else 1.0
            vmax = vmax if vmax > 0 else 1.0
            cs = ax.contourf(U1, U2, np.ma.masked_invalid(values),
                              levels=np.linspace(0, vmax, 101), cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(str(label))
        fig.colorbar(cs, ax=ax, label=cbar_label)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    save_or_show(fig, path)
    return fig, axes


def plot_mode_heatmap_sweep(U1, U2, series: dict, xlabel: str, ylabel: str, title: str, path: str = None,
                             diverging: bool = False, cbar_label: str = ''):
    """2D counterpart of ``plot_mode_unit_sweep``/``plot_mode_scalar_sweep``:
    one heatmap panel per mode (``full`` unit's own value only -- a
    heatmap grid doesn't have a line plot's spare room for a per-unit
    breakdown).

    series : {mode_label: 2D array, same shape as U1/U2}
    """
    return _heatmap_grid(U1, U2, series, xlabel, ylabel, title, path, diverging=diverging, cbar_label=cbar_label)


def plot_triad_heatmap_sweep(U1, U2, series: dict, xlabel: str, ylabel: str, title: str, path: str = None,
                              cbar_label: str = ''):
    """2D counterpart of ``plot_triad_sweep``: one heatmap panel per triad.

    series : {triad_label: 2D array, same shape as U1/U2}
    """
    return _heatmap_grid(U1, U2, series, xlabel, ylabel, title, path, diverging=False, cbar_label=cbar_label)
