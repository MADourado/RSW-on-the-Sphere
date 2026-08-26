"""P-measure and filtering-error (F2) map rendering.

Compute lives in rsw_sphere.utilities.pmeasure; this module only draws.

Run from the command line (output under outputs/figures/wave_sets/):

    python rsw_sphere/plotting/pmeasure_map.py outputs/figures/wave_sets/quartet_rossby_kelvin_pmeasure.png --wave-set quartet_rossby_kelvin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rsw_sphere.plotting.style import save_or_show
from rsw_sphere.plotting.labels import _mode_label


def plot_p_measure_map(U1, U2, P, xlabel: str = None, ylabel: str = None,
                        title: str = None, vlim: float = 100.0,
                        path: str = None, ax=None):
    """Diverging-colormap P-measure map. vlim: symmetric clip (%, default 100).

    Returns (fig, ax, cs, n_clipped) -- n_clipped: grid points clipped at vlim.
    """
    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    n_clipped = int(np.sum(np.abs(P) > vlim))
    P_clipped = np.clip(P, -vlim, vlim)

    # levels must span [-vlim, vlim] explicitly, else an all-positive P
    # field silently loses the diverging scale.
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    levels = np.linspace(-vlim, vlim, 101)
    cs = ax.contourf(U1, U2, P_clipped, levels=levels, cmap='RdBu_r', norm=norm)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label='P (%)')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs, n_clipped


def plot_filtering_error_map(U1, U2, F2, xlabel: str = None, ylabel: str = None,
                              title: str = None, vmax: float = None,
                              path: str = None, ax=None):
    """Sequential-colormap F2 map (F2 >= 0, no zero-crossing to preserve).

    Blank cells: MIN_REFERENCE_DEK gate left F2 as NaN.
    Returns (fig, ax, cs).
    """
    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    finite = np.isfinite(F2)
    if vmax is None:
        vmax = float(np.nanmax(F2)) if np.any(finite) else 1.0

    cs = ax.contourf(U1, U2, np.ma.masked_invalid(F2), levels=np.linspace(0, vmax, 101), cmap='viridis')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=r'$\mathcal{F}_2^a$')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs


def plot_frequency_shift_map(U1, U2, shift, xlabel: str = None, ylabel: str = None,
                              title: str = None, vlim: float = None,
                              agree=None, path: str = None, ax=None):
    """Diverging-colormap dominant-period %-shift map (period can lengthen
    or shorten). vlim: symmetric clip; default the data's own max |shift|.

    `agree`: optional bool array, same shape as `shift`
    (FreqShiftAgree from wave_set_diagnostics_sweep) -- where False,
    marks that exact grid point (small dot) rather than hiding the
    value. Disagreement between the two period estimators is advisory,
    not an error (JFM-template.tex Sec. 3.3.5: it can flag a genuine
    second spectral component, not just a bad measurement) -- shift is
    real and plotted either way; markers are per-sample-point, not
    interpolated, since agreement isn't a smooth field.

    Returns (fig, ax, cs).
    """
    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    finite = np.isfinite(shift)
    if vlim is None:
        vlim = float(np.nanmax(np.abs(shift))) if np.any(finite) else 1.0
    shift_clipped = np.clip(shift, -vlim, vlim)

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    levels = np.linspace(-vlim, vlim, 101)
    cs = ax.contourf(U1, U2, np.ma.masked_invalid(shift_clipped), levels=levels, cmap='RdBu_r', norm=norm)
    if agree is not None:
        disagree = (~np.asarray(agree, dtype=bool)) & finite
        if np.any(disagree):
            ax.scatter(np.asarray(U1)[disagree], np.asarray(U2)[disagree],
                       s=6, c='k', marker='.', alpha=0.6, linewidths=0,
                       label='estimators disagree')
            ax.legend(loc='upper right', fontsize='small', framealpha=0.8)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label='dominant period shift (%)')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs


def plot_fmax_map(U1, U2, Fmax, xlabel: str = None, ylabel: str = None,
                   title: str = None, vlim: float = None,
                   path: str = None, ax=None):
    """Diverging-colormap peak-kinetic-energy %-difference map (signed,
    full wave set vs. reference triad). vlim: symmetric clip; default the
    data's own max |Fmax|.

    Returns (fig, ax, cs).
    """
    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    finite = np.isfinite(Fmax)
    if vlim is None:
        vlim = float(np.nanmax(np.abs(Fmax))) if np.any(finite) else 1.0
    Fmax_clipped = np.clip(Fmax, -vlim, vlim)

    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    levels = np.linspace(-vlim, vlim, 101)
    cs = ax.contourf(U1, U2, np.ma.masked_invalid(Fmax_clipped), levels=levels, cmap='RdBu_r', norm=norm)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label=r'$\mathcal{F}_{max}^a$ (%)')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs


def plot_novelty_period_map(U1, U2, NoveltyPeriod, xlabel: str = None, ylabel: str = None,
                             title: str = None, vmax: float = None,
                             path: str = None, ax=None):
    """Sequential-colormap novelty-period map (period in days, always
    positive -- see rsw_sphere.utilities.periods.novel_frequency_content).

    Blank cells: dEK_sub too small (MIN_REFERENCE_DEK gate), or no novel
    frequency survived the prominence threshold at that grid point --
    both cases leave NoveltyPeriod NaN, same convention as F2/FreqShift.
    NoveltyRelevance (companion array from wave_set_diagnostics_sweep,
    same shape) is not drawn here by default -- available for a caller to
    overlay (e.g. as a contour or alpha channel) if wanted.

    Returns (fig, ax, cs).
    """
    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    finite = np.isfinite(NoveltyPeriod)
    if vmax is None:
        vmax = float(np.nanmax(NoveltyPeriod)) if np.any(finite) else 1.0

    cs = ax.contourf(U1, U2, np.ma.masked_invalid(NoveltyPeriod), levels=np.linspace(0, vmax, 101),
                      cmap='viridis')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    fig.colorbar(cs, ax=ax, label='Novelty period (days)')

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, cs


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
    from rsw_sphere.utilities.pmeasure import p_measure_sweep

    parser = argparse.ArgumentParser(
        description="Compute (with .npz caching) and plot a P-measure "
                    "sweep for a quartet/quintet example from the registry.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
                         default="quartet_rossby_kelvin")
    parser.add_argument("--swept", nargs=2, type=str, default=None,
                         help="two mode keys to sweep; default: auto-detected.")
    parser.add_argument("--target", nargs="+", type=str, default=None,
                         help="mode keys to report P for; default: reference triad's members.")
    parser.add_argument("--n-grid", dest="n_grid", type=int, default=40)
    parser.add_argument("--tf", dest="tf_days", type=float, default=10)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--fixed", type=float, default=30.0,
                         help="velocity (m/s) for modes neither swept nor targeted.")
    parser.add_argument("--cache", dest="cache_path", default=None)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[args.wave_set]
    triads = [spec.triad_indices(i) for i in range(spec.n_triads())]

    if args.swept:
        swept_indices = tuple(spec.index(k) for k in args.swept)
    else:
        ref = triads[spec.reference_triad]
        non_ref = [i for i in range(spec.n_modes()) if i not in ref]
        swept_indices = (ref[1], non_ref[0]) if non_ref else (ref[1], ref[2])

    if args.target:
        target_indices = [spec.index(k) for k in args.target]
    else:
        ref = triads[spec.reference_triad]
        target_indices = [ref[1], ref[2]]

    fixed_velocities = {i: args.fixed for i in range(spec.n_modes())
                         if i not in swept_indices}

    result = p_measure_sweep(
        spec.modes, triads, spec.h_e, swept_indices, fixed_velocities,
        target_indices, reference_triad=spec.reference_triad,
        n_grid=args.n_grid, tf_days=args.tf_days, h=args.h,
        cache_path=args.cache_path, verbose=True, progress_label=args.wave_set)

    n_targets = len(target_indices)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5))
    if n_targets == 1:
        axes = [axes]
    label1 = _mode_label(*spec.modes[swept_indices[0]])
    label2 = _mode_label(*spec.modes[swept_indices[1]])
    for k, ax in enumerate(axes):
        plot_p_measure_map(result['U1'], result['U2'], result['P'][..., k],
                            xlabel=f'{label1} - zonal velocity (m/s)',
                            ylabel=f'{label2} - zonal velocity (m/s)',
                            title=f"P: {result['labels'][k]}", ax=ax)
    fig.tight_layout()
    save_or_show(fig, args.path)


if __name__ == "__main__":
    main()
