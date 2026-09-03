"""Precession-frequency + efficiency figure rendering.

Compute lives in rsw_sphere.utilities.precession; this module only draws.
"""
import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style, add_outward_twin_axis, save_or_show


def plot_dual_axis_frequency_efficiency(result, spec, plot_triad=None,
                                         xlabel='', title='', plot_u_max=None, path=None,
                                         efficiency_label=r'Efficiency $\mathcal{E}$'):
    """Twin-axis "precession frequency (dotted) + efficiency (solid)" figure.

    result : output of precession_frequency_efficiency.
    spec : WaveSetSpec -- derives each triad's legend label.
    plot_triad : int or None -- draw only that triad's frequency curve
        (every triad is still computed regardless).
    plot_u_max : float or None -- crop the plotted/auto-scaled range.
    efficiency_label : right-axis label/legend text -- override when
        `result['efficiency']` is a different quantity than the plain
        efficiency share (e.g. the efficiency variation Delta-E_a).
    """
    u_values = result['u_values']
    freq_by_triad = dict(result['freq_by_triad'])
    efficiency = result['efficiency']
    low_freq_power = result.get('low_freq_power')
    labels = result['triad_labels']

    if plot_u_max is not None:
        mask = u_values <= plot_u_max
        u_values = u_values[mask]
        freq_by_triad = {lbl: v[mask] for lbl, v in freq_by_triad.items()}
        if efficiency is not None:
            efficiency = efficiency[mask]
        if low_freq_power is not None:
            low_freq_power = low_freq_power[mask]

    apply_house_style()
    markers = ['o', 's', '^', 'v']
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for i, lbl in enumerate(labels):
        if plot_triad is not None and i != plot_triad:
            continue
        ax.plot(u_values, np.abs(freq_by_triad[lbl]), markers[i % len(markers)] + ':', ms=3,
                color='C0', label=f'Prec. freq. {lbl}', alpha=1.0 if i == 0 else 0.6)
    ax.axhline(0.01, color='grey', ls=':', lw=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$|$precession frequency$|$ (rad/day)', color='C0')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_title(title)

    lines1, labels1 = ax.get_legend_handles_labels()
    all_lines, all_labels = list(lines1), list(labels1)

    if efficiency is not None:
        ax2 = ax.twinx()
        ax2.plot(u_values, 100 * efficiency, 'd-', ms=3, color='C3', label=efficiency_label)
        ax2.set_ylabel(f'{efficiency_label} (%)', color='C3')
        ax2.tick_params(axis='y', labelcolor='C3')
        ax2.axhline(0, color='C3', ls=':', lw=0.8, alpha=0.5)
        lines2, labels2 = ax2.get_legend_handles_labels()
        all_lines += lines2
        all_labels += labels2

    if low_freq_power is not None:
        ax3 = add_outward_twin_axis(ax, u_values, low_freq_power, marker_style='^-',
                                     color='C2', ylabel='Low-freq. power (target mode)',
                                     label='Low-freq. power')
        lines3, labels3 = ax3.get_legend_handles_labels()
        all_lines += lines3
        all_labels += labels3

    ax.legend(all_lines, all_labels, loc='lower left')
    fig.tight_layout()
    save_or_show(fig, path)
    return fig, ax


def plot_phase_trace(phi_list, T_days_list, labels, title='', ylabel=r'$\Phi$ (rad)', path=None):
    """One or more phi(t) traces -- works for both combined dynamical
    phase (dynamical_phase) and individual raw mode phase (individual_phase).
    """
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for phi, T_days, lbl in zip(phi_list, T_days_list, labels):
        ax.plot(T_days, phi, label=lbl)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save_or_show(fig, path)
    return fig, ax


def main():
    import argparse
    import numpy as np
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
    from rsw_sphere.utilities.precession import precession_frequency_efficiency

    parser = argparse.ArgumentParser(
        description="Compute (with trajectory caching) and plot a "
                    "precession-frequency + efficiency sweep for a wave "
                    "set from the registry.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
                         default="quartet_rh_preference")
    parser.add_argument("--sweep-mode", default="d", help="mode key to sweep")
    parser.add_argument("--target", default=None, help="mode key to report efficiency for")
    parser.add_argument("--plot-triad", type=int, default=None)
    parser.add_argument("--u-min", type=float, default=10.0)
    parser.add_argument("--u-max", type=float, default=150.0)
    parser.add_argument("--n-points", type=int, default=15)
    parser.add_argument("--tf", dest="tf_days", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[args.wave_set]

    u_values = np.linspace(args.u_min, args.u_max, args.n_points)
    result = precession_frequency_efficiency(
        spec, args.sweep_mode, u_values, target_mode_key=args.target,
        tf_days=args.tf_days, h=args.h)

    plot_dual_axis_frequency_efficiency(
        result, spec, plot_triad=args.plot_triad,
        xlabel=f"{args.sweep_mode} driving velocity (m/s)",
        title=spec.display_label or spec.key, path=args.path)

    min_freq = {lbl: np.min(np.abs(v)) for lbl, v in result['freq_by_triad'].items()}
    print(f"{args.wave_set}: min |precession_freq| per triad: {min_freq}")


if __name__ == "__main__":
    main()
