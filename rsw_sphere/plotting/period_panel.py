"""Power-spectrum figure for a wave set's kinetic-energy time series.

Compute lives in rsw_sphere.utilities.periods; this module only draws.
"""
import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style, mode_color, save_or_show
from rsw_sphere.utilities.periods import dominant_periods


def wave_set_period_panel(t_days, E, mode_labels, mode_mnalpha, max_period_days: float = None,
                           path: str = None, ax=None):
    """Power spectrum, dominant/local-max periods marked, one line per mode.

    Returns (fig, ax, results) -- results: one dominant_periods() dict per mode.
    """
    results = []
    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    for j in range(E.shape[1]):
        r = dominant_periods(t_days, E[:, j], max_period_days=max_period_days)
        results.append(r)
        m, n, alpha = mode_mnalpha[j]
        color = mode_color(m, n, alpha)
        ax.plot(r['periods_days'], r['power'], label=mode_labels[j], color=color)
        ax.axvline(r['period_global'], color=color, ls='--', lw=1, alpha=0.6)

    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Spectral power (a.u.)')
    ax.set_xlim(left=0)
    ax.legend(loc='upper right', fontsize=7)

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, results


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
    from rsw_sphere.plotting.energy_evolution import wave_set_energy_evolution_from_spec

    parser = argparse.ArgumentParser(
        description="Plot the power-spectrum (period) figure for a "
                    "quartet/quintet example from the wave-set registry.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
                         default="quartet_rh_preference")
    parser.add_argument("--tf", dest="tf_days", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--max-period", dest="max_period_days", type=float, default=None)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[args.wave_set]

    # Throwaway axes: only used to grab the integrated trajectory.
    _, throwaway_ax = plt.subplots()
    r = wave_set_energy_evolution_from_spec(spec, tf_days=args.tf_days, h=args.h, ax=throwaway_ax)
    plt.close(throwaway_ax.figure)

    _, _, results = wave_set_period_panel(
        r['t'], r['E'], r['labels'], list(spec.modes),
        max_period_days=args.max_period_days, path=args.path)

    for label, pr in zip(r['labels'], results):
        flag = " [HORIZON-LIMITED]" if pr['horizon_limited'] else ""
        print(f"{label}: period_global={pr['period_global']:.3g}d, "
              f"period_local_max={pr['period_local_max']}{flag}")


if __name__ == "__main__":
    main()
