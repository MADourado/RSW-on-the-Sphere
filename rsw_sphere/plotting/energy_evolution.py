"""Energy-evolution figures for a wave set (triad/quartet/quintet).

Panels plot raw, unnormalized kinetic energy |A_j|^2 (and raw
E_total=E2+E3), not normalized by initial total energy -- a wave set with
2+ constituent triads does not conserve energy exactly (see
rsw_sphere.dynamics.wave_sets).

``plot_energy_evolution`` draws from an already-integrated trajectory
(used by run_dynamics.py, which integrates via trajectory_cache). The
``wave_set_energy_evolution``/``wave_set_comparison_panel`` functions
below still integrate ad hoc (not cached) for existing callers.
"""
import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.plotting.style import mode_color, apply_house_style, save_or_show, TOTAL_ENERGY_COLOR
from rsw_sphere.plotting.labels import _mode_label

G = 9.8

#: Highlighted mode solid, everything else dashed (one dashed style for
#: all non-highlighted modes; style.MODE_COLORS carries identity).
_HIGHLIGHT_LINESTYLE = '-'
_OTHER_LINESTYLE = '--'


def plot_energy_evolution(t, E, E_total, labels, modes, highlight: int = None,
                           path: str = None, ax=None):
    """Draw |A_j(t)|^2 per mode + E_total from an already-integrated trajectory.

    Parameters
    ----------
    t : ndarray (days)
    E : ndarray (len(t), n_modes)
    E_total : ndarray (len(t),)
    labels : sequence of str
    modes : sequence of (m, n, alpha) -- for mode_color()
    highlight : int or None -- index drawn solid; others dashed.
    path, ax : see save_or_show.

    Returns (fig, ax, lines) -- lines: {label: Line2D}.
    """
    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    lines = {}
    for j, (m, n, alpha) in enumerate(modes):
        ls = _HIGHLIGHT_LINESTYLE if (highlight is None or j == highlight) else _OTHER_LINESTYLE
        lines[labels[j]], = ax.plot(t, E[:, j], label=labels[j], color=mode_color(m, n, alpha), ls=ls)
    lines['Total'], = ax.plot(t, E_total, label='Total', color=TOTAL_ENERGY_COLOR, ls=':', lw=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(r'$|A|^2$ (nondimensional)')
    ax.legend(loc='upper right', fontsize=7)

    if own_fig:
        save_or_show(fig, path)
    return fig, ax, lines


def wave_set_energy_evolution(modes, triads, velocities, h_e: float = 10000,
                               t0: float = 0, tf_days: float = 10, h: float = 0.01,
                               N: int = 10, deg: int = 300,
                               highlight: int = None,
                               path: str = None, ax=None):
    """Integrate (ad hoc, not cached) and plot a wave set's raw kinetic energy.

    Returns dict: t (days), E, E_total, labels, drift, dEK.
    """
    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)

    t_f = tf_days * 4 * np.pi
    Y, T = RK44(ws, t0, t_f, h, A0)

    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)
    E = np.real(Y * np.conj(Y))
    t = np.linspace(0, t_f / (4 * np.pi), len(T))

    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    dEK = E.max(axis=0) - E.min(axis=0)

    labels = [_mode_label(*m) for m in modes]
    result = {'t': t, 'E': E, 'E_total': E_total, 'labels': labels,
              'drift': drift, 'dEK': dEK}

    _, ax_used, _ = plot_energy_evolution(t, E, E_total, labels, modes, highlight=highlight,
                                           path=path if ax is None else None, ax=ax)
    return result


def wave_set_comparison_panel(modes, triads, velocities, h_e: float = 10000,
                               triad_labels=None, wave_set_label: str = 'Wave set',
                               t0: float = 0, tf_days: float = 10, h: float = 0.01,
                               N: int = 10, deg: int = 300,
                               highlight=None, highlight_full: int = None,
                               path: str = None):
    """"Triad 1 / triad 2 [/ triad 3] / wave set" comparison row.

    Each constituent triad is its own 3-mode WaveSet (sum mode last), not
    triad_dynamics.triad_energy_evolution -- that normalizes by initial
    energy, which would put a sub-triad panel in different units.

    highlight : int, or one per triad (sequence of len(triads)), or None.
    highlight_full : highlight for the full-wave-set panel (only used
    when highlight is a sequence).

    Returns list of wave_set_energy_evolution results, one per triad then
    the full wave set (last element).
    """
    n_triads = len(triads)
    if highlight is not None and not isinstance(highlight, int):
        per_triad_highlight = list(highlight)
        if len(per_triad_highlight) != n_triads:
            raise ValueError(
                f"highlight sequence has {len(per_triad_highlight)} entries, "
                f"expected one per triad ({n_triads})")
        full_highlight = highlight_full
    else:
        per_triad_highlight = [highlight] * n_triads
        full_highlight = highlight

    apply_house_style()
    fig, axes = plt.subplots(1, n_triads + 1, figsize=(4.3 * (n_triads + 1), 4.2), sharey=True)

    results = []
    for i, (i_sum, i_p, i_q) in enumerate(triads):
        sub_modes = [modes[i_p], modes[i_q], modes[i_sum]]
        sub_velocities = [velocities[i_p], velocities[i_q], velocities[i_sum]]
        local_map = {i_p: 0, i_q: 1, i_sum: 2}
        h_i = per_triad_highlight[i]
        local_highlight = local_map.get(h_i) if h_i is not None else None

        r = wave_set_energy_evolution(
            sub_modes, [(2, 0, 1)], sub_velocities, h_e=h_e,
            t0=t0, tf_days=tf_days, h=h, N=N, deg=deg,
            highlight=local_highlight, ax=axes[i])
        title = triad_labels[i] if triad_labels else f"Triad {i + 1}"
        axes[i].set_title(title, fontsize=10)
        results.append(r)

    r_full = wave_set_energy_evolution(
        modes, triads, velocities, h_e=h_e,
        t0=t0, tf_days=tf_days, h=h, N=N, deg=deg,
        highlight=full_highlight, ax=axes[-1])
    axes[-1].set_title(wave_set_label, fontsize=10)
    results.append(r_full)

    fig.tight_layout()
    save_or_show(fig, path)
    return results


def wave_set_energy_evolution_from_spec(spec, tf_days: float = None, h: float = None,
                                         N: int = 10, deg: int = 300,
                                         highlight: int = None, path: str = None, ax=None):
    """Registry wrapper for wave_set_energy_evolution."""
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    settings = spec.settings
    return wave_set_energy_evolution(
        spec.modes, triad_indices, spec.velocities, h_e=spec.h_e,
        tf_days=tf_days if tf_days is not None else settings.get('tf_days', 10),
        h=h if h is not None else settings.get('h', 0.01),
        N=N, deg=deg, highlight=highlight, path=path, ax=ax)


def wave_set_comparison_panel_from_spec(spec, tf_days: float = None, h: float = None,
                                         N: int = 10, deg: int = 300,
                                         highlight=None, highlight_full: int = None,
                                         path: str = None):
    """Registry wrapper for wave_set_comparison_panel."""
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    triad_labels = [t.display_label for t in spec.triads]
    settings = spec.settings
    return wave_set_comparison_panel(
        spec.modes, triad_indices, spec.velocities, h_e=spec.h_e,
        triad_labels=triad_labels, wave_set_label=spec.display_label,
        tf_days=tf_days if tf_days is not None else settings.get('tf_days', 10),
        h=h if h is not None else settings.get('h', 0.01),
        N=N, deg=deg, highlight=highlight, highlight_full=highlight_full, path=path)


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs

    parser = argparse.ArgumentParser(
        description="Plot the energy-integration figure for a quartet/"
                    "quintet example from the wave-set registry.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
                         default="quartet_rh_preference")
    parser.add_argument("--panel", action="store_true",
                         help="plot the full triad(s)/wave-set comparison row.")
    parser.add_argument("--tf", dest="tf_days", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[args.wave_set]

    if args.panel:
        results = wave_set_comparison_panel_from_spec(
            spec, tf_days=args.tf_days, h=args.h, path=args.path)
        r_full = results[-1]
    else:
        r_full = wave_set_energy_evolution_from_spec(
            spec, tf_days=args.tf_days, h=args.h, path=args.path)

    print(f"{args.wave_set}: drift={r_full['drift']:.3e}, "
          f"dEK={dict(zip(r_full['labels'], r_full['dEK']))}")


if __name__ == "__main__":
    main()
