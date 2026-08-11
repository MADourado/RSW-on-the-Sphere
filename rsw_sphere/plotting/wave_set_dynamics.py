"""Energy-integration figures for quartet/quintet ("wave set") examples in
the paper's merged §Coupled Triads section.

Mirrors ``rsw_sphere.plotting.triad_dynamics`` (§2.2's single-triad energy
figure) but for ``WaveSet`` (``rsw_sphere.dynamics.wave_sets``) instead of
``TRIAD``, and follows **convention 15**: panels plot the raw,
unnormalized kinetic energy ``|A_j|^2`` (and raw ``E_total = E2+E3``), not
normalized by the initial total energy -- a wave set with 2+ constituent
triads does not conserve energy (see ``rsw_sphere.dynamics.wave_sets``'s
module docstring), so §2.2's "normalize by E_0, total stays at 1" framing
does not apply here. Every result dict below reports the energy ``drift``
(the departure from conservation) alongside the per-mode energy variation
``dEK``, per the plan's C6 validity-gate discipline -- callers must not
silently trust a ``ΔEK``/``P``-derived number without checking
``drift / dEK`` is small for the mode in question.

Run from the command line (output written under
``outputs/figures/wave_sets/`` by convention):

    python rsw_sphere/plotting/wave_set_dynamics.py outputs/figures/wave_sets/quartet_rh_preference_energy.png --wave-set quartet_rh_preference
    python -m rsw_sphere.plotting.wave_set_dynamics outputs/figures/wave_sets/quartet_gravity_kelvin_panel.png --wave-set quartet_gravity_kelvin --panel

or import and call it from another script:

    from rsw_sphere.plotting.wave_set_dynamics import wave_set_energy_evolution
    from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
    spec = load_wave_set_specs()['quartet_rh_preference']
    wave_set_energy_evolution(spec.modes, [spec.triad_indices(i) for i in range(spec.n_triads())],
                               spec.velocities, h_e=spec.h_e, tf_days=30,
                               path="outputs/figures/wave_sets/quartet_rh_preference_energy.png")
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.plotting.style import mode_color, apply_house_style, TOTAL_ENERGY_COLOR
from rsw_sphere.plotting.labels import _mode_label

G = 9.8

#: Line styles by role -- matches ``triad_dynamics``'s convention (target/
#: highlighted mode solid, everything else dashed). Unlike the 3-mode triad
#: case, a wave set can have 4-5 modes, too many for distinct dash patterns
#: to stay legible -- every non-highlighted mode shares one dashed style and
#: relies on ``style.MODE_COLORS`` for identity (already the primary
#: identity channel across every §2.2/§3 figure).
_HIGHLIGHT_LINESTYLE = '-'
_OTHER_LINESTYLE = '--'


def wave_set_energy_evolution(modes, triads, velocities, h_e: float = 10000,
                               t0: float = 0, tf_days: float = 10, h: float = 0.01,
                               N: int = 10, deg: int = 300,
                               highlight: int = None,
                               path: str = None, ax=None):
    """Integrate and plot the raw kinetic energy of a quartet/quintet
    ("wave set") example.

    Parameters
    ----------
    modes : sequence of (m, n, alpha) int triples
        One entry per mode (see ``rsw_sphere.dynamics.wave_sets.WaveSet``).
    triads : sequence of (i_sum, i_p, i_q) int triples
        Constituent triads, indices into ``modes`` (paper convention: sum
        mode first). A plain triad is ``triads=[(2, 0, 1)]`` on 3 modes.
    velocities : sequence of float, same length as ``modes``
        Initial zonal velocities (m/s).
    h_e : float, optional
        Equivalent height, m. Default ``10000``.
    t0 : float, optional
        Initial nondimensional time. Default ``0``.
    tf_days : float, optional
        Final time in days, converted via ``t_f = tf_days * 4*pi`` (same
        convention as ``triad_dynamics.triad_energy_evolution``). Default
        ``10``.
    h : float, optional
        RK33 step size (nondimensional time). Default ``0.01`` (coarser
        than §2.2's ``0.001`` default -- wave-set runs are more expensive
        per step; tighten for a final, hi-res pass).
    N, deg : int, optional
        Hough truncation / quadrature degree. Default ``N=10, deg=300``.
    highlight : int or None, optional
        Index of the mode to draw solid while every other mode is dashed
        (see module docstring for why this replaces §2.2's ``target``).
        ``None`` (default): every mode drawn solid.
    path : str or None, optional
        If given, the figure is saved (PNG, 200 dpi) and closed. If
        ``None`` and ``ax`` is also ``None``, shown interactively.
    ax : matplotlib.axes.Axes or None, optional
        Plot into this axes instead of creating a new figure (for
        composing multi-panel figures). The figure is neither saved nor
        shown when ``ax`` is given -- the caller owns it.

    Returns
    -------
    dict
        ``t`` (days), ``E`` (shape ``(len(t), n_modes)``, raw ``|A_j|^2``),
        ``E_total`` (raw, ``E2+E3``), ``labels`` (paper-facing mode
        labels), ``drift`` (``max_t|E_total(t)-E_total(0)| /
        |E_total(0)|``), ``dEK`` (per mode, ``max - min`` of ``E``).
    """
    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)

    t_f = tf_days * 4 * np.pi
    Y, T = RK33(ws, t0, t_f, h, A0)

    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)
    E = np.real(Y * np.conj(Y))
    t = np.linspace(0, t_f / (4 * np.pi), len(T))

    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    dEK = E.max(axis=0) - E.min(axis=0)

    labels = [_mode_label(*m) for m in modes]
    result = {'t': t, 'E': E, 'E_total': E_total, 'labels': labels,
              'drift': drift, 'dEK': dEK}

    own_fig = ax is None
    if own_fig:
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    for j, (m, n, alpha) in enumerate(modes):
        ls = _HIGHLIGHT_LINESTYLE if (highlight is None or j == highlight) else _OTHER_LINESTYLE
        ax.plot(t, E[:, j], label=labels[j], color=mode_color(m, n, alpha), ls=ls)
    ax.plot(t, E_total, label='Total', color=TOTAL_ENERGY_COLOR, ls=':', lw=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(r'$|A|^2$ (nondimensional)')
    ax.legend(loc='upper right', fontsize=7)

    if not own_fig:
        return result

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return result


def wave_set_comparison_panel(modes, triads, velocities, h_e: float = 10000,
                               triad_labels=None, wave_set_label: str = 'Wave set',
                               t0: float = 0, tf_days: float = 10, h: float = 0.01,
                               N: int = 10, deg: int = 300,
                               highlight: int = None,
                               path: str = None):
    """"Triad 1 / triad 2 [/ triad 3] / wave set" comparison row -- the
    figure layout §3/§4 repeat for every quartet and the quintet.

    Each constituent triad is plotted from its **own 3-mode WaveSet**
    (built via ``wave_set_energy_evolution`` on that triad alone, sum mode
    last), *not* via ``triad_dynamics.triad_energy_evolution`` -- the
    latter normalizes by the triad's own initial energy, which would put a
    sub-triad panel in different units from the wave-set panel next to it
    (see the module docstring's convention-15 note and
    ``PLAN-section-3.md``'s Phase C physics note).

    Parameters
    ----------
    modes, triads, velocities, h_e, t0, tf_days, h, N, deg, highlight :
        See ``wave_set_energy_evolution``. ``highlight`` is a *global*
        mode index (into ``modes``); it is remapped to each sub-triad's
        own local index automatically, and omitted from a sub-triad panel
        that doesn't contain that mode.
    triad_labels : sequence of str or None, optional
        Per-triad subplot titles (e.g. "Triad 1 (RH-only)"). Defaults to
        "Triad 1", "Triad 2", ...
    wave_set_label : str, optional
        Title for the final (full wave-set) subplot.
    path : str or None, optional
        If given, the composed figure is saved (PNG, 200 dpi) and closed.
        If ``None``, shown interactively.

    Returns
    -------
    list of dict
        One ``wave_set_energy_evolution`` result per constituent triad (in
        order), followed by the full wave set's result (last element).
    """
    n_triads = len(triads)
    apply_house_style()
    fig, axes = plt.subplots(1, n_triads + 1, figsize=(4.3 * (n_triads + 1), 4.2), sharey=False)

    results = []
    for i, (i_sum, i_p, i_q) in enumerate(triads):
        sub_modes = [modes[i_p], modes[i_q], modes[i_sum]]
        sub_velocities = [velocities[i_p], velocities[i_q], velocities[i_sum]]
        local_map = {i_p: 0, i_q: 1, i_sum: 2}
        local_highlight = local_map.get(highlight) if highlight is not None else None

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
        highlight=highlight, ax=axes[-1])
    axes[-1].set_title(wave_set_label, fontsize=10)
    results.append(r_full)

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return results


def wave_set_energy_evolution_from_spec(spec, tf_days: float = None, h: float = None,
                                         N: int = 10, deg: int = 300,
                                         highlight: int = None, path: str = None, ax=None):
    """Registry convenience wrapper for ``wave_set_energy_evolution`` --
    thin layer over the explicit-config core (see module/plan's
    "generality rule": registry lookup is never the only entry point).
    ``tf_days``/``h`` default to ``spec.settings`` when not given.
    """
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    settings = spec.settings
    return wave_set_energy_evolution(
        spec.modes, triad_indices, spec.velocities, h_e=spec.h_e,
        tf_days=tf_days if tf_days is not None else settings.get('tf_days', 10),
        h=h if h is not None else settings.get('h', 0.01),
        N=N, deg=deg, highlight=highlight, path=path, ax=ax)


def wave_set_comparison_panel_from_spec(spec, tf_days: float = None, h: float = None,
                                         N: int = 10, deg: int = 300,
                                         highlight: int = None, path: str = None):
    """Registry convenience wrapper for ``wave_set_comparison_panel``."""
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    triad_labels = [t.display_label for t in spec.triads]
    settings = spec.settings
    return wave_set_comparison_panel(
        spec.modes, triad_indices, spec.velocities, h_e=spec.h_e,
        triad_labels=triad_labels, wave_set_label=spec.display_label,
        tf_days=tf_days if tf_days is not None else settings.get('tf_days', 10),
        h=h if h is not None else settings.get('h', 0.01),
        N=N, deg=deg, highlight=highlight, path=path)


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs

    parser = argparse.ArgumentParser(
        description="Plot the energy-integration figure for a quartet/"
                    "quintet example loaded from the wave-set registry YAML "
                    "(rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs).")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output image path. If omitted, shown interactively.")
    parser.add_argument(
        "--specs", default=DEFAULT_WAVESETS_PATH,
        help=f"path to the wave-set registry YAML (default: {DEFAULT_WAVESETS_PATH}).")
    parser.add_argument(
        "--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
        default="quartet_rh_preference",
        help="which registered wave set (role key) to integrate.")
    parser.add_argument(
        "--panel", action="store_true",
        help="plot the full 'triad 1 / triad 2 [/ triad 3] / wave set' "
             "comparison row instead of just the wave set's own energy figure.")
    parser.add_argument(
        "--tf", dest="tf_days", type=float, default=None,
        help="final integration time, in days (default: from registry settings).")
    parser.add_argument(
        "--h", type=float, default=None,
        help="RK33 step size, nondimensional time (default: from registry settings).")
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
