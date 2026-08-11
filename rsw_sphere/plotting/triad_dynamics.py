"""Energy-integration figure for a single resonant-triad example (§2.2,
"Resonant Triads").

Integrates the three-wave amplitude equations (``TRIAD.f`` /
``RK33``, ``rsw_sphere.dynamics.dynamic_triads``) forward in time from
given initial zonal velocities and plots the normalized kinetic energy of
each mode plus the (conserved) total energy.

Run from the command line (output written under ``outputs/figures/triads/``
by convention; nothing is written outside ``outputs/`` automatically):

    python rsw_sphere/plotting/triad_dynamics.py outputs/figures/triads/triad_gravity_with_rossby_catalyst_energy.png --triad triad_gravity_with_rossby_catalyst
    python rsw_sphere/plotting/triad_dynamics.py outputs/figures/triads/triad_rossby_only_non_resonant_energy.png --triad triad_rossby_only_non_resonant --tf 3
    python -m rsw_sphere.plotting.triad_dynamics outputs/figures/triads/triad_kelvin_rossby_flow_energy.png --triad triad_kelvin_rossby_flow

or import and call it from another script:

    from rsw_sphere.plotting.triad_dynamics import triad_energy_evolution
    from rsw_sphere.dynamics.triad_specs import load_triad_specs
    spec = load_triad_specs()['triad_gravity_with_rossby_catalyst']
    triad_energy_evolution(spec.modes, spec.velocities, spec.h_e,
                            t0=0, tf_days=10, h=0.001,
                            path="outputs/figures/triads/triad_gravity_with_rossby_catalyst_energy.png")
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.hough_harmonics.normalization import norm_component
from rsw_sphere.dynamics.dynamic_triads import TRIAD, RK33, Energy_0
from rsw_sphere.plotting.style import mode_color, TOTAL_ENERGY_COLOR

G = 9.8


#: Line styles by role, so the target mode is identifiable without relying
#: on color alone (paper-review follow-up, 2026-08-11: colors alone were
#: too similar between some modes; solid vs. dashed is also
#: grayscale/colorblind-safe). The two non-target modes get *different*
#: dash patterns from each other so they remain distinguishable where their
#: curves overlap.
_TARGET_LINESTYLE = '-'
_OTHER_LINESTYLES = ['--', ':']


def triad_energy_evolution(modes, velocities, h_e: float = 10000,
                            t0: float = 0, tf_days: float = 10, h: float = 0.001,
                            N: int = 10, deg: int = 300,
                            target: int = None,
                            path: str = None, ax=None):
    """Integrate and plot the normalized kinetic energy of a resonant triad.

    Parameters
    ----------
    modes : sequence of 3 (m, n, alpha) int triples
        Mode a, b, c (see ``rsw_sphere.dynamics.triad_specs``).
    velocities : sequence of 3 float
        Initial zonal velocities (m/s) for modes a, b, c.
    h_e : float, optional
        Equivalent height in metres. Default ``10000``.
    t0 : float, optional
        Initial nondimensional time. Default ``0``.
    tf_days : float, optional
        Final time, in days. Converted to nondimensional time via
        ``t_f = tf_days * 4*pi`` (the convention used by
        ``rsw_sphere.dynamics.dynamic_triads.Triad_dynamics`` and
        ``run_dynamics.py``). Default ``10``.
    h : float, optional
        RK33 step size (nondimensional time). Default ``0.001``.
    target : int or None, optional
        Index (0/1/2, i.e. a/b/c) of the target mode -- the one whose
        efficiency is the point of the figure. If given, its curve is
        drawn solid and the other two dashed/dotted (see
        ``_TARGET_LINESTYLE``/``_OTHER_LINESTYLES``), so the target is
        identifiable without relying on color alone. If ``None``
        (default), all three modes are drawn solid.
    N : int, optional
        Hough-mode expansion truncation order. Default ``10``.
    deg : int, optional
        Gaussian-quadrature degree, must match ``norm_Hough``'s. Default ``300``.
    path : str or None, optional
        If given, the figure is saved to this path (PNG, 200 dpi) and the
        figure is closed. If ``None`` (default) and ``ax`` is also
        ``None``, the figure is shown interactively with ``plt.show()``.
    ax : matplotlib.axes.Axes or None, optional
        If given, plot into this axes instead of creating a new figure
        (used to compose multi-panel figures). When ``ax`` is given, the
        figure is neither saved nor shown by this function -- the caller
        owns the figure.

    Returns
    -------
    dict
        ``t`` (days), ``E_a``, ``E_b``, ``E_c``, ``E_total`` (all
        normalized by the initial total energy), plus ``efficiency_a/b/c``
        (max - min of each normalized energy).
    """
    eps, gamma = gamma_from_he(h_e, g=G)

    (m_a, n_a, alpha_a), (m_b, n_b, alpha_b), (m_c, n_c, alpha_c) = modes
    Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c, N, deg)

    u_a, u_b, u_c = velocities
    nu_a = norm_component(Triad.uvh_a[0]) * np.sqrt(G * h_e)
    nu_b = norm_component(Triad.uvh_b[0]) * np.sqrt(G * h_e)
    nu_c = norm_component(Triad.uvh_c[0]) * np.sqrt(G * h_e)

    A_0 = np.array([u_a / nu_a, u_b / nu_b, u_c / nu_c])

    t_f = tf_days * 4 * np.pi

    E_02, E_03 = Energy_0(Triad, A_0)
    E_0 = E_02 + E_03

    Y, T = RK33(Triad, t0, t_f, h, A_0)
    Y_a, Y_b, Y_c = Y[:, 0], Y[:, 1], Y[:, 2]

    Z = np.array([Y_a, Y_b, Y_c])
    E_2, E_3 = Energy_0(Triad, Z)
    E_T = np.real((E_2 + E_3) / E_0)

    t = np.linspace(0, t_f / (4 * np.pi), len(T))

    E_a = np.real(Y_a * np.conj(Y_a) / E_0)
    E_b = np.real(Y_b * np.conj(Y_b) / E_0)
    E_c = np.real(Y_c * np.conj(Y_c) / E_0)

    result = {
        't': t, 'E_a': E_a, 'E_b': E_b, 'E_c': E_c, 'E_total': E_T,
        'efficiency_a': E_a.max() - E_a.min(),
        'efficiency_b': E_b.max() - E_b.min(),
        'efficiency_c': E_c.max() - E_c.min(),
    }

    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
    else:
        fig = ax.figure

    # TRIAD.label_* uses the code's internal EIG/WIG shorthand; the paper
    # consistently uses EG/WG, so relabel for display without touching the
    # shared TRIAD class (used elsewhere with the EIG/WIG convention intact).
    def _paper_label(m, n, alpha, fallback):
        return {1: 'EG', 2: 'WG', 3: 'RH'}.get(alpha, fallback[:2]) + f'({m},{n})'

    if target is None:
        ls_a = ls_b = ls_c = _TARGET_LINESTYLE
    else:
        styles = [None, None, None]
        others = iter(_OTHER_LINESTYLES)
        for i in range(3):
            styles[i] = _TARGET_LINESTYLE if i == target else next(others)
        ls_a, ls_b, ls_c = styles

    ax.plot(t, E_a, label=_paper_label(m_a, n_a, alpha_a, Triad.label_a), color=mode_color(m_a, n_a, alpha_a), ls=ls_a)
    ax.plot(t, E_b, label=_paper_label(m_b, n_b, alpha_b, Triad.label_b), color=mode_color(m_b, n_b, alpha_b), ls=ls_b)
    ax.plot(t, E_c, label=_paper_label(m_c, n_c, alpha_c, Triad.label_c), color=mode_color(m_c, n_c, alpha_c), ls=ls_c)
    ax.plot(t, E_T, label='Total', color=TOTAL_ENERGY_COLOR, ls='--', lw=1)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Energy (nondimensional)')
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc='upper right', fontsize=8)

    if not own_fig:
        return result

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return result


def main():
    import argparse
    from rsw_sphere.dynamics.triad_specs import DEFAULT_SPECS_PATH, load_triad_specs

    parser = argparse.ArgumentParser(
        description="Plot the energy-integration figure for a resonant-"
                    "triad example loaded from a triad-registry YAML "
                    "(rsw_sphere.dynamics.triad_specs.load_triad_specs).")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output image path (e.g. outputs/figures/triads/"
             "triad_gravity_with_rossby_catalyst_energy.png). "
             "If omitted, the figure is shown interactively.")
    parser.add_argument(
        "--specs", default=DEFAULT_SPECS_PATH,
        help=f"path to the triad-registry YAML (default: {DEFAULT_SPECS_PATH}).")
    parser.add_argument(
        "--triad", choices=list(load_triad_specs(DEFAULT_SPECS_PATH)),
        default="triad_gravity_with_rossby_catalyst",
        help="which registered triad (role key) to integrate, from the "
             "default registry YAML (default: "
             "triad_gravity_with_rossby_catalyst). If --specs points at a "
             "YAML with different keys, pass the matching role key here.")
    parser.add_argument(
        "--tf", dest="tf_days", type=float, default=10,
        help="final integration time, in days (default: 10).")
    parser.add_argument(
        "--h", type=float, default=0.001,
        help="RK33 step size, nondimensional time (default: 0.001).")
    args = parser.parse_args()

    specs = load_triad_specs(args.specs)
    spec = specs[args.triad]
    triad_energy_evolution(spec.modes, spec.velocities, h_e=spec.h_e,
                            tf_days=args.tf_days, h=args.h, path=args.path)


if __name__ == "__main__":
    main()
