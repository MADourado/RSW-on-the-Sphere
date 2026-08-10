"""Efficiency-of-energy-transfer sweep for a resonant-triad example (§2.2,
"Resonant Triads").

For a grid of initial zonal velocities of modes a and b (mode c held
fixed), integrates the three-wave amplitude equations and records the
efficiency (max - min of normalized kinetic energy, eq. "effor" in the
dissertation) of a chosen target mode. This is a compute-heavy sweep (a
40x40 grid takes ~1.6e6 RK33 steps at the defaults, ~1e7 at the
dissertation's 100x100/t_f=100/h=0.001 settings), so the compute
(``efficiency_sweep``) is split from the plotting (``plot_efficiency_map``)
and cached to ``.npz`` -- figure styling can then be iterated without
re-running the sweep.

Run from the command line (output written under ``outputs/figures/triads/``
by convention; nothing is written outside ``outputs/`` automatically):

    python rsw_sphere/plotting/triad_efficiency.py outputs/figures/triads/gravity_catalyst_efficiency.png --triad gravity_catalyst --cache outputs/figures/triads/gravity_catalyst_eff.npz
    python rsw_sphere/plotting/triad_efficiency.py outputs/figures/triads/rossby_pump_efficiency.png --triad rossby_pump --n-grid 20
    python -m rsw_sphere.plotting.triad_efficiency outputs/figures/triads/kelvin_rh_flow_efficiency.png --triad kelvin_rh_flow

or import and call it from another script:

    from rsw_sphere.plotting.triad_efficiency import efficiency_sweep, plot_efficiency_map
    U1, U2, EFF = efficiency_sweep(modes, h_e=10000, u1_range=(0, 100),
                                    u2_range=(0, 100), cache_path="sweep.npz")
    plot_efficiency_map(U1, U2, EFF, path="output.png")
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

G = 9.8


def efficiency_sweep(modes, h_e: float = 10000, u1_range=(1, 100), u2_range=(1, 100),
                      target: int = 0, fixed_velocity: float = 10.0, fixed_index: int = 2,
                      n_grid: int = 40, t_f: float = 100, h: float = 0.001,
                      N: int = 10, deg: int = 300, cache_path: str = None,
                      verbose: bool = False, progress_label: str = ""):
    """Pure-compute sweep of energy-transfer efficiency vs. two initial
    zonal velocities.

    Modes are indexed 0=a, 1=b, 2=c. Two of the three modes are swept over
    a velocity grid (``u1_range`` for mode index 0, ``u2_range`` for mode
    index 1, by default -- i.e. modes a and b), the third is held fixed at
    ``fixed_velocity`` (mode ``fixed_index``, default mode c), and the
    efficiency (max - min of the normalized kinetic energy, eq. "effor" in
    the dissertation) of ``target`` is recorded at every grid point.

    Parameters
    ----------
    modes : sequence of 3 (m, n, alpha) int triples
        Mode a, b, c.
    h_e : float, optional
        Equivalent height in metres. Default ``10000``.
    u1_range, u2_range : (float, float), optional
        Zonal-velocity ranges (m/s) for the two swept modes (a and b).
        Default ``(1, 100)`` for both.
    target : int, optional
        Index (0/1/2 = a/b/c) of the mode whose efficiency is recorded.
        Default ``0`` (mode a).
    fixed_velocity : float, optional
        Zonal velocity (m/s) of the mode held fixed. Default ``10.0``.
    fixed_index : int, optional
        Index (0/1/2) of the mode held fixed. Default ``2`` (mode c).
    n_grid : int, optional
        Grid resolution along each swept axis. Default ``40``.
    t_f : float, optional
        Integration horizon, nondimensional time. Default ``100`` (the
        dissertation's ``Triad_Precession`` value).
    h : float, optional
        RK33 step size, nondimensional time. Default ``0.001``.
    N, deg : int, optional
        Hough-mode truncation / quadrature degree; ``deg`` must match
        ``norm_Hough``'s. Default ``10``/``300``.
    cache_path : str or None, optional
        If given and the file exists, the sweep is loaded from it instead
        of recomputed. If given and the file does not exist, the result is
        computed and then saved there. If ``None``, no caching is done.
        Essential in practice: at ``n_grid=40, t_f=100, h=0.001`` this
        sweep is ~1.6e7 RK33 steps and can take minutes.
    verbose : bool, optional
        If ``True``, print one progress line per completed grid row
        (elapsed time, ETA). Default ``False``.
    progress_label : str, optional
        Prefix for progress lines (e.g. a triad name), only used if
        ``verbose``.

    Returns
    -------
    U1, U2, EFF : ndarray, ndarray, ndarray
        Meshgrid of the two swept velocities (m/s) and the resulting
        efficiency (fraction, not percent) of the target mode.
    """
    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return data['U1'], data['U2'], data['EFF']

    eps, gamma = gamma_from_he(h_e, g=G)
    (m_a, n_a, alpha_a), (m_b, n_b, alpha_b), (m_c, n_c, alpha_c) = modes
    Triad = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c, N, deg)

    nu = [
        norm_component(Triad.uvh_a[0]) * np.sqrt(G * h_e),
        norm_component(Triad.uvh_b[0]) * np.sqrt(G * h_e),
        norm_component(Triad.uvh_c[0]) * np.sqrt(G * h_e),
    ]

    u1 = np.linspace(u1_range[0], u1_range[1], n_grid)
    u2 = np.linspace(u2_range[0], u2_range[1], n_grid)
    U1, U2 = np.meshgrid(u1, u2)

    swept_indices = [i for i in range(3) if i != fixed_index]
    idx1, idx2 = swept_indices

    EFF = np.empty_like(U1)
    if verbose:
        import time
        t_start = time.time()
    for i in range(n_grid):
        for j in range(n_grid):
            A0 = [0j, 0j, 0j]
            A0[fixed_index] = fixed_velocity / nu[fixed_index]
            A0[idx1] = U1[i, j] / nu[idx1]
            A0[idx2] = U2[i, j] / nu[idx2]
            A0 = np.array(A0)

            E_02, E_03 = Energy_0(Triad, A0)
            E_0 = E_02 + E_03

            if E_0 == 0:
                # All three modes start at zero velocity (can happen at a
                # sweep-grid corner when the held-fixed mode is also 0,
                # e.g. gravity_catalyst's mode c) -- nothing to normalize
                # by and no energy to exchange, so efficiency is 0 by
                # convention rather than 0/0 (NaN).
                EFF[i, j] = 0.0
                continue

            Y, T = RK33(Triad, 0, t_f, h, A0)
            Y_t = Y[:, target] * np.conj(Y[:, target]) / E_0
            Y_t = np.real(Y_t)
            EFF[i, j] = Y_t.max() - Y_t.min()

        if verbose:
            done_rows = i + 1
            elapsed = time.time() - t_start
            eta = elapsed / done_rows * (n_grid - done_rows)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"    {prefix}row {done_rows}/{n_grid} "
                  f"({100 * done_rows / n_grid:.0f}%) "
                  f"elapsed {elapsed:.0f}s, eta {eta:.0f}s", flush=True)

    if cache_path:
        np.savez(cache_path, U1=U1, U2=U2, EFF=EFF)

    return U1, U2, EFF


def plot_efficiency_map(U1, U2, EFF, xlabel='mode b - zonal velocity (m/s)',
                         ylabel='mode a - zonal velocity (m/s)',
                         title='Efficiency of Energy Transfer (%)',
                         path: str = None, ax=None):
    """Plot a precomputed efficiency sweep as a filled-contour map.

    Parameters
    ----------
    U1, U2, EFF : ndarray
        Output of ``efficiency_sweep``.
    xlabel, ylabel, title : str, optional
        Axis/plot labels.
    path : str or None, optional
        If given, the figure is saved to this path (PNG, 200 dpi) and the
        figure is closed. If ``None`` (default) and ``ax`` is also
        ``None``, the figure is shown interactively with ``plt.show()``.
    ax : matplotlib.axes.Axes or None, optional
        If given, plot into this axes instead of creating a new figure.

    Returns
    -------
    matplotlib.contour.QuadContourSet
    """
    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    cs = ax.contourf(U1, U2, 100 * EFF, levels=100, cmap='terrain')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.colorbar(cs, ax=ax)

    if not own_fig:
        return cs

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return cs


def main():
    import argparse
    from rsw_sphere.dynamics.triad_specs import DEFAULT_SPECS_PATH, load_triad_specs

    parser = argparse.ArgumentParser(
        description="Compute (with .npz caching) and plot the efficiency-"
                    "of-energy-transfer sweep for a resonant-triad example "
                    "loaded from a triad-registry YAML "
                    "(rsw_sphere.dynamics.triad_specs.load_triad_specs).")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output image path (e.g. "
             "outputs/figures/triads/gravity_catalyst_efficiency.png). "
             "If omitted, the figure is shown interactively.")
    parser.add_argument(
        "--specs", default=DEFAULT_SPECS_PATH,
        help=f"path to the triad-registry YAML (default: {DEFAULT_SPECS_PATH}).")
    parser.add_argument(
        "--triad", choices=list(load_triad_specs(DEFAULT_SPECS_PATH)),
        default="gravity_catalyst",
        help="which registered triad (role key) to sweep, from the "
             "default registry YAML (default: gravity_catalyst). If "
             "--specs points at a YAML with different keys, pass the "
             "matching role key here.")
    parser.add_argument(
        "--target", type=int, default=0, choices=[0, 1, 2],
        help="index of the target mode (0=a, 1=b, 2=c) whose efficiency is "
             "plotted (default: 0).")
    parser.add_argument(
        "--n-grid", dest="n_grid", type=int, default=40,
        help="grid resolution along each swept axis (default: 40).")
    parser.add_argument(
        "--tf", dest="t_f", type=float, default=100,
        help="integration horizon, nondimensional time (default: 100).")
    parser.add_argument(
        "--h", type=float, default=0.001,
        help="RK33 step size, nondimensional time (default: 0.001).")
    parser.add_argument(
        "--cache", dest="cache_path", default=None,
        help="path to an .npz cache file (computed once, reloaded after).")
    args = parser.parse_args()

    specs = load_triad_specs(args.specs)
    spec = specs[args.triad]
    U1, U2, EFF = efficiency_sweep(
        spec.modes, h_e=spec.h_e, target=args.target,
        n_grid=args.n_grid, t_f=args.t_f, h=args.h, cache_path=args.cache_path)
    plot_efficiency_map(U1, U2, EFF, path=args.path)


if __name__ == "__main__":
    main()
