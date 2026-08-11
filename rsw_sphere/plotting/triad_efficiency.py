"""Efficiency-of-energy-transfer sweep for a resonant-triad example (§2.2,
"Resonant Triads").

**Target mode** is the standing term (matching the dissertation's own
methodology, ``Chapter2.tex:341``) for the mode whose efficiency is
measured: it is held at rest (zero initial zonal velocity) while the
*other two* modes are swept over a velocity grid, and the efficiency (max -
min of normalized kinetic energy, eq. "effor" in the dissertation) of the
target mode itself is recorded. This is a compute-heavy sweep (a 40x40 grid
takes ~1.6e6 RK33 steps at the defaults, ~1e7 at the dissertation's
100x100/t_f=100/h=0.001 settings), so the compute (``efficiency_sweep``) is
split from the plotting (``plot_efficiency_map``) and cached to ``.npz``
-- figure styling can then be iterated without re-running the sweep.

Run from the command line (output written under ``outputs/figures/triads/``
by convention; nothing is written outside ``outputs/`` automatically):

    python rsw_sphere/plotting/triad_efficiency.py outputs/figures/triads/triad_gravity_with_rossby_catalyst_efficiency.png --triad triad_gravity_with_rossby_catalyst --cache outputs/figures/triads/triad_gravity_with_rossby_catalyst_eff.npz
    python rsw_sphere/plotting/triad_efficiency.py outputs/figures/triads/triad_rossby_only_non_resonant_efficiency.png --triad triad_rossby_only_non_resonant --n-grid 20
    python -m rsw_sphere.plotting.triad_efficiency outputs/figures/triads/triad_kelvin_rossby_flow_efficiency.png --triad triad_kelvin_rossby_flow

or import and call it from another script:

    from rsw_sphere.plotting.triad_efficiency import efficiency_sweep, plot_efficiency_map
    U1, U2, EFF = efficiency_sweep(modes, h_e=10000, target=0, cache_path="sweep.npz")
    plot_efficiency_map(U1, U2, EFF, modes=modes, target=0, path="output.png")
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import hashlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.hough_harmonics.normalization import norm_component
from rsw_sphere.dynamics.dynamic_triads import TRIAD, RK33, Energy_0

G = 9.8

#: Velocity-sweep range caps by mode family (paper-review item 1): Rossby
#: (RH, alpha=3) modes are swept up to 100 m/s (jet-stream-strength winds);
#: gravity (EG/WG, alpha=1/2) modes are capped at 50 m/s (realistic
#: Kelvin/inertia-gravity wind-anomaly amplitudes are a few-20 m/s; higher
#: values used in earlier drafts were unrealistic).
RH_VELOCITY_RANGE = (0.0, 100.0)
EG_VELOCITY_RANGE = (0.0, 50.0)


def default_velocity_range(alpha):
    """Mode-family-aware default sweep range (m/s): RH_VELOCITY_RANGE for
    alpha=3 (Rossby-Haurwitz), EG_VELOCITY_RANGE for alpha=1/2 (EIG/WIG).
    """
    return RH_VELOCITY_RANGE if alpha == 3 else EG_VELOCITY_RANGE


def _mode_label(m, n, alpha):
    return {1: 'EG', 2: 'WG', 3: 'RH'}[alpha] + f'({m},{n})'


def efficiency_sweep(modes, h_e: float = 10000, u1_range=None, u2_range=None,
                      target: int = 0, fixed_velocity: float = 0.0, fixed_index: int = None,
                      n_grid: int = 40, t_f: float = 100, h: float = 0.001,
                      N: int = 10, deg: int = 300, cache_path: str = None,
                      verbose: bool = False, progress_label: str = ""):
    """Pure-compute sweep of energy-transfer efficiency vs. two initial
    zonal velocities.

    Modes are indexed 0=a, 1=b, 2=c. Following the dissertation's own
    methodology (``Chapter2.tex:341``: "both RH waves ... initialized with
    zero amplitude, and the efficiency of energy transfer in the latter
    modes is analyzed"), the **target mode** -- the one whose efficiency is
    measured -- is held at rest (``fixed_velocity=0`` by default) and the
    *other two* modes are swept over a velocity grid (``u1_range`` for the
    lower-indexed swept mode, ``u2_range`` for the higher-indexed one).
    ``fixed_index`` defaults to ``target`` itself, so by default this
    function reproduces exactly that convention; pass ``fixed_index``
    explicitly to hold a *different* mode fixed instead (e.g. to reproduce
    a swept-target sensitivity study).

    Parameters
    ----------
    modes : sequence of 3 (m, n, alpha) int triples
        Mode a, b, c.
    h_e : float, optional
        Equivalent height in metres. Default ``10000``.
    u1_range, u2_range : (float, float) or None, optional
        Zonal-velocity ranges (m/s) for the two swept modes, in index
        order. Default ``None``: each defaults independently to
        ``RH_VELOCITY_RANGE`` (0-100 m/s) or ``EG_VELOCITY_RANGE``
        (0-50 m/s) according to that swept mode's own family (alpha),
        per ``default_velocity_range``.
    target : int, optional
        Index (0/1/2 = a/b/c) of the mode whose efficiency is recorded --
        the *target mode*. Default ``0`` (mode a).
    fixed_velocity : float, optional
        Zonal velocity (m/s) of the mode held fixed. Default ``0.0`` (the
        target mode at rest, per the dissertation's methodology).
    fixed_index : int or None, optional
        Index (0/1/2) of the mode held fixed. Default ``None``: uses
        ``target`` (i.e. the target mode itself is held fixed at
        ``fixed_velocity``, and the *other two* modes are swept) -- this
        is the fix for the mismatch noted in the plan's "Known issues"
        review (bug 3): the sweep must hold the target at rest, not sweep
        it as one of the two axes.
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
        sweep is ~1.6e7 RK33 steps and can take minutes. Callers that want
        the cache to auto-invalidate on parameter changes should build
        ``cache_path`` from ``cache_key_hash(...)`` rather than a bare
        triad name (see that function's docstring).
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
    if fixed_index is None:
        fixed_index = target

    swept_indices = [i for i in range(3) if i != fixed_index]
    idx1, idx2 = swept_indices
    if u1_range is None:
        u1_range = default_velocity_range(modes[idx1][2])
    if u2_range is None:
        u2_range = default_velocity_range(modes[idx2][2])

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
                # All three modes start at zero velocity. With the target
                # mode now held at rest by default (fixed_velocity=0, per
                # the dissertation's own methodology -- see this
                # function's docstring), this triggers routinely at the
                # (0, 0) corner of every sweep, not just as an edge case
                # for one triad's registered velocities as before: both
                # swept axes start at 0 m/s and the fixed target is also
                # 0 m/s, so there is genuinely no energy to exchange and
                # nothing to normalize by. Efficiency is 0 by convention
                # (not 0/0 -> NaN) -- this is the correct value at that
                # corner, not a numerical artifact to distinguish from a
                # "real" NaN elsewhere in the grid.
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


def cache_key_hash(modes, h_e, target, fixed_index, fixed_velocity,
                    u1_range, u2_range, n_grid, t_f, h, N=10, deg=300):
    """Short hash of every parameter that changes ``efficiency_sweep``'s
    result, for building a ``.npz`` cache filename that auto-invalidates
    when a sweep parameter changes.

    Fixes the stale-cache bug logged in the plan's "Known issues" section:
    caching by triad name alone silently serves an old sweep's result
    after any of ``target``/``fixed_index``/velocity ranges/``n_grid``/
    ``t_f``/``h`` change (this happened in practice -- e.g. a
    ``gravity_catalyst`` cache computed before the ``E_0==0`` guard was
    added kept a stale NaN baked into a shipped figure).

    Returns
    -------
    str
        8 hex characters, stable across runs for identical parameters
        (``hashlib.sha1`` of the repr of every argument).
    """
    payload = repr((tuple(tuple(m) for m in modes), h_e, target, fixed_index,
                     fixed_velocity, tuple(u1_range), tuple(u2_range),
                     n_grid, t_f, h, N, deg))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


#: "Nice" ceiling values efficiency-map panels may autoscale to (paper-
#: review follow-up, 2026-08-11: a single fixed 0-100% scale shared across
#: every panel flattened the low-efficiency triads -- e.g. Triad A/C, whose
#: actual maxima are ~20%/~7% -- into a near-uniform color regardless of
#: colormap choice, since the *data range*, not the color family, was the
#: bottleneck). ``plot_efficiency_map`` now autoscales each panel's vmax to
#: the smallest of these at or above that panel's own max, so each panel
#: uses its full color range; pass an explicit ``vmax`` to force a shared
#: scale across a set of panels when a direct visual comparison is wanted.
_NICE_VMAX_STEPS = [1, 2, 5, 10, 20, 25, 50, 75, 100]


def _nice_vmax(data_max_pct):
    for step in _NICE_VMAX_STEPS:
        if data_max_pct <= step:
            return step
    return 100


def plot_efficiency_map(U1, U2, EFF, modes=None, target: int = 0,
                         fixed_index: int = None, display_label: str = '',
                         xlabel: str = None, ylabel: str = None,
                         title: str = None, vmax: float = None,
                         path: str = None, ax=None):
    """Plot a precomputed efficiency sweep as a filled-contour map.

    Parameters
    ----------
    U1, U2, EFF : ndarray
        Output of ``efficiency_sweep``.
    modes : sequence of 3 (m, n, alpha) int triples, or None, optional
        The triad's modes, used (with ``target``/``fixed_index``) to derive
        axis labels and the title from actual mode identity (e.g.
        ``"RH(4,5)"``) rather than generic "mode a"/"mode b" strings --
        fixes the axis-label/mode-identity bugs logged in the plan's
        "Known issues" review. If ``None``, falls back to the explicit
        ``xlabel``/``ylabel``/``title`` (or generic defaults).
    target : int, optional
        Index (0/1/2) of the target mode (held fixed, efficiency shown).
        Default ``0``.
    fixed_index : int or None, optional
        Index (0/1/2) of the mode actually held fixed in the sweep that
        produced ``EFF``. Default ``None``: assumed equal to ``target``
        (``efficiency_sweep``'s own default target-at-rest convention).
    display_label : str, optional
        Triad display tag (e.g. ``"Triad C"``) prepended to the title if
        given.
    xlabel, ylabel, title : str or None, optional
        Explicit overrides. If ``None`` and ``modes`` is given, derived
        automatically; if ``None`` and ``modes`` is also ``None``, falls
        back to generic labels.
    vmax : float or None, optional
        Colorbar ceiling (percent). If ``None`` (default), autoscaled per
        panel to the smallest of ``_NICE_VMAX_STEPS`` at or above this
        panel's own data max -- see the module-level note above ``_NICE_VMAX_STEPS``.
        Pass an explicit value (e.g. ``100``) to force a shared scale
        across multiple panels for direct visual comparison.
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
    if modes is not None:
        if fixed_index is None:
            fixed_index = target
        swept_indices = [i for i in range(3) if i != fixed_index]
        idx1, idx2 = swept_indices
        label1 = _mode_label(*modes[idx1])
        label2 = _mode_label(*modes[idx2])
        target_label = _mode_label(*modes[target])
        if xlabel is None:
            xlabel = f'{label1} - zonal velocity (m/s)'
        if ylabel is None:
            ylabel = f'{label2} - zonal velocity (m/s)'
        if title is None:
            prefix = f'{display_label}: ' if display_label else ''
            title = f'{prefix}target: {target_label} -- efficiency (%)'
    else:
        if xlabel is None:
            xlabel = 'swept mode 1 - zonal velocity (m/s)'
        if ylabel is None:
            ylabel = 'swept mode 2 - zonal velocity (m/s)'
        if title is None:
            title = 'Efficiency of Energy Transfer (%)'

    own_fig = ax is None
    if own_fig:
        from rsw_sphere.plotting.style import apply_house_style
        apply_house_style()
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure

    # Perceptually-uniform, colorblind-safe, grayscale-safe colormap
    # (paper-review item 7) with a power-law norm (gamma<1 stretches the
    # low end). vmax is autoscaled per panel (see _nice_vmax) rather than
    # fixed at 100 for every panel: a shared fixed scale flattened
    # low-efficiency triads into a near-uniform color regardless of
    # colormap choice, since the data range -- not the color family -- was
    # the bottleneck (paper-review follow-up, 2026-08-11).
    eff_pct = 100 * EFF
    if vmax is None:
        vmax = _nice_vmax(float(np.nanmax(eff_pct)))
    norm = PowerNorm(gamma=0.45, vmin=0, vmax=vmax)
    cs = ax.contourf(U1, U2, eff_pct, levels=100, cmap='cividis', norm=norm)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar_ticks = [t for t in [0, 1, 2, 5, 10, 20, 25, 50, 75, 100] if t <= vmax]
    if cbar_ticks[-1] != vmax:
        cbar_ticks.append(vmax)
    cbar = fig.colorbar(cs, ax=ax, ticks=cbar_ticks)
    cbar.ax.set_yticklabels([str(t) for t in cbar_ticks])

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
        help="output image path (e.g. outputs/figures/triads/"
             "triad_gravity_with_rossby_catalyst_efficiency.png). "
             "If omitted, the figure is shown interactively.")
    parser.add_argument(
        "--specs", default=DEFAULT_SPECS_PATH,
        help=f"path to the triad-registry YAML (default: {DEFAULT_SPECS_PATH}).")
    parser.add_argument(
        "--triad", choices=list(load_triad_specs(DEFAULT_SPECS_PATH)),
        default="triad_gravity_with_rossby_catalyst",
        help="which registered triad (role key) to sweep, from the "
             "default registry YAML (default: "
             "triad_gravity_with_rossby_catalyst). If --specs points at a "
             "YAML with different keys, pass the matching role key here.")
    parser.add_argument(
        "--target", type=int, default=0, choices=[0, 1, 2],
        help="index of the target mode (0=a, 1=b, 2=c) -- held at rest, "
             "efficiency plotted -- while the other two modes are swept "
             "(default: 0).")
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
    plot_efficiency_map(U1, U2, EFF, modes=spec.modes, target=args.target,
                         display_label=spec.display_label, path=args.path)


if __name__ == "__main__":
    main()
