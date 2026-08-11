"""P-measure (energy-transfer enhancement/inhibition) sweep for a quartet/
quintet ("wave set") example -- §4's ``fig: 4ef``/``fig: 4ef222``/
``fig: rh30`` figures.

The P-measure (paper eq. ``Pa``) compares a target mode's kinetic-energy
variation in the **full wave set** against its variation in **one
constituent triad in isolation**:

    P (%) = 100 * (dEK_wave_set - dEK_triad) / dEK_triad

A positive P means the extra mode(s) *enhance* the target's energy
exchange relative to its own triad; negative means they *inhibit* it.
Three things that the dissertation's own methodology treats as essential
but that a naive scalar-only API would push into ad-hoc plotting-time
special cases (this is exactly how the §2.2 target/fixed-index bug
happened -- see ``triad_efficiency.py``'s docstring):

1. **Per-mode ``triad_index``.** Different target modes are compared
   against *different* constituent triads (``tab: cap4ex2``'s own
   methodology note: RH(4,5)/RH(3,4) use the RH-only triad's ΔEK as
   denominator; RH(1,2)/EG(1,1), each private to only one triad, use their
   own triad's ΔEK). ``p_measure``'s default picks, per target mode, the
   ``reference_triad`` if the mode belongs to it, else the first
   constituent triad that contains it -- matching the dissertation's rule
   -- but accepts an explicit override.
2. **``dEK_triad == 0`` -> NaN, guarded explicitly.** Fires whenever every
   mode in a target's reference triad starts at zero velocity (the (0, 0)
   sweep corner) -- the analogue of ``triad_efficiency.py``'s
   ``E_0 == 0`` guard.
3. **Row-level denominator caching.** In a 2-axis sweep, typically only
   one of the two swept modes is even a member of a given target's
   reference triad (e.g. sweeping RH(1,2) and a private gravity mode: the
   RH-only reference triad doesn't contain the gravity mode at all). When
   the second swept axis doesn't touch a target's reference triad, that
   triad's ΔEK is recomputed once per *row* (fixed first-axis velocity)
   and reused across the row, not recomputed at every grid point.

Follows **convention 14**: ``plot_p_measure_map`` uses a *diverging*
colormap (``RdBu_r``, ``TwoSlopeNorm(vcenter=0)``) rather than §2.2's
sequential ``cividis``+``PowerNorm`` -- a sequential map would destroy the
zero-crossing between inhibition (negative P) and enhancement (positive
P), which is the entire point of this figure.

Run from the command line (output written under
``outputs/figures/wave_sets/`` by convention):

    python rsw_sphere/plotting/wave_set_pmeasure.py outputs/figures/wave_sets/quartet_gravity_kelvin_pmeasure.png --wave-set quartet_gravity_kelvin
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.sweeps import cache_key_hash
from rsw_sphere.plotting.triad_efficiency import default_velocity_range

G = 9.8


def _default_triad_index_for_mode(triads, reference_triad, mode_idx):
    """Which constituent triad serves as mode ``mode_idx``'s P-measure
    denominator, by the dissertation's own rule (see module docstring):
    its ``reference_triad`` if it's a member, else the first triad
    (in registration order) that contains it. ``None`` if the mode is in
    no triad at all (should not happen for a validly-built wave set).
    """
    if mode_idx in triads[reference_triad]:
        return reference_triad
    for t, tri in enumerate(triads):
        if mode_idx in tri:
            return t
    return None


def _dEK_for_triad(gamma, modes, triad, velocities, h_e, t0, t_f, h, N, deg, mode_idx):
    """Integrate constituent ``triad`` (indices into ``modes``) alone as
    its own 3-mode ``WaveSet`` and return ``mode_idx``'s kinetic-energy
    variation (``max - min`` of raw ``|A|^2``) within it.
    """
    i_sum, i_p, i_q = triad
    sub_modes = [modes[i_p], modes[i_q], modes[i_sum]]
    sub_velocities = [velocities[i_p], velocities[i_q], velocities[i_sum]]
    local = {i_p: 0, i_q: 1, i_sum: 2}[mode_idx]

    sub_ws = WaveSet(gamma, sub_modes, [(2, 0, 1)], N=N, deg=deg)
    A0 = sub_ws.amplitudes_from_velocities(sub_velocities, h_e, g=G)
    Y, _ = RK33(sub_ws, t0, t_f, h, A0)
    E = np.real(Y[:, local] * np.conj(Y[:, local]))
    return E.max() - E.min()


def p_measure(modes, triads, velocities, h_e: float = 10000,
              target_indices=None, reference_triad: int = 0, triad_index=None,
              t0: float = 0, tf_days: float = 10, h: float = 0.01,
              N: int = 10, deg: int = 300):
    """P-measure (%) for one or more target modes of a wave set, at one
    fixed initial condition.

    Parameters
    ----------
    modes, triads : see ``rsw_sphere.dynamics.wave_sets.WaveSet``.
    velocities : sequence of float, same length as ``modes``
        Initial zonal velocities (m/s).
    h_e : float, optional
    target_indices : sequence of int or None, optional
        Mode indices to compute P for. Default: all modes.
    reference_triad : int, optional
        Default denominator triad index for a mode that belongs to it
        (see module docstring). Default ``0``.
    triad_index : dict of int -> int, or None, optional
        Explicit per-mode denominator-triad override (``{mode_idx:
        triad_idx}``). Modes not present fall back to
        ``_default_triad_index_for_mode``.
    t0, tf_days, h, N, deg : see ``wave_set_dynamics.wave_set_energy_evolution``.

    Returns
    -------
    dict
        ``P`` (percent, NaN where the denominator ΔEK is 0), ``dEK_full``,
        ``dEK_triad``, ``triad_index_used`` (per target), ``drift``
        (wave-set energy drift, see ``rsw_sphere.dynamics.wave_sets``),
        ``labels`` (per target mode).
    """
    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)

    t_f = tf_days * 4 * np.pi
    Y, T = RK33(ws, t0, t_f, h, A0)
    E = np.real(Y * np.conj(Y))
    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)
    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])

    if target_indices is None:
        target_indices = list(range(ws.n_modes))
    triad_index = dict(triad_index or {})

    P = np.full(len(target_indices), np.nan)
    dEK_full = np.full(len(target_indices), np.nan)
    dEK_triad = np.full(len(target_indices), np.nan)
    triad_index_used = []

    for k, tgt in enumerate(target_indices):
        t_idx = triad_index.get(tgt, _default_triad_index_for_mode(triads, reference_triad, tgt))
        triad_index_used.append(t_idx)
        dEK_full[k] = E[:, tgt].max() - E[:, tgt].min()
        if t_idx is None:
            continue
        dEK_triad[k] = _dEK_for_triad(gamma, modes, triads[t_idx], velocities, h_e,
                                       t0, t_f, h, N, deg, tgt)
        if dEK_triad[k] > 0:
            P[k] = 100 * (dEK_full[k] - dEK_triad[k]) / dEK_triad[k]

    labels = [_mode_label(*modes[tgt]) for tgt in target_indices]
    return {'P': P, 'dEK_full': dEK_full, 'dEK_triad': dEK_triad,
            'triad_index_used': triad_index_used, 'drift': drift, 'labels': labels}


def p_measure_sweep(modes, triads, h_e: float, swept_indices, fixed_velocities: dict,
                     target_indices, u1_range=None, u2_range=None,
                     reference_triad: int = 0, triad_index=None,
                     n_grid: int = 40, tf_days: float = 10, h: float = 0.01,
                     N: int = 10, deg: int = 300, cache_path: str = None,
                     verbose: bool = False, progress_label: str = ""):
    """Pure-compute 2D sweep of the P-measure for one or more target modes,
    over the initial zonal velocities of two other modes.

    Parameters
    ----------
    modes, triads : see ``rsw_sphere.dynamics.wave_sets.WaveSet``.
    h_e : float
    swept_indices : (int, int)
        Mode indices whose velocities are the two swept axes.
    fixed_velocities : dict of int -> float
        Velocities (m/s) for every mode **not** in ``swept_indices`` or
        ``target_indices`` need not be present if they coincide with a
        swept/target mode; every other mode must have an entry.
    target_indices : sequence of int
        Mode indices to report P for (each may use a different
        denominator triad, resolved the same way as ``p_measure``).
    u1_range, u2_range : (float, float) or None, optional
        Velocity ranges (m/s) for the two swept axes. Default: each
        defaults to ``triad_efficiency.default_velocity_range`` by that
        mode's own family (RH vs. gravity), same caps as §2.2.
    reference_triad, triad_index : see ``p_measure``.
    n_grid : int, optional
        Grid resolution per axis. Default ``40``.
    tf_days, h, N, deg : see ``p_measure``.
    cache_path : str or None, optional
        ``.npz`` cache (see ``triad_efficiency.efficiency_sweep`` -- same
        pattern). Build from ``cache_key_hash`` for auto-invalidation.
    verbose, progress_label : see ``triad_efficiency.efficiency_sweep``.

    Returns
    -------
    dict
        ``U1``, ``U2`` (meshgrid, m/s), ``P`` (shape ``(n_grid, n_grid,
        len(target_indices))``, percent), ``drift`` (shape ``(n_grid,
        n_grid)``), ``labels`` (per target).
    """
    idx1, idx2 = swept_indices
    if u1_range is None:
        u1_range = default_velocity_range(modes[idx1][2])
    if u2_range is None:
        u2_range = default_velocity_range(modes[idx2][2])

    if cache_path and os.path.exists(cache_path):
        data = np.load(cache_path)
        return {'U1': data['U1'], 'U2': data['U2'], 'P': data['P'],
                'drift': data['drift'], 'labels': list(data['labels'])}

    gamma = gamma_from_he(h_e, g=G)[1]
    ws = WaveSet(gamma, modes, triads, N=N, deg=deg)
    t_f = tf_days * 4 * np.pi

    triad_index = dict(triad_index or {})
    t_idx_for_target = [
        triad_index.get(tgt, _default_triad_index_for_mode(triads, reference_triad, tgt))
        for tgt in target_indices
    ]
    # Whether a target's denominator triad depends on the 2nd swept axis
    # (idx2) at all -- if not, that triad's dEK can be cached per row
    # (fixed idx1) and reused across every column (varying idx2 only
    # perturbs modes outside that triad, per the module docstring's point 3).
    axis2_in_triad = [
        (t_idx is not None and idx2 in triads[t_idx]) for t_idx in t_idx_for_target
    ]

    u1 = np.linspace(u1_range[0], u1_range[1], n_grid)
    u2 = np.linspace(u2_range[0], u2_range[1], n_grid)
    U1, U2 = np.meshgrid(u1, u2)

    P = np.full((n_grid, n_grid, len(target_indices)), np.nan)
    DRIFT = np.empty((n_grid, n_grid))

    if verbose:
        import time
        t_start = time.time()

    for i in range(n_grid):
        row_cache = {}  # triad_idx -> dEK (scalar), valid across the whole row
        for j in range(n_grid):
            velocities = np.empty(ws.n_modes)
            for m in range(ws.n_modes):
                if m == idx1:
                    velocities[m] = U1[i, j]
                elif m == idx2:
                    velocities[m] = U2[i, j]
                else:
                    velocities[m] = fixed_velocities[m]

            A0 = ws.amplitudes_from_velocities(velocities, h_e, g=G)
            Y, _ = RK33(ws, 0, t_f, h, A0)
            E = np.real(Y * np.conj(Y))
            E2, E3 = ws.energy(Y)
            E_total = np.real(E2 + E3)
            DRIFT[i, j] = np.max(np.abs(E_total - E_total[0])) / np.maximum(np.abs(E_total[0]), 1e-300)

            for k, tgt in enumerate(target_indices):
                t_idx = t_idx_for_target[k]
                dEK_full = E[:, tgt].max() - E[:, tgt].min()
                if t_idx is None:
                    continue
                if (not axis2_in_triad[k]) and t_idx in row_cache:
                    dEK_triad = row_cache[t_idx]
                else:
                    dEK_triad = _dEK_for_triad(gamma, modes, triads[t_idx], velocities, h_e,
                                                0, t_f, h, N, deg, tgt)
                    if not axis2_in_triad[k]:
                        row_cache[t_idx] = dEK_triad
                if dEK_triad > 0:
                    P[i, j, k] = 100 * (dEK_full - dEK_triad) / dEK_triad

        if verbose:
            done_rows = i + 1
            elapsed = time.time() - t_start
            eta = elapsed / done_rows * (n_grid - done_rows)
            prefix = f"[{progress_label}] " if progress_label else ""
            print(f"    {prefix}row {done_rows}/{n_grid} "
                  f"({100 * done_rows / n_grid:.0f}%) "
                  f"elapsed {elapsed:.0f}s, eta {eta:.0f}s", flush=True)

    labels = [_mode_label(*modes[tgt]) for tgt in target_indices]
    if cache_path:
        np.savez(cache_path, U1=U1, U2=U2, P=P, drift=DRIFT, labels=np.array(labels))

    return {'U1': U1, 'U2': U2, 'P': P, 'drift': DRIFT, 'labels': labels}


def plot_p_measure_map(U1, U2, P, xlabel: str = None, ylabel: str = None,
                        title: str = None, vlim: float = 100.0,
                        path: str = None, ax=None):
    """Plot one target mode's P-measure sweep as a diverging-colormap map
    (**convention 14** -- see module docstring for why sequential
    ``cividis`` is wrong here).

    Parameters
    ----------
    U1, U2 : ndarray
        Meshgrid of the two swept velocities (m/s).
    P : ndarray, same shape as ``U1``
        P-measure (percent) for one target mode (index a single target
        out of ``p_measure_sweep``'s ``P[..., k]`` before calling this).
    xlabel, ylabel, title : str or None, optional
    vlim : float, optional
        Symmetric color-scale limit (percent); values are clipped to
        ``[-vlim, vlim]`` (paper's own stated ceiling of 1, i.e. 100%, for
        the P-measure). Default ``100.0``.
    path : str or None, optional
    ax : matplotlib.axes.Axes or None, optional

    Returns
    -------
    (QuadContourSet, int)
        The contour set and ``n_clipped`` -- the number of grid points
        whose |P| exceeded ``vlim`` and were clipped, so the caption can
        state it (paper-review convention: never clip silently).
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

    # `levels` must span [-vlim, vlim] explicitly -- passing a bare int to
    # contourf makes it choose level boundaries from the *data*'s own
    # min/max, not the norm's, so an all-positive P field (common: most of
    # the domain enhances rather than inhibits) silently loses the
    # diverging scale and renders as if it were a sequential colormap
    # (caught by inspecting a first-draft figure: the colorbar only spanned
    # the data's actual range instead of the full +-100% scale).
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

    if not own_fig:
        return cs, n_clipped

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return cs, n_clipped


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs

    parser = argparse.ArgumentParser(
        description="Compute (with .npz caching) and plot a P-measure "
                    "sweep for a quartet/quintet example from the wave-set "
                    "registry YAML.")
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--wave-set", choices=list(load_wave_set_specs(DEFAULT_WAVESETS_PATH)),
                         default="quartet_gravity_kelvin")
    parser.add_argument("--swept", nargs=2, type=str, default=None,
                         help="two mode keys (e.g. c d) to sweep; default: "
                              "the two non-reference-triad-only members, "
                              "auto-detected.")
    parser.add_argument("--target", nargs="+", type=str, default=None,
                         help="mode keys to report P for; default: the "
                              "reference triad's two RH members.")
    parser.add_argument("--n-grid", dest="n_grid", type=int, default=40)
    parser.add_argument("--tf", dest="tf_days", type=float, default=10)
    parser.add_argument("--h", type=float, default=0.01)
    parser.add_argument("--fixed", type=float, default=30.0,
                         help="velocity (m/s) for modes that are neither "
                              "swept nor targeted (default: 30).")
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
    if args.path:
        fig.savefig(args.path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
