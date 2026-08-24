"""General "precession frequency vs. driving velocity" figure for any
registered wave set, config-driven (examples/precession_sweep_figures.yaml)
rather than one hardcoded script per figure -- e.g. this paper's own
Quartet A (RH(3,6) swept) and its RH(3,8) variant are two entries in
that one registry, not two near-duplicate scripts.

Compute and plot are two separate functions (``sweep``/``plot_sweep``),
matching this repo's own established pattern for expensive figures
(``rsw_sphere.plotting.triad_efficiency``'s ``efficiency_sweep``/
``plot_efficiency_map``, ``wave_set_pmeasure``'s equivalent split): the
sweep -- ~10+ minutes at this figure's own default resolution -- always
writes its raw result to a ``.npz`` cache, and plotting always reads
from that cache rather than from a live sweep. Labels, colors, titles
can then be iterated by editing ``plot_sweep`` and re-running it alone,
never re-paying the sweep's own cost.

Per-triad legend labels (which modes each constituent triad contains)
are derived directly from the wave-set registry at plot time -- never
hardcoded -- so they cannot drift out of sync with the registry itself.

Each entry may set ``target_mode`` (a mode key, e.g. ``c``): if present,
the figure gains a second panel showing that mode's own energy-transfer
efficiency (``quartet_precession_sweep.precession_sweep``'s own
``efficiency``, normalized by the trajectory's time-averaged total
energy rather than its initial value, since a quartet does not conserve
energy exactly) alongside the precession-frequency panel, from the SAME
already-integrated sweep -- no extra integration cost. Omitting
``target_mode`` reproduces the original single-panel figure exactly.

Run:

    python examples/precession_sweep_figure.py quartet_a_rh36 outputs/figures/quartet_a_rh36_precession_cache.npz outputs/figures/quartet_a_rh36_precession.png
    python examples/precession_sweep_figure.py quartet_a_rh36 outputs/figures/quartet_a_rh36_precession_cache.npz outputs/figures/quartet_a_rh36_precession.png --plot-only
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import yaml
import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs, DEFAULT_WAVESETS_PATH
from quartet_precession_sweep import precession_sweep

DEFAULT_FIGURES_PATH = os.path.join(_ROOT, "precession_sweep_figures.yaml")


def triad_mode_labels(wave_set_key, yaml_path=DEFAULT_WAVESETS_PATH):
    """{triad display_label: "RH(a)+RH(b)+RH(c)"-style mode-list string},
    derived from the wave-set registry itself -- not a hardcoded/YAML
    duplicate, so it cannot go stale relative to the registry.
    """
    spec = load_wave_set_specs(yaml_path)[wave_set_key]
    labels = {}
    for i, t in enumerate(spec.triads):
        i_sum, i_p, i_q = spec.triad_indices(i)
        mode_str = "+".join(_mode_label(*spec.modes[j]) for j in (i_sum, i_p, i_q))
        labels[t.display_label] = mode_str
    return labels


def sweep(entry_key, cache_path, figures_yaml=DEFAULT_FIGURES_PATH, wave_sets_yaml=DEFAULT_WAVESETS_PATH):
    """Pure compute: run the sweep for ``entry_key`` and write it to
    ``cache_path`` (.npz). Does no plotting. Re-running this with an
    already-existing ``cache_path`` is a no-op (skips the sweep) --
    delete the cache file first to force a recompute (required after
    adding/changing this entry's own ``target_mode``, since that changes
    what gets computed, not just how it's plotted).

    Returns
    -------
    u_values : ndarray
    f_by_triad : dict of {triad display_label: ndarray}
    energy_drift : ndarray
    efficiency : ndarray or None
        ``None`` iff this entry has no ``target_mode``.
    """
    with open(figures_yaml) as f:
        cfg = yaml.safe_load(f)[entry_key]
    labels = triad_mode_labels(cfg['wave_set'], wave_sets_yaml)
    target_mode = cfg.get('target_mode')

    if os.path.exists(cache_path):
        d = np.load(cache_path)
        f_by_triad = {lbl: d[f'f_{i}'] for i, lbl in enumerate(labels)}
        efficiency = d['efficiency'] if 'efficiency' in d else None
        return d['u_values'], f_by_triad, d['energy_drift'], efficiency

    u_values = np.linspace(cfg['u_min'], cfg['u_max'], cfg['n_points'])
    f_by_triad = {lbl: [] for lbl in labels}
    energy_drift = []
    efficiency = [] if target_mode is not None else None
    for u in u_values:
        r = precession_sweep(cfg['wave_set'], cfg['sweep_mode'], [u], tf_days=cfg['tf_days'],
                              target_mode_key=target_mode)[0]
        for lbl in labels:
            f_by_triad[lbl].append(r[lbl]['precession_freq'])
        energy_drift.append(r['energy_drift'])
        if target_mode is not None:
            efficiency.append(r['efficiency'])
    f_by_triad = {lbl: np.array(v) for lbl, v in f_by_triad.items()}
    energy_drift = np.array(energy_drift)
    efficiency = np.array(efficiency) if efficiency is not None else None

    save_kwargs = {'u_values': u_values, 'energy_drift': energy_drift}
    save_kwargs.update({f'f_{i}': f_by_triad[lbl] for i, lbl in enumerate(labels)})
    if efficiency is not None:
        save_kwargs['efficiency'] = efficiency
    np.savez(cache_path, **save_kwargs)
    return u_values, f_by_triad, energy_drift, efficiency


def plot_sweep(entry_key, cache_path, path=None, figures_yaml=DEFAULT_FIGURES_PATH,
                wave_sets_yaml=DEFAULT_WAVESETS_PATH):
    """Pure plotting: read ``cache_path`` (must already exist -- see
    ``sweep``) and draw the figure. Never runs the sweep itself, so
    labels/styling can be iterated freely without re-paying its cost.

    Two panels (precession frequency, efficiency) if the cache has an
    ``efficiency`` array (i.e. this entry has a ``target_mode``); one
    panel (the original figure, unchanged) otherwise.
    """
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"{cache_path} does not exist -- run sweep({entry_key!r}, {cache_path!r}) first.")
    with open(figures_yaml) as f:
        cfg = yaml.safe_load(f)[entry_key]
    labels = triad_mode_labels(cfg['wave_set'], wave_sets_yaml)
    d = np.load(cache_path)
    u_values = d['u_values']
    f_by_triad = {lbl: d[f'f_{i}'] for i, lbl in enumerate(labels)}
    efficiency = d['efficiency'] if 'efficiency' in d else None

    apply_house_style()
    markers = ['o', 's', '^', 'v']

    if efficiency is not None:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    else:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    for i, (lbl, mode_str) in enumerate(labels.items()):
        ax.plot(u_values, np.abs(f_by_triad[lbl]), markers[i % len(markers)] + '-', ms=3,
                label=f'{lbl} ({mode_str})', alpha=1.0 if i == 0 else 0.6)
    ax.axhline(0.01, color='grey', ls=':', lw=1)
    ax.set_xlabel(cfg['xlabel'])
    ax.set_ylabel(r'$|$precession frequency$|$ (rad/day)')
    ax.set_title(cfg['title'])
    ax.legend(fontsize=8)

    if efficiency is not None:
        ax2.plot(u_values, 100 * efficiency, 'o-', ms=3, color='C3')
        ax2.set_xlabel(cfg['xlabel'])
        ax2.set_ylabel(r'Efficiency $\mathcal{E}_{\mathrm{avg}}$ (\%)')
        ax2.set_title('Target mode efficiency\n(time-averaged $E_{total}$ normalization)', fontsize=10)

    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("entry", help="entry name in precession_sweep_figures.yaml")
    parser.add_argument("cache", help="path to the .npz cache (computed if missing, else reused)")
    parser.add_argument("path", nargs="?", default=None, help="output PNG path")
    parser.add_argument("--figures-yaml", default=DEFAULT_FIGURES_PATH)
    parser.add_argument("--wave-sets-yaml", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--plot-only", action="store_true",
                         help="skip sweep() entirely and error if the cache is missing, "
                              "instead of computing it -- use once you know the cache exists.")
    args = parser.parse_args()

    if not args.plot_only:
        sweep(args.entry, args.cache, args.figures_yaml, args.wave_sets_yaml)
    plot_sweep(args.entry, args.cache, args.path, args.figures_yaml, args.wave_sets_yaml)
    print(f"Saved to {args.path}" if args.path else "Shown interactively")
