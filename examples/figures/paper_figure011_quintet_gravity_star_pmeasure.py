"""Figure ``fig: 4eff3`` (JFM-template.tex, ``sec: quintet``,
``Figures/quintet_gravity_star_pmeasure.png``): P-measure (%) for the RH
modes (4,5) and (3,4) in the star quintet, sweeping RH(1,2) and EG(1,1)'s
own initial zonal velocities (the three RH modes' registered velocities
are fixed at 30 m/s, EG(7,9) fixed at its registered 0 m/s). Coarse grid
(10x10), matching the figure's own caption.

Built on ``rsw_sphere.utilities.pmeasure.p_measure_sweep`` (cached to
``.npz``) + ``rsw_sphere.plotting.pmeasure_map.plot_p_measure_map`` for
the registered ``quintet_gravity_star`` wave set.

Run:

    python examples/figures/paper_figure011_quintet_gravity_star_pmeasure.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.pmeasure_map import plot_p_measure_map
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.sweeps import wave_set_cache_key_hash
from rsw_sphere.utilities.pmeasure import p_measure_sweep

WAVE_SET_KEY = "quintet_gravity_star"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure011_quintet_gravity_star_pmeasure.png")
SWEPT_KEYS = ("c", "d")   # RH(1,2), EG(1,1)
TARGET_KEYS = ("a", "b")  # RH(4,5), RH(3,4)
N_GRID = 10               # caption: "Coarse grid (10x10)"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--n-grid", type=int, default=N_GRID)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[WAVE_SET_KEY]
    triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
    swept_indices = tuple(spec.index(k) for k in SWEPT_KEYS)
    target_indices = [spec.index(k) for k in TARGET_KEYS]
    tf_days = spec.settings["tf_days"]
    h = spec.settings["h"]
    t_f_nondim = tf_days * 4 * 3.141592653589793

    fixed_velocities = {
        i: spec.velocities[i] for i in range(spec.n_modes())
        if i not in swept_indices and i not in target_indices
    }
    for i in target_indices:
        if i not in swept_indices:
            fixed_velocities.setdefault(i, spec.velocities[i])

    out_dir = os.path.dirname(args.path)
    os.makedirs(out_dir, exist_ok=True)
    cache_hash = wave_set_cache_key_hash(
        spec.modes, triads, spec.h_e, swept_indices, fixed_velocities,
        target_indices, spec.reference_triad, args.n_grid, t_f_nondim, h)
    cache_path = os.path.join(out_dir, f"pmeasure_{cache_hash}.npz")
    if args.clear_cache and os.path.exists(cache_path):
        os.remove(cache_path)

    print(f"P-measure sweep (n_grid={args.n_grid}) -> cache {os.path.basename(cache_path)}")
    result = p_measure_sweep(
        spec.modes, triads, spec.h_e, swept_indices, fixed_velocities,
        target_indices, reference_triad=spec.reference_triad,
        n_grid=args.n_grid, tf_days=tf_days, h=h, cache_path=cache_path,
        verbose=True, progress_label=WAVE_SET_KEY)

    label1 = _mode_label(*spec.modes[swept_indices[0]])
    label2 = _mode_label(*spec.modes[swept_indices[1]])

    apply_house_style()
    fig, axes = plt.subplots(1, len(target_indices), figsize=(6 * len(target_indices), 5))
    for k, ax in enumerate(axes):
        _, _, cs, n_clipped = plot_p_measure_map(
            result['U1'], result['U2'], result['P'][..., k],
            xlabel=f'{label1} - zonal velocity (m/s)',
            ylabel=f'{label2} - zonal velocity (m/s)',
            title=f"P: {result['labels'][k]}", ax=ax)
        if n_clipped:
            print(f"  {result['labels'][k]}: {n_clipped} grid point(s) clipped to +-100%")
    fig.tight_layout()
    fig.savefig(args.path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {os.path.abspath(args.path)}")
    print(f"drift range: [{result['drift'].min():.3e}, {result['drift'].max():.3e}]")


if __name__ == "__main__":
    main()
