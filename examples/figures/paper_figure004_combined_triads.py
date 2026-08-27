"""Figure ``fig: combined`` (JFM-template.tex, ``sec: combined``):
combined Rossby-gravity triads. Top row: Triad C (EG(1,1)+RH(3,4)+RH(4,5)),
target EG(1,1). Middle row: the same Triad C, retargeted to RH(3,4).
Bottom row: Triad D (EG(6,9)+RH(1,7)+EG(7,9)), target EG(6,9). Each row is
(efficiency map, energy time integration).

Built directly on the current registry/driver machinery
(``_triad_panel_row.triad_row``, itself ``run_sweep.py``'s own unified 2D
engine + ``rsw_sphere.plotting.energy_evolution``) applied to the two
registered ``triad_kelvin_rossby_flow``/``triad_gravity_with_rossby_catalyst``
wave sets -- the dedicated per-figure script the LaTeX's own stale comment
(pointing at the composite-panel assembler
``examples_legacy/make_section22_figures.py``) should reference instead.

Run:

    python examples/figures/paper_figure004_combined_triads.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EXAMPLES_FIGURES = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_FIGURES not in sys.path:
    sys.path.insert(0, _EXAMPLES_FIGURES)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.style import apply_house_style
from _triad_panel_row import triad_row

#: (wave-set key, target-mode index, energy-panel velocity override).
#: triad_kelvin_rossby_flow's spec.modes: 0=RH(4,5) (sum), 1=EG(1,1),
#: 2=RH(3,4) -- targets 1/2 per the paper's two worked directions.
#: triad_gravity_with_rossby_catalyst's spec.modes: 0=EG(7,9) (sum, at
#: rest by registration), 1=EG(6,9) (target), 2=RH(1,7). With the target
#: forced to rest, the registered velocities alone would leave only
#: RH(1,7) driven (too weakly coupled to move EG(6,9)) -- the override
#: below puts the initial energy on EG(7,9) instead, reproducing the
#: near-total EG<->EG exchange the caption describes.
ROWS = [
    ("triad_kelvin_rossby_flow", 1, None),               # Triad C, target EG(1,1)
    ("triad_kelvin_rossby_flow", 2, None),                # Triad C, target RH(3,4)
    ("triad_gravity_with_rossby_catalyst", 1, (50.0, 0.0, 20.0)),  # Triad D, target EG(6,9)
]

DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "paper", "paper_figure004_combined_triads.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--n-grid", type=int, default=15)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)

    apply_house_style()
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(12, 4.5 * len(ROWS)), squeeze=False)
    for row, (key, target, energy_velocities) in zip(axes, ROWS):
        print(f"computing {key} (target={target}) ...")
        triad_row(specs[key], target, row[0], row[1], n_grid=args.n_grid,
                  energy_velocities=list(energy_velocities) if energy_velocities else None)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    fig.savefig(args.path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(args.path)}")


if __name__ == "__main__":
    main()
