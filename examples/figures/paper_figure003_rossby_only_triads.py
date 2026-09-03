"""Figure ``fig: rossby_only`` (JFM-template.tex, ``sec: rossbyonly``):
Rossby-only triads. Top row: Triad A (RH(3,10)+RH(1,2)+RH(4,5)), target
RH(3,10); bottom row: Triad B (RH(3,4)+RH(1,2)+RH(4,5)), target RH(3,4).
Each row is (efficiency map, energy time integration).

Built directly on the current registry/driver machinery
(``_triad_panel_row.triad_row``, itself ``run_sweep.py``'s own unified 2D
engine + ``rsw_sphere.plotting.energy_evolution``) applied to the two
registered ``triad_rossby_only_*`` wave sets -- the dedicated per-figure
script the LaTeX's own stale comment (pointing at the composite-panel
assembler ``examples_legacy/make_section22_figures.py``) should reference
instead.

Run:

    python examples/figures/paper_figure003_rossby_only_triads.py
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

#: (wave-set key, target-mode index into spec.modes). spec.modes index 0
#: is the sum mode RH(4,5) for both triads; the paper's own target is the
#: private "b" mode (index 1: RH(3,10) for Triad A, RH(3,4) for Triad B).
ROWS = [
    ("triad_rossby_only_near_resonant", 1),  # Triad A, target RH(3,10)
    ("triad_rossby_only_non_resonant", 1),   # Triad B, target RH(3,4)
]

DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "paper", "paper_figure003_rossby_only_triads.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--n-grid", type=int, default=15)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)

    apply_house_style(base_size=14)
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(12, 4.5 * len(ROWS)), squeeze=False)
    for row, (key, target) in zip(axes, ROWS):
        print(f"computing {key} (target={target}) ...")
        triad_row(specs[key], target, row[0], row[1], n_grid=args.n_grid)
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    fig.savefig(args.path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.abspath(args.path)}")


if __name__ == "__main__":
    main()
