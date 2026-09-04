"""Figure ``fig: hough_examples`` (JFM-template.tex, four subfigures
``fig: hough_rh``/``hough_eg``/``hough_wig``/``hough_rh34``): spatial
structure (height field + velocity arrows) of four representative Hough
harmonics -- RH(1,2), EG(1,1), WG(2,4), RH(3,4) -- at h_e=10 000m.

Thin wrapper around ``rsw_sphere.plotting.hough_spatial_ev.hough_spatial_ev``
(also installed as the ``rsw-hough-mode`` console script), one call per
subfigure, per each subfigure's own ``% python rsw_sphere/plotting/
hough_spatial_ev.py ...`` comment in the LaTeX.

Run:

    python examples/figures/paper_figure002_hough_examples.py
"""
import os

import _bootstrap  # noqa: F401 -- repo root on sys.path

from rsw_sphere.plotting.hough_spatial_ev import hough_spatial_ev

#: (output filename, m, n, alpha) -- matches each subfigure's own LaTeX
#: comment exactly (alpha: 1=EG, 2=WG, 3=RH).
MODES = [
    ("hough_RH_1_2.png", 1, 2, 3),
    ("hough_EG_1_1.png", 1, 1, 1),
    ("hough_WIG_2_4.png", 2, 4, 2),
    ("hough_RH_3_4.png", 3, 4, 3),
]

DEFAULT_OUT_DIR = os.path.join(_bootstrap.ROOT, "outputs", "figures", "paper", "hough_examples")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", nargs="?", default=DEFAULT_OUT_DIR)
    parser.add_argument("--he", type=float, default=10000.0, help="equivalent height (m)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for fname, m, n, alpha in MODES:
        path = os.path.join(args.out_dir, fname)
        hough_spatial_ev(m, n, alpha, h_e=args.he, path=path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
