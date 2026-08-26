"""Figure ``fig: 1`` (JFM-template.tex, ``\\label{fig: 1}``, end of the
Hough-harmonics background section): dispersion relation of the RSW
equations on the sphere at h_e=10 000m -- frequency and period vs. zonal
wavenumber for the EG/WG/RH wave families.

Thin wrapper around ``rsw_sphere.plotting.dispersion_relation_fancy``
(also installed as the ``rsw-dispersion`` console script), per that
figure's own ``% Generated with ...`` comment in the LaTeX.

Run:

    python examples/figures/paper_figure001_dispersion_relation.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.plotting.dispersion_relation_fancy import dispersion_relation

DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "paper", "paper_figure001_dispersion_relation.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--he", type=float, default=10000.0, help="equivalent height (m)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    dispersion_relation(h_e=args.he, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")


if __name__ == "__main__":
    main()
