"""Plot the dispersion relation (plain + publication-quality "fancy" variant)
and per-mode Hough harmonic diagnostics -- latitudinal profile + derivatives
and the full spatial (lambda, phi) eigenvector pattern -- (no time
integration).

For triad amplitude-equation integration and energy exchange, see
run_dynamics.py instead — both scripts read the same configs.yaml, each
using only the section relevant to it.

Run from the command line:

    python run_diagnostics.py --config configs.yaml
"""
import argparse
import os

import yaml

from rsw_sphere.plotting.dispersion_relation import dispersion_relation
from rsw_sphere.plotting.dispersion_relation_fancy import dispersion_relation as dispersion_relation_fancy
from rsw_sphere.plotting.hough_and_derivatives import hough_and_derivatives, mode_tag
from rsw_sphere.plotting.hough_spatial_ev import hough_spatial_ev


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default="configs.yaml",
        help="path to a YAML config (see the default configs.yaml). "
             "Default: configs.yaml"
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        parser.error(f"config file not found: {args.config!r}")

    try:
        output = config['OUTPUT_PATH']
        h_e = config['h_e']
        triad = config['Triad']
    except KeyError as e:
        parser.error(f"{args.config!r} is missing required key: {e}")

    os.makedirs(output, exist_ok=True)
    print(f'Output directory: {os.path.abspath(output)}')

    if config.get('dispersion_relation'):
        dispersion_path = f'{output}/dispersion_relation.png'
        dispersion_relation(h_e, dispersion_path)
        print(f'  wrote {os.path.abspath(dispersion_path)}')

        dispersion_fancy_path = f'{output}/dispersion_relation_fancy.png'
        dispersion_relation_fancy(h_e=h_e, path=dispersion_fancy_path)
        print(f'  wrote {os.path.abspath(dispersion_fancy_path)}')

    for name, mode in triad.items():
        if not mode.get('show_mode'):
            continue
        m, n, alpha = mode['m'], mode['n'], mode['alpha']
        tag = mode_tag(m, n, alpha)
        print(f'Plotting Hough mode {name}: (m={m}, n={n}, alpha={alpha}) [{tag}]')
        mode_path = f'{output}/{tag}/'
        os.makedirs(mode_path, exist_ok=True)

        hough_and_derivatives(m, n, alpha, h_e, mode_path)
        print(f'  wrote {os.path.abspath(mode_path + f"Hough_harmonic_{tag}.png")}')
        print(f'  wrote {os.path.abspath(mode_path + f"derivatives_{tag}.png")}')

        spatial_path = f'{mode_path}Hough_spatial_{tag}.png'
        hough_spatial_ev(m, n, alpha, h_e=h_e, path=spatial_path)
        print(f'  wrote {os.path.abspath(spatial_path)}')

    print('Diagnostics finished')


if __name__ == "__main__":
    main()
