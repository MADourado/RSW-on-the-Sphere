"""Plot the dispersion relation and per-mode Hough harmonic diagnostics
(no time integration).

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
from rsw_sphere.plotting.hough_and_derivatives import hough_and_derivatives


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

    if config.get('dispersion_relation'):
        print('Plotting dispersion relation')
        dispersion_relation(h_e, f'{output}/dispersion_relation.png')

    for name, mode in triad.items():
        if not mode.get('show_mode'):
            continue
        print(f'Plotting Hough mode {name}: (m={mode["m"]}, n={mode["n"]}, alpha={mode["alpha"]})')
        mode_path = f'{output}/{name}/'
        os.makedirs(mode_path, exist_ok=True)
        hough_and_derivatives(mode['m'], mode['n'], mode['alpha'], h_e, mode_path)

    print('Diagnostics finished')


if __name__ == "__main__":
    main()
