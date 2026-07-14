"""Integrate a resonant triad's nonlinear amplitude equations and plot the
energy exchange between its three Hough modes.

For dispersion-relation and single-mode diagnostic plots (no time
integration), see run_diagnostics.py instead — both scripts read the same
configs.yaml, each using only the section relevant to it.

Run from the command line:

    python run_dynamics.py --config configs.yaml
"""
import argparse
import os

import numpy as np
import yaml

from rsw_sphere.plotting.dynamic_three_waves import triad_evolution


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
        dynamics = config['Dynamics']
    except KeyError as e:
        parser.error(f"{args.config!r} is missing required key: {e}")

    os.makedirs(output, exist_ok=True)

    if not dynamics['show_dynamics']:
        print('show_dynamics is false in the config — nothing to do.')
        return

    m_a, n_a, alpha_a = triad['mode_a']['m'], triad['mode_a']['n'], triad['mode_a']['alpha']
    m_b, n_b, alpha_b = triad['mode_b']['m'], triad['mode_b']['n'], triad['mode_b']['alpha']
    m_c, n_c, alpha_c = triad['mode_c']['m'], triad['mode_c']['n'], triad['mode_c']['alpha']
    u_a, u_b, u_c = triad['mode_a']['u'], triad['mode_b']['u'], triad['mode_c']['u']

    t0 = dynamics['t0']
    tf = dynamics['tf'] * 4 * np.pi
    h = dynamics['h']

    print('Starting Triad dynamics')
    triad_evolution(h_e, m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c,
                     u_a, u_b, u_c, t0, tf, h, f'{output}/dynamics.png')
    print('Dynamics finished')


if __name__ == "__main__":
    main()
