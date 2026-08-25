"""Integrate a resonant triad's nonlinear amplitude equations and plot the
energy exchange between its three Hough modes.

For dispersion-relation and single-mode diagnostic plots (no time
integration), see run_diagnostics.py instead — both scripts read the same
configs.yaml, each using only the section relevant to it.

Run from the command line:

    python run_dynamics.py --config configs.yaml

A plain triad (docs/code_guide.md: "a plain triad is the degenerate
3-modes/1-triad case" of WaveSet) is configs.yaml's own single hardcoded
Triad:/Dynamics: block. Pass --wave-set instead to integrate a registered
quartet/quintet (rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs)
-- tf/h then default to that wave set's own registry settings rather than
configs.yaml's Dynamics: block:

    python run_dynamics.py --wave-set quartet_rh_preference
    python run_dynamics.py --wave-set quartet_rh_preference --specs examples/wave_sets_section_3.yaml
"""
import argparse
import os

import numpy as np
import yaml

from rsw_sphere.plotting.dynamic_three_waves import triad_evolution
from rsw_sphere.plotting.wave_set_dynamics import wave_set_energy_evolution_from_spec
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs


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
    parser.add_argument(
        "--wave-set", type=str, default=None,
        help="registry role key (rsw_sphere.dynamics.wave_set_specs) -- "
             "when given, integrates that quartet/quintet instead of "
             "configs.yaml's Triad:/Dynamics: block.")
    parser.add_argument(
        "--specs", type=str, default=DEFAULT_WAVESETS_PATH,
        help=f"wave-set registry YAML, only used with --wave-set. "
             f"Default: {DEFAULT_WAVESETS_PATH}")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        parser.error(f"config file not found: {args.config!r}")

    try:
        output = config['OUTPUT_PATH']
    except KeyError as e:
        parser.error(f"{args.config!r} is missing required key: {e}")

    os.makedirs(output, exist_ok=True)
    print(f'Output directory: {os.path.abspath(output)}')

    if args.wave_set:
        specs = load_wave_set_specs(args.specs)
        if args.wave_set not in specs:
            parser.error(f"--wave-set {args.wave_set!r} not found in {args.specs!r} "
                         f"(available: {list(specs)})")
        spec = specs[args.wave_set]

        print(f'Starting WaveSet dynamics: {args.wave_set} ({spec.display_label})')
        dynamics_path = f'{output}/{args.wave_set}_dynamics.png'
        result = wave_set_energy_evolution_from_spec(spec, path=dynamics_path)
        print(f'  wrote {os.path.abspath(dynamics_path)}')
        print(f'  drift={result["drift"]:.3e}, dEK={dict(zip(result["labels"], result["dEK"]))}')
        print('Dynamics finished')
        return

    try:
        h_e = config['h_e']
        triad = config['Triad']
        dynamics = config['Dynamics']
    except KeyError as e:
        parser.error(f"{args.config!r} is missing required key: {e}")

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
    dynamics_path = f'{output}/dynamics.png'
    triad_evolution(h_e, m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c,
                     u_a, u_b, u_c, t0, tf, h, dynamics_path)
    print(f'  wrote {os.path.abspath(dynamics_path)}')
    print('Dynamics finished')


if __name__ == "__main__":
    main()
