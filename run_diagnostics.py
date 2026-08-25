"""Plot the dispersion relation (plain + publication-quality "fancy" variant)
and per-mode Hough harmonic diagnostics -- latitudinal profile + derivatives
and the full spatial (lambda, phi) eigenvector pattern -- (no time
integration).

For triad amplitude-equation integration and energy exchange, see
run_dynamics.py instead — both scripts read the same configs.yaml, each
using only the section relevant to it.

Run from the command line:

    python run_diagnostics.py --config configs.yaml

A plain triad (docs/code_guide.md: "a plain triad is the degenerate
3-modes/1-triad case" of WaveSet) is configs.yaml's own single hardcoded
Triad: block. Pass --wave-set instead to source modes from the WaveSet
registry (rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs) --
every mode in the wave set is plotted (the registry has no per-mode
show_mode flag):

    python run_diagnostics.py --wave-set quartet_rh_preference
    python run_diagnostics.py --wave-set quartet_rh_preference --specs examples/wave_sets_section_3.yaml
"""
import argparse
import os

import yaml

from rsw_sphere.plotting.dispersion_relation import dispersion_relation
from rsw_sphere.plotting.dispersion_relation_fancy import dispersion_relation as dispersion_relation_fancy
from rsw_sphere.plotting.hough_and_derivatives import hough_and_derivatives, mode_tag
from rsw_sphere.plotting.hough_spatial_ev import hough_spatial_ev
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _plot_mode(name, m, n, alpha, h_e, output):
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
             "when given, modes are sourced from the WaveSet registry "
             "instead of configs.yaml's Triad: block. Every mode in the "
             "wave set is plotted.")
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

    if args.wave_set:
        specs = load_wave_set_specs(args.specs)
        if args.wave_set not in specs:
            parser.error(f"--wave-set {args.wave_set!r} not found in {args.specs!r} "
                         f"(available: {list(specs)})")
        spec = specs[args.wave_set]
        h_e = spec.h_e
    else:
        try:
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

    if args.wave_set:
        for mk, (m, n, alpha) in zip(spec.mode_keys, spec.modes):
            _plot_mode(mk, m, n, alpha, h_e, output)
    else:
        for name, mode in triad.items():
            if not mode.get('show_mode'):
                continue
            _plot_mode(name, mode['m'], mode['n'], mode['alpha'], h_e, output)

    print('Diagnostics finished')


if __name__ == "__main__":
    main()
