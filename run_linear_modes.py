"""Plot the dispersion relation (plain + publication-quality "fancy" variant)
and per-mode Hough harmonic diagnostics -- latitudinal profile + derivatives
and the full spatial (lambda, phi) eigenvector pattern -- (no time
integration). Per-mode plots go under <output_root>/figures/linear/<mode_tag>/.

For triad/wave-set integration, see run_dynamics.py instead.

Modes come from the WaveSet registry (rsw_sphere.dynamics.wave_set_specs) --
a triad is just the 1-triad case of a wave set, so there is no separate
plain-Triad: config path anymore. Every mode of one --wave-set is plotted:

    python run_linear_modes.py --wave-set triad_kelvin_rossby_flow
    python run_linear_modes.py --wave-set quartet_rossby_kelvin --specs examples/wave_sets_custom.yaml

Pass --run-all instead of --wave-set to plot every mode of every wave set
in the registry (--specs), shared modes plotted once:

    python run_linear_modes.py --run-all
"""
import argparse
import os

from rsw_sphere.plotting.dispersion_relation import dispersion_relation
from rsw_sphere.plotting.dispersion_relation_fancy import dispersion_relation as dispersion_relation_fancy
from rsw_sphere.plotting.hough_and_derivatives import hough_and_derivatives, mode_tag
from rsw_sphere.plotting.hough_spatial_ev import hough_spatial_ev
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs


def _plot_mode(name, m, n, alpha, h_e, output):
    tag = mode_tag(m, n, alpha)
    print(f'Plotting Hough mode {name}: (m={m}, n={n}, alpha={alpha}) [{tag}]')
    mode_path = f'{output}/linear/{tag}/'
    os.makedirs(mode_path, exist_ok=True)

    hough_and_derivatives(m, n, alpha, h_e, mode_path)
    print(f'  wrote {os.path.abspath(mode_path + f"Hough_harmonic_{tag}.png")}')
    print(f'  wrote {os.path.abspath(mode_path + f"derivatives_{tag}.png")}')

    spatial_path = f'{mode_path}Hough_spatial_{tag}.png'
    hough_spatial_ev(m, n, alpha, h_e=h_e, path=spatial_path)
    print(f'  wrote {os.path.abspath(spatial_path)}')


def _plot_wave_set(spec, output):
    for mk, (m, n, alpha) in zip(spec.mode_keys, spec.modes):
        _plot_mode(mk, m, n, alpha, spec.h_e, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wave-set", type=str, default=None,
        help="registry role key (rsw_sphere.dynamics.wave_set_specs) -- "
             "every mode in the wave set is plotted. Required unless --run-all.")
    parser.add_argument(
        "--specs", type=str, default=DEFAULT_WAVESETS_PATH,
        help=f"wave-set registry YAML. Default: {DEFAULT_WAVESETS_PATH}")
    parser.add_argument(
        "--run-all", action="store_true",
        help="plot every mode of every wave set in the registry (--specs), "
             "instead of one --wave-set.")
    parser.add_argument(
        "--output-root", type=str, default="outputs",
        help="figures go under <output-root>/figures/. Default: outputs")
    parser.add_argument(
        "--no-dispersion-relation", action="store_true",
        help="skip the dispersion-relation plot(s), only plot per-mode Hough harmonics.")
    args = parser.parse_args()

    if args.run_all and args.wave_set:
        parser.error("--run-all and --wave-set are mutually exclusive")
    if not args.run_all and not args.wave_set:
        parser.error("either --wave-set or --run-all is required")

    specs = load_wave_set_specs(args.specs)
    if args.wave_set and args.wave_set not in specs:
        parser.error(f"--wave-set {args.wave_set!r} not found in {args.specs!r} "
                     f"(available: {list(specs)})")

    output = os.path.join(args.output_root, "figures")
    os.makedirs(output, exist_ok=True)
    print(f'Output directory: {os.path.abspath(output)}')

    if not args.no_dispersion_relation:
        active_specs = specs.values() if args.run_all else [specs[args.wave_set]]
        h_e_values = sorted({s.h_e for s in active_specs})
        for h_e_i in h_e_values:
            suffix = "" if len(h_e_values) == 1 else f"_he{int(h_e_i)}"
            dispersion_path = f'{output}/dispersion_relation{suffix}.png'
            dispersion_relation(h_e_i, dispersion_path)
            print(f'  wrote {os.path.abspath(dispersion_path)}')

            dispersion_fancy_path = f'{output}/dispersion_relation_fancy{suffix}.png'
            dispersion_relation_fancy(h_e=h_e_i, path=dispersion_fancy_path)
            print(f'  wrote {os.path.abspath(dispersion_fancy_path)}')

    if args.run_all:
        for key, spec in specs.items():
            print(f'=== {key} ({spec.display_label or key}) ===')
            _plot_wave_set(spec, output)
    else:
        _plot_wave_set(specs[args.wave_set], output)

    print('Linear-mode plots finished')


if __name__ == "__main__":
    main()
