"""Item 4, generalized: frequency/period comparison for ANY target mode
in quartet_rossby_gravity_influence, between that mode integrated alone in each
constituent triad that contains it (auto-discovered -- 1 sub-triad for a
private mode, 2 for a shared mode, member or sum role alike) and the same
mode in the full four-wave quartet.

Supersedes rh45_frequency_comparison.py's hardcoded triad0/triad1 --
kept as a fixed worked example; this script is the general version.

See rh45_frequency_comparison.py's own docstring for the normalization
rationale (peak-normalization, not total-power) and the known open
problem with naive argmax(|difference|) picking up a peak-position-shift
dipole artifact rather than genuinely new spectral content -- this
script prints the SAME argmax number for reference but does not attempt
to fix that artifact yet (still an open design question).

Run:

    python examples/freqshift_novelty/target_frequency_comparison.py --target "RH(3,4)"
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.utilities.periods import dominant_periods, _power_spectrum
from run_dynamics import run_dynamics

WAVE_SET_KEY = "quartet_rossby_gravity_influence"


def _peak_normalized_spectrum(t_days, E_j):
    periods, power = _power_spectrum(t_days, E_j)
    peak = power.max() if len(power) else 1.0
    return periods, power / peak if peak > 0 else power


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help='e.g. "RH(3,4)"')
    parser.add_argument("--wave-set", default=WAVE_SET_KEY)
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("--xmax", type=float, default=3.0)
    args = parser.parse_args()

    specs = load_wave_set_specs()
    spec = specs[args.wave_set]
    config = RunConfig.from_wave_set(spec, plot=False)
    results = run_dynamics(config, write_table=False)

    full = results["full"]
    sub_units = {name: r for name, r in results.items() if name != "full"}
    containing = {name: r for name, r in sub_units.items() if args.target in r["labels"]}
    if not containing:
        raise SystemExit(f"{args.target!r} not found in any sub-triad of {args.wave_set!r} "
                          f"(available: {full['labels']})")

    j_full = full["labels"].index(args.target)
    t_full, E_full = full["t"], full["E"][:, j_full]
    r_full = dominant_periods(t_full, E_full)
    p_full, pow_full_n = _peak_normalized_spectrum(t_full, E_full)

    print(f"=== {args.target} in {args.wave_set}: period comparison ===")
    print(f"  full quartet: period_global={r_full['period_global']:.4f}d  "
          f"period_local_max={r_full['period_local_max']}")

    common_periods = np.linspace(0.01, args.xmax, 4000)
    interp_full = np.interp(common_periods, p_full[::-1], pow_full_n[::-1])

    apply_house_style()
    fig, axes = plt.subplots(3, 1, figsize=(8, 11))
    colors = ["tab:orange", "tab:green", "tab:purple"]

    axes[0].plot(t_full, E_full, color="black", label="full quartet")
    axes[1].plot(p_full, pow_full_n, color="black", label="full quartet")
    axes[2].axhline(0, color="gray", lw=0.7)

    for color, (name, unit) in zip(colors, containing.items()):
        j_sub = unit["labels"].index(args.target)
        t_sub, E_sub = unit["t"], unit["E"][:, j_sub]
        r_sub = dominant_periods(t_sub, E_sub)
        shift = 100 * (r_full['period_global'] - r_sub['period_global']) / r_sub['period_global']
        print(f"  {name} alone: period_global={r_sub['period_global']:.4f}d  "
              f"(full vs. this triad's own dominant period: {shift:+.1f}%)")

        p_sub, pow_sub_n = _peak_normalized_spectrum(t_sub, E_sub)
        interp_sub = np.interp(common_periods, p_sub[::-1], pow_sub_n[::-1])
        diff = interp_full - interp_sub
        novelty_period = common_periods[np.argmax(np.abs(diff))]
        print(f"    naive argmax(|full - {name}|) novelty period: {novelty_period:.4f}d "
              "(may be a peak-shift dipole artifact, not genuine new content -- inspect the plot)")

        axes[0].plot(t_sub, E_sub, color=color, ls="--", alpha=0.8, label=f"{name} alone")
        axes[1].plot(p_sub, pow_sub_n, color=color, ls="--", label=f"{name} alone")
        axes[2].plot(common_periods, diff, color=color,
                     label=f"full − {name} (argmax peak={novelty_period:.3f}d)")

    axes[0].set_xlabel("Time (days)"); axes[0].set_ylabel("KE = |A|^2 (nondim.)")
    axes[0].set_title(f"{args.target}: time series"); axes[0].legend(fontsize=8)
    axes[1].set_xlim(0, args.xmax); axes[1].set_xlabel("Period (days)")
    axes[1].set_ylabel("Peak-normalized power")
    axes[1].set_title(f"{args.target}: spectrum (each normalized by its own peak)")
    axes[1].legend(fontsize=8)
    axes[2].set_xlim(0, args.xmax); axes[2].set_xlabel("Period (days)")
    axes[2].set_ylabel("Normalized power difference")
    axes[2].set_title(f"{args.target}: difference spectrum (novelty-period prototype)")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{args.target} in {args.wave_set}: isolated-triad vs. full-quartet frequency comparison")
    fig.tight_layout()
    path = args.path or os.path.join(
        _ROOT, "outputs", "figures", "freqshift_novelty",
        f"{args.target.replace('(', '').replace(')', '').replace(',', '_')}_frequency_comparison.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
