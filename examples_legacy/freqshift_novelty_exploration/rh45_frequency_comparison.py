"""Item 4, focused on the single target RH(4,5) in quartet_rossby_gravity_influence:
frequency/period comparison between RH(4,5) integrated alone in each of
its two constituent triads (triad0: RH(3,4)+RH(4,5)+WG(7,9); triad1:
WG(3,9)+RH(4,5)+WG(7,9)) and RH(4,5) in the full four-wave quartet.

Three views, all on the KE = |A|^2 time series (see draft_spectra_amp_vs_ke.py
for the KE-vs-|A| check that motivated this choice):

1. Time series overlay (does the full-quartet trajectory look different
   from either triad-alone trajectory, by eye).
2. Power-spectrum overlay (all three spectra, normalized by their own
   peak so shapes are comparable despite different absolute amplitudes --
   see the module docstring below for why peak- rather than total-power
   normalization was used here).
3. Difference spectrum (full quartet's own normalized spectrum minus each
   sub-triad's own normalized spectrum) and the "novelty period": the
   dominant peak of that difference. This is the actual prototype of the
   metric item 4 is designing -- draft/exploratory, not yet wired into
   rsw_sphere.utilities.periods or the sweep_2d registry.

Normalization choice (open question in the plan, decided here from the
data): peak-normalization (divide by max power) was used rather than
total-power normalization, because RH(4,5)'s own total power is
dominated by whichever peak happens to be tallest -- total-power
normalization would let a mode with one huge, narrow peak and no other
structure swamp a difference spectrum against a mode with two comparably-
sized peaks, even though the SHAPE difference (an extra peak) is the
signal of interest here, not the overall power scale. Peak-normalization
puts both spectra's tallest features on the same 0-1 scale first, so a
genuinely new peak stands out in the difference regardless of the two
runs' different absolute energies.

Run:

    python examples/freqshift_novelty/rh45_frequency_comparison.py
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
TARGET_LABEL = "RH(4,5)"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "freqshift_novelty",
                               "rh45_frequency_comparison.png")


def _peak_normalized_spectrum(t_days, E_j):
    periods, power = _power_spectrum(t_days, E_j)
    peak = power.max() if len(power) else 1.0
    return periods, power / peak if peak > 0 else power


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--xmax", type=float, default=3.0)
    args = parser.parse_args()

    specs = load_wave_set_specs()
    spec = specs[WAVE_SET_KEY]
    config = RunConfig.from_wave_set(spec, plot=False)
    results = run_dynamics(config, write_table=False)

    full = results["full"]
    containing_names = [name for name in results if name != "full" and TARGET_LABEL in results[name]["labels"]]
    triad0_name, triad1_name = containing_names
    triad0, triad1 = results[triad0_name], results[triad1_name]

    j_full = full["labels"].index(TARGET_LABEL)
    j_t0 = triad0["labels"].index(TARGET_LABEL)
    j_t1 = triad1["labels"].index(TARGET_LABEL)

    t_full, E_full = full["t"], full["E"][:, j_full]
    t_t0, E_t0 = triad0["t"], triad0["E"][:, j_t0]
    t_t1, E_t1 = triad1["t"], triad1["E"][:, j_t1]

    r_full = dominant_periods(t_full, E_full)
    r_t0 = dominant_periods(t_t0, E_t0)
    r_t1 = dominant_periods(t_t1, E_t1)

    print(f"=== {TARGET_LABEL} in {WAVE_SET_KEY}: period comparison ===")
    print(f"  full quartet:      period_global={r_full['period_global']:.4f}d  "
          f"period_local_max={r_full['period_local_max']}")
    print(f"  {triad0_name} alone: period_global={r_t0['period_global']:.4f}d")
    print(f"  {triad1_name} alone: period_global={r_t1['period_global']:.4f}d")
    shift_vs_t0 = 100 * (r_full['period_global'] - r_t0['period_global']) / r_t0['period_global']
    shift_vs_t1 = 100 * (r_full['period_global'] - r_t1['period_global']) / r_t1['period_global']
    print(f"  full vs. triad0 own dominant period: {shift_vs_t0:+.1f}%")
    print(f"  full vs. triad1 own dominant period: {shift_vs_t1:+.1f}%")
    print("  (small/large here is not yet the point -- the FULL spectrum plainly contains BOTH "
          "triads' own timescales at once, which a single scalar period-shift number can't represent; "
          "see the difference-spectrum panel below.)")

    # Peak-normalized spectra + difference spectra (interpolated onto a
    # common period grid so full/sub can be subtracted point-by-point).
    p_full, pow_full_n = _peak_normalized_spectrum(t_full, E_full)
    p_t0, pow_t0_n = _peak_normalized_spectrum(t_t0, E_t0)
    p_t1, pow_t1_n = _peak_normalized_spectrum(t_t1, E_t1)

    common_periods = np.linspace(0.01, args.xmax, 4000)
    interp_full = np.interp(common_periods, p_full[::-1], pow_full_n[::-1])
    interp_t0 = np.interp(common_periods, p_t0[::-1], pow_t0_n[::-1])
    interp_t1 = np.interp(common_periods, p_t1[::-1], pow_t1_n[::-1])

    diff_vs_t0 = interp_full - interp_t0
    diff_vs_t1 = interp_full - interp_t1
    novelty_period_vs_t0 = common_periods[np.argmax(np.abs(diff_vs_t0))]
    novelty_period_vs_t1 = common_periods[np.argmax(np.abs(diff_vs_t1))]
    print(f"  novelty period (full - triad0, argmax|diff|): {novelty_period_vs_t0:.4f}d")
    print(f"  novelty period (full - triad1, argmax|diff|): {novelty_period_vs_t1:.4f}d")

    apply_house_style()
    fig, axes = plt.subplots(3, 1, figsize=(8, 11))

    axes[0].plot(t_full, E_full, color="black", label="full quartet")
    axes[0].plot(t_t0, E_t0, color="tab:orange", ls="--", alpha=0.8, label=f"{triad0_name} alone")
    axes[0].plot(t_t1, E_t1, color="tab:green", ls="--", alpha=0.8, label=f"{triad1_name} alone")
    axes[0].set_xlabel("Time (days)"); axes[0].set_ylabel("KE = |A|^2 (nondim.)")
    axes[0].set_title(f"{TARGET_LABEL}: time series"); axes[0].legend(fontsize=8)

    axes[1].plot(p_full, pow_full_n, color="black", label="full quartet")
    axes[1].plot(p_t0, pow_t0_n, color="tab:orange", ls="--", label=f"{triad0_name} alone")
    axes[1].plot(p_t1, pow_t1_n, color="tab:green", ls="--", label=f"{triad1_name} alone")
    axes[1].set_xlim(0, args.xmax); axes[1].set_xlabel("Period (days)")
    axes[1].set_ylabel("Peak-normalized power")
    axes[1].set_title(f"{TARGET_LABEL}: spectrum (each normalized by its own peak)")
    axes[1].legend(fontsize=8)

    axes[2].axhline(0, color="gray", lw=0.7)
    axes[2].plot(common_periods, diff_vs_t0, color="tab:orange",
                 label=f"full − triad0 (novelty peak={novelty_period_vs_t0:.3f}d)")
    axes[2].plot(common_periods, diff_vs_t1, color="tab:green",
                 label=f"full − triad1 (novelty peak={novelty_period_vs_t1:.3f}d)")
    axes[2].set_xlim(0, args.xmax); axes[2].set_xlabel("Period (days)")
    axes[2].set_ylabel("Normalized power difference")
    axes[2].set_title(f"{TARGET_LABEL}: difference spectrum (novelty-period prototype)")
    axes[2].legend(fontsize=8)

    fig.suptitle(f"{TARGET_LABEL} in {WAVE_SET_KEY}: isolated-triad vs. full-quartet frequency comparison")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    fig.savefig(args.path, dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.abspath(args.path)}")


if __name__ == "__main__":
    main()
