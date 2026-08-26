"""Thin CLI wrapper around the promoted novelty-frequency-detection core
(``rsw_sphere.utilities.periods.novel_frequency_content_multi`` /
``rsw_sphere.utilities.novelty_frequency`` /
``rsw_sphere.plotting.novelty_frequency_panel``).

This script started as the item-4 prototype (design/validation history:
.claude/plans/nifty-puzzling-meteor.md, PLAN-paper-4.2-audit-and-
freqshift-redesign-2026-08-26.md item 4) and is kept as a worked example
against ``quartet_rossby_gravity_influence`` rather than deleted -- the
actual algorithm now lives in ``rsw_sphere`` (this file has no logic of
its own, to avoid two copies of the same algorithm drifting apart).

Run:

    python examples/freqshift_novelty/novelty_detection.py --target "RH(4,5)" --plot
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.utilities.novelty_frequency import novelty_for_target, novelty_combined_for_target
from rsw_sphere.plotting.novelty_frequency_panel import novelty_frequency_figure
from run_dynamics import run_dynamics

WAVE_SET_KEY = "quartet_rossby_gravity_influence"


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help='e.g. "RH(3,4)"')
    parser.add_argument("--wave-set", default=WAVE_SET_KEY)
    parser.add_argument("--xmax", type=float, default=3.0)
    parser.add_argument("--min-prominence", type=float, default=0.02)
    parser.add_argument("--exclusion-frac", type=float, default=0.20)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    specs = load_wave_set_specs()
    spec = specs[args.wave_set]
    config = RunConfig.from_wave_set(spec, plot=False)
    results = run_dynamics(config, write_table=False)

    # Per-(target, sub-triad) breakdown, for a detailed text report.
    per_sub = novelty_for_target(results, args.target, xmax=args.xmax,
                                  min_prominence=args.min_prominence,
                                  exclusion_frac=args.exclusion_frac)

    print(f"=== novelty-frequency detection: {args.target} in {args.wave_set} ===")
    for sub_name, result in per_sub.items():
        print(f"  vs. {sub_name}:")
        if not result['novel_peaks']:
            print(f"    no novel frequency survives prominence>={args.min_prominence} after exclusion")
            continue
        for i, p in enumerate(result['novel_peaks']):
            tag = "DOMINANT" if i == 0 else f"  #{i + 1}"
            print(f"    [{tag}] period={p['period_days']:.4f}d  "
                  f"relevance={p['relevance_pct']:.2f}%  "
                  f"(band {p['band_days'][0]:.3f}-{p['band_days'][1]:.3f}d, "
                  f"prominence={p['prominence']:.4f})")

    if args.plot:
        # Combined figure: one file for this target, every containing
        # sub-triad drawn together (rsw_sphere.plotting.novelty_frequency_panel).
        combined = novelty_combined_for_target(results, args.target, xmax=args.xmax,
                                                min_prominence=args.min_prominence,
                                                exclusion_frac=args.exclusion_frac)
        fname = f"novelty_{args.target.replace('(', '').replace(')', '').replace(',', '_')}.png"
        path = os.path.join(_ROOT, "outputs", "figures", "freqshift_novelty", fname)
        novelty_frequency_figure(results, args.target, combined, path, xmax=args.xmax)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
