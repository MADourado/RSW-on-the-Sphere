"""Figure ``fig: quartet_a_precession`` (JFM-template.tex, subfigure (a) of
``fig: precession_frequency``, ``sec: quartet_rh_precession``,
``Figures/quartet_a_rh36_precession.png``): precession frequency and
target-mode efficiency for Quartet A's target triad RH(4,5)+RH(1,2)+RH(3,4),
sweeping RH(3,6)'s own driving velocity while the shared edge stays fixed.

Thin wrapper around ``run_sweep.run_sweep`` (the same function
``run_sweep.py --wave-set quartet_rh_preference`` calls), reading the
1D sweep/plot/target_mode/plot_triad settings straight from the
registered ``quartet_rh_preference`` entry in ``wave_sets_default.yaml``
-- no separate config file needed.

Run:

    python examples/figures/paper_figure006_quartet_a_precession.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import yaml

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH
from run_sweep import run_sweep

WAVE_SET_KEY = "quartet_rh_preference"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure006_quartet_a_precession.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--no-per-point", action="store_true",
                         help="skip the per-grid-point run_dynamics pass (diagnostics only)")
    args = parser.parse_args()

    with open(args.specs) as f:
        raw = yaml.safe_load(f)[WAVE_SET_KEY]
    config = RunConfig.from_registry_entry(WAVE_SET_KEY, args.specs)

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    print(f"Running precession sweep for wave set {WAVE_SET_KEY!r}...")
    run_sweep(config, args.path, raw.get("plot", {}), run_per_point=not args.no_per_point,
              target_mode=raw.get("target_mode"), plot_triad=raw.get("plot_triad"))
    print(f"wrote {os.path.abspath(args.path)}")


if __name__ == "__main__":
    main()
