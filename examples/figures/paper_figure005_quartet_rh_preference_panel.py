"""Figure ``fig: cap42`` (JFM-template.tex, ``sec: quartet_rh_preference``,
``Figures/quartet_rh_preference_panel.png``): energy-integration comparison
row for Quartet A -- (left) the triad completed with RH(3,6), (middle) the
triad completed with RH(3,4), (right) the full four-wave configuration.

Thin wrapper around ``rsw_sphere.plotting.energy_evolution.
wave_set_comparison_panel_from_spec`` (also installed as the
``rsw-waveset`` console script's own ``--panel`` mode) on the registered
``quartet_rh_preference`` wave set.

Run:

    python examples/figures/paper_figure005_quartet_rh_preference_panel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.energy_evolution import wave_set_comparison_panel_from_spec

WAVE_SET_KEY = "quartet_rh_preference"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure005_quartet_rh_preference_panel.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[WAVE_SET_KEY]

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    results = wave_set_comparison_panel_from_spec(spec, path=args.path)
    r_full = results[-1]
    print(f"wrote {os.path.abspath(args.path)}")
    print(f"{WAVE_SET_KEY}: drift={r_full['drift']:.3e}, "
          f"dEK={dict(zip(r_full['labels'], r_full['dEK']))}")


if __name__ == "__main__":
    main()
