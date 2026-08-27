"""Figure ``fig: cap43panel`` (JFM-template.tex,
``sec: quartet_rossby_gravity_fast``, ``Figures/quartet_gravity_79_panel.png``):
energy-evolution comparison row for Quartet D -- (left) the RH-only triad,
(middle) the triad with the higher-frequency gravity mode EG(7,9), (right)
the full four-wave configuration.

Thin wrapper around ``rsw_sphere.plotting.energy_evolution.
wave_set_comparison_panel_from_spec`` on the registered
``quartet_gravity_79`` wave set, highlighting EG(7,9) (mode key "d"). No
companion period or P-measure figure is cited for this wave set.

Run:

    python examples/figures/paper_figure009_quartet_gravity_79_panel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.energy_evolution import wave_set_comparison_panel_from_spec

WAVE_SET_KEY = "quartet_gravity_79"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure009_quartet_gravity_79_panel.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[WAVE_SET_KEY]
    highlight = spec.index("d")  # EG(7,9)

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    results = wave_set_comparison_panel_from_spec(
        spec, highlight=highlight, highlight_full=highlight, path=args.path)
    r_full = results[-1]
    print(f"wrote {os.path.abspath(args.path)}")
    print(f"{WAVE_SET_KEY}: drift={r_full['drift']:.3e}, "
          f"dEK={dict(zip(r_full['labels'], r_full['dEK']))}")


if __name__ == "__main__":
    main()
