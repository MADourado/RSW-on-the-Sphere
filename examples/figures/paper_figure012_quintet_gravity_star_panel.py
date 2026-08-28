"""Figure ``fig: quintetpanel`` (JFM-template.tex, ``sec: quintet``,
``Figures/quintet_gravity_star_panel.png``): energy-evolution comparison
row for the star quintet -- its three constituent triads (RH-only, with
EG(1,1), with EG(7,9)) followed by the full five-wave configuration.

Thin wrapper around ``rsw_sphere.plotting.energy_evolution.
wave_set_comparison_panel_from_spec`` on the registered
``quintet_gravity_star`` wave set. Each constituent triad highlights its
own private gravity mode (Triad 1 is RH-only, no highlight); the full
panel is left all-solid since the three sub-panels highlight different
modes.

Run:

    python examples/figures/paper_figure012_quintet_gravity_star_panel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.energy_evolution import wave_set_comparison_panel_from_spec

WAVE_SET_KEY = "quintet_gravity_star"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure012_quintet_gravity_star_panel.png")

#: Per-triad highlight, by mode key -- Triad 1 (RH-only) has no private
#: gravity mode to highlight, Triad 2/3 highlight their own EG(1,1)/EG(7,9).
_HIGHLIGHT_BY_LABEL = {"Triad 1 (RH-only)": None, "Triad 2 (with EG(1,1))": "d",
                        "Triad 3 (with EG(7,9))": "e"}


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[WAVE_SET_KEY]
    triad_labels = [t.display_label for t in spec.triads]
    highlight = [spec.index(_HIGHLIGHT_BY_LABEL[label]) if _HIGHLIGHT_BY_LABEL[label] is not None else None
                 for label in triad_labels]

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    results = wave_set_comparison_panel_from_spec(
        spec, highlight=highlight, highlight_full=None, path=args.path)
    r_full = results[-1]
    print(f"wrote {os.path.abspath(args.path)}")
    print(f"{WAVE_SET_KEY}: drift={r_full['drift']:.3e}, "
          f"dEK={dict(zip(r_full['labels'], r_full['dEK']))}")


if __name__ == "__main__":
    main()
