"""Figure ``fig: cap4ex1`` (JFM-template.tex, ``sec: quartet_rh_gravity``,
``Figures/quartet_rossby_kelvin_panel.png``): energy-evolution comparison
row for Quartet C -- (left) the RH-only triad, (middle) the triad with
the Kelvin-like mode EG(1,1), (right) the full four-wave configuration.

Thin wrapper around ``rsw_sphere.plotting.energy_evolution.
wave_set_comparison_panel_from_spec`` on the registered
``quartet_rossby_kelvin`` wave set, highlighting EG(1,1) (mode key "d").
Exposes ``compute()`` so ``paper_figure008_quartet_rossby_kelvin_periods.py``
(the companion power-spectrum figure, ``fig: power1``) can reuse the same
integrated trajectory rather than re-deriving it.

Run:

    python examples/figures/paper_figure007_quartet_rossby_kelvin_panel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.energy_evolution import wave_set_comparison_panel_from_spec

WAVE_SET_KEY = "quartet_rossby_kelvin"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure007_quartet_rossby_kelvin_panel.png")


def compute(specs_path: str = DEFAULT_WAVESETS_PATH, path: str = None):
    """Integrate and (if ``path`` is given) plot the comparison panel.
    Returns (spec, results) -- results[-1] is the full-wave-set result
    dict (t, E, E_total, labels, drift, dEK).
    """
    specs = load_wave_set_specs(specs_path)
    spec = specs[WAVE_SET_KEY]
    highlight = spec.index("d")  # EG(1,1)
    results = wave_set_comparison_panel_from_spec(
        spec, highlight=highlight, highlight_full=highlight, path=path)
    return spec, results


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    spec, results = compute(args.specs, path=args.path)
    r_full = results[-1]
    print(f"wrote {os.path.abspath(args.path)}")
    print(f"{WAVE_SET_KEY}: drift={r_full['drift']:.3e}, "
          f"dEK={dict(zip(r_full['labels'], r_full['dEK']))}")


if __name__ == "__main__":
    main()
