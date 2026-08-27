"""Figure ``fig: power1`` (JFM-template.tex, ``sec: quartet_rh_gravity``,
``Figures/quartet_rossby_kelvin_periods.png``): power spectrum (FFT of the
kinetic-energy time series) for every mode of Quartet C, corresponding to
the comparison panel ``fig: cap4ex1``.

Reuses ``paper_figure007_quartet_rossby_kelvin_panel.compute()``'s own
integrated trajectory (same registered ``quartet_rossby_kelvin`` wave
set, full four-wave configuration) instead of re-integrating.

Run:

    python examples/figures/paper_figure008_quartet_rossby_kelvin_periods.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EXAMPLES_FIGURES = os.path.dirname(os.path.abspath(__file__))
if _EXAMPLES_FIGURES not in sys.path:
    sys.path.insert(0, _EXAMPLES_FIGURES)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.period_panel import wave_set_period_panel
from paper_figure007_quartet_rossby_kelvin_panel import compute

WAVE_SET_KEY = "quartet_rossby_kelvin"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure008_quartet_rossby_kelvin_periods.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    args = parser.parse_args()

    spec, results = compute(args.specs)
    r_full = results[-1]

    apply_house_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    _, _, period_results = wave_set_period_panel(
        r_full['t'], r_full['E'], r_full['labels'], list(spec.modes), ax=ax)
    ax.set_title(spec.display_label)

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    fig.savefig(args.path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {os.path.abspath(args.path)}")
    for label, pr in zip(r_full['labels'], period_results):
        flag = " [HORIZON-LIMITED]" if pr['horizon_limited'] else ""
        print(f"  {label}: period_global={pr['period_global']:.3g}d, "
              f"period_local_max={pr['period_local_max']}{flag}")


if __name__ == "__main__":
    main()
