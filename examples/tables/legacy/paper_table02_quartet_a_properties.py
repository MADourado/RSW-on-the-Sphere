"""Table ``tab: cap41`` (JFM-template.tex, "Rossby-only quartet" ->
"Partner preference", ``sec: quartet_rh_preference``): properties of
Quartet A (RH(4,5)+RH(1,2)+RH(3,4)/RH(3,6)) -- each mode's frequency,
linear period, initial zonal velocity/amplitude, and the two per-triad
coupling coefficients.

Thin wrapper around ``rsw_sphere.plotting.wave_set_table.wave_set_properties``
applied to the registered ``quartet_rh_preference`` entry in
``wave_sets_default.yaml`` (per that table's own ``% regenerate:`` comment
in the LaTeX).

Run:

    python examples/tables/paper_table02_quartet_a_properties.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_table

WAVE_SET_KEY = "quartet_rh_preference"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "tables", "paper_table02_quartet_a_properties.tex")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--fmt", choices=["latex", "csv", "markdown"], default="latex")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    text = wave_set_table({WAVE_SET_KEY: specs[WAVE_SET_KEY]}, fmt=args.fmt, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")
    return text


if __name__ == "__main__":
    main()
