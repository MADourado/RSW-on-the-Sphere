"""Table 1 (``\\label{tab: master}``, JFM-template.tex, section "Resonant
triads" / ``sec: resonant``): each mode's frequency, linear period, and
coupling coefficient, together with the mismatch and pump mode, for the
four single-triad examples discussed in that section (Triad A/B in
``sec: rossbyonly``, Triad C/D in ``sec: combined``).

The dedicated ``rsw_sphere/plotting/triad_table.py`` this table's LaTeX
comment used to name was retired in the examples re-factor -- its job is
now the general-purpose ``wave_set_table.wave_set_properties`` (a triad
is just the 1-triad case of a wave set), applied here to the four
registered triad_* entries in ``wave_sets_default.yaml``.

Run:

    python examples/tables/paper_table01_resonant_triads.py
    python examples/tables/paper_table01_resonant_triads.py --fmt markdown
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_table

#: The four triads of Table 1, in the paper's own order (Triad A/B/C/D).
TRIAD_KEYS = [
    "triad_rossby_only_near_resonant",   # Triad A
    "triad_rossby_only_non_resonant",    # Triad B
    "triad_kelvin_rossby_flow",          # Triad C
    "triad_gravity_with_rossby_catalyst",  # Triad D
]

DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "tables", "paper_table01_resonant_triads.tex")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--fmt", choices=["latex", "csv", "markdown"], default="latex")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    triads = {k: specs[k] for k in TRIAD_KEYS}

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    text = wave_set_table(triads, fmt=args.fmt, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")
    return text


if __name__ == "__main__":
    main()
