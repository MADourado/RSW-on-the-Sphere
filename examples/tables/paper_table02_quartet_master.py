"""Combined quartet master table (JFM-template.tex, ``sec:coupled``):
one merged table across Quartets A/B/C/D/E, replacing the three separate
per-quartet tables ``tab: cap41``/``cap42``/``cap43`` (Quartet A/C/D)
that used to sit inline in each quartet's own subsection -- mirroring
Table ``tab: master``'s existing hand-merged, multi-group style for the
single triads (\\S sec: resonant), generated here instead of hand-merged.
Quartet E (``quartet_rossby_gravity_influence_high``, registered
2026-08-28) is included here alongside A-D.

Quartet B's own ``tab: precession_comparison`` (a differently-shaped
barotropic-vs-RSW efficiency/precession comparison) is untouched and
stays separate -- this table only adds Quartet B's coefficient
properties as a 4th group alongside A/C/D, via the same
``wave_set_properties()`` call the others already use.

Run:

    python examples/tables/paper_table02_quartet_master.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_master_table, wave_set_properties

DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "tables", "paper_table02_quartet_master.tex")

#: Registry key -> paper order (Quartet A, B, C, D, E).
KEYS = ["quartet_rh_preference", "quartet_rh_borrowed_topology",
        "quartet_rossby_kelvin", "quartet_rossby_gravity_influence",
        "quartet_rossby_gravity_influence_high"]

CAPTION = (
    r"Properties of every quartet examined in \S\ref{sec:coupled} "
    r"(Figure \ref{fig: topology_overview}): each mode's frequency "
    r"$\omega$, linear period (days), and coupling coefficient within "
    r"whichever constituent triad(s) contain it (Coeff.$_1$/Coeff.$_2$, "
    r"$-$ where a mode isn't in that triad), the triad's mismatch "
    r"$\delta$, and each triad's own pump mode (largest coupling "
    r"coefficient). Quartet B's own precession-comparison table "
    r"(Table \ref{tab: precession_comparison}) is separate."
)


def main():
    specs_all = load_wave_set_specs()
    specs = {k: specs_all[k] for k in KEYS}

    text = wave_set_master_table(specs, fmt='latex', path=None,
                                  caption=CAPTION, label='quartet_master')
    print(text)

    os.makedirs(os.path.dirname(DEFAULT_OUTPUT), exist_ok=True)
    wave_set_master_table(specs, fmt='latex', path=DEFAULT_OUTPUT,
                           caption=CAPTION, label='quartet_master')
    print(f"\nSaved to {os.path.relpath(DEFAULT_OUTPUT, _ROOT)}")

    print("\n=== periods (days), for prose ===")
    for k in KEYS:
        p = wave_set_properties(specs[k])
        for lbl, period in zip(p['mode_labels'], p['period_days']):
            print(f"  {specs[k].display_label} {lbl}: {period:.3f}d")


if __name__ == "__main__":
    main()
