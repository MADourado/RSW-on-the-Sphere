"""Table ``tab: quartet_master`` (JFM-template.tex, ``sec:coupled``): one
merged table across Quartets A/B/C/D/E -- each mode's frequency, linear
period, initial zonal velocity/amplitude, and per-triad coupling
coefficients, one group per quartet. Mirrors Table ``tab: master``'s
multi-group style for the single triads (\\S sec: resonant).

Quartet B's own ``tab: precession_comparison`` (a differently-shaped
barotropic-vs-RSW efficiency/precession comparison) stays separate; this
table carries only Quartet B's coefficient properties, as a 4th group.

Run:

    python examples/tables/paper_table02_quartet_master.py
"""
import os

import _bootstrap  # noqa: F401 -- repo root on sys.path

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_master_table, wave_set_properties

DEFAULT_OUTPUT = os.path.join(_bootstrap.ROOT, "outputs", "tables", "paper_table02_quartet_master.tex")

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
    print(f"\nSaved to {os.path.relpath(DEFAULT_OUTPUT, _bootstrap.ROOT)}")

    print("\n=== periods (days), for prose ===")
    for k in KEYS:
        p = wave_set_properties(specs[k])
        for lbl, period in zip(p['mode_labels'], p['period_days']):
            print(f"  {specs[k].display_label} {lbl}: {period:.3f}d")


if __name__ == "__main__":
    main()
