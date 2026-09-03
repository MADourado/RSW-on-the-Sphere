"""Combined quintet master table (JFM-template.tex, ``sec:5waves``): one
merged table across Quintet A and Quintet B, mirroring Table 2's own
``tab: quartet_master`` (\\S sec:coupled) via the same
``wave_set_master_table`` machinery -- both quintets have 3 constituent
triads, so they share one column layout (Coeff. 1/2/3) the way the
quartets share Coeff. 1/2.

Run:

    python examples/tables/paper_table05_quintet_master.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_master_table, wave_set_properties

DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "tables", "paper_table05_quintet_master.tex")

#: Registry key -> paper order (Quintet A, B).
KEYS = ["quintet_gravity_star", "quintet_gravity_influence_star"]

CAPTION = (
    r"Properties of both quintets examined in \S\ref{sec:5waves} "
    r"(Figure \ref{fig: quintet_topology}): each mode's frequency "
    r"$\omega$, linear period (days), and coupling coefficient within "
    r"whichever constituent triad(s) contain it (Coeff.$_1$/Coeff.$_2$/"
    r"Coeff.$_3$, $-$ where a mode isn't in that triad), each triad's "
    r"mismatch $\delta$, and its own pump mode (largest coupling "
    r"coefficient)."
)


def main():
    specs_all = load_wave_set_specs()
    specs = {k: specs_all[k] for k in KEYS}

    text = wave_set_master_table(specs, fmt='latex', path=None,
                                  caption=CAPTION, label='quintet_master')
    print(text)

    os.makedirs(os.path.dirname(DEFAULT_OUTPUT), exist_ok=True)
    wave_set_master_table(specs, fmt='latex', path=DEFAULT_OUTPUT,
                           caption=CAPTION, label='quintet_master')
    print(f"\nSaved to {os.path.relpath(DEFAULT_OUTPUT, _ROOT)}")

    print("\n=== periods (days), for prose ===")
    for k in KEYS:
        p = wave_set_properties(specs[k])
        for lbl, period in zip(p['mode_labels'], p['period_days']):
            print(f"  {specs[k].display_label} {lbl}: {period:.3f}d")


if __name__ == "__main__":
    main()
