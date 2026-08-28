"""Figure for `sec: quartet_rossby_gravity_fast` (JFM-template.tex),
opening Quartet C's gravity-mode wavenumber screen: as slot ``d`` is swept
over EG(1,n)/WG(1,n), n=1..15 odd, in place of the registered EG(1,1), each
candidate's OWN efficiency variation Delta-E (eq: effvar) is tracked -- its
own share of the quartet's total energy, compared against its own isolated
Triad 2 (RH(4,5)+RH(3,4)+candidate; target_mode defaults to the swept slot
itself, run_sweep_sets.run_sweep_sets's own convention). NOT the RH modes'
own efficiency variation -- those aren't targeted by this sweep at all.
Motivates dropping EG(1,1)/WG(1,1) as a private 4th mode in favor of the
higher-frequency shared-edge topologies (Quartets D/E) discussed in the
rest of this subsubsection.

Thin wrapper around run_sweep_sets.py's own candidate-substitution engine
(``quartet_rossby_kelvin``'s registered ``alternative_modes.d`` block) --
only the plotting differs from ``run_sweep_sets.py``'s own generic
``plot_candidate_scalar`` (which deliberately draws unconnected points,
since candidates usually have no natural ordering): here EG(1,n) and
WG(1,n) each *do* have a natural ordering (wavenumber n), so they're drawn
as two connected lines instead, against n on the x-axis rather than the
candidate's mode label.

Run:

    python examples/figures/paper_figure013_quartet_rossby_kelvin_gravity_wavenumber.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt

from rsw_sphere.plotting.style import apply_house_style, BLUE
from run_sweep_sets import run_sweep_sets

WAVE_SET_KEY = "quartet_rossby_kelvin"
SLOT = "d"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure013_quartet_rossby_kelvin_gravity_wavenumber.png")

#: Family line colors -- EG in style.py's own gravity BLUE, WG in the same
#: teal used for WG(7,9) in style.MODE_COLORS (WG's own cool sub-range).
EG_COLOR = BLUE
WG_COLOR = '#17a589'


def compute():
    """One efficiency_var point per registered candidate (EG(1,n)/WG(1,n),
    n=1..15 odd) -- same call ``run_sweep_sets.py``'s own CLI makes for
    ``--wave-set quartet_rossby_kelvin --slot d``.
    """
    return run_sweep_sets(WAVE_SET_KEY, SLOT, diagnostics_override=("efficiency_var",))


def plot(results, path: str = None):
    apply_house_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    eg = sorted((r for r in results if r["alpha"] == 1 and "error" not in r), key=lambda r: r["n"])
    wg = sorted((r for r in results if r["alpha"] == 2 and "error" not in r), key=lambda r: r["n"])
    ax.plot([r["n"] for r in eg], [r["efficiency_var (%)"] for r in eg],
            marker='o', ms=5, lw=1.5, color=EG_COLOR, label="EG(1,n)")
    ax.plot([r["n"] for r in wg], [r["efficiency_var (%)"] for r in wg],
            marker='s', ms=5, lw=1.5, color=WG_COLOR, label="WG(1,n)")

    ax.axhline(0, color='grey', lw=0.8, ls='--', zorder=1)
    ax.set_xlabel("Gravity mode wavenumber $n$")
    ax.set_ylabel(r"Candidate's own efficiency variation $\Delta\mathcal{E}$ (final, %)")
    ax.set_title("Quartet C: private gravity mode's own efficiency variation vs. its wavenumber")
    ax.legend(fontsize=9)
    fig.tight_layout()
    if path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    print(f"Running candidate substitution for {WAVE_SET_KEY!r}, slot {SLOT!r}...")
    results = compute()
    plot(results, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")

    print(f"\n{'mode':>10} {'efficiency_var (%)':>20}")
    for r in sorted(results, key=lambda r: (r["alpha"], r["n"])):
        v = r.get("efficiency_var (%)")
        print(f"{r['mode']:>10} {v if v is not None else float('nan'):>20.3f}")


if __name__ == "__main__":
    main()
