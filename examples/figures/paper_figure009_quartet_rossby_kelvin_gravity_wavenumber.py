"""Figure for `sec: quartet_rossby_gravity_fast` (JFM-template.tex),
opening Quartet C's gravity-mode wavenumber screen: as slot ``d`` is swept
over EG(1,n)/WG(1,n), n=1..15 odd, in place of the registered EG(1,1),
this shows the mechanism (left panel: the candidate's own coupling
coefficient to the shared edge RH(4,5)+RH(3,4)) alongside its effect
(right panel: RH(1,2)'s own efficiency variation) -- two panels, not a
shared twin-axis plot, since the two quantities need genuinely different
scales (the coefficient spans 4 orders of magnitude, log; the efficiency
variation is a modest +/-11%, linear) and forcing them onto one shared
axis pair made it impossible to tell which curve belonged to which axis.

Left panel (log scale): |coupling coefficient alpha| of the candidate's
own triad (member of Triad 2, closed with the shared edge) -- a purely
linear-algebra quantity, no time integration, immune to any energy-budget
normalization choice. Right panel (linear, %, zero centered): RH(1,2)'s
own efficiency variation against Triad 1 (RH-only, its only containing
triad) -- the dynamical effect the coefficient decay is meant to explain.
Plotting both side by side, sharing the same x-axis, lets a reader see
directly that RH(1,2)'s own effect tracks the coupling coefficient's
decay.

Run:

    python examples/figures/paper_figure009_quartet_rossby_kelvin_gravity_wavenumber.py
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
                               "paper_figure009_quartet_rossby_kelvin_gravity_wavenumber.png")

#: Family line colors -- EG in style.py's own gravity BLUE, WG in the same
#: teal used for WG(7,9) in style.MODE_COLORS (WG's own cool sub-range).
EG_COLOR = BLUE
WG_COLOR = '#17a589'


def compute():
    """Two calls over the SAME registered candidate list (EG(1,n)/WG(1,n),
    n=1..15 odd): one for the candidate's own coupling coefficient (static
    property, always computed regardless of `diagnostics:` -- no time
    integration needed for this column), one for RH(1,2)'s own efficiency
    variation (the registry's own default alternative_modes.d.target_mode: c).
    """
    coeff = run_sweep_sets(WAVE_SET_KEY, SLOT, diagnostics_override=(),
                            target_mode_override=SLOT)
    rh12 = run_sweep_sets(WAVE_SET_KEY, SLOT, diagnostics_override=("efficiency_var",))
    return coeff, rh12


def _family_rows(results, alpha):
    return sorted((r for r in results if r["alpha"] == alpha and "error" not in r), key=lambda r: r["n"])


def plot(coeff, rh12, path: str = None):
    apply_house_style(base_size=14)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for alpha, fam, color in ((1, "EG(1,n)", EG_COLOR), (2, "WG(1,n)", WG_COLOR)):
        eg_coeff = _family_rows(coeff, alpha)
        eg_rh12 = _family_rows(rh12, alpha)
        ax1.plot([r["n"] for r in eg_coeff], [r["coeff"] for r in eg_coeff],
                 marker='o', ms=5, lw=1.5, color=color, label=fam)
        ax2.plot([r["n"] for r in eg_rh12], [r["efficiency_var (%)"] for r in eg_rh12],
                 marker='s', ms=5, lw=1.5, color=color, label=fam)

    ax1.set_yscale('log')
    ax1.set_xlabel("Gravity mode wavenumber $n$")
    ax1.set_ylabel(r"$|\alpha|$")
    ax1.set_title("(a) Coupling coefficient")
    ax1.legend()

    ymax = max(abs(v) for v in ax2.get_ylim())
    ax2.set_ylim(-ymax, ymax)
    ax2.axhline(0, color='grey', lw=0.8, ls=':', zorder=1)
    ax2.set_xlabel("Gravity mode wavenumber $n$")
    ax2.set_ylabel(r"RH(1,2) efficiency variation $\Delta\mathcal{E}$ (%)")
    ax2.set_title("(b) Effect on RH(1,2)")
    ax2.legend()

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
    coeff, rh12 = compute()
    plot(coeff, rh12, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")

    print(f"\n{'mode':>10} {'coeff':>14} {'RH(1,2) eff_var (%)':>22}")
    rh12_by_mode = {r["mode"]: r for r in rh12}
    for r in sorted(coeff, key=lambda r: (r["alpha"], r["n"])):
        ev = rh12_by_mode.get(r["mode"], {}).get("efficiency_var (%)", float("nan"))
        print(f"{r['mode']:>10} {r['coeff']:>14.6e} {ev:>22.3f}")


if __name__ == "__main__":
    main()
