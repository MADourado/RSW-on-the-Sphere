"""Figure for `sec: quartet_rossby_gravity_fast` (JFM-template.tex),
opening Quartet C's gravity-mode wavenumber screen: as slot ``d`` is swept
over EG(1,n)/WG(1,n), n=1..15 odd, in place of the registered EG(1,1), both
the candidate's OWN "final" diagnostics (target_mode_override=SLOT) and
RH(1,2)'s own (the registry's own alternative_modes.d.target_mode: c) are
tracked side by side -- 2x2 grid, efficiency variation (top row) and
p_measure (bottom row), gravity mode (left column) vs. RH(1,2) (right
column). p_measure was reinstated here (2026-08-28) after an earlier pass
dropped it from several figures in favor of efficiency_var alone --
inadequate on its own, since efficiency_var normalizes by each
configuration's own (here candidate-dependent) total energy budget and can
drift for that reason alone even when a mode's own raw energy swing
(p_measure) does not; see the total_energy diagnostic in
run_sweep_sets.py/run_sweep.py for the general version of this check.

Thin wrapper around run_sweep_sets.py's own candidate-substitution engine
(``quartet_rossby_kelvin``'s registered ``alternative_modes.d`` block) --
only the plotting differs from ``run_sweep_sets.py``'s own generic
``plot_candidate_scalar`` (which deliberately draws unconnected points,
since candidates usually have no natural ordering): here EG(1,n) and
WG(1,n) each *do* have a natural ordering (wavenumber n), so they're drawn
as two connected lines instead, against n on the x-axis rather than the
candidate's mode label.

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

_DIAGNOSTICS = ("efficiency_var", "p_measure_final")
_COLS = {"efficiency_var (%)": r"$\Delta\mathcal{E}$ (final, %)",
         "p_measure_final (%)": r"$\mathcal{P}$ (final, %)"}


def compute():
    """Two calls over the SAME registered candidate list (EG(1,n)/WG(1,n),
    n=1..15 odd): one targeting the swept slot itself (the gravity mode's
    own diagnostics), one targeting the registry's own default
    (alternative_modes.d.target_mode: c, i.e. RH(1,2)).
    """
    gravity = run_sweep_sets(WAVE_SET_KEY, SLOT, diagnostics_override=_DIAGNOSTICS,
                              target_mode_override=SLOT)
    rh12 = run_sweep_sets(WAVE_SET_KEY, SLOT, diagnostics_override=_DIAGNOSTICS)
    return gravity, rh12


def _plot_panel(ax, results, col, ylabel, title):
    eg = sorted((r for r in results if r["alpha"] == 1 and "error" not in r), key=lambda r: r["n"])
    wg = sorted((r for r in results if r["alpha"] == 2 and "error" not in r), key=lambda r: r["n"])
    ax.plot([r["n"] for r in eg], [r[col] for r in eg],
            marker='o', ms=5, lw=1.5, color=EG_COLOR, label="EG(1,n)")
    ax.plot([r["n"] for r in wg], [r[col] for r in wg],
            marker='s', ms=5, lw=1.5, color=WG_COLOR, label="WG(1,n)")
    ax.axhline(0, color='grey', lw=0.8, ls='--', zorder=1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def plot(gravity, rh12, path: str = None):
    apply_house_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    _plot_panel(axes[0, 0], gravity, "efficiency_var (%)", _COLS["efficiency_var (%)"],
                "Gravity mode's own efficiency variation")
    _plot_panel(axes[0, 1], rh12, "efficiency_var (%)", "", "RH(1,2)'s own efficiency variation")
    _plot_panel(axes[1, 0], gravity, "p_measure_final (%)", _COLS["p_measure_final (%)"],
                "Gravity mode's own p_measure")
    _plot_panel(axes[1, 1], rh12, "p_measure_final (%)", "", "RH(1,2)'s own p_measure")

    for ax in axes[:, 1]:
        ax.set_ylabel("")
    for ax in axes[1, :]:
        ax.set_xlabel("Gravity mode wavenumber $n$")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Quartet C: private gravity mode swept over wavenumber $n$")
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
    gravity, rh12 = compute()
    plot(gravity, rh12, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")

    for label, results in (("gravity mode (own)", gravity), ("RH(1,2)", rh12)):
        print(f"\n{label}:")
        print(f"{'mode':>10} {'efficiency_var (%)':>20} {'p_measure_final (%)':>20}")
        for r in sorted(results, key=lambda r: (r["alpha"], r["n"])):
            ev = r.get("efficiency_var (%)", float("nan"))
            pm = r.get("p_measure_final (%)", float("nan"))
            print(f"{r['mode']:>10} {ev:>20.3f} {pm:>20.3f}")


if __name__ == "__main__":
    main()
