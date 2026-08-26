"""Item 4 exploratory step (.claude/PLAN-paper-4.2-audit-and-freqshift-redesign-
2026-08-26.md): for quartet_rossby_gravity_influence, overlay each mode's own FFT power
spectrum (of KE = |A|^2) from the full-quartet run against its spectrum in
EVERY sub-triad that contains it -- not just one arbitrary baseline. This
matters for a mode shared as the SUM in both constituent triads (WG(7,9)
here), which has no single unambiguous "its own" sub-triad the way a
shared MEMBER mode (RH(4,5) here) does: compare against every containing
sub-triad and let the reader see which one it actually tracks.

Draft/exploratory groundwork for the new frequency-shift/novelty-period
metric -- NOT a paper figure, not yet the final metric implementation
(open questions: KE vs. complex-amplitude input, spectrum normalization,
still being decided against these plots).

Run:

    python examples/freqshift_novelty/draft_spectra.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.utilities.periods import dominant_periods
from run_dynamics import run_dynamics

WAVE_SET_KEY = "quartet_rossby_gravity_influence"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "freqshift_novelty", "draft_spectra.png")


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--xmax", type=float, default=3.0, help="period axis upper limit (days)")
    args = parser.parse_args()

    specs = load_wave_set_specs()
    spec = specs[WAVE_SET_KEY]
    config = RunConfig.from_wave_set(spec, plot=False)
    results = run_dynamics(config, write_table=False)

    full = results["full"]
    sub_units = {name: r for name, r in results.items() if name != "full"}

    # For each full-quartet mode, every sub-triad unit whose own labels
    # include it (private modes: 1 match; shared modes, member OR sum
    # role alike: as many matches as triads they belong to).
    mode_to_units = {
        label: [u for u in sub_units.values() if label in u["labels"]]
        for label in full["labels"]
    }

    apply_house_style()
    n = len(mode_to_units)
    ncols = 2
    nrows = (n + 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    axes = axes.ravel()

    colors = ["tab:orange", "tab:green", "tab:purple"]
    for ax, (label, units) in zip(axes, mode_to_units.items()):
        j_full = full["labels"].index(label)
        r_full = dominant_periods(full["t"], full["E"][:, j_full])
        ax.plot(r_full["periods_days"], r_full["power"], color="black",
                label=f"full quartet (peak={r_full['period_global']:.3f}d)")
        for color, u in zip(colors, units):
            j_sub = u["labels"].index(label)
            r_sub = dominant_periods(u["t"], u["E"][:, j_sub])
            n_units_containing = len(units)
            tag = f"sub-triad {u['name']}" + (" (shared)" if n_units_containing > 1 else "")
            ax.plot(r_sub["periods_days"], r_sub["power"], color=color, ls="--",
                     label=f"{tag} (peak={r_sub['period_global']:.3f}d)")
        ax.set_title(label)
        ax.set_xlabel("Period (days)")
        ax.set_ylabel("Spectral power of |A|^2 (a.u.)")
        ax.set_xlim(0, args.xmax)
        ax.legend(fontsize=8)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"{WAVE_SET_KEY}: per-mode KE spectrum, full quartet vs. EVERY containing sub-triad")
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    fig.savefig(args.path, dpi=150, bbox_inches="tight")
    print(f"wrote {os.path.abspath(args.path)}")

    for label, units in mode_to_units.items():
        j_full = full["labels"].index(label)
        r_full = dominant_periods(full["t"], full["E"][:, j_full])
        print(f"{label}: full quartet peak={r_full['period_global']:.4f}d")
        for u in units:
            j_sub = u["labels"].index(label)
            r_sub = dominant_periods(u["t"], u["E"][:, j_sub])
            print(f"    vs. {u['name']}: peak={r_sub['period_global']:.4f}d")


if __name__ == "__main__":
    main()
