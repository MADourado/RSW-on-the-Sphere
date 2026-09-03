"""Figure for `sec: quartet_rossby_gravity_fast` (JFM-template.tex),
Quartet D's efficiency-variation sensitivity to WG(3,9)'s driving
velocity: (a) 1D sweep, Rossby modes only; (b) 2D heatmap, RH(4,5) target.

Panel (a): efficiency_var vs. WG(3,9) velocity, 0-60 m/s, n_grid=31 --
step exactly 2 m/s, every grid point an integer (n_grid=30 over this same
range does NOT give integer steps, checked directly: step=60/29=2.069...).
Only the two Rossby modes (RH(4,5)/RH(3,4)) are drawn, for focus -- the
two gravity modes' own efficiency_var covers a noticeably wider range
over the same sweep (WG(3,9) itself: -11% to +124%, checked directly
2026-09-03) and would be a distraction on the same axes as the much
smoother, near-identical RH(4,5)/RH(3,4) curves, not because it is
numerically unstable (both are well above MIN_REFERENCE_DEK throughout).

Panel (b): efficiency_var 2D heatmap for target RH(4,5) alone, sweeping
WG(3,9) (0-60) and RH(3,4) (0-100) together. Both axes share n_grid=11 --
compute_2d_grid's own engine uses one n_grid for both axes, so 11 was
chosen (rather than two different per-axis counts) as the largest value
giving an all-integer step on BOTH axes at once (step=6 on WG(3,9),
step=10 on RH(3,4); gcd(60,100)=20, and 11-1=10 divides both cleanly).

Run:

    python examples/figures/paper_figure011_quartet_rossby_gravity_influence_efficiency.py
"""
import dataclasses
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from rsw_sphere.dynamics.run_config import RunConfig, SweepConfig, SweepAxis
from rsw_sphere.plotting.style import apply_house_style, mode_color
from run_sweep import run_sweep, compute_2d_grid

WAVE_SET_KEY = "quartet_rossby_gravity_influence"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure011_quartet_rossby_gravity_influence_efficiency.png")

_ROSSBY_LABELS = ["RH(4,5)", "RH(3,4)"]
_ROSSBY_MNALPHA = {"RH(4,5)": (4, 5, 3), "RH(3,4)": (3, 4, 3)}


def compute_1d():
    """1D efficiency_var sweep, WG(3,9) 0-60 m/s, n_grid=31 (integer step=2)."""
    config = RunConfig.from_registry_entry(WAVE_SET_KEY)
    sweep = SweepConfig(axes=(SweepAxis(mode="d", min=0.0, max=60.0),),
                         n_grid=31, diagnostics=("efficiency_var",))
    config = dataclasses.replace(config, sweep=sweep)
    return run_sweep(config, plot_per_point=False)


def compute_2d():
    """2D efficiency_var grid, WG(3,9) 0-60 x RH(3,4) 0-100, n_grid=11
    (integer steps 6/10 on both axes -- see module docstring).
    """
    config = RunConfig.from_registry_entry(WAVE_SET_KEY)
    sweep = SweepConfig(axes=(SweepAxis(mode="d", min=0.0, max=60.0),
                               SweepAxis(mode="c", min=0.0, max=100.0)),
                         n_grid=11, diagnostics=("efficiency_var",))
    config = dataclasses.replace(config, sweep=sweep)
    return compute_2d_grid(config, plot_per_point=False)


def plot(sweep_1d, U1, U2, grid_results, path: str = None):
    apply_house_style(base_size=14)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    u_values = sweep_1d["efficiency_var"]["u_values"]
    series = sweep_1d["efficiency_var"]["series"]
    for label in _ROSSBY_LABELS:
        ax.plot(u_values, series[label], marker="o", ms=3, lw=1.3,
                color=mode_color(*_ROSSBY_MNALPHA[label]), label=label)
    ax.set_xlabel("WG(3,9) initial velocity (m/s)")
    ax.set_ylabel("Efficiency variation (final, %)")
    ax.set_title("(a) Rossby modes only")
    ax.legend()

    ax2 = axes[1]
    n_grid = U1.shape[0]
    values = np.array([[grid_results[i, j]["final"]["RH(4,5)"]["efficiency_var"]
                         for j in range(n_grid)] for i in range(n_grid)])
    vlim = float(np.max(np.abs(values))) or 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim)
    cs = ax2.contourf(U1, U2, values, levels=np.linspace(-vlim, vlim, 101), cmap="RdBu_r", norm=norm)
    fig.colorbar(cs, ax=ax2, label="RH(4,5) efficiency variation (final, %)")
    ax2.set_xlabel("WG(3,9) initial velocity (m/s)")
    ax2.set_ylabel("RH(3,4) initial velocity (m/s)")
    ax2.set_title("(b) RH(4,5) target")

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

    print(f"Running 1D sweep (WG(3,9), n_grid=31) for {WAVE_SET_KEY!r}...")
    sweep_1d = compute_1d()
    print(f"Running 2D sweep (WG(3,9) x RH(3,4), n_grid=11) for {WAVE_SET_KEY!r}...")
    U1, U2, grid_results = compute_2d()

    plot(sweep_1d, U1, U2, grid_results, path=args.path)
    print(f"wrote {os.path.abspath(args.path)}")

    series = sweep_1d["efficiency_var"]["series"]
    u_values = sweep_1d["efficiency_var"]["u_values"]
    print("\n=== RH(4,5)/RH(3,4) efficiency_var full 1D sweep ===")
    for u, v45, v34 in zip(u_values, series["RH(4,5)"], series["RH(3,4)"]):
        print(f"  u={u:5.1f}  RH(4,5)={v45:7.3f}%  RH(3,4)={v34:7.3f}%")

    n_grid = U1.shape[0]
    values = np.array([[grid_results[i, j]["final"]["RH(4,5)"]["efficiency_var"]
                         for j in range(n_grid)] for i in range(n_grid)])
    i_min, j_min = np.unravel_index(np.argmin(values), values.shape)
    i_max, j_max = np.unravel_index(np.argmax(values), values.shape)
    print("\n=== RH(4,5) 2D panel (WG(3,9), RH(3,4)) ===")
    print(f"  min = {values[i_min, j_min]:.3f}% at (WG(3,9)={U1[i_min, j_min]:.1f}, "
          f"RH(3,4)={U2[i_min, j_min]:.1f})")
    print(f"  max = {values[i_max, j_max]:.3f}% at (WG(3,9)={U1[i_max, j_max]:.1f}, "
          f"RH(3,4)={U2[i_max, j_max]:.1f})")
    i54 = int(np.argmin(np.abs(U1[0, :] - 54)))
    j10 = int(np.argmin(np.abs(U2[:, 0] - 10)))
    print(f"  near (54,10): WG(3,9)={U1[j10, i54]:.1f}, RH(3,4)={U2[j10, i54]:.1f}, "
          f"value={values[j10, i54]:.3f}%")


if __name__ == "__main__":
    main()
