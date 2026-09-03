"""Figure ``fig: quintetpanel`` (JFM-template.tex, ``sec: quintet``,
``Figures/quintet_gravity_star_panel.png``): Quintet A (``quintet_gravity_star``,
Quartet C + EG(7,9) closing the shared edge RH(4,5)+RH(3,4) a second
time) at its registered velocities.

2x3 grid -- top row: the three constituent triads (Triad 1 RH-only,
Triad 2 with EG(1,1), Triad 3 with EG(7,9)), each integrated alone;
bottom row: the full quintet, then RH(4,5)'s and RH(1,2)'s own
novelty-frequency spectra (§ diagnostics) against whichever sub-triad(s)
contain them -- RH(3,4) is skipped to keep the grid at 2x3 (mirrors
paper_figure008_quartet_rossby_kelvin_panel.py's own top-row-of-triads +
bottom-row-of-spectra layout, extended by one triad/column). tf_days=150
(vs. the registry's own 60d) for a cleaner spectrum -- resolves ~36
cycles of the ~4.2d dominant period instead of ~14.

Run:

    python examples/figures/paper_figure014_quintet_gravity_star_panel.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.trajectory_cache import _mode_slug
from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.energy_evolution import plot_energy_evolution
from rsw_sphere.plotting.novelty_frequency_panel import novelty_frequency_figure
from rsw_sphere.utilities.novelty_frequency import novelty_combined_for_target

WAVE_SET_KEY = "quintet_gravity_star"
TF_DAYS = 150.0
SPECTRUM_LABELS = ["RH(4,5)", "RH(1,2)"]
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure014_quintet_gravity_star_panel.png")

#: Display-only time-axis limit (days) for the top row -- tuned by eye,
#: doesn't affect the tf_days=150 integration.
EVOLUTION_XMAX_DAYS = 25.0
SPECTRUM_XMAX_DAYS = 8.0


def compute():
    """Run (cached) the full quintet + all three constituent triads at
    the registry's own velocities, but tf_days=150. Returns (spec, results).
    """
    from run_dynamics import run_dynamics
    spec0 = RunConfig.from_registry_entry(WAVE_SET_KEY).wave_set_spec
    config = RunConfig.from_wave_set(spec0, tf_days=TF_DAYS, h=0.01, plot=False, parallel=False)
    results = run_dynamics(config)
    return spec0, results


def _unit_modes(spec, name: str):
    if name == "full":
        return spec.modes
    for i in range(spec.n_triads()):
        member_p, member_q, _ = spec.sub_triad_modes(i)
        if name == f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}":
            return spec.sub_triad_modes(i)
    raise ValueError(f"no constituent triad of {spec.key!r} matches unit name {name!r}")


def plot(spec, results, path: str = None,
         evolution_xmax: float = EVOLUTION_XMAX_DAYS, spectrum_xmax: float = SPECTRUM_XMAX_DAYS):
    apply_house_style(base_size=16)
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))

    triad_names = []
    for i in range(spec.n_triads()):
        member_p, member_q, _ = spec.sub_triad_modes(i)
        triad_names.append(f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}")

    for ax, name in zip(axes[0, :], triad_names):
        r = results[name]
        modes = _unit_modes(spec, name)
        plot_energy_evolution(r["t"], r["E"], r["E_total"], r["labels"], modes, ax=ax)
        ax.set_title(r["title"])
        ax.set_xlim(0, evolution_xmax)

    r_full = results["full"]
    plot_energy_evolution(r_full["t"], r_full["E"], r_full["E_total"], r_full["labels"],
                           _unit_modes(spec, "full"), ax=axes[1, 0])
    axes[1, 0].set_title(spec.display_label)
    axes[1, 0].set_xlim(0, evolution_xmax)

    for ax, label in zip(axes[1, 1:], SPECTRUM_LABELS):
        novelty_result = novelty_combined_for_target(results, label)
        novelty_frequency_figure(results, label, novelty_result, ax=ax, xmax=spectrum_xmax,
                                  full_label="Quintet", show_excluded_in_legend=False)

    for ax in list(axes[0, 1:]) + list(axes[1, 1:]):
        ax.set_ylabel("")

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
    parser.add_argument("--evolution-xmax", type=float, default=EVOLUTION_XMAX_DAYS)
    parser.add_argument("--spectrum-xmax", type=float, default=SPECTRUM_XMAX_DAYS)
    args = parser.parse_args()

    spec, results = compute()
    plot(spec, results, path=args.path,
         evolution_xmax=args.evolution_xmax, spectrum_xmax=args.spectrum_xmax)
    print(f"wrote {os.path.abspath(args.path)}")

    report = compute_diagnostics_report(results, spec)
    print("\n=== per-unit dominant period, spectrum-panel Rossby modes ===")
    for label in SPECTRUM_LABELS:
        for name in results:
            if label not in results[name]["labels"]:
                continue
            m = report["per_mode_unit"][name][label]
            print(f"  [{name}] {label}: period_global={m['period_global']:.4f}d")

    print("\n=== final (combined) diagnostics, every mode ===")
    for d in report["final"]:
        novelty = (f"{d['novelty_period_final_days']:.4f}d ({d['novelty_relevance_final_pct']:.2f}%)"
                   if d["novelty_period_final_days"] == d["novelty_period_final_days"] else "none detected")
        eff = (f"{d['efficiency_var_final_pct']:+.2f}%" if d['efficiency_var_final_pct'] == d['efficiency_var_final_pct']
               else "n/a")
        sd = (f"{d['spectral_dev_final_pct']:.2f}%" if d['spectral_dev_final_pct'] == d['spectral_dev_final_pct']
              else "n/a")
        print(f"  {d['mode']}: efficiency_var={eff}  spectral_dev={sd}  vs={d['vs']}  "
              f"novelty_period={novelty}")

    r_full = results["full"]
    print(f"\n=== drift ===\n  full: drift={r_full['drift']:.3e}")


if __name__ == "__main__":
    main()
