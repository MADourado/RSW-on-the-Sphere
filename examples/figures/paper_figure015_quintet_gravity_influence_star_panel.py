"""Figure ``fig: quintetpanel_b`` (JFM-template.tex, ``sec: quintet_b``,
``Figures/quintet_gravity_influence_star_panel.png``): Quintet B
(``quintet_gravity_influence_star``), Quartet D
(quartet_rossby_gravity_influence) with a third private member EG(3,9)
closing the shared sum mode WG(7,9) a third time.

2x3 grid -- top row: the three constituent triads (Triad 1
RH(3,4)+RH(4,5)+WG(7,9), Triad 2 WG(3,9)+RH(4,5)+WG(7,9), Triad 3
EG(3,9)+RH(4,5)+WG(7,9)), each integrated alone; bottom row: the full
quintet, then RH(3,4)'s and RH(4,5)'s own novelty-frequency spectra
against whichever sub-triad(s) contain them -- mirrors
paper_figure014_quintet_gravity_star_panel.py's own layout.

Run:

    python examples/figures/paper_figure015_quintet_gravity_influence_star_panel.py
"""
import os

import _bootstrap  # noqa: F401 -- repo root on sys.path

import matplotlib.pyplot as plt

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.trajectory_cache import _mode_slug
from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report
from rsw_sphere.plotting.style import apply_house_style
from rsw_sphere.plotting.energy_evolution import plot_energy_evolution
from rsw_sphere.plotting.novelty_frequency_panel import novelty_frequency_figure
from rsw_sphere.utilities.novelty_frequency import novelty_combined_for_target

WAVE_SET_KEY = "quintet_gravity_influence_star"
SPECTRUM_LABELS = ["RH(3,4)", "RH(4,5)"]
DEFAULT_OUTPUT = os.path.join(_bootstrap.ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure015_quintet_gravity_influence_star_panel.png")

#: Display-only time-axis limits (days) -- tuned by eye, doesn't affect
#: the registry's own tf_days=30 integration.
EVOLUTION_XMAX_DAYS = 15.0
SPECTRUM_XMAX_DAYS = 4.0


def compute():
    """Run (cached) the full quintet + all three constituent triads at
    the registry's own velocities/tf_days. Returns (spec, results).
    """
    from run_dynamics import run_dynamics
    config = RunConfig.from_registry_entry(WAVE_SET_KEY)
    results = run_dynamics(config)
    return config.wave_set_spec, results


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

    print("\n=== dynamical phase (precession frequency, cycles/day) ===")
    for triad_label, d in report["precession"].items():
        alone = f"{d['freq_alone_cpd']:.5f}" if d['freq_alone_cpd'] is not None else "n/a"
        print(f"  {triad_label}: full={d['freq_full_cpd']:.5f}  alone={alone}  "
              f"phase_variation={d['phase_variation_pct']:+.2f}%")

    r_full = results["full"]
    print(f"\n=== drift ===\n  full: drift={r_full['drift']:.3e}")


if __name__ == "__main__":
    main()
