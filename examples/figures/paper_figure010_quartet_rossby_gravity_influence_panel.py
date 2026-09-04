"""Figure for ``sec: quartet_rossby_gravity_fast`` (JFM-template.tex),
Quartet D (``quartet_rossby_gravity_influence``): WG(7,9) is the shared
SUM mode of both constituent triads (RH(3,4)+RH(4,5)+WG(7,9) and
WG(3,9)+RH(4,5)+WG(7,9)), with RH(4,5) the other shared member.

2x2 grid -- top row: Triad 1 and Triad 2 energy evolution (one shared
``run_dynamics()`` call, not two separate integrations); bottom row: the
full quartet's own evolution, and RH(3,4)'s novelty-frequency spectrum
(``rsw_sphere.plotting.novelty_frequency_panel``) against Triad 1 (its
only containing triad -- RH(3,4) is private, not a member of Triad 2).
RH(4,5) is highlighted in the evolution panels instead (it's the mode
shared by both triads, so the only one that appears in all three); the
spectrum target is chosen separately since RH(3,4) picks up Triad 2's own
characteristic frequency despite not being one of its members, the more
telling illustration of coupling through the shared sum mode.

Run:

    python examples/figures/paper_figure010_quartet_rossby_gravity_influence_panel.py
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

WAVE_SET_KEY = "quartet_rossby_gravity_influence"
HIGHLIGHT_LABEL = "RH(4,5)"     # shared member -- highlighted in every evolution panel
SPECTRUM_TARGET_LABEL = "RH(3,4)"  # private to Triad 1 -- the novelty-spectrum panel's target
DEFAULT_OUTPUT = os.path.join(_bootstrap.ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure010_quartet_rossby_gravity_influence_panel.png")

#: Display-only time-axis limit (days) for the three evolution panels --
#: tuned by eye for this figure, doesn't affect the registry's own
#: tf_days. Unlike paper_figure008's own row-shared y-labels, every panel
#: here keeps its own y-axis label: the 2x2 grid mixes an evolution panel
#: with a spectrum panel in the bottom row, so the two axes in that row
#: don't share a y-axis meaning.
EVOLUTION_XMAX_DAYS = 8.0


def compute():
    """Run (cached) the full quartet + both constituent triads. Returns
    (spec, results) -- results keyed 'full'/'triad_<member1>_<member2>',
    the exact shape run_dynamics() returns.
    """
    from run_dynamics import run_dynamics
    config = RunConfig.from_registry_entry(WAVE_SET_KEY)
    results = run_dynamics(config)
    return config.wave_set_spec, results


def _unit_modes(spec, name: str):
    """(m, n, alpha) tuples for one unit's own modes, in the same order
    as that unit's own r['labels'] -- 'full' is spec.modes; a sub-triad
    unit is spec.sub_triad_modes(i) (member_p, member_q, sum), matched by
    name via the same triad_<member_p>_<member_q> convention run_dynamics
    itself builds units with.
    """
    if name == "full":
        return spec.modes
    for i in range(spec.n_triads()):
        member_p, member_q, _ = spec.sub_triad_modes(i)
        if name == f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}":
            return spec.sub_triad_modes(i)
    raise ValueError(f"no constituent triad of {spec.key!r} matches unit name {name!r}")


def plot(spec, results, path: str = None, evolution_xmax: float = EVOLUTION_XMAX_DAYS):
    apply_house_style(base_size=15)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9))

    triad_names = []
    for i in range(spec.n_triads()):
        member_p, member_q, _ = spec.sub_triad_modes(i)
        triad_names.append(f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}")

    for ax, name in zip(axes[0], triad_names):
        r = results[name]
        modes = _unit_modes(spec, name)
        highlight = r["labels"].index(HIGHLIGHT_LABEL)
        plot_energy_evolution(r["t"], r["E"], r["E_total"], r["labels"], modes,
                               highlight=highlight, ax=ax)
        ax.set_title(r["title"])
        ax.set_xlim(0, evolution_xmax)

    r_full = results["full"]
    plot_energy_evolution(r_full["t"], r_full["E"], r_full["E_total"], r_full["labels"],
                           _unit_modes(spec, "full"), highlight=r_full["labels"].index(HIGHLIGHT_LABEL),
                           ax=axes[1, 0])
    axes[1, 0].set_title(spec.display_label)
    axes[1, 0].set_xlim(0, evolution_xmax)

    novelty_result = novelty_combined_for_target(results, SPECTRUM_TARGET_LABEL)
    novelty_frequency_figure(results, SPECTRUM_TARGET_LABEL, novelty_result, ax=axes[1, 1],
                              full_label="Quartet", show_excluded_in_legend=False)

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
    parser.add_argument("--evolution-xmax", type=float, default=EVOLUTION_XMAX_DAYS,
                         help="days shown on each evolution panel's time axis (display only, "
                              "tuned by eye -- doesn't affect the registry's own tf_days)")
    args = parser.parse_args()

    spec, results = compute()
    plot(spec, results, path=args.path, evolution_xmax=args.evolution_xmax)
    print(f"wrote {os.path.abspath(args.path)}")

    triad_names = [n for n in results if n != "full"]
    report = compute_diagnostics_report(results, spec)

    print(f"\n=== headline numbers: {HIGHLIGHT_LABEL} across triads/full ({WAVE_SET_KEY}) ===")
    for name in triad_names + ["full"]:
        m = report["per_mode_unit"][name][HIGHLIGHT_LABEL]
        r = results[name]
        E_share_peak = 100 * r["E"][:, r["labels"].index(HIGHLIGHT_LABEL)].max() / r["E_total"].mean()
        print(f"[{name}] dEK={m['dEK']:.4e}  dominant_period={m['period_global']:.4f}d  "
              f"peak_share_of_mean_total={E_share_peak:.2f}%")

    print(f"\n=== pairwise diagnostics: {HIGHLIGHT_LABEL} full vs. each containing triad ===")
    for d in report["pairwise"]:
        if d["mode"] != HIGHLIGHT_LABEL:
            continue
        novelty = (f"{d['novelty_period_days']:.4f}d ({d['novelty_relevance_pct']:.2f}%)"
                   if d["novelty_period_days"] == d["novelty_period_days"] else "none detected")
        print(f"  vs {d['vs']}: efficiency_var={d['efficiency_var_pct']:+.2f}%  "
              f"spectral_dev={d['spectral_dev_pct']:.2f}%  novelty_period={novelty}")

    print(f"\n=== final (combined) diagnostics: {HIGHLIGHT_LABEL} ===")
    for d in report["final"]:
        if d["mode"] != HIGHLIGHT_LABEL:
            continue
        novelty = (f"{d['novelty_period_final_days']:.4f}d ({d['novelty_relevance_final_pct']:.2f}%)"
                   if d["novelty_period_final_days"] == d["novelty_period_final_days"] else "none detected")
        print(f"  efficiency_var_final={d['efficiency_var_final_pct']:+.2f}%  "
              f"spectral_dev_final={d['spectral_dev_final_pct']:.2f}%  vs={d['vs']}  "
              f"novelty_period_final={novelty}")

    print(f"\n=== spectrum-panel target: {SPECTRUM_TARGET_LABEL} (private to Triad 1) ===")
    for d in report["pairwise"]:
        if d["mode"] != SPECTRUM_TARGET_LABEL:
            continue
        novelty = (f"{d['novelty_period_days']:.4f}d ({d['novelty_relevance_pct']:.2f}%)"
                   if d["novelty_period_days"] == d["novelty_period_days"] else "none detected")
        print(f"  vs {d['vs']}: efficiency_var={d['efficiency_var_pct']:+.2f}%  "
              f"spectral_dev={d['spectral_dev_pct']:.2f}%  novelty_period={novelty}")
    for d in report["final"]:
        if d["mode"] != SPECTRUM_TARGET_LABEL:
            continue
        novelty = (f"{d['novelty_period_final_days']:.4f}d ({d['novelty_relevance_final_pct']:.2f}%)"
                   if d["novelty_period_final_days"] == d["novelty_period_final_days"] else "none detected")
        print(f"  efficiency_var_final={d['efficiency_var_final_pct']:+.2f}%  "
              f"spectral_dev_final={d['spectral_dev_final_pct']:.2f}%  vs={d['vs']}  "
              f"novelty_period_final={novelty}")


if __name__ == "__main__":
    main()
