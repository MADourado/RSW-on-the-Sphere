"""Figure for ``sec: quartet_rossby_gravity_fast`` (JFM-template.tex),
Quartet E (``quartet_rossby_gravity_influence_high``): EG(7,9) plays a
double role -- sum mode closing the RH-only triad (Triad 1, with
RH(4,5)+RH(3,4)), and a member of a second,
higher triad closed by the even-higher-frequency gravity mode EG(11,11)
(Triad 2, with RH(4,5)). Structurally the same "shared edge" pattern as
Quartet D's own panel (P1=EG(7,9), P2=RH(4,5), apexes RH(3,4)/EG(11,11)),
just with EG(7,9) itself also playing the SUM role for Triad 1 rather
than being one of the two apexes.

2x2 grid -- top row: Triad 1 and Triad 2 energy evolution; bottom row:
the full quartet's own evolution, and RH(3,4)'s novelty-frequency
spectrum against Triad 1, its only containing triad (RH(3,4) is private,
not a member of Triad 2). RH(4,5) is highlighted in the evolution panels
(the mode shared by both triads). RH(3,4) develops a spectral peak near
0.54d in the full quartet, matching Triad 2's own dominant period almost
exactly -- Triad 2's periodicity transmitting into a mode that isn't one
of its members, the same mechanism as Quartet D's own RH(3,4) story (see
that figure's module docstring). This peak sits inside Triad 2's own
excluded window, so it is NOT flagged by the novelty-detection algorithm
(correctly: it's already explained by Triad 2, not a genuinely new
period) -- visible in the spectrum panel itself, not in
report['final']['RH(3,4)']['novelty_period_final_days'].

Run:

    python examples/figures/paper_figure012_quartet_rossby_gravity_influence_high_panel.py
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

WAVE_SET_KEY = "quartet_rossby_gravity_influence_high"
HIGHLIGHT_LABEL = "RH(4,5)"     # shared member -- highlighted in every evolution panel
SPECTRUM_TARGET_LABEL = "RH(3,4)"  # private to Triad 1 -- the novelty-spectrum panel's target
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure012_quartet_rossby_gravity_influence_high_panel.png")

#: Display-only time-axis limit (days) for the three evolution panels --
#: wide enough to show several cycles of the slower (~0.54d) modulation
#: RH(4,5)/EG(7,9)/EG(11,11) carry in Triad 2/the full quartet, not just
#: the much faster (~0.15d) dominant oscillation.
EVOLUTION_XMAX_DAYS = 5.0

#: Display-only period-axis limit (days) for the spectrum panel -- the
#: auto-derived default (sqrt(tf_days/2), 3d at tf_days=20) leaves mostly
#: empty space past the 0.148d dominant peak and 0.54d secondary bump,
#: both well inside 1.5d.
SPECTRUM_XMAX_DAYS = 1.5


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


def plot(spec, results, path: str = None, evolution_xmax: float = EVOLUTION_XMAX_DAYS,
         spectrum_xmax: float = SPECTRUM_XMAX_DAYS):
    apply_house_style()
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
        ax.set_title(r["title"], fontsize=10)
        ax.set_xlim(0, evolution_xmax)

    r_full = results["full"]
    plot_energy_evolution(r_full["t"], r_full["E"], r_full["E_total"], r_full["labels"],
                           _unit_modes(spec, "full"), highlight=r_full["labels"].index(HIGHLIGHT_LABEL),
                           ax=axes[1, 0])
    axes[1, 0].set_title(spec.display_label, fontsize=10)
    axes[1, 0].set_xlim(0, evolution_xmax)

    novelty_result = novelty_combined_for_target(results, SPECTRUM_TARGET_LABEL)
    novelty_frequency_figure(results, SPECTRUM_TARGET_LABEL, novelty_result, ax=axes[1, 1], xmax=spectrum_xmax)
    axes[1, 1].set_title(f"Frequency spectrum: {SPECTRUM_TARGET_LABEL}", fontsize=10)

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
    parser.add_argument("--spectrum-xmax", type=float, default=SPECTRUM_XMAX_DAYS,
                         help="days shown on the spectrum panel's period axis (display only)")
    args = parser.parse_args()

    spec, results = compute()
    plot(spec, results, path=args.path, evolution_xmax=args.evolution_xmax, spectrum_xmax=args.spectrum_xmax)
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

    print(f"\n=== spectrum-panel target: {SPECTRUM_TARGET_LABEL} (private to Triad 1) ===")
    for name in triad_names + ["full"]:
        if SPECTRUM_TARGET_LABEL not in results[name]["labels"]:
            continue
        m = report["per_mode_unit"][name][SPECTRUM_TARGET_LABEL]
        print(f"[{name}] {SPECTRUM_TARGET_LABEL}: dominant_period={m['period_global']:.4f}d  "
              f"top_peaks={[(round(p['period_days'], 4), round(p['power_frac'], 2)) for p in m['top_peaks']]}")
    for d in report["final"]:
        if d["mode"] != SPECTRUM_TARGET_LABEL:
            continue
        novelty = (f"{d['novelty_period_final_days']:.4f}d ({d['novelty_relevance_final_pct']:.2f}%)"
                   if d["novelty_period_final_days"] == d["novelty_period_final_days"] else "none detected")
        print(f"  p_measure_final={d['p_measure_final_pct']:+.2f}%  "
              f"spectral_dev_final={d['spectral_dev_final_pct']:.2f}%  vs={d['vs']}  "
              f"novelty_period_final={novelty}")


if __name__ == "__main__":
    main()
