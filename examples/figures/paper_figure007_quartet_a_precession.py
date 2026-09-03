"""Figure ``fig: quartet_a_precession`` (JFM-template.tex, subfigure (a) of
``fig: precession_frequency``, ``sec: quartet_rh_precession``,
``Figures/quartet_a_rh36_precession.png``): precession frequency and
target-mode efficiency variation for Quartet A's target triad
RH(4,5)+RH(1,2)+RH(3,4), sweeping RH(3,6)'s own driving velocity while the
shared edge stays fixed.

Calls run_sweep.run_sweep once for [dynamical_phase, efficiency_var] on the
registered ``quartet_rh_preference`` entry (one shared per-grid-point
computation -- the same pair of standalone figures
``run_sweep.py --wave-set quartet_rh_preference`` would produce), then
composes this dual-axis figure itself from the two results, reusing
``rsw_sphere.plotting.precession_plot.plot_dual_axis_frequency_efficiency``
(only its ``efficiency_label`` overridden, to `Delta-E_a`). No config file
needed beyond the registry entry's own ``sweep:``/``settings:`` blocks; the
paper-specific framing (which mode's efficiency to show, which triad's own
frequency curve, axis labels) lives here, not in the registry --
run_sweep.py itself has no paper-specific concept of either.

Run:

    python examples/figures/paper_figure007_quartet_a_precession.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.plotting.precession_plot import plot_dual_axis_frequency_efficiency
from run_sweep import run_sweep

WAVE_SET_KEY = "quartet_rh_preference"
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "figures", "wave_sets", WAVE_SET_KEY,
                               "paper_figure007_quartet_a_precession.png")
TARGET_MODE_KEY = "c"    # RH(3,4) -- efficiency variation shown on the right axis
PLOT_TRIAD = 0           # Triad 1 -- only its own precession-frequency curve is drawn
XLABEL = "RH(3,6) driving velocity $u$ (m/s)"
TITLE = "Quartet A"
PLOT_U_MAX = 120.0


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--no-plot-per-point", action="store_true",
                         help="skip the per-grid-point run_dynamics pass (diagnostics only)")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    spec = specs[WAVE_SET_KEY]
    config = RunConfig.from_registry_entry(WAVE_SET_KEY, args.specs)

    print(f"Running sweep for wave set {WAVE_SET_KEY!r} (dynamical_phase + efficiency_var)...")
    sweep_results = run_sweep(config, plot_per_point=not args.no_plot_per_point)

    target_label = _mode_label(*spec.modes[spec.index(TARGET_MODE_KEY)])
    # dynamical_phase's own series keys are the long, self-describing
    # labels _build_units() constructs ("Triad 1 (RH(3,4)+RH(1,2)+RH(4,5))"),
    # not the registry's own short display_label ("Triad 1") --
    # plot_dual_axis_frequency_efficiency expects the short form (it
    # builds its own long legend text internally from spec.triads), and
    # _build_units() preserves registry triad order, so zip them back
    # together. efficiency_var's own series is keyed by mode label alone
    # (a "final" scalar, not a per-(mode,unit) pair) and is already a %,
    # matching run_sweep.py's own convention -- plot_dual_axis_frequency_efficiency
    # does its own *100 for display, so undo it here to avoid a double
    # percentage conversion.
    freq_by_triad = dict(zip((t.display_label for t in spec.triads),
                              sweep_results['dynamical_phase']['series'].values()))
    result = {
        'u_values': sweep_results['dynamical_phase']['u_values'],
        'freq_by_triad': freq_by_triad,
        'efficiency': sweep_results['efficiency_var']['series'][target_label] / 100,
        'triad_labels': [t.display_label for t in spec.triads],
    }

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    plot_dual_axis_frequency_efficiency(
        result, spec, plot_triad=PLOT_TRIAD, xlabel=XLABEL, title=TITLE,
        plot_u_max=PLOT_U_MAX, path=args.path,
        efficiency_label=f'Effic. var. {target_label}')
    print(f"  figure -> {os.path.abspath(args.path)}")

    min_freq = {lbl: float(np.min(vals)) for lbl, vals in result['freq_by_triad'].items()}
    print(f"  min |precession_freq| per triad: {min_freq}")


if __name__ == "__main__":
    main()
