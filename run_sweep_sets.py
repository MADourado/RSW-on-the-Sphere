"""Loop a diagnostic over a LIST of wave-set variants (a mode fills
one slot). Candidates and their diagnostics are read straight from a wave set's own registered
``alternative_modes`` block (wave_sets_default.yaml)

Registry schema (one block per swapped slot, inside a
wave_sets_default.yaml entry):

    alternative_modes:
      d:                              # mode key being substituted
        target_mode: d                # optional, default: the slot itself
        diagnostics: [p_measure]      # subset of {p_measure, p_measure_final,
                                       # efficiency_var, spectral_dev_var,
                                       # novelty_period, efficiency,
                                       # low_frequency_energy}
        candidate_velocity: 30.0      # optional: drive every candidate at
                                       # this velocity instead of the slot's
                                       # own registered one
        tf_days: 20                   # optional override (default: the
        h: 0.01                       # wave set's own settings block)
        candidates:                   # m/n/alpha triples -- paste straight
          - {m: 1, n: 1, alpha: 1}    # from run_mode_search.py's own output
          - {m: 1, n: 3, alpha: 1}    # (drop role/required_m/coup_*/pump)

Every requested diagnostic, plus each candidate's own static ``delta``
(frequency mismatch) / ``coeff`` (coupling coefficient) / isolated-triad
``efficiency`` becomes both a CSV column and its own
point-wise figure, one point per candidate.

Run:

    python run_sweep_sets.py --wave-set quartet_rossby_kelvin --slot d
"""
import argparse
import csv
import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import yaml

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.dynamics.run_config import RunConfig, default_max_workers
from rsw_sphere.dynamics.diagnostics_report import (compute_diagnostics_report, pairwise_value_for_target,
                                                      write_diagnostics_files)
from rsw_sphere.dynamics.trajectory_cache import _mode_slug
from rsw_sphere.utilities.pmeasure import _default_triad_index_for_mode
from rsw_sphere.plotting.labels import _mode_label, mode_fs_label
from rsw_sphere.plotting.wave_set_table import wave_set_properties
from rsw_sphere.plotting.sweep_diagnostics import plot_candidate_scalar

from run_dynamics import run_dynamics

#: run_sweep_sets.py's own diagnostic vocabulary. p_measure/novelty_period
#: are the PAIRWISE flavor (against target_idx's own resolved reference
#: triad, rsw_sphere.utilities.pmeasure._default_triad_index_for_mode) --
#: p_measure_final/efficiency_var are the "final" combined-across-every-
#: containing-sub-triad flavor (report['final']), same meaning run_sweep.py's
#: own unified engine uses.
_KNOWN_DIAGNOSTICS = frozenset({"p_measure", "p_measure_final", "efficiency_var", "spectral_dev_var",
                                 "novelty_period", "efficiency", "low_frequency_energy"})
_ROW_LABELS = {"p_measure": "p_measure (%)", "p_measure_final": "p_measure_final (%)",
               "efficiency_var": "efficiency_var (%)", "spectral_dev_var": "spectral_dev_var (%)",
               "novelty_period": "novelty_period (days)"}
#: report['pairwise']'s own field name for each pairwise diagnostic.
_PAIRWISE_FIELDS = {"p_measure": "p_measure_pct", "novelty_period": "novelty_period_days"}
#: Short, paper-facing display name per plotted column -- keeps figure titles
#: readable; falls back to the raw column name (_write_plots) for anything
#: not listed here.
_COL_TITLES = {
    "delta": "Mismatch δ", "coeff": "Coupling coefficient",
    "omega": "Frequency ω", "period_days": "Period (days)",
    "isolated_triad_efficiency (%)": "Isolated-triad efficiency (%)",
    "efficiency_var (%)": "Efficiency variation (%)",
    "p_measure (%)": "P-measure (%)", "p_measure_final (%)": "P-measure, final (%)",
    "spectral_dev_var (%)": "Spectral deviation, final (%)",
    "novelty_period (days)": "Novelty period (days)", "efficiency (%)": "Efficiency (%)",
    "low_frequency_energy": "Low-frequency energy",
}
#: Columns present on every row that aren't a diagnostic/property to plot.
_BASE_COLUMNS = frozenset({"mode", "m", "n", "alpha", "error", "run_dir", "figure_path",
                            "diag_pairwise", "diag_final", "novelty_figures"})


def _build_candidate_spec(spec, candidate_slot, mode_triple, velocity=None):
    idx = spec.index(candidate_slot)
    modes = list(spec.modes)
    modes[idx] = mode_triple
    if velocity is None:
        return dataclasses.replace(spec, modes=tuple(modes))
    velocities = list(spec.velocities)
    velocities[idx] = velocity
    return dataclasses.replace(spec, modes=tuple(modes), velocities=tuple(velocities))


def _one_candidate(args):
    """Worker: one diagnostic point for one candidate (module-level, picklable).
    Invalid/non-convergent candidates (e.g. no Hough mode at this (m,n,alpha),
    or a selection-rule violation) return {'error': str(exc)} instead of
    raising, so one bad candidate doesn't crash the whole batch.
    """
    try:
        return _one_candidate_inner(args)
    except Exception as exc:
        return {"error": str(exc)}


def _one_candidate_inner(args):
    """One run_dynamics() + compute_diagnostics_report() call at the
    candidate's own static registered velocities (no sweep at all).
    Every requested diagnostic is read off the SAME report, no
    per-diagnostic extra integration. delta/coeff/isolated-triad
    efficiency are read off the target's own resolved reference triad
    (same resolution rule pairwise_value_for_target uses).
    """
    cand_spec, target_idx, diagnostics, tf_days, h, output_root, plot_dynamics = args
    target_label = _mode_label(*cand_spec.modes[target_idx])

    config = RunConfig.from_wave_set(cand_spec, tf_days=tf_days, h=h, output_root=output_root,
                                      plot=plot_dynamics, parallel=False)
    results = run_dynamics(config)
    report = compute_diagnostics_report(results, cand_spec)

    row = {}
    if plot_dynamics:
        # run_dir/figure_path are already unique per candidate (trajectory_cache.ic_label
        # encodes every mode's own identity+velocity) and land in the SAME shared
        # outputs/dynamics/<wave_set_key>/ tree every other driver uses -- not a
        # sweep-specific subtree, so trajectory caching (keyed off this same
        # output_root) stays shared/reusable. These columns are this table's
        # own index back into that tree, not a relocation of it.
        run_dir = results["full"]["run_dir"]
        run_label = os.path.basename(run_dir)
        row["run_dir"] = run_dir
        row["figure_path"] = results["full"]["figure_path"]
        # Same bundle run_dynamics.py --diagnostics writes (diag_evol/diag_prec_freq/
        # diag_pairwise/diag_final CSVs + diag_freq_novel_*.png spectra), reusing the
        # SAME report already computed above (write_diagnostics_files's own docstring
        # calls this reuse out explicitly: "factored out for reuse by a sweep's own
        # per-point diagnostics bundle").
        diag_paths = write_diagnostics_files(results, report, run_dir, run_label, cand_spec, diagnostics=True)
        row["diag_pairwise"] = diag_paths.get("diag_pairwise")
        row["diag_final"] = diag_paths.get("diag_final")
        row["novelty_figures"] = ";".join(diag_paths.get("novelty", []))
    triads = [cand_spec.triad_indices(i) for i in range(cand_spec.n_triads())]
    t_idx = _default_triad_index_for_mode(triads, cand_spec.reference_triad, target_idx)
    if t_idx is not None:
        props = wave_set_properties(cand_spec)
        row["delta"] = props["triads"][t_idx]["delta"]
        row["coeff"] = abs(props["coef"][target_idx, t_idx])
        row["omega"] = props["omega"][target_idx]
        row["period_days"] = props["period_days"][target_idx]
        member_p, member_q, _ = cand_spec.sub_triad_modes(t_idx)
        unit_name = f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}"
        row["isolated_triad_efficiency (%)"] = 100 * report["per_mode_unit"][unit_name][target_label]["efficiency"]

    for d in diagnostics:
        if d in _PAIRWISE_FIELDS:
            row[_ROW_LABELS[d]] = pairwise_value_for_target(
                report, cand_spec, target_idx, cand_spec.reference_triad, _PAIRWISE_FIELDS[d])
        elif d == "p_measure_final":
            final_row = next((r for r in report["final"] if r["mode"] == target_label), None)
            row[_ROW_LABELS[d]] = final_row["p_measure_final_pct"] if final_row else float("nan")
        elif d == "efficiency_var":
            final_row = next((r for r in report["final"] if r["mode"] == target_label), None)
            row[_ROW_LABELS[d]] = final_row["efficiency_var_final_pct"] if final_row else float("nan")
        elif d == "spectral_dev_var":
            final_row = next((r for r in report["final"] if r["mode"] == target_label), None)
            row[_ROW_LABELS[d]] = final_row["spectral_dev_final_pct"] if final_row else float("nan")
        elif d == "efficiency":
            row["efficiency (%)"] = 100 * report["per_mode_unit"]["full"][target_label]["efficiency"]
        elif d == "low_frequency_energy":
            row["low_frequency_energy"] = report["per_mode_unit"]["full"][target_label]["low_freq_power"]

    return row


def load_alternative_modes(spec_key: str, slot: str, specs_path: str = DEFAULT_WAVESETS_PATH) -> dict:
    """The raw ``alternative_modes[slot]`` block from one
    wave_sets_default.yaml entry -- read straight from the raw YAML dict
    (like ``sweep``/``plot``), not folded into WaveSetSpec since only this
    driver consumes it.
    """
    with open(specs_path) as f:
        raw = yaml.safe_load(f)
    if spec_key not in raw:
        raise ValueError(f"{spec_key!r} not found in {specs_path!r} (available: {list(raw)})")
    alt = raw[spec_key].get("alternative_modes", {})
    if slot not in alt:
        raise ValueError(f"{spec_key!r} has no alternative_modes for slot {slot!r} "
                          f"(available: {list(alt)})")
    return alt[slot]


def run_sweep_sets(spec_key: str, slot: str, specs_path: str = DEFAULT_WAVESETS_PATH,
                    output_root: str = "outputs", max_workers: int = None,
                    diagnostics_override=None, tf_days_override: float = None,
                    h_override: float = None, plot_dynamics: bool = False) -> list:
    """Run one static diagnostic point per candidate registered under
    ``<spec_key>``'s own ``alternative_modes[slot]`` block.

    Parameters
    ----------
    plot_dynamics : bool, optional
        Also save each candidate's own run_dynamics energy-evolution
        figures (full wave set + every constituent triad), for visually
        inspecting a surprising diagnostic value instead of trusting the
        scalar alone. Figures land in the SAME shared ``outputs/dynamics/
        <spec_key>/<ic_label>_tf..._h.../`` tree every other driver uses
        (unique per candidate already, since ``ic_label`` encodes every
        mode's own identity+velocity) -- not a sweep-specific subtree,
        so trajectory caching stays shared/reusable rather than
        duplicated. ``run_dir``/``figure_path`` columns on each returned
        row (and ``candidates.csv``) are this sweep's own index back into
        that tree.

    Returns
    -------
    list of dict
        One row per candidate: ``mode``/``m``/``n``/``alpha`` (candidate
        identity), ``delta``/``coeff``/``isolated_triad_efficiency (%)``
        (static properties of the target's own resolved reference triad),
        plus every requested diagnostic, plus ``run_dir``/``figure_path``
        if ``plot_dynamics`` -- or ``{'error': str}`` for a candidate that
        failed to build/integrate.
    """
    spec = load_wave_set_specs(specs_path)[spec_key]
    cfg = load_alternative_modes(spec_key, slot, specs_path)

    target_idx = spec.index(cfg.get("target_mode", slot))
    candidates = [(c["m"], c["n"], c["alpha"]) for c in cfg["candidates"]]

    diagnostics = tuple(diagnostics_override) if diagnostics_override is not None \
        else tuple(cfg.get("diagnostics", ("p_measure",)))
    unknown = set(diagnostics) - _KNOWN_DIAGNOSTICS
    if unknown:
        raise ValueError(f"unknown diagnostic(s) {unknown} -- must be a subset of {_KNOWN_DIAGNOSTICS}")

    tf_days = tf_days_override if tf_days_override is not None \
        else cfg.get("tf_days", spec.settings.get("tf_days", 10))
    h = h_override if h_override is not None else cfg.get("h", spec.settings.get("h", 0.01))
    candidate_velocity = cfg.get("candidate_velocity")

    cand_specs = [_build_candidate_spec(spec, slot, m, velocity=candidate_velocity) for m in candidates]
    args = [(cs, target_idx, diagnostics, tf_days, h, output_root, plot_dynamics) for cs in cand_specs]

    workers = max_workers or default_max_workers()
    n = len(args)
    rows = [None] * n
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one_candidate, a): i for i, a in enumerate(args)}
        for done, fut in enumerate(as_completed(futures), start=1):
            i = futures[fut]
            row = fut.result()
            rows[i] = row
            suffix = f" -> {row['run_dir']}" if plot_dynamics and "run_dir" in row else ""
            print(f"  [{done}/{n}] {_mode_label(*candidates[i])} done{suffix}")

    results = []
    for mode_triple, row in zip(candidates, rows):
        entry = {"mode": _mode_label(*mode_triple), "m": mode_triple[0],
                  "n": mode_triple[1], "alpha": mode_triple[2]}
        entry.update(row)
        results.append(entry)
    return results


def _write_table(results, path):
    if not results:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _write_plots(results, out_dir, target_label, fixed_labels, registered_fs, target_fs):
    """One point-wise (no connecting line) figure per non-base column --
    every requested diagnostic plus the static delta/coeff/isolated-triad
    efficiency columns, each against the candidate's own mode label.

    registered_fs/target_fs (rsw_sphere.plotting.labels.mode_fs_label,
    e.g. ``EG1_1``/``RH1_2``) are embedded in every filename so a PNG
    copied out of ``out_dir`` on its own (e.g. into the paper repo) still
    says which slot's own originally-registered mode these candidates
    replace and which mode's own diagnostic is plotted -- same
    self-containment rationale as run_dynamics.py's own evol_*.png
    filenames. The title stays short (that context already lives in the
    filename/directory) -- just "<diagnostic> of <target> in quartet with
    <fixed>+x".
    """
    ok = [r for r in results if "error" not in r]
    if not ok:
        return
    labels = [r["mode"] for r in ok]
    for col in ok[0]:
        if col in _BASE_COLUMNS:
            continue
        values = [r[col] for r in ok]
        fname = col.split(" ")[0].replace("/", "_")
        path = os.path.join(out_dir, f"candidates_{fname}_{registered_fs}_target_{target_fs}.png")
        title = f"{_COL_TITLES.get(col, col)} of {target_label} in quartet with {fixed_labels}+x"
        plot_candidate_scalar(labels, values, col, title, path)
        print(f"figure -> {os.path.abspath(path)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave-set", required=True)
    parser.add_argument("--slot", required=True,
                         help="role letter to swap candidates into (e.g. 'd'), "
                              "matching a key in the wave set's own modes: and "
                              "alternative_modes: blocks -- not a mode label like RH(2,3)")
    parser.add_argument("--specs", default=DEFAULT_WAVESETS_PATH)
    parser.add_argument("--diagnostics", default=None,
                         help="comma-separated, overrides the registry's own "
                              "alternative_modes.<slot>.diagnostics")
    parser.add_argument("--tf-days", type=float, default=None)
    parser.add_argument("--h", type=float, default=None)
    parser.add_argument("--table", default=None)
    parser.add_argument("--plot-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--dynamics-plots", action="store_true",
                         help="also save each candidate's own run_dynamics energy-evolution "
                              "figures (full wave set + every constituent triad), for visually "
                              "checking a surprising diagnostic value. Figures land in the "
                              "normal shared outputs/dynamics/<wave-set>/ tree (unique per "
                              "candidate already); run_dir/figure_path columns on "
                              "candidates.csv index straight into them.")
    args = parser.parse_args()

    diagnostics_override = tuple(args.diagnostics.split(",")) if args.diagnostics else None

    print(f"Running candidate substitution for {args.wave_set!r}, slot {args.slot!r}...")
    results = run_sweep_sets(args.wave_set, args.slot, specs_path=args.specs,
                              output_root=args.output_root, max_workers=args.max_workers,
                              diagnostics_override=diagnostics_override,
                              tf_days_override=args.tf_days, h_override=args.h,
                              plot_dynamics=args.dynamics_plots)

    spec = load_wave_set_specs(args.specs)[args.wave_set]

    # Which mode a diagnostic is actually computed FOR -- defaults to the
    # slot itself, but the registry can point it elsewhere (see
    # run_sweep_sets()'s own target_idx resolution above); must be
    # resolved the same way here so the title/filenames stay accurate.
    cfg = load_alternative_modes(args.wave_set, args.slot, args.specs)
    target_mode_key = cfg.get("target_mode", args.slot)
    target_mnalpha = spec.modes[spec.index(target_mode_key)]
    target_label = _mode_label(*target_mnalpha)
    target_fs = mode_fs_label(*target_mnalpha)

    registered_mnalpha = spec.modes[spec.index(args.slot)]
    registered_paper_label = _mode_label(*registered_mnalpha)
    registered_fs = mode_fs_label(*registered_mnalpha)
    # Dir-safe (no parens/commas, unlike registered_fs/target_fs's own
    # single-underscore convention) -- system tools choke on "()" in a
    # bare directory name.
    registered_dir_label = registered_paper_label.replace('(', '_').replace(')', '').replace(',', '_')

    fixed_labels = "+".join(_mode_label(*spec.modes[i]) for i, key in enumerate(spec.mode_keys)
                             if key not in (args.slot, target_mode_key))

    # Named after the slot's own ORIGINALLY REGISTERED mode (e.g. EG_1_1), not the
    # bare slot letter (e.g. "d") -- a slot letter alone isn't informative when
    # browsing outputs/sweep_sets/ later.
    out_dir = args.plot_dir or os.path.join("outputs", "sweep_sets", args.wave_set,
                                             f"alternatives_to_mode_{registered_dir_label}")
    table_path = args.table or os.path.join(out_dir, "candidates.csv")
    _write_table(results, table_path)
    print(f"table -> {os.path.abspath(table_path)}")

    if not args.no_plot:
        _write_plots(results, out_dir, target_label, fixed_labels, registered_fs, target_fs)

    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
