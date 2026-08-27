"""Loop a diagnostic over a LIST of wave-set variants (which mode fills
one slot), orthogonal to run_sweep.py's IC sweep -- generalizes the kind
of hand-rolled candidate-mode catalogue enumeration that used to live in
one-off scripts (e.g. the now-deleted examples_legacy/special_runs/
gate_i2_map_extension.py's own find_catalogue()). One point per candidate,
at the base wave set's own
registered velocities, unless candidate_velocity overrides the
candidate slot's own velocity (not swept either way).

Config:

    base_wave_set: quartet_rossby_kelvin
    specs_path: wave_sets_default.yaml
    candidate_slot: d                          # mode key swapped per candidate
    candidate_velocity: 30.0                   # optional: drive the candidate at
                                                 # this velocity instead of its base
                                                 # spec's own registered one (often 0 --
                                                 # p_measure/filtering_error still pick up
                                                 # a passively, nonlinearly excited candidate)
    candidates_from: {max_n: 15}                # m inferred from candidate_slot's
                                                 # own triad selection rule; n in [m, max_n]; or:
    candidates: [{m: 1, n: 1, alpha: 1}, ...]   # explicit list
    alphas: [1, 2]                              # candidates_from only; default EG+WG
    diagnostics: [p_measure, filtering_error]   # subset of registry.ALL_2D
    tf_days: 20
    h: 0.01
    table: outputs/tables/quartet_rossby_kelvin_candidates.csv
    output: outputs/figures/wave_sets/quartet_rossby_kelvin_candidates.png
    output_root: outputs

Run:

    python run_sweep_sets.py --config examples/candidates_quartet_rossby_kelvin.yaml
"""
import argparse
import csv
import dataclasses
import os
from concurrent.futures import ProcessPoolExecutor

import yaml

from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs
from rsw_sphere.plotting.labels import _mode_label
from rsw_sphere.utilities.pmeasure import wave_set_diagnostics_sweep, _DIAGNOSTIC_ARRAY_KEYS as _PAIRWISE_KEYS
from rsw_sphere.utilities.functional import functional_diagnostics_sweep
from rsw_sphere.utilities.registry import PAIRWISE, FUNCTIONAL, ALL_2D


def _required_m_for_slot(spec, candidate_slot):
    """The zonal wavenumber a candidate must have to keep candidate_slot's
    own triad(s) selection-rule-valid (m_sum = m_p + m_q), holding every
    OTHER mode in that triad at its current, fixed (m,n,alpha). Raises if
    candidate_slot appears in more than one triad with different implied m
    (ambiguous -- pass `candidates:` explicitly instead).
    """
    required = set()
    for i, t in enumerate(spec.triads):
        i_sum, i_p, i_q = spec.triad_indices(i)
        idx = spec.index(candidate_slot)
        if idx == i_sum:
            required.add(spec.modes[i_p][0] + spec.modes[i_q][0])
        elif idx == i_p:
            required.add(spec.modes[i_sum][0] - spec.modes[i_q][0])
        elif idx == i_q:
            required.add(spec.modes[i_sum][0] - spec.modes[i_p][0])
    if len(required) != 1:
        raise ValueError(
            f"{candidate_slot!r} implies {len(required)} different required "
            f"m across its own triad(s) ({required}) -- pass `candidates:` explicitly.")
    return required.pop()


def _candidates_from_edge(spec, candidate_slot, max_n, alphas=(1, 2)):
    m = _required_m_for_slot(spec, candidate_slot)
    return [(m, n, alpha) for alpha in alphas for n in range(m, max_n + 1)]


def _build_candidate_spec(spec, candidate_slot, mode_triple, velocity=None):
    idx = spec.index(candidate_slot)
    modes = list(spec.modes)
    modes[idx] = mode_triple
    if velocity is None:
        return dataclasses.replace(spec, modes=tuple(modes))
    velocities = list(spec.velocities)
    velocities[idx] = velocity
    return dataclasses.replace(spec, modes=tuple(modes), velocities=tuple(velocities))


_ROW_LABELS = {"p_measure": "p_measure (%)", "filtering_error": "filtering_error"}


def _one_candidate(args):
    """Worker: one diagnostic point for one candidate (module-level, picklable).
    Invalid/non-convergent candidates (e.g. no Hough mode at this (m,n,alpha))
    return {'error': str(exc)} instead of raising, so one bad candidate
    doesn't crash the whole batch.
    """
    try:
        return _one_candidate_inner(args)
    except Exception as exc:
        return {"error": str(exc)}


def _one_candidate_inner(args):
    cand_spec, target_idx, diagnostics, tf_days, h = args
    triad_indices = [cand_spec.triad_indices(i) for i in range(cand_spec.n_triads())]

    row = {}
    pairwise = [d for d in diagnostics if d in PAIRWISE]
    if pairwise:
        u_fixed = cand_spec.velocities[target_idx]
        r = wave_set_diagnostics_sweep(
            cand_spec.modes, triad_indices, cand_spec.h_e, (target_idx, target_idx),
            {i: cand_spec.velocities[i] for i in range(cand_spec.n_modes()) if i != target_idx},
            [target_idx], diagnostics=tuple(pairwise),
            u1_range=(u_fixed, u_fixed), u2_range=(u_fixed, u_fixed),
            reference_triad=cand_spec.reference_triad, n_grid=1, tf_days=tf_days, h=h)
        for d in pairwise:
            row[_ROW_LABELS[d]] = r[_PAIRWISE_KEYS[d]][0, 0, 0]

    functional = [d for d in diagnostics if d in FUNCTIONAL]
    if functional:
        r = functional_diagnostics_sweep(
            cand_spec.modes, triad_indices, cand_spec.h_e, (target_idx, target_idx),
            {i: cand_spec.velocities[i] for i in range(cand_spec.n_modes()) if i != target_idx},
            [target_idx], diagnostics=tuple(functional),
            u1_range=(cand_spec.velocities[target_idx], cand_spec.velocities[target_idx]),
            u2_range=(cand_spec.velocities[target_idx], cand_spec.velocities[target_idx]),
            n_grid=1, tf_days=tf_days, h=h)
        if "efficiency" in functional:
            row["efficiency (%)"] = 100 * r["Efficiency"][0, 0, 0]
        if "low_frequency_energy" in functional:
            row["low_frequency_energy"] = r["LowFreqEnergy"][0, 0, 0]

    return row


def run_sweep_sets(config: dict) -> list:
    """Run one diagnostic point per candidate. Returns a list of row dicts
    (candidate label + mode + requested diagnostic values)."""
    specs_path = config.get("specs_path", DEFAULT_WAVESETS_PATH)
    spec = load_wave_set_specs(specs_path)[config["base_wave_set"]]
    slot = config["candidate_slot"]
    # target_mode is the mode whose own diagnostic value is reported --
    # usually a DIFFERENT, already-driven mode (e.g. the RH partner), not
    # candidate_slot itself: candidate_slot typically starts at rest, so
    # its own P-measure/F2 against its own (single) reference triad is a
    # trivial near-zero comparison, not the screening question of
    # interest ("how much does swapping in this candidate affect the
    # already-driven target mode").
    target_idx = spec.index(config.get("target_mode", slot))

    if "candidates" in config:
        candidates = [(c["m"], c["n"], c["alpha"]) for c in config["candidates"]]
    else:
        cf = config["candidates_from"]
        candidates = _candidates_from_edge(spec, slot, cf["max_n"],
                                            tuple(config.get("alphas", (1, 2))))

    diagnostics = tuple(config.get("diagnostics", ("p_measure", "filtering_error")))
    unknown = set(diagnostics) - ALL_2D
    if unknown:
        raise ValueError(f"unknown diagnostic(s) {unknown} -- must be a subset of {ALL_2D}")

    tf_days = config.get("tf_days", spec.settings.get("tf_days", 10))
    h = config.get("h", spec.settings.get("h", 0.01))

    candidate_velocity = config.get("candidate_velocity")
    cand_specs = [_build_candidate_spec(spec, slot, m, velocity=candidate_velocity) for m in candidates]
    args = [(cs, target_idx, diagnostics, tf_days, h) for cs in cand_specs]

    max_workers = config.get("max_workers") or max(1, (os.cpu_count() or 2) // 2)
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(_one_candidate, args))

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(f"Running candidate sweep for {config['base_wave_set']!r}, slot {config['candidate_slot']!r}...")
    results = run_sweep_sets(config)

    table_path = config.get("table")
    if table_path:
        _write_table(results, table_path)
        print(f"wrote {os.path.abspath(table_path)}")

    for r in results:
        print(f"  {r}")


if __name__ == "__main__":
    main()
