"""CSV table writers for driver output.

Not a LaTeX formatter (see rsw_sphere.plotting.wave_set_table for the
existing paper-table LaTeX/CSV builder) -- this is the plain summary CSV
run_dynamics.py writes alongside its own figures, for
examples/tables/*.py scripts to read and format into a paper table.
"""
import csv
import os


def dynamics_summary_rows(result: dict, spec) -> list:
    """One row per (topology unit, mode) from a run_dynamics() result dict.

    Parameters
    ----------
    result : dict
        run_dynamics.run_dynamics's own return value (unit name -> dict
        with title/labels/dEK/drift).
    spec : WaveSetSpec
        Only used for spec.key (the 'wave_set' column).

    Returns
    -------
    list of dict
        wave_set, unit, title, mode, dEK, drift.
    """
    rows = []
    for unit_name, r in result.items():
        for label, dEK in zip(r['labels'], r['dEK']):
            rows.append({
                'wave_set': spec.key, 'unit': unit_name, 'title': r['title'],
                'mode': label, 'dEK': float(dEK), 'drift': r['drift'],
            })
    return rows


def write_csv(rows: list, path: str):
    """Write rows (list of same-keys dicts) to path, creating parent dirs."""
    if not rows:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
