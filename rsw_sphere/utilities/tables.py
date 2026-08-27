"""CSV table writer for driver output.

Not a LaTeX formatter (see rsw_sphere.plotting.wave_set_table for the
existing paper-table LaTeX/CSV builder) -- this is the plain ``rows: list
of dict -> csv`` helper run_dynamics.py uses for its own diag_*.csv
tables.
"""
import csv
import os


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
