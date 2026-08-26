"""Table ``tab: rhfamily`` (JFM-template.tex, "Rossby-only quartet" ->
"Partner preference", ``sec: quartet_rh_preference``): the RH(4,5)+RH(1,2)
driving pair against every even-n RH(3,n) target, n=4..16 -- isolated
triad efficiency E plus the P-measure P_a once RH(3,4) is restored as a
four-wave competitor.

Two current-driver sources, combined here by n:

- Isolated-triad columns (omega, period, A0, coeff, E) --
  ``examples/rh_partner_family.py``'s own ``run_family`` over the
  registered ``rh_partner_family`` family (``examples/triad_families.yaml``).
  ``tf_days=240`` is used uniformly (not the family's own tf_days=30
  default) since n=10 is not converged at 30d -- confirmed by that
  script's own tf-convergence self-check (30d: 0.84%, 240d: 12.64%,
  matching the paper's own reported value; every other member is
  already flat between 30d and 240d).
- P_a column -- ``run_sweep_sets.py``'s own ``run_sweep_sets`` applied to
  ``examples/candidates_rh_partner_family.yaml`` (RH(3,4) fixed in the
  quartet, RH(3,n) as the swapped-in competitor, p_measure diagnostic,
  same tf_days=240).

n=4 has no P_a entry (it is the quartet's own fixed private mode, not a
candidate); odd n are omitted (selection rule forces zero coupling).

Run:

    python examples/tables/paper_table03_rh_partner_family.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_EXAMPLES = os.path.join(_ROOT, "examples")
if _EXAMPLES not in sys.path:
    sys.path.insert(0, _EXAMPLES)

import numpy as np
import yaml

from rsw_sphere.physics import linear_period_days
from rh_partner_family import run_family
from run_sweep_sets import run_sweep_sets

TF_DAYS = 240.0
CANDIDATES_CONFIG = os.path.join(_EXAMPLES, "candidates_rh_partner_family.yaml")
DEFAULT_OUTPUT = os.path.join(_ROOT, "outputs", "tables", "paper_table03_rh_partner_family.csv")


def build_table(tf_days: float = TF_DAYS, candidates_config: str = CANDIDATES_CONFIG):
    """Returns a list of row dicts, one per even n=4..16, in table order."""
    triad_results = run_family("rh_partner_family", tf_days=tf_days)
    by_n = {r["n"]: r for r in triad_results}

    with open(candidates_config) as f:
        config = yaml.safe_load(f)
    config["tf_days"] = tf_days
    p_results = run_sweep_sets(config)
    p_by_n = {r["n"]: r["p_measure (%)"] for r in p_results}

    rows = []
    for n in (4, 6, 8, 10, 12, 14, 16):
        r = by_n[n]
        period = np.real(linear_period_days(r["omega_target"]))
        rows.append({
            "n": n,
            "delta": float(r["delta"].real),
            "omega": float(r["omega_target"].real),
            "period_days": float(period),
            "coeff": float(r["alpha_target"]),
            "efficiency_pct": float(100 * np.real(r["efficiency"])),
            "P_a_pct": None if n == 4 else float(p_by_n.get(n, float("nan"))),
        })
    return rows


def main():
    import argparse
    import csv
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=DEFAULT_OUTPUT)
    parser.add_argument("--tf-days", type=float, default=TF_DAYS)
    args = parser.parse_args()

    rows = build_table(tf_days=args.tf_days)

    os.makedirs(os.path.dirname(args.path), exist_ok=True)
    with open(args.path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {os.path.abspath(args.path)}")

    print(f"{'n':>3} {'delta':>10} {'period(d)':>10} {'coeff':>10} {'E(%)':>10} {'P_a(%)':>10}")
    for r in rows:
        pa = f"{r['P_a_pct']:.2f}" if r["P_a_pct"] is not None else "--"
        print(f"{r['n']:>3} {r['delta']:>10.4f} {r['period_days']:>10.3f} "
              f"{r['coeff']:>10.4f} {r['efficiency_pct']:>10.4f} {pa:>10}")


if __name__ == "__main__":
    main()
