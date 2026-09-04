"""Table ``tab: rhfamily`` (JFM-template.tex, "Rossby-only quartet" ->
"Partner preference", ``sec: quartet_rh_preference``): the RH(4,5)+RH(1,2)
driving pair against every even-n RH(3,n) target, n=4..16 -- isolated
triad efficiency E plus the efficiency variation Delta-E_a (eq: effvar)
once RH(3,4) is restored as a four-wave competitor.

Both column groups come from one source: ``run_sweep_sets.py``'s own
``run_sweep_sets`` applied to ``quartet_rh_preference``'s registered
``alternative_modes.d`` block (``wave_sets_default.yaml``) -- RH(3,n)
substituted into slot d, RH(3,4) fixed in slot c as the quartet's own
four-wave competitor. Each candidate's own resolved reference triad
(sum=a, members=[b, d]) is the RH(4,5)+RH(1,2)+RH(3,n) triad evaluated
standalone. The registered ``tf_days: 240`` is what this table's own
tf-convergence check settled on -- do not lower it.

n=4 has no efficiency-variation entry (it is the quartet's own fixed
private mode, not a candidate); odd n are omitted (selection rule forces
zero coupling).

Run:

    python examples/tables/paper_table03_rh_partner_family.py
"""
import os

import _bootstrap  # noqa: F401 -- repo root on sys.path

from run_sweep_sets import run_sweep_sets

TF_DAYS = 240.0
DEFAULT_OUTPUT = os.path.join(_bootstrap.ROOT, "outputs", "tables", "paper_table03_rh_partner_family.csv")


def build_table(tf_days: float = TF_DAYS):
    """Returns a list of row dicts, one per even n=4..16, in table order."""
    results = run_sweep_sets(
        "quartet_rh_preference", "d", diagnostics_override=("efficiency_var",), tf_days_override=tf_days)
    by_n = {r["n"]: r for r in results}

    rows = []
    for n in (4, 6, 8, 10, 12, 14, 16):
        r = by_n[n]
        rows.append({
            "n": n,
            "delta": float(r["delta"]),
            "omega": float(r["omega"]),
            "period_days": float(r["period_days"]),
            "coeff": float(r["coeff"]),
            "efficiency_pct": float(r["isolated_triad_efficiency (%)"]),
            "eff_var_pct": None if n == 4 else float(r["efficiency_var (%)"]),
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

    print(f"{'n':>3} {'delta':>10} {'period(d)':>10} {'coeff':>10} {'E(%)':>10} {'dEff(%)':>10}")
    for r in rows:
        dev = f"{r['eff_var_pct']:.2f}" if r["eff_var_pct"] is not None else "--"
        print(f"{r['n']:>3} {r['delta']:>10.4f} {r['period_days']:>10.3f} "
              f"{r['coeff']:>10.4f} {r['efficiency_pct']:>10.4f} {dev:>10}")


if __name__ == "__main__":
    main()
