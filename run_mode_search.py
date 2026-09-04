"""CLI for rsw_sphere.utilities.mode_search: given a fixed edge (2 modes)
or pivot (1 mode), list candidate modes that complete a valid triad with
them -- for scouting a new wave_sets_default.yaml entry or a
run_sweep_sets.py `candidates:` block without hand-deriving the
selection rule.

Cheap by default (wavenumber + symmetry-parity rules only); --coupling
also computes actual TRIAD coefficients (slower -- builds Hough
eigenvectors per candidate).

Background: a "mode" here is a normal mode of the linearized rotating
shallow-water equations on the sphere (a Hough harmonic), identified by
3 integers `m,n,alpha`: 
`m` is the zonal wavenumber (latitude),
`n` is the total wavenumber (>= m; meridional/north-south structure)
`alpha` selects the wave family: 
1 = eastward inertia-gravity (EG), 
2 = westward inertia-gravity (WG), 
3 = Rossby-Haurwitz (RH). 

Three modes A, B, C form a resonant "triad"  only if their zonal
wavenumbers satisfy m_C = m_A + m_B for some choice of which one is
called "the sum" -- this script searches for modes that satisfy that
condition (plus a second, cheap necessary condition on meridional
symmetry) given some of the triad already fixed.

Options:

    --edge MODE_P MODE_Q   2 modes you already have, each written as
                            "m,n,alpha" (e.g. "4,5,3" = m=4, n=5,
                            alpha=3 = RH(4,5)). The script finds every
                            THIRD mode that could join these two into a
                            valid triad -- as the "sum" mode, or as the
                            other "member". This is also how you build a
                            quartet/quintet: run it once per constituent
                            triad on the same fixed edge, and pick a
                            different candidate each time.
    --pivot MODE            1 mode you already have ("m,n,alpha"). The
                            script finds every PAIR of new modes (P, Q)
                            that could form an independent triad through
                            it -- for an "hourglass" topology (two
                            triads sharing only this one mode, not a
                            shared edge). Exactly one of --edge/--pivot
                            must be given.
    --max-n N               Upper bound on the candidates' own `n`
                            (their meridional wavenumber); candidates are
                            searched over n from the minimum physically
                            required value up to N. Larger N = a wider,
                            slower search. Default: 15.
    --alphas LIST            Comma-separated wave families to search,
                            using the same 1=EG/2=WG/3=RH codes as
                            above, e.g. "1,2" to only look for gravity-
                            wave candidates. Default: "1,2,3" (all three).
    --coupling               Also compute each candidate's actual
                            coupling strength (not just whether it's
                            allowed to be nonzero) -- slower, since it
                            requires building the candidate's full Hough
                            eigenvector.
    --csv PATH               Also write the results table to this CSV path.
    --h-e                    Equivalent depth (m) used only when
                            --coupling is given. Default: 10000.

Run:

    # RH(4,5) + RH(3,4), searching gravity-wave coupling candidates up to n=9
    python run_mode_search.py --edge 4,5,3 3,4,3 --max-n 9 --alphas 1,2

    # same, but also rank by actual coupling strength, and save to CSV
    python run_mode_search.py --edge 4,5,3 3,4,3 --max-n 9 --alphas 1 \\
        --coupling --csv outputs/_scratch/candidates.csv

    # hourglass-quintet: pairs completing an independent triad through
    # the single mode RH(4,5)
    python run_mode_search.py --pivot 4,5,3 --max-n 6 --alphas 3
"""
import argparse

from rsw_sphere.utilities.mode_search import edge_completions, pivot_completions


def _parse_mode(s):
    parts = s.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"expected 'm,n,alpha', got {s!r}")
    m, n, alpha = (int(p) for p in parts)
    return (m, n, alpha)


def _parse_alphas(s):
    return tuple(int(a) for a in s.split(","))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--edge", nargs=2, type=_parse_mode, metavar=("MODE_P", "MODE_Q"),
                         help="2 fixed modes 'm,n,alpha', e.g. 4,5,3 3,4,3 -- "
                              "find candidate third modes (triads/quartets/star-quintets).")
    parser.add_argument("--pivot", type=_parse_mode, metavar="MODE",
                         help="1 fixed mode 'm,n,alpha' -- find candidate (P, Q) pairs "
                              "completing an independent triad through it (hourglass-quintets).")
    parser.add_argument("--max-n", type=int, default=15)
    parser.add_argument("--alphas", type=_parse_alphas, default=(1, 2, 3),
                         help="comma-separated wave families to scan, e.g. 1,2. Default: 1,2,3")
    parser.add_argument("--coupling", action="store_true",
                         help="also compute actual TRIAD coupling coefficients (slower).")
    parser.add_argument("--h-e", type=float, default=10000.0)
    parser.add_argument("--csv", default=None, help="write results to this CSV path.")
    args = parser.parse_args()

    if bool(args.edge) == bool(args.pivot):
        parser.error("exactly one of --edge or --pivot is required")

    if args.edge:
        candidates = edge_completions(*args.edge, max_n=args.max_n, alphas=args.alphas,
                                       compute_coupling=args.coupling, h_e=args.h_e,
                                       table=True, csv_path=args.csv)
    else:
        candidates = pivot_completions(args.pivot, max_n=args.max_n, alphas=args.alphas,
                                        compute_coupling=args.coupling, h_e=args.h_e,
                                        table=True, csv_path=args.csv)

    print(f"\n{len(candidates)} candidate(s)")


if __name__ == "__main__":
    main()
