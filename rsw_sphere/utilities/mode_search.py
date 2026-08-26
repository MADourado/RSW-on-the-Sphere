"""Mode-interaction discovery: given 1-2 fixed modes, enumerate candidate
modes that complete a valid triad/quartet/quintet with them -- so a new
wave_sets_default.yaml entry doesn't require hand-deriving the selection
rule, and a result list can be pasted straight into a run_sweep_sets.py
`candidates:` block.

Two primitives cover every topology in this repository:

- ``edge_completions(mode_p, mode_q, ...)`` -- 2 modes fixed (an "edge"),
  enumerate candidate third modes. Covers triads directly; a quartet or
  star-quintet (rsw_sphere.dynamics.wave_sets docstring: modes coupled
  through triads sharing one edge) is built by calling this once per
  constituent triad on the same edge and picking a different candidate
  each time.
- ``pivot_completions(mode, ...)`` -- 1 mode fixed (a "pivot"), enumerate
  candidate (P, Q) PAIRS completing an independent triad through it.
  Covers an "hourglass" quintet (two triads sharing a single mode, not
  an edge) -- not currently used by any registered wave set, but a valid
  WaveSet topology (verified: WaveSet accepts a mode shared by exactly
  one index across two otherwise-disjoint triads).

Cheap by default: only the wavenumber selection rule (m_sum = m_p + m_q)
and meridional-symmetry parity (rsw_sphere.hough_harmonics...symetry) --
both O(1), no Hough/eigenvalue computation. Verified empirically
(2026-08-26, quartet_gravity_kelvin's own edge) that an odd count of
equatorially-antisymmetric modes among the three always gives exactly
zero coupling -- a real second selection rule, not just TRIAD's `fat`
sign convention (see wave_sets.py's own docstring on that distinction).
Pass ``compute_coupling=True`` to additionally compute actual TRIAD
coupling coefficients (expensive: builds Hough eigenvectors per
candidate) and rank/filter survivors by coupling strength, not just
whether it's structurally allowed to be nonzero.
"""
import csv
import os
import warnings

from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import symetry
from rsw_sphere.plotting.labels import _mode_label

_FAMILY_NAME = {1: "EG", 2: "WG", 3: "RH"}
_ALL_ALPHAS = (1, 2, 3)


def _candidate_dict(m, n, alpha, role, required_m):
    return {"m": m, "n": n, "alpha": alpha, "label": _mode_label(m, n, alpha),
            "role": role, "required_m": required_m}


def _scan_m(m_target, max_n, alphas, role, exclude=()):
    """Every (m_target, n, alpha) with n in [m_target, max_n], alpha in
    alphas, not in exclude -- m_target <= 0 yields nothing (m=0 excluded:
    no registered mode uses it, and every role formula here would need a
    separate zonally-symmetric-mode convention this repo doesn't have)."""
    if m_target <= 0:
        return []
    out = []
    for alpha in alphas:
        for n in range(m_target, max_n + 1):
            cand = (m_target, n, alpha)
            if cand in exclude:
                continue
            out.append(_candidate_dict(m_target, n, alpha, role, m_target))
    return out


def _symmetry_allows(*modes):
    """True if the count of equatorially-antisymmetric modes among
    ``modes`` is even -- the necessary condition for nonzero coupling
    (see module docstring)."""
    n_antisym = sum(1 for m in modes if not symetry(*m))
    return n_antisym % 2 == 0


def _with_coupling(candidates, mode_a, mode_b, gamma, N, deg):
    """Attach coup_a/coup_b/coup_c (each mode's OWN coefficient in its own
    amplitude equation, abs value -- see module docstring) and pump
    ("a"/"b"/"c": which of the three is the sum/"pump" mode) to each
    candidate sharing edge (mode_a, mode_b) -- mode_a/mode_b are the
    edge_completions caller's own mode_p/mode_q, candidate is "c"."""
    from rsw_sphere.dynamics.dynamic_triads import TRIAD

    out = []
    for cand in candidates:
        c = (cand["m"], cand["n"], cand["alpha"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if cand["role"] == "sum":
                    # slot1=a, slot2=b, slot3=c (pump).
                    t = TRIAD(gamma, *mode_a, *mode_b, *c, N=N, deg=deg)
                    coup_a, coup_b, coup_c = abs(t.coef_ABC), abs(t.coef_BAC), abs(t.coef_CAB)
                    pump = "c"
                elif mode_a[0] > mode_b[0]:
                    # a is the pump: slot1=b, slot2=c, slot3=a.
                    t = TRIAD(gamma, *mode_b, *c, *mode_a, N=N, deg=deg)
                    coup_b, coup_c, coup_a = abs(t.coef_ABC), abs(t.coef_BAC), abs(t.coef_CAB)
                    pump = "a"
                else:
                    # b is the pump: slot1=a, slot2=c, slot3=b.
                    t = TRIAD(gamma, *mode_a, *c, *mode_b, N=N, deg=deg)
                    coup_a, coup_c, coup_b = abs(t.coef_ABC), abs(t.coef_BAC), abs(t.coef_CAB)
                    pump = "b"
        except Exception as exc:
            out.append(dict(cand, coup_a=None, coup_b=None, coup_c=None, pump=None, error=str(exc)))
            continue
        out.append(dict(cand, coup_a=coup_a, coup_b=coup_b, coup_c=coup_c, pump=pump))
    return out


def edge_completions(mode_p, mode_q, max_n=15, alphas=_ALL_ALPHAS,
                      compute_coupling=False, gamma=None, h_e=10000.0, N=10, deg=300,
                      table=False, csv_path=None):
    """Candidate third modes completing a triad with the fixed edge
    (mode_p, mode_q) -- as the sum mode (m = m_p + m_q), or as the other
    member (m = |m_p - m_q|, whichever of p/q would then be the sum).

    Parameters
    ----------
    mode_p, mode_q : (m, n, alpha) int triples
    max_n : int -- candidate n in [required_m, max_n].
    alphas : iterable of int -- wave families to scan (default: all 3).
    compute_coupling : bool -- also compute each mode's own actual TRIAD
        coefficient (coup_a, coup_b, coup_c) and which one is the sum/
        "pump" mode (pump); expensive, see module docstring.
    gamma, h_e, N, deg : only used if compute_coupling -- gamma takes
        precedence if given, else derived from h_e.
    table, csv_path : bool, str or None -- print a terminal table and/or
        write a CSV (rsw_sphere.utilities.tables.write_csv) of the result.

    Returns
    -------
    list of dict
        {m, n, alpha, label, role ("sum"/"member"), required_m
        [, coup_a, coup_b, coup_c, pump]} -- coup_a/coup_b are mode_p's/
        mode_q's own coefficients, coup_c the candidate's own; pump is
        "a"/"b"/"c", whichever of the three is the sum mode for this
        candidate. Drop role/required_m/coup_*/pump to use a result
        directly as a run_sweep_sets.py `candidates:` entry.
    """
    m_sum_role = mode_p[0] + mode_q[0]
    m_member_role = abs(mode_p[0] - mode_q[0])

    candidates = (
        _scan_m(m_sum_role, max_n, alphas, "sum", exclude=(mode_p, mode_q)) +
        _scan_m(m_member_role, max_n, alphas, "member", exclude=(mode_p, mode_q))
    )
    candidates = [c for c in candidates
                  if _symmetry_allows(mode_p, mode_q, (c["m"], c["n"], c["alpha"]))]

    if compute_coupling:
        if gamma is None:
            from rsw_sphere.physics import gamma_from_he, G
            gamma = gamma_from_he(h_e, g=G)[1]
        candidates = _with_coupling(candidates, mode_p, mode_q, gamma, N, deg)

    if table or csv_path:
        print_candidates(candidates, csv_path=csv_path,
                          base_labels={"a": _mode_label(*mode_p), "b": _mode_label(*mode_q)})
    return candidates


def pivot_completions(mode, max_n=15, alphas=_ALL_ALPHAS,
                       compute_coupling=False, gamma=None, h_e=10000.0, N=10, deg=300,
                       table=False, csv_path=None):
    """Candidate (P, Q) PAIRS completing an independent triad through the
    fixed pivot mode -- pivot as sum (P, Q both members, m_P + m_Q =
    m_pivot) or pivot as a member (P is the sum, m_P = m_pivot + m_Q, Q
    the other member, any m_Q in [1, max_n]).

    Much larger search space than edge_completions (two free modes, not
    one) -- keep max_n small.

    Returns
    -------
    list of dict
        {p: {m,n,alpha,label}, q: {...}, role ("pivot_is_sum"/
        "pivot_is_member") [, coup_pivot, coup_p, coup_q, pump]} -- each
        coup_* is that mode's own coefficient; pump is "pivot"/"p"/"q",
        whichever is the sum mode for this pair.
    """
    pairs = []

    # pivot is sum: every unordered (m_P, m_Q) with m_P + m_Q = m_pivot.
    m_pivot = mode[0]
    for m_p in range(1, m_pivot):
        m_q = m_pivot - m_p
        if m_q < m_p:
            continue
        for alpha_p in alphas:
            for n_p in range(m_p, max_n + 1):
                for alpha_q in alphas:
                    for n_q in range(m_q, max_n + 1):
                        p, q = (m_p, n_p, alpha_p), (m_q, n_q, alpha_q)
                        if p == q or p == mode or q == mode:
                            continue
                        if m_p == m_q and p > q:
                            continue  # (p, q) and (q, p) are the same pair when m_p == m_q
                        if _symmetry_allows(mode, p, q):
                            pairs.append({"p": _mode_dict(p), "q": _mode_dict(q),
                                          "role": "pivot_is_sum"})

    # pivot is a member: P is the sum (m_P = m_pivot + m_Q), Q any other member.
    for m_q in range(1, max_n + 1):
        m_p = m_pivot + m_q
        for alpha_q in alphas:
            for n_q in range(m_q, max_n + 1):
                for alpha_p in alphas:
                    for n_p in range(m_p, max_n + 1):
                        p, q = (m_p, n_p, alpha_p), (m_q, n_q, alpha_q)
                        if p == mode or q == mode:
                            continue
                        if _symmetry_allows(mode, p, q):
                            pairs.append({"p": _mode_dict(p), "q": _mode_dict(q),
                                          "role": "pivot_is_member"})

    if compute_coupling:
        if gamma is None:
            from rsw_sphere.physics import gamma_from_he, G
            gamma = gamma_from_he(h_e, g=G)[1]
        pairs = _pivot_with_coupling(pairs, mode, gamma, N, deg)

    if table or csv_path:
        print_candidates(pairs, csv_path=csv_path, base_labels={"pivot": _mode_label(*mode)})
    return pairs


def _mode_dict(m):
    return {"m": m[0], "n": m[1], "alpha": m[2], "label": _mode_label(*m)}


def _pivot_with_coupling(pairs, pivot, gamma, N, deg):
    """Attach coup_pivot/coup_p/coup_q (each mode's OWN coefficient, abs
    value) and pump ("pivot"/"p"/"q") to each pair."""
    from rsw_sphere.dynamics.dynamic_triads import TRIAD

    out = []
    for pair in pairs:
        p = (pair["p"]["m"], pair["p"]["n"], pair["p"]["alpha"])
        q = (pair["q"]["m"], pair["q"]["n"], pair["q"]["alpha"])
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if pair["role"] == "pivot_is_sum":
                    # slot1=p, slot2=q, slot3=pivot (pump).
                    t = TRIAD(gamma, *p, *q, *pivot, N=N, deg=deg)
                    coup_p, coup_q, coup_pivot = abs(t.coef_ABC), abs(t.coef_BAC), abs(t.coef_CAB)
                    pump = "pivot"
                else:
                    # P is the sum/pump (see pivot_completions docstring):
                    # slot1=q, slot2=pivot, slot3=p.
                    t = TRIAD(gamma, *q, *pivot, *p, N=N, deg=deg)
                    coup_q, coup_pivot, coup_p = abs(t.coef_ABC), abs(t.coef_BAC), abs(t.coef_CAB)
                    pump = "p"
        except Exception as exc:
            out.append(dict(pair, coup_pivot=None, coup_p=None, coup_q=None, pump=None, error=str(exc)))
            continue
        out.append(dict(pair, coup_pivot=coup_pivot, coup_p=coup_p, coup_q=coup_q, pump=pump))
    return out


_PIVOT_COUPLING_KEYS = ("coup_pivot", "coup_p", "coup_q", "pump", "error")


def _display_value(v):
    """4-significant-figure rendering for the terminal table -- CSV output
    (write_csv, below) always gets the untruncated value."""
    if isinstance(v, float):
        return f"{v:.4g}"
    return v


def print_candidates(candidates, csv_path=None, base_labels=None):
    """Print a terminal table (values rounded to 4 significant figures)
    and, if csv_path is given, write a full-precision CSV
    (rsw_sphere.utilities.tables.write_csv) -- either edge_completions's
    flat dicts or pivot_completions's {p, q, role} dicts.

    base_labels : dict or None -- the FIXED input mode(s)' own paper
        labels, e.g. {"a": "RH(4,5)", "b": "RH(3,4)"} or
        {"pivot": "RH(4,5)"}, printed above the table so the column
        names (which just say "a"/"b"/"pump", not which physical mode
        that is) are unambiguous.
    """
    if base_labels:
        print(", ".join(f"{k} = {v}" for k, v in base_labels.items()))
    if not candidates:
        print("(no candidates)")
        return

    rows = candidates
    if "p" in candidates[0]:  # pivot_completions: flatten p/q for display/CSV
        rows = [
            {"p": c["p"]["label"], "q": c["q"]["label"], "role": c["role"],
             **{k: c[k] for k in _PIVOT_COUPLING_KEYS if k in c}}
            for c in candidates
        ]

    columns = list(rows[0].keys())
    display_rows = [{col: _display_value(r[col]) for col in columns} for r in rows]
    widths = {col: max(len(col), max(len(str(r[col])) for r in display_rows)) for col in columns}
    header = "  ".join(col.ljust(widths[col]) for col in columns)
    print(header)
    print("  ".join("-" * widths[col] for col in columns))
    for r in display_rows:
        print("  ".join(str(r[col]).ljust(widths[col]) for col in columns))

    if csv_path:
        from rsw_sphere.utilities.tables import write_csv
        write_csv(rows, csv_path)
        print(f"wrote {os.path.abspath(csv_path)}")
