"""edge_completions/pivot_completions: selection-rule + symmetry-parity
filtering, and (opt-in) coupling-coefficient computation."""
import numpy as np

from rsw_sphere.utilities.mode_search import edge_completions, pivot_completions

A_MODE, B_MODE = (4, 5, 3), (3, 4, 3)  # quartet_gravity_kelvin's own edge


def test_member_role_matches_quartet_gravity_kelvin():
    # d=EG(1,1) is quartet_gravity_kelvin's own registered member-role
    # completion of this edge -- locks each mode's own coupling value
    # against the already-validated 2026-08-26 session result. A_MODE
    # (m=4) is the pump here (bigger than B_MODE's m=3).
    cands = edge_completions(A_MODE, B_MODE, max_n=1, alphas=(1,), compute_coupling=True)
    assert len(cands) == 1
    assert cands[0]["label"] == "EG(1,1)"
    assert cands[0]["role"] == "member"
    assert cands[0]["pump"] == "a"
    assert np.isclose(cands[0]["coup_a"], 0.5580101250865662)
    assert np.isclose(cands[0]["coup_b"], 0.5911164970447396)
    assert cands[0]["coup_c"] is not None


def test_sum_role_matches_quartet_gravity_79():
    cands = edge_completions(A_MODE, B_MODE, max_n=9, alphas=(1,))
    sum_labels = {c["label"] for c in cands if c["role"] == "sum"}
    assert "EG(7,9)" in sum_labels  # quartet_gravity_79's own registered d


def test_sum_role_candidate_is_the_pump():
    cands = edge_completions(A_MODE, B_MODE, max_n=9, alphas=(1,), compute_coupling=True)
    eg79 = next(c for c in cands if c["label"] == "EG(7,9)")
    assert eg79["pump"] == "c"
    assert all(v is not None for v in (eg79["coup_a"], eg79["coup_b"], eg79["coup_c"]))


def test_pivot_completions_coupling_matches_edge_style_values():
    # RH(4,5) pivot, pivot_is_sum with p=RH(1,1)/q=RH(3,3): coup_p/coup_pivot
    # here are the same physical coefficients as edge_completions' old
    # coup_p/coup_s convention for the equivalent (p, q, sum=pivot) triad.
    pairs = pivot_completions((4, 5, 3), max_n=4, alphas=(3,), compute_coupling=True)
    pair = next(p for p in pairs
                if p["role"] == "pivot_is_sum" and p["p"]["label"] == "RH(1,1)"
                and p["q"]["label"] == "RH(3,3)")
    assert pair["pump"] == "pivot"
    assert np.isclose(pair["coup_p"], 0.20750824435672854)
    assert np.isclose(pair["coup_pivot"], 0.14201117224227347)
    assert pair["coup_q"] is not None


def test_symmetry_filter_excludes_even_n():
    cands = edge_completions(A_MODE, B_MODE, max_n=9, alphas=(1,))
    even_n_labels = {c["label"] for c in cands if c["n"] % 2 == 0}
    assert even_n_labels == set()


def test_edge_completions_excludes_the_fixed_modes_themselves():
    cands = edge_completions(A_MODE, B_MODE, max_n=10, alphas=(3,))
    labels = {c["label"] for c in cands}
    assert "RH(4,5)" not in labels
    assert "RH(3,4)" not in labels


def test_pivot_completions_no_self_pairs_or_swapped_duplicates():
    pairs = pivot_completions((4, 5, 3), max_n=4, alphas=(3,))
    for pair in pairs:
        assert pair["p"] != pair["q"]
    seen = set()
    for pair in pairs:
        key = frozenset([tuple(pair["p"].values()), tuple(pair["q"].values())])
        assert key not in seen, f"duplicate pair {pair}"
        seen.add(key)


def test_pivot_completions_excludes_the_pivot_itself():
    pivot = (4, 5, 3)
    pairs = pivot_completions(pivot, max_n=6, alphas=(3,))
    for pair in pairs:
        assert (pair["p"]["m"], pair["p"]["n"], pair["p"]["alpha"]) != pivot
        assert (pair["q"]["m"], pair["q"]["n"], pair["q"]["alpha"]) != pivot


def test_print_candidates_writes_csv(tmp_path):
    from rsw_sphere.utilities.mode_search import print_candidates
    cands = edge_completions(A_MODE, B_MODE, max_n=1, alphas=(1,))
    csv_path = tmp_path / "candidates.csv"
    print_candidates(cands, csv_path=str(csv_path))
    assert csv_path.exists()
    assert "EG(1,1)" in csv_path.read_text()
