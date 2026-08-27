"""run_sweep_sets: candidate-mode substitution + m-inference + one-point
diagnostic screening."""
import pytest

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs

import run_sweep_sets as rss


def test_required_m_for_member_slot():
    # quartet_rossby_kelvin: triad2 = {sum: a (m=4), members: [b (m=3), d]}
    # -> m_d = m_a - m_b = 1.
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    assert rss._required_m_for_slot(spec, "d") == 1


def test_build_candidate_spec_only_changes_the_slot():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    cand = rss._build_candidate_spec(spec, "d", (1, 3, 2))
    idx = spec.index("d")
    assert cand.modes[idx] == (1, 3, 2)
    other = [i for i in range(spec.n_modes()) if i != idx]
    assert [cand.modes[i] for i in other] == [spec.modes[i] for i in other]
    assert cand.velocities == spec.velocities  # unchanged


def test_build_candidate_spec_velocity_override():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    idx = spec.index("d")
    cand = rss._build_candidate_spec(spec, "d", (1, 3, 2), velocity=30.0)
    assert cand.velocities[idx] == 30.0
    other = [i for i in range(spec.n_modes()) if i != idx]
    assert [cand.velocities[i] for i in other] == [spec.velocities[i] for i in other]


@pytest.mark.slow
def test_run_sweep_sets_end_to_end_small():
    config = {
        "base_wave_set": "quartet_rossby_kelvin",
        "candidate_slot": "d",
        "target_mode": "b",
        "candidates": [{"m": 1, "n": 1, "alpha": 1}, {"m": 1, "n": 1, "alpha": 2}],
        "diagnostics": ["p_measure"],
        "tf_days": 1.0, "h": 0.05,
    }
    results = rss.run_sweep_sets(config)
    assert len(results) == 2
    assert results[0]["mode"] == "EG(1,1)"
    assert results[1]["mode"] == "WG(1,1)"
    assert "p_measure (%)" in results[0]
    assert "error" not in results[0]
