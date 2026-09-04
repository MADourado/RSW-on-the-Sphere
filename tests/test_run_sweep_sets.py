"""run_sweep_sets: candidate-mode substitution + one-point diagnostic
screening, sourced from a wave set's own registered alternative_modes."""
import pytest

from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs

import run_sweep_sets as rss


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


def test_load_alternative_modes_reads_registered_slot():
    """The slot's own registered block is returned verbatim, including a
    target_mode that deliberately differs from the slot being swept."""
    cfg = rss.load_alternative_modes("quartet_rossby_kelvin", "d")
    assert cfg["target_mode"] == "c"
    assert {"m": 1, "n": 1, "alpha": 1} in cfg["candidates"]


def test_load_alternative_modes_target_mode_can_equal_the_slot():
    cfg = rss.load_alternative_modes("quartet_rh_preference", "d")
    assert cfg["target_mode"] == "d"


def test_load_alternative_modes_missing_slot_raises():
    with pytest.raises(ValueError, match="alternative_modes"):
        rss.load_alternative_modes("quartet_rossby_kelvin", "a")


@pytest.mark.slow
def test_run_sweep_sets_end_to_end_small(tmp_path):
    results = rss.run_sweep_sets(
        "quartet_rossby_kelvin", "d", output_root=str(tmp_path),
        diagnostics_override=("efficiency_var",), tf_days_override=1.0, h_override=0.05)
    assert len(results) == 16
    by_mode = {r["mode"]: r for r in results}
    assert "EG(1,1)" in by_mode and "WG(1,1)" in by_mode
    assert "efficiency_var (%)" in by_mode["EG(1,1)"]
    assert "error" not in by_mode["EG(1,1)"]
    assert "delta" in by_mode["EG(1,1)"] and "coeff" in by_mode["EG(1,1)"]
