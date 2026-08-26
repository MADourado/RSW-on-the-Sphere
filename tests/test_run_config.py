"""RunConfig/SweepConfig parsing: registry key, inline wave set, and
auto-derived sweep axes."""
import pytest
import yaml

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs


_TINY_REGISTRY = {
    "tiny_quartet": {
        "h_e": 10000,
        "modes": {
            "a": {"m": 4, "n": 5, "alpha": 3, "u": 30.0},
            "b": {"m": 3, "n": 4, "alpha": 3, "u": 30.0},
            "c": {"m": 1, "n": 2, "alpha": 3, "u": 30.0},
            "d": {"m": 1, "n": 1, "alpha": 1, "u": 0.0},
        },
        "triads": [
            {"sum": "a", "members": ["b", "c"]},
            {"sum": "a", "members": ["b", "d"]},
        ],
        "reference_triad": 0,
        "settings": {"tf_days": 20, "h": 0.01, "n_grid": 10},
        # sweep-specific tf_days deliberately differs from settings.tf_days,
        # matching quartet_rh_preference's own registry entry (run_dynamics.py
        # needs a short horizon, its own precession sweep needs a long one).
        "tf_days": 150.0,
        "sweep": {"axes": [{"mode": "d", "min": 0.0, "max": 50.0}], "diagnostics": ["precession"]},
        "target_mode": "c",
        "output": "outputs/figures/wave_sets/tiny_quartet_sweep.png",
    }
}


def test_from_wave_set_uses_registry_settings():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    config = RunConfig.from_wave_set(spec)
    assert config.tf_days == spec.settings["tf_days"]
    assert config.h == spec.settings["h"]
    assert config.sweep is None


def test_has_subtriads_true_for_quartet_false_for_triad():
    specs = load_wave_set_specs()
    assert specs["quartet_rossby_kelvin"].has_subtriads() is True


def test_from_yaml_registry_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"wave_set": "quartet_rossby_kelvin", "tf_days": 5, "h": 0.02}))
    config = RunConfig.from_yaml(str(p))
    assert config.wave_set_spec.key == "quartet_rossby_kelvin"
    assert config.tf_days == 5
    assert config.h == 0.02


def test_from_yaml_inline_wave_set_is_single_triad(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "modes": {
            "a": {"m": 4, "n": 5, "alpha": 3, "u": 30.0},
            "b": {"m": 3, "n": 4, "alpha": 3, "u": 30.0},
            "c": {"m": 1, "n": 2, "alpha": 3, "u": 0.0},
        },
        "triads": [{"sum": "a", "members": ["b", "c"]}],
        "tf_days": 5, "h": 0.02,
    }))
    config = RunConfig.from_yaml(str(p))
    assert config.wave_set_spec.n_modes() == 3
    assert config.wave_set_spec.has_subtriads() is False


def test_sweep_axes_auto_derived_from_private_modes(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "wave_set": "quartet_rossby_kelvin",
        "sweep": {"n_grid": 5, "diagnostics": ["p_measure"]},
    }))
    config = RunConfig.from_yaml(str(p))
    assert {a.mode for a in config.sweep.axes} == {"c", "d"}
    assert config.sweep.n_grid == 5
    assert config.sweep.diagnostics == ("p_measure",)


def test_sweep_axes_explicit_override(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "wave_set": "quartet_rossby_kelvin",
        "sweep": {"axes": [{"mode": "c", "min": 10.0, "max": 20.0}], "diagnostics": ["p_measure"]},
    }))
    config = RunConfig.from_yaml(str(p))
    assert len(config.sweep.axes) == 1
    axis = config.sweep.axes[0]
    assert (axis.mode, axis.min, axis.max) == ("c", 10.0, 20.0)


def test_from_registry_entry_reads_its_own_sweep_and_tf_days(tmp_path):
    p = tmp_path / "specs.yaml"
    p.write_text(yaml.safe_dump(_TINY_REGISTRY))
    config = RunConfig.from_registry_entry("tiny_quartet", str(p))
    assert config.tf_days == 150.0  # entry's own tf_days, not settings.tf_days=20
    assert len(config.sweep.axes) == 1
    assert config.sweep.axes[0].mode == "d"
    assert config.sweep.diagnostics == ("precession",)


def test_from_registry_entry_does_not_affect_from_wave_set(tmp_path):
    """run_dynamics.py's own RunConfig.from_wave_set(spec) must keep using
    settings.tf_days, unaffected by a sibling top-level tf_days: added for
    run_sweep.py's own from_registry_entry."""
    p = tmp_path / "specs.yaml"
    p.write_text(yaml.safe_dump(_TINY_REGISTRY))
    spec = load_wave_set_specs(str(p))["tiny_quartet"]
    config = RunConfig.from_wave_set(spec)
    assert config.tf_days == 20  # settings.tf_days, not the entry's own 150.0


def test_from_registry_entry_missing_key_raises(tmp_path):
    p = tmp_path / "specs.yaml"
    p.write_text(yaml.safe_dump(_TINY_REGISTRY))
    with pytest.raises(ValueError, match="not_a_real_key"):
        RunConfig.from_registry_entry("not_a_real_key", str(p))
