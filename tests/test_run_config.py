"""RunConfig/SweepConfig parsing: registry key, inline wave set, and
auto-derived sweep axes."""
import yaml

from rsw_sphere.dynamics.run_config import RunConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs


def test_from_wave_set_uses_registry_settings():
    spec = load_wave_set_specs()["quartet_gravity_kelvin"]
    config = RunConfig.from_wave_set(spec)
    assert config.tf_days == spec.settings["tf_days"]
    assert config.h == spec.settings["h"]
    assert config.sweep is None


def test_has_subtriads_true_for_quartet_false_for_triad():
    specs = load_wave_set_specs()
    assert specs["quartet_gravity_kelvin"].has_subtriads() is True


def test_from_yaml_registry_key(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({"wave_set": "quartet_gravity_kelvin", "tf_days": 5, "h": 0.02}))
    config = RunConfig.from_yaml(str(p))
    assert config.wave_set_spec.key == "quartet_gravity_kelvin"
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
        "wave_set": "quartet_gravity_kelvin",
        "sweep": {"n_grid": 5, "diagnostics": ["p_measure"]},
    }))
    config = RunConfig.from_yaml(str(p))
    assert {a.mode for a in config.sweep.axes} == {"c", "d"}
    assert config.sweep.n_grid == 5
    assert config.sweep.diagnostics == ("p_measure",)


def test_sweep_axes_explicit_override(tmp_path):
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump({
        "wave_set": "quartet_gravity_kelvin",
        "sweep": {"axes": [{"mode": "c", "min": 10.0, "max": 20.0}], "diagnostics": ["p_measure"]},
    }))
    config = RunConfig.from_yaml(str(p))
    assert len(config.sweep.axes) == 1
    axis = config.sweep.axes[0]
    assert (axis.mode, axis.min, axis.max) == ("c", 10.0, 20.0)
