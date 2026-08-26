"""Shared config for the four root drivers (run_linear_modes.py,
run_dynamics.py, run_sweep.py, run_sweep_sets.py).

Run as a quick self-check:

    python -m rsw_sphere.dynamics.run_config
"""
import os
import sys
from dataclasses import dataclass, field

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rsw_sphere.dynamics.wave_set_specs import WaveSetSpec, DEFAULT_WAVESETS_PATH, load_wave_set_specs


@dataclass(frozen=True)
class SweepAxis:
    """One swept mode: role key + velocity range (m/s)."""
    mode: str
    min: float
    max: float


@dataclass(frozen=True)
class SweepConfig:
    """run_sweep.py's own config block. Dimensionality = len(axes) (1 or 2)."""
    axes: tuple
    n_grid: int = 10
    diagnostics: tuple = ()
    save_point_figures: bool = True


@dataclass(frozen=True)
class RunConfig:
    """Config shared by all four drivers. Only run_sweep.py reads `sweep`."""
    wave_set_spec: WaveSetSpec
    tf_days: float
    h: float
    output_root: str = "outputs"
    plot: bool = True
    parallel: bool = True
    max_workers: int = None
    sweep: SweepConfig = None

    @classmethod
    def from_yaml(cls, path: str) -> "RunConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)

        if "wave_set" in cfg:
            specs_path = cfg.get("specs_path", DEFAULT_WAVESETS_PATH)
            specs = load_wave_set_specs(specs_path)
            key = cfg["wave_set"]
            if key not in specs:
                raise ValueError(f"{key!r} not found in {specs_path!r} (available: {list(specs)})")
            spec = specs[key]
        elif "modes" in cfg:
            spec = _load_inline_spec(cfg)
        else:
            raise ValueError(f"{path!r}: needs either 'wave_set' (registry key) or 'modes' (inline spec)")

        return cls._from_cfg(cfg, spec)

    @classmethod
    def from_registry_entry(cls, key: str, specs_path: str = DEFAULT_WAVESETS_PATH) -> "RunConfig":
        """Build straight from one wave_sets_default.yaml entry's own
        tf_days/h/sweep (if it has one) -- no separate wrapper config file
        needed, e.g. for run_sweep.py --wave-set KEY. Falls back to the
        entry's settings block for tf_days/h if not overridden at the
        entry's own top level (same fallback from_yaml uses)."""
        with open(specs_path) as f:
            raw = yaml.safe_load(f)
        if key not in raw:
            raise ValueError(f"{key!r} not found in {specs_path!r} (available: {list(raw)})")
        spec = load_wave_set_specs(specs_path)[key]
        return cls._from_cfg(raw[key], spec)

    @classmethod
    def _from_cfg(cls, cfg: dict, spec: WaveSetSpec) -> "RunConfig":
        return cls.from_wave_set(
            spec,
            tf_days=cfg.get("tf_days", spec.settings.get("tf_days", 10)),
            h=cfg.get("h", spec.settings.get("h", 0.01)),
            output_root=cfg.get("output_root", "outputs"),
            plot=cfg.get("plot", True),
            parallel=cfg.get("parallel", True),
            max_workers=cfg.get("max_workers"),
            sweep=_sweep_from_dict(cfg.get("sweep"), spec),
        )

    @classmethod
    def from_wave_set(cls, spec: WaveSetSpec, tf_days: float = None, h: float = None, **overrides) -> "RunConfig":
        return cls(
            wave_set_spec=spec,
            tf_days=tf_days if tf_days is not None else spec.settings.get("tf_days", 10),
            h=h if h is not None else spec.settings.get("h", 0.01),
            output_root=overrides.get("output_root", "outputs"),
            plot=overrides.get("plot", True),
            parallel=overrides.get("parallel", True),
            max_workers=overrides.get("max_workers"),
            sweep=overrides.get("sweep"),
        )


def _sweep_from_dict(raw, spec: WaveSetSpec):
    if not raw:
        return None

    from rsw_sphere.utilities.efficiency import default_velocity_range

    if "axes" in raw:
        axes = tuple(SweepAxis(mode=a["mode"], min=float(a["min"]), max=float(a["max"])) for a in raw["axes"])
    else:
        _, private = spec.shared_and_private_modes()
        if len(private) not in (1, 2):
            raise ValueError(
                f"{spec.key!r} has {len(private)} private mode(s) -- "
                "auto-derived sweep axes need 1 or 2; pass 'axes' explicitly otherwise.")
        axes = tuple(
            SweepAxis(mode=spec.mode_keys[i], min=default_velocity_range(spec.modes[i][2])[0],
                      max=default_velocity_range(spec.modes[i][2])[1])
            for i in private
        )

    return SweepConfig(
        axes=axes,
        n_grid=int(raw.get("n_grid", 10)),
        diagnostics=tuple(raw.get("diagnostics", ())),
        save_point_figures=raw.get("save_point_figures", True),
    )


def _load_inline_spec(cfg) -> WaveSetSpec:
    """One inline wave-set entry (no registry key) -- reuses
    load_wave_set_specs' own YAML-dict parser via a single-item wrap.
    """
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump({"_inline": cfg}, f)
        tmp_path = f.name
    try:
        return load_wave_set_specs(tmp_path)["_inline"]
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    config = RunConfig.from_wave_set(spec)
    assert config.tf_days == spec.settings["tf_days"]
    assert config.h == spec.settings["h"]
    assert config.sweep is None
    print(f"run_config self-check OK: {config.wave_set_spec.key!r}, tf_days={config.tf_days}, h={config.h}")
