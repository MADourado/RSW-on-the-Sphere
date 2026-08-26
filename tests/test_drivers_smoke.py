"""End-to-end smoke tests for all four root drivers, at small/fast
parameters (short tf_days, coarse h, tiny n_grid) -- checks each driver
runs correctly end-to-end, not full-resolution physics (that's
examples/check_wave_set_physics.py's job)."""
import os
import subprocess
import sys
import textwrap

import numpy as np

from rsw_sphere.dynamics.run_config import RunConfig, SweepAxis, SweepConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_run_linear_modes_smoke(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_linear_modes.py"),
         "--wave-set", "triad_kelvin_rossby_flow", "--output-root", str(tmp_path)],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "dispersion_relation.png").exists()
    assert (tmp_path / "figures" / "linear" / "RH-4-5" / "Hough_harmonic_RH-4-5.png").exists()


def test_run_linear_modes_run_all_smoke(tmp_path):
    specs_path = tmp_path / "specs.yaml"
    specs_path.write_text(textwrap.dedent("""\
        one:
          h_e: 10000
          modes:
            a: {m: 4, n: 5, alpha: 3, u: 30.0}
            b: {m: 3, n: 4, alpha: 3, u: 30.0}
            c: {m: 1, n: 2, alpha: 3, u: 30.0}
          triads:
            - {sum: a, members: [b, c], display_label: "Triad 1", triad_key: null}
          reference_triad: 0
        two:
          h_e: 10000
          modes:
            a: {m: 4, n: 5, alpha: 3, u: 30.0}
            b: {m: 1, n: 6, alpha: 3, u: 30.0}
            c: {m: 3, n: 6, alpha: 3, u: 30.0}
          triads:
            - {sum: a, members: [b, c], display_label: "Triad 1", triad_key: null}
          reference_triad: 0
        """))
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_linear_modes.py"), "--run-all",
         "--specs", str(specs_path), "--output-root", str(tmp_path), "--no-dispersion-relation"],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode == 0, result.stderr
    assert "=== one " in result.stdout
    assert "=== two " in result.stdout
    # shared mode RH(4,5) plotted once, reused by both wave sets
    assert (tmp_path / "figures" / "linear" / "RH-4-5" / "Hough_harmonic_RH-4-5.png").exists()
    assert (tmp_path / "figures" / "linear" / "RH-1-6" / "Hough_harmonic_RH-1-6.png").exists()



    
def test_run_linear_modes_requires_wave_set_or_run_all(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_linear_modes.py"), "--output-root", str(tmp_path)],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode != 0
    assert "required" in result.stderr


def test_run_dynamics_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_gravity_kelvin"]
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05,
                                      output_root=str(tmp_path), plot=True, parallel=False)
    from run_dynamics import run_dynamics
    result = run_dynamics(config)
    assert set(result) == {"full", "triad0", "triad1"}
    assert all(np.isfinite(r["drift"]) for r in result.values())
    assert all(os.path.exists(r["trajectory_path"]) for r in result.values())
    assert all(os.path.exists(r["figure_path"]) for r in result.values())
    assert os.path.exists(os.path.join(str(tmp_path), "tables", f"{spec.key}.csv"))


def test_run_sweep_1d_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_rh_preference"]
    sweep = SweepConfig(axes=(SweepAxis(mode="d", min=80.0, max=90.0),), n_grid=2,
                         diagnostics=("precession",), save_point_figures=False)
    config = RunConfig.from_wave_set(spec, tf_days=2.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    output = str(tmp_path / "sweep_1d.png")
    result = run_sweep(config, output, run_per_point=False)
    assert os.path.exists(output)
    assert len(result["u_values"]) == 2


def test_run_sweep_2d_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_gravity_kelvin"]
    sweep = SweepConfig(axes=(SweepAxis(mode="c", min=0.0, max=20.0),
                               SweepAxis(mode="d", min=0.0, max=10.0)),
                         n_grid=2, diagnostics=("p_measure",), save_point_figures=False)
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    output = str(tmp_path / "sweep_2d.png")
    result = run_sweep(config, output, run_per_point=False)
    assert os.path.exists(output)
    assert result["P"].shape == (2, 2, 2)


def test_run_sweep_sets_smoke():
    import run_sweep_sets as rss
    config = {
        "base_wave_set": "quartet_gravity_kelvin",
        "candidate_slot": "d",
        "target_mode": "b",
        "candidates": [{"m": 1, "n": 1, "alpha": 1}, {"m": 1, "n": 1, "alpha": 2}],
        "diagnostics": ["p_measure"],
        "tf_days": 1.0, "h": 0.05,
    }
    results = rss.run_sweep_sets(config)
    assert len(results) == 2
    assert all("p_measure (%)" in r and "error" not in r for r in results)
