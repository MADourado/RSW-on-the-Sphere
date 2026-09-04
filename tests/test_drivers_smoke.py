"""End-to-end smoke tests for the root drivers, at small/fast
parameters (short tf_days, coarse h, tiny n_grid) -- checks each driver
runs correctly end-to-end, not full-resolution physics (that's
rsw_sphere/utilities/check_wave_set_physics.py's job). run_sweep_sets.py
is covered by tests/test_run_sweep_sets.py instead -- no separate smoke
test here, it would just re-run the same function with trivially
different parameters."""
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from rsw_sphere.dynamics.run_config import RunConfig, SweepAxis, SweepConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_run_linear_modes_smoke(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_linear_modes.py"),
         "--wave-set", "triad_kelvin_rossby_flow", "--output-root", str(tmp_path)],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "figures" / "dispersion_relation.png").exists()
    assert (tmp_path / "figures" / "linear" / "RH-4-5" / "Hough_harmonic_RH-4-5.png").exists()


@pytest.mark.slow
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


@pytest.mark.slow
def test_run_dynamics_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05,
                                      output_root=str(tmp_path), plot=True, parallel=False)
    from run_dynamics import run_dynamics
    result = run_dynamics(config)
    assert set(result) == {"full", "triad_rh34_rh12", "triad_rh34_eg11"}
    assert all(np.isfinite(r["drift"]) for r in result.values())
    assert all(os.path.exists(r["trajectory_path"]) for r in result.values())
    assert all(os.path.exists(r["figure_path"]) for r in result.values())


@pytest.mark.slow
def test_run_sweep_1d_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_rh_preference"]
    sweep = SweepConfig(axes=(SweepAxis(mode="d", min=80.0, max=90.0),), n_grid=2,
                         diagnostics=("efficiency", "dynamical_phase"))
    config = RunConfig.from_wave_set(spec, tf_days=2.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    result = run_sweep(config, plot_per_point=False)
    assert set(result) == {"efficiency", "dynamical_phase"}
    for r in result.values():
        assert os.path.exists(r["path"])
        assert os.path.exists(r["csv_path"])
        assert len(r["u_values"]) == 2


@pytest.mark.slow
def test_run_sweep_2d_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    sweep = SweepConfig(axes=(SweepAxis(mode="c", min=0.0, max=20.0),
                               SweepAxis(mode="d", min=0.0, max=10.0)),
                         n_grid=2, diagnostics=("efficiency_var",))
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    result = run_sweep(config, plot_per_point=False)
    assert set(result) == {"efficiency_var"}
    assert os.path.exists(result["efficiency_var"]["path"])
    assert os.path.exists(result["efficiency_var"]["csv_path"])
    assert result["efficiency_var"]["U1"].shape == (2, 2)
    # one heatmap panel per mode -- quartet_rossby_kelvin has 4
    assert len(result["efficiency_var"]["series"]) == 4


@pytest.mark.slow
def test_run_sweep_wave_set_cli_smoke(tmp_path):
    specs_path = tmp_path / "specs.yaml"
    specs_path.write_text(textwrap.dedent("""\
        tiny_quartet:
          h_e: 10000
          modes:
            a: {m: 4, n: 5, alpha: 3, u: 30.0}
            b: {m: 3, n: 4, alpha: 3, u: 30.0}
            c: {m: 1, n: 2, alpha: 3, u: 30.0}
            d: {m: 1, n: 1, alpha: 1, u: 0.0}
          triads:
            - {sum: a, members: [b, c], display_label: "Triad 1", triad_key: null}
            - {sum: a, members: [b, d], display_label: "Triad 2", triad_key: null}
          reference_triad: 0
          settings: {tf_days: 1, h: 0.05}
          sweep:
            axes: [{mode: c, min: 0.0, max: 20.0}, {mode: d, min: 0.0, max: 10.0}]
            n_grid: 2
            diagnostics: [efficiency_var]
          plot: {title: "tiny test"}
        """))
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_sweep.py"), "--wave-set", "tiny_quartet",
         "--specs", str(specs_path), "--output-root", str(tmp_path),
         "--no-plot-per-point"],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode == 0, result.stderr
    out_dir = tmp_path / "sweep" / "tiny_quartet"
    pngs = list(out_dir.glob("sweep_diag_efficiency_var_*.png"))
    csvs = list(out_dir.glob("sweep_diag_efficiency_var_*.csv"))
    assert len(pngs) == 1, list(out_dir.iterdir())
    assert len(csvs) == 1


def test_run_sweep_requires_wave_set(tmp_path):
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_sweep.py")],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode != 0
    assert "required" in result.stderr


def test_run_mode_search_edge_smoke():
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_mode_search.py"),
         "--edge", "4,5,3", "3,4,3", "--max-n", "1", "--alphas", "1"],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode == 0, result.stderr
    assert "EG(1,1)" in result.stdout


def test_run_mode_search_pivot_smoke():
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_mode_search.py"),
         "--pivot", "4,5,3", "--max-n", "3", "--alphas", "3"],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode == 0, result.stderr
    assert "pivot_is_sum" in result.stdout or "pivot_is_member" in result.stdout


def test_run_mode_search_requires_edge_or_pivot():
    result = subprocess.run(
        [sys.executable, os.path.join(_ROOT, "run_mode_search.py")],
        capture_output=True, text=True, cwd=_ROOT)
    assert result.returncode != 0
    assert "required" in result.stderr
