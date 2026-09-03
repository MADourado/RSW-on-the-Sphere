"""run_sweep.py's unified 1D/2D diagnostics: name normalization/'all'-
expansion/warn-on-mismatch logic (fast, no integration) and end-to-end
smoke tests (slow, real integration)."""
import os

import numpy as np
import pytest

from rsw_sphere.dynamics.run_config import RunConfig, SweepAxis, SweepConfig
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from run_sweep import _normalize_diagnostics, _ALL_DIAGNOSTICS


def test_normalize_expands_all():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    result = _normalize_diagnostics(("all",), spec)
    assert result == list(_ALL_DIAGNOSTICS)


def test_normalize_alias():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    result = _normalize_diagnostics(("energy_var",), spec)
    assert result == ["efficiency_var"]


def test_normalize_novelty_aliases():
    # 'novelty_period'/'novelty_freq' also name a DIFFERENT, now-retired
    # pairwise diagnostic (rsw_sphere.utilities.registry.ALL_2D, deleted
    # once run_sweep_sets.py/_triad_panel_row.py migrated off it) --
    # aliased here to this engine's own final/combined variant anyway,
    # on request.
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    result = _normalize_diagnostics(("novelty_period", "novelty_freq"), spec)
    assert result == ["novel_period", "novel_freq"]


def test_normalize_dedupes():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    # "energy_var" aliases to "efficiency_var" -- both requested at once dedupe to one.
    result = _normalize_diagnostics(("efficiency_var", "energy_var"), spec)
    assert result == ["efficiency_var"]


def test_normalize_warns_and_skips_scalar_diagnostics_on_plain_triad(capsys):
    spec = load_wave_set_specs()["triad_kelvin_rossby_flow"]
    assert spec.has_subtriads() is False
    result = _normalize_diagnostics(("all",), spec)
    assert set(result) == {"efficiency", "dominant_freq", "dominant_period", "low_frequency_energy",
                            "dynamical_phase", "total_energy"}
    out = capsys.readouterr().out
    for name in ("efficiency_var", "spectral_dev_var", "novel_freq", "novel_period"):
        assert f"diagnostic {name!r}" in out


def test_normalize_raises_on_unknown_name():
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    with pytest.raises(ValueError, match="unknown sweep diagnostic"):
        _normalize_diagnostics(("not_a_real_diagnostic",), spec)


def test_normalize_rejects_retired_p_measure_name():
    # "p_measure" (2026-09-03: recognized as the same quantity as
    # "efficiency_var" once both sides of the ratio share one reference
    # energy budget -- see rsw_sphere.utilities.pmeasure's own module
    # docstring) is retired, not aliased -- callers must migrate to
    # "efficiency_var" explicitly.
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    with pytest.raises(ValueError, match="unknown sweep diagnostic"):
        _normalize_diagnostics(("p_measure",), spec)


@pytest.mark.slow
def test_compute_diagnostics_report_shape(tmp_path):
    from run_dynamics import run_dynamics
    from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report

    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False)
    results = run_dynamics(config)
    report = compute_diagnostics_report(results, spec)

    assert set(report["per_mode_unit"]) == set(results)
    for unit, per_mode in report["per_mode_unit"].items():
        assert set(per_mode) == set(results[unit]["labels"])
        for m in per_mode.values():
            assert np.isfinite(m["dEK"])
            assert m["period_global"] > 0
            assert m["low_freq_power"] >= 0

    # quartet_rossby_kelvin has 2 sub-triads -> "final" diagnostics defined
    assert spec.has_subtriads()
    assert {d["mode"] for d in report["final"]} == set(results["full"]["labels"])
    for d in report["final"]:
        assert d["vs"] in ("triad_rh34_rh12", "triad_rh34_eg11")


@pytest.mark.slow
def test_pairwise_value_for_target_matches_default_triad_selection(tmp_path):
    """pairwise_value_for_target must reproduce the OLD engine's own
    _default_triad_index_for_mode selection (reference_triad if the
    target is one of its members, else the first containing triad) --
    run_sweep_sets.py relies on this exact selection to preserve
    paper_table03's own efficiency_var numbers."""
    from run_dynamics import run_dynamics
    from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report, pairwise_value_for_target
    from rsw_sphere.utilities.pmeasure import _default_triad_index_for_mode

    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False)
    results = run_dynamics(config)
    report = compute_diagnostics_report(results, spec)

    from rsw_sphere.dynamics.trajectory_cache import _mode_slug

    triads = [spec.triad_indices(i) for i in range(spec.n_triads())]
    for target_idx in range(spec.n_modes()):
        t_idx = _default_triad_index_for_mode(triads, spec.reference_triad, target_idx)
        member_p, member_q, _ = spec.sub_triad_modes(t_idx)
        expected_unit = f"triad_{_mode_slug(*member_p)}_{_mode_slug(*member_q)}"
        target_label = results["full"]["labels"][target_idx]
        expected = next(r["efficiency_var_pct"] for r in report["pairwise"]
                         if r["mode"] == target_label and r["vs"] == expected_unit)

        got = pairwise_value_for_target(report, spec, target_idx, spec.reference_triad, "efficiency_var_pct")
        if np.isnan(expected):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(expected)


@pytest.mark.slow
def test_compute_diagnostics_report_empty_for_plain_triad(tmp_path):
    from run_dynamics import run_dynamics
    from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report

    spec = load_wave_set_specs()["triad_kelvin_rossby_flow"]
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False)
    results = run_dynamics(config)
    report = compute_diagnostics_report(results, spec)

    assert list(results) == ["full"]
    assert report["pairwise"] == []
    assert report["final"] == []
    assert report["precession"] == {}
    # per_mode_unit is still computed unconditionally
    assert set(report["per_mode_unit"]["full"]) == set(results["full"]["labels"])


@pytest.mark.slow
def test_run_sweep_1d_diagnostics_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    sweep = SweepConfig(axes=(SweepAxis(mode="d", min=0.0, max=20.0),), n_grid=2,
                         diagnostics=("all",))
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    result = run_sweep(config, plot_per_point=False)

    assert set(result) == set(_ALL_DIAGNOSTICS)
    out_dir = tmp_path / "sweep" / "quartet_rossby_kelvin"
    for name, r in result.items():
        assert os.path.exists(r["path"]), name
        assert os.path.exists(r["csv_path"]), name
        assert os.path.dirname(r["path"]) == str(out_dir)
        assert os.path.basename(r["path"]).startswith(f"sweep_diag_{name}_")
        assert len(r["u_values"]) == 2


@pytest.mark.slow
def test_run_sweep_1d_diagnostics_plain_triad_smoke(tmp_path):
    spec = load_wave_set_specs()["triad_kelvin_rossby_flow"]
    sweep = SweepConfig(axes=(SweepAxis(mode="b", min=0.0, max=20.0),), n_grid=2,
                         diagnostics=("all",))
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    result = run_sweep(config, plot_per_point=False)
    assert set(result) == {"efficiency", "dominant_freq", "dominant_period", "low_frequency_energy",
                            "dynamical_phase", "total_energy"}


@pytest.mark.slow
def test_run_sweep_2d_diagnostics_smoke(tmp_path):
    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    sweep = SweepConfig(axes=(SweepAxis(mode="c", min=0.0, max=20.0), SweepAxis(mode="d", min=0.0, max=20.0)),
                         n_grid=2, diagnostics=("efficiency_var", "dynamical_phase"))
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    result = run_sweep(config, plot_per_point=False)

    assert set(result) == {"efficiency_var", "dynamical_phase"}
    out_dir = tmp_path / "sweep" / "quartet_rossby_kelvin"
    for name, r in result.items():
        assert os.path.exists(r["path"]), name
        assert os.path.exists(r["csv_path"]), name
        assert r["U1"].shape == (2, 2)
        assert r["U2"].shape == (2, 2)
    # one heatmap panel per mode for the scalar diagnostics (4 modes)
    assert len(result["efficiency_var"]["series"]) == 4
    # one heatmap panel per triad for dynamical_phase (2 constituent triads)
    assert len(result["dynamical_phase"]["series"]) == 2


@pytest.mark.slow
def test_run_sweep_2d_diagnostics_matches_1d_for_same_point(tmp_path):
    """1D and 2D dispatch through the exact same _run_sweep_point worker
    -- spot-check they agree on a shared physical configuration (same
    modes/velocities/tf/h) rather than trusting that by construction."""
    from run_dynamics import run_dynamics
    from rsw_sphere.dynamics.diagnostics_report import compute_diagnostics_report

    spec = load_wave_set_specs()["quartet_rossby_kelvin"]
    config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                      plot=False, parallel=False)
    results = run_dynamics(config)
    report = compute_diagnostics_report(results, spec)
    direct = {d["mode"]: d["efficiency_var_final_pct"] for d in report["final"]}

    sweep = SweepConfig(axes=(SweepAxis(mode="c", min=spec.velocities[spec.index("c")],
                                         max=spec.velocities[spec.index("c")]),), n_grid=1,
                         diagnostics=("efficiency_var",))
    sweep_config = RunConfig.from_wave_set(spec, tf_days=1.0, h=0.05, output_root=str(tmp_path),
                                            plot=False, parallel=False, sweep=sweep)
    from run_sweep import run_sweep
    swept = run_sweep(sweep_config, plot_per_point=False)
    for mode_label, value in direct.items():
        assert swept["efficiency_var"]["series"][mode_label][0] == pytest.approx(value)
