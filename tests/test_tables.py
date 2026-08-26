"""dynamics_summary_rows / write_csv."""
import csv

from rsw_sphere.utilities.tables import dynamics_summary_rows, write_csv


class _FakeSpec:
    key = "fake_wave_set"


def test_dynamics_summary_rows_one_row_per_mode_per_unit():
    result = {
        "full": {"title": "Quartet", "labels": ["RH(4,5)", "RH(3,4)"],
                  "dEK": [0.01, 0.02], "drift": 1e-4},
        "triad0": {"title": "Triad 1", "labels": ["RH(4,5)", "RH(3,4)", "RH(1,2)"],
                    "dEK": [0.01, 0.02, 0.03], "drift": 1e-13},
    }
    rows = dynamics_summary_rows(result, _FakeSpec())
    assert len(rows) == 2 + 3
    assert rows[0] == {"wave_set": "fake_wave_set", "unit": "full", "title": "Quartet",
                        "mode": "RH(4,5)", "dEK": 0.01, "drift": 1e-4}


def test_write_csv_round_trip(tmp_path):
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    path = tmp_path / "sub" / "out.csv"
    write_csv(rows, str(path))
    with open(path) as f:
        read_back = list(csv.DictReader(f))
    assert read_back == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_write_csv_empty_rows_no_error(tmp_path):
    write_csv([], str(tmp_path / "empty.csv"))
    assert not (tmp_path / "empty.csv").exists()
