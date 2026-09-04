"""write_csv."""
import csv

from rsw_sphere.utilities.tables import write_csv


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
