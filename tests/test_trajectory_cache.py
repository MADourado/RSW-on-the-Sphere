"""run_and_cache/ic_label/topology_folder: cache hit/miss correctness and
readable-filename convention. Mirrors trajectory_cache.py's own __main__
self-check, as a pytest."""
import numpy as np
import pytest

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.trajectory_cache import ic_label, topology_folder, run_and_cache

H_E = 10000.0


def test_topology_folder_by_mode_count():
    assert topology_folder(3) == "triads"
    assert topology_folder(4) == "quartets"
    assert topology_folder(5) == "quintets"
    assert topology_folder(6) == "n6modes"


def test_ic_label_sorted_by_mode_not_registration_order():
    modes_a = [(4, 5, 3), (1, 2, 3)]
    modes_b = [(1, 2, 3), (4, 5, 3)]  # same modes, listed in reverse
    assert ic_label(modes_a, [30.0, 10.0]) == ic_label(modes_b, [10.0, 30.0])


@pytest.mark.slow
def test_run_and_cache_hit_returns_identical_trajectory(tmp_path):
    gamma = gamma_from_he(H_E, g=G)[1]
    modes = [(4, 5, 3), (3, 4, 3), (1, 2, 3)]
    velocities = [30.0, 30.0, 30.0]
    ws = WaveSet(gamma, modes, [(0, 1, 2)], N=10, deg=300)
    A0 = ws.amplitudes_from_velocities(velocities, H_E, g=G)
    t_f = 5 * 4 * np.pi

    Y1, T1, path1 = run_and_cache(ws, A0, t_f, 0.01, velocities=velocities, output_root=str(tmp_path))
    Y2, T2, path2 = run_and_cache(ws, A0, t_f, 0.01, velocities=velocities, output_root=str(tmp_path))

    assert path1 == path2
    assert "triads" in path1
    assert np.array_equal(T1, T2)
    assert np.allclose(Y1, Y2)
