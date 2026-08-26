"""Direct TRIAD energy-conservation test -- the independent reference
implementation WaveSet is cross-checked against (rsw_sphere/utilities/check_wave_set_physics.py's
C1), tested here on its own rather than only indirectly."""
import numpy as np
import pytest

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.dynamics.dynamic_triads import TRIAD, Energy_0
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet

pytestmark = pytest.mark.slow

H_E = 10000.0


def test_triad_energy_conservation():
    """TRIAD.__init__'s calling convention: mode_c is the sum ("pump")
    mode, mode_a/mode_b its two members -- Energy_0(triad, A) mirrors
    WaveSet.energy()'s (E2, E3) split."""
    gamma = gamma_from_he(H_E, g=G)[1]
    t = TRIAD(gamma, 1, 2, 3, 3, 4, 3, 4, 5, 3, N=10, deg=300)

    ws = WaveSet(gamma, [(4, 5, 3), (1, 2, 3), (3, 4, 3)], [(0, 1, 2)], N=10, deg=300)
    A0_sum, A0_p, A0_q = ws.amplitudes_from_velocities([30.0, 20.0, 25.0], H_E, g=G)
    A0 = np.array([A0_p, A0_q, A0_sum], dtype=complex)

    t_f = 5 * 4 * np.pi
    Y, T = RK44(t, 0, t_f, 0.01, A0)
    E2, E3 = Energy_0(t, Y.T)
    E_total = np.real(E2 + E3)

    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    assert drift < 1e-8


def test_triad_matches_waveset_on_same_topology():
    """Same triad, same IC: TRIAD and WaveSet's single-triad case must
    produce numerically identical trajectories (mode order permuted:
    WaveSet is sum-first, TRIAD is sum-last)."""
    gamma = gamma_from_he(H_E, g=G)[1]
    t = TRIAD(gamma, 1, 2, 3, 3, 4, 3, 4, 5, 3, N=10, deg=300)
    ws = WaveSet(gamma, [(4, 5, 3), (1, 2, 3), (3, 4, 3)], [(0, 1, 2)], N=10, deg=300)

    A0_sum, A0_p, A0_q = ws.amplitudes_from_velocities([30.0, 20.0, 25.0], H_E, g=G)
    A0_triad = np.array([A0_p, A0_q, A0_sum], dtype=complex)
    A0_ws = np.array([A0_sum, A0_p, A0_q], dtype=complex)

    t_f = 5 * 4 * np.pi
    Y_t, T_t = RK44(t, 0, t_f, 0.01, A0_triad)
    Y_ws, T_ws = RK44(ws, 0, t_f, 0.01, A0_ws)

    assert np.allclose(Y_t[:, 0], Y_ws[:, 1])  # member p
    assert np.allclose(Y_t[:, 1], Y_ws[:, 2])  # member q
    assert np.allclose(Y_t[:, 2], Y_ws[:, 0])  # sum
