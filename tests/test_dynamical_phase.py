"""``rsw_sphere.dynamics.dynamical_phase.individual_phase`` reduces to a
straight line of slope ``-omega_j`` when coupling is negligible."""
import numpy as np
import pytest

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet

pytestmark = pytest.mark.slow
from rsw_sphere.dynamics.dynamical_phase import individual_phase

G, H_E = 9.8, 10000.0
N, DEG = 6, 300  # low N for speed; deg MUST stay 300 -- see tests/test_wave_sets.py


def test_individual_phase_matches_linear_limit():
    gamma = gamma_from_he(H_E, g=G)[1]
    modes = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]
    ws = WaveSet(gamma, modes, [(0, 1, 2)], N=N, deg=DEG)

    A0 = np.zeros(3, dtype=complex)
    A0[1] = 1e-6  # one mode, negligible amplitude -> coupling term ~1e-12, negligible
    t_f = 20 * 4 * np.pi
    Y, T = RK44(ws, 0, t_f, 0.01, A0)

    phi = individual_phase(Y, 1)
    slope = np.polyfit(T, phi, 1)[0]
    # abs=1e-5, not tighter: RK44's own fixed-step discretization error over
    # t_f, not a looseness in the identity itself (exact in the h->0 limit).
    assert slope == pytest.approx(-ws.omega[1], abs=1e-5)
