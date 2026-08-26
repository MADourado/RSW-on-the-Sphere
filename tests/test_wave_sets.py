"""Structural/exact invariants for ``rsw_sphere.dynamics.wave_sets.WaveSet``
-- regression guards for two real bugs this repo hit before, not accuracy
checks (hence the deliberately low Hough-truncation resolution: these
invariants hold regardless of resolution).
"""
import numpy as np
import pytest

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet

G, H_E = 9.8, 10000.0
N, DEG = 6, 300  # low N -- fast, exact identities regardless of resolution.
# deg=300 is NOT arbitrary/lowerable here: velocity_to_amplitude's own
# norm_component uses a hardcoded deg=300 quadrature independent of
# whatever deg a WaveSet was built with (examples_legacy/raphaldini2022_compare/precession_resonance_broad_search.py
# documents this same constraint) -- deg MUST stay 300 wherever
# amplitudes_from_velocities is used, or norm_component's quadrature
# shape mismatches the mode's own uvh arrays.


@pytest.mark.slow
def test_triad_energy_conservation():
    """A 3-mode/1-triad WaveSet's total energy (E2+E3) is conserved to
    ~machine precision -- an *exact* identity for any 3-wave truncation
    regardless of resonance (re-confirmed numerically after an initial
    mistake; see paper-nonlinear-interactions-SWE-sphere's own memory
    'Paper §2.2 refactor status' / project history)."""
    gamma = gamma_from_he(H_E, g=G)[1]
    modes = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]
    ws = WaveSet(gamma, modes, [(0, 1, 2)], N=N, deg=DEG)
    A0 = ws.amplitudes_from_velocities([30.0, 20.0, 25.0], H_E, g=G)

    t_f = 5 * 4 * np.pi
    Y, T = RK44(ws, 0, t_f, 0.01, A0)
    E2, E3 = ws.energy(Y)
    E_total = np.real(E2 + E3)

    drift = np.max(np.abs(E_total - E_total[0])) / np.abs(E_total[0])
    assert drift < 1e-8


def test_rejects_invalid_modes():
    """Constructing a WaveSet with n < m must raise, not silently return a
    duplicate/wrong eigenvector -- a real bug this repo hit before this
    guard was added (norm_Hough does not raise on n < m; it silently
    returns a zero-padded/degenerate eigenvector instead)."""
    gamma = gamma_from_he(H_E, g=G)[1]
    # sum mode's wavenumber constraint (m_sum = m_p + m_q = 1+3 = 4) still
    # holds -- isolating the n < m check from the (separate) wavenumber
    # check, which WaveSet.__init__ validates first.
    modes = [(1, 2, 3), (3, 4, 3), (4, 3, 3)]  # last mode: n=3 < m=4
    with pytest.raises(ValueError, match="n < m"):
        WaveSet(gamma, modes, [(2, 0, 1)], N=N, deg=DEG)
