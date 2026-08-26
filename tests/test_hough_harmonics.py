"""Regression test for the eigenvalue problem core (Hough_harmonic) --
every downstream number in this repo depends on it, so a silent
regression here would be invisible everywhere else."""
import numpy as np

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import Hough_harmonic

_GAMMA_HE10000 = gamma_from_he(10000, g=G)[1]


def test_rh45_eigenvalue_and_field_locked():
    # RH(4,5), h_e=10000 -- reference values saved 2026-08-26.
    U, V, Z, DU, DV, DZ, eigen = Hough_harmonic(4, 5, 3, _GAMMA_HE10000, np.pi / 4, N=10)
    assert np.isclose(eigen, -0.12524039009585008)
    assert np.isclose(U, -0.3369455973302844)
    assert np.isclose(V, -0.590588198637625)
    assert np.isclose(Z, 0.19671355578307934)


def test_eigenvalue_independent_of_latitude():
    # The eigenvalue is a property of the mode, not of where it's sampled.
    eigens = [Hough_harmonic(4, 5, 3, _GAMMA_HE10000, phi, N=10)[6]
              for phi in (0.0, np.pi / 4, np.pi / 2)]
    assert np.allclose(eigens, eigens[0])
