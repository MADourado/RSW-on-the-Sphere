"""Generalized N-wave amplitude-equation system: quartets, quintets, and
(as a degenerate 1-triad case) triads, all as instances of one class.

``WaveSet`` generalizes ``TRIAD`` (``rsw_sphere.dynamics.dynamic_triads``)
to an arbitrary set of modes coupled through an arbitrary set of
constituent triads: a quartet is 4 modes / 2 triads sharing one edge, a
quintet is 5 modes / (2 or 3) triads, a triad is the degenerate case of 3
modes / 1 triad.

``TRIAD`` is deliberately left as an independent implementation -- it is
the reference ``WaveSet`` is proven against, see
``rsw_sphere/utilities/check_wave_set_physics.py``, checks C1-C4.

The permutation, precisely
---------------------------
``TRIAD(gamma, m_a,n_a,alpha_a, m_b,n_b,alpha_b, m_c,n_c,alpha_c)`` treats
mode **c** as the "sum" mode (``mismatch = -freq_c + freq_b + freq_a``,
i.e. conceptually :math:`m_c = m_a + m_b`). Because
``rsw_sphere.hough_harmonics.inner_product.inner_product`` is symmetric
under swapping its 2nd/3rd mode arguments, and ``S_abc`` is symmetric under
swapping its 1st/2nd arguments, a constituent triad with sum mode ``s`` and
member modes ``p``, ``q`` (in *either* order) reproduces
``TRIAD(gamma, mode_p, mode_q, mode_s, ...)`` exactly -- **the sum mode
goes in TRIAD's slot c (last).** ``WaveSet`` stores this permutation once,
in ``_triad_local_indices``, and nowhere else needs to know it.

The symmetry factor ``fat``
----------------------------
``TRIAD`` applies ``fat = -1`` to all three coupling coefficients (and to
``Sabc``) when all three modes are equatorially symmetric
(``symetry() == True``). Flipping one triad's ``fat``
is the gauge transformation ``A_j -> s_j A_j`` with ``s_p s_q s_r = -1``;
since every configuration used in this paper is a *star* sharing a single
edge (the sum mode's siblings are each private to one triad), this map is
surjective and the choice of ``fat_policy`` cannot change any energy at any
instant (see the docstring of ``rsw_sphere/utilities/check_wave_set_physics.py``,
check C4, for the precise claim and how it is verified rather than
assumed). Default ``fat_policy='symmetry'`` matches ``TRIAD``, which is
required for efficiency variation's numerator (a wave set) and
denominator (a ``TRIAD``) to be on the same convention.

Non-conservation
-----------------
Unlike a single triad, a wave set with 2+ constituent triads does **not**
conserve total energy in general (paper Appendix "Energy conservation of
the four-wave truncation" / eq. ``con``) -- there is no wave-set-level
analogue of the per-triad residual check. ``WaveSet`` does not compute or
expose one; only ``energy_drift`` (a diagnostic, not an identity) is
provided. Callers must report drift alongside any energy-variation-derived
quantity (e.g. efficiency variation) rather than assume it is negligible.

Run as a quick sanity check:

    python -m rsw_sphere.dynamics.wave_sets
"""


import numpy as np

from rsw_sphere.hough_harmonics.normalization import norm_Hough, velocity_to_amplitude
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import symetry
from rsw_sphere.hough_harmonics.inner_product import inner_product, S_abc
from rsw_sphere.dynamics.dynamic_triads import TRIAD, label as _internal_label

import warnings
warnings.filterwarnings("ignore", category=np.ComplexWarning)


class WaveSet:
    """A set of Hough modes coupled through one or more resonant triads.

    Parameters
    ----------
    gamma : float
        Lamb's parameter (``rsw_sphere.physics.gamma_from_he``).
    modes : sequence of (m, n, alpha) int triples
        One entry per mode, in a fixed order that all other arguments/
        outputs index into. ``alpha``: 1=EG, 2=WG, 3=RH.
    triads : sequence of (i_sum, i_p, i_q) int triples
        Each names one constituent triad by index into ``modes``:
        ``i_sum`` is the "sum" mode (paper convention: the mode whose
        wavenumber is the sum of the other two, e.g. mode ``a`` in
        ``eq :4sys1``), ``i_p``/``i_q`` its two members, in either order.
        A plain triad is ``modes`` of length 3, ``triads=[(2, 0, 1)]``
        (mode 2 is the sum) -- but for a triad, prefer ``TRIAD`` directly
        unless you specifically need ``WaveSet``'s uniform interface.
    N : int, optional
        Hough-mode expansion truncation order. Default 10.
    deg : int, optional
        Gaussian-quadrature degree. Must match the ``deg`` used inside
        ``norm_Hough`` for every mode -- **no default is applied silently
        across a triad/wave-set boundary**; pass it explicitly whenever
        comparing a ``WaveSet`` against a ``TRIAD`` (see
        ``sub_triad``, which forwards this same value).
    fat_policy : {'symmetry', 'off', 'always'}, optional
        Per-triad symmetry sign factor. ``'symmetry'`` (default) matches
        ``TRIAD``'s own rule (``fat=-1`` iff all three modes are
        equatorially symmetric). ``'off'`` never applies it. ``'always'``
        applies ``fat=-1``
        unconditionally, for probing the gauge claim (check C4) beyond
        just the symmetric-mode case. Does not affect any energy at any
        instant (see module docstring) -- provided *only* for the physics
        cross-checks; there is no reason to pass anything but the default
        in application code.

    Attributes
    ----------
    n_modes, n_triads : int
    labels : list of str
        Internal EIG/WIG/RH labels (``rsw_sphere.dynamics.dynamic_triads.label``),
        one per mode. For paper-facing EG/WG labels use
        ``rsw_sphere.plotting.labels._mode_label(*modes[j])``.
    omega : ndarray, shape (n_modes,)
    symmetric : list of bool, length n_modes
    alpha : ndarray, shape (n_triads, 3)
        Per triad, ``[alpha_p, alpha_q, alpha_sum]`` -- the coupling
        coefficients in the same role order as ``TRIAD.coef_ABC``,
        ``coef_BAC``, ``coef_CAB`` for that triad's ``(p, q, sum)``.
    delta : ndarray, shape (n_triads,)
        Per-triad mismatch, ``-omega[sum] + omega[q] + omega[p]``.
    S : ndarray, shape (n_triads,)
    fat : ndarray, shape (n_triads,)
    residual : ndarray, shape (n_triads,)
        ``(alpha_p + alpha_q - alpha_sum) + delta * S`` per triad; must be
        ~0 for a physically consistent triad (warns if not). There is no
        wave-set-level analogue -- see module docstring.
    """

    def __init__(self, gamma, modes, triads, N=10, deg=300, fat_policy='symmetry'):
        if fat_policy not in ('symmetry', 'off', 'always'):
            raise ValueError(f"fat_policy must be 'symmetry'/'off'/'always', got {fat_policy!r}")

        self.gamma = gamma
        self.modes = [tuple(m) for m in modes]
        self.triads = [tuple(t) for t in triads]
        self.N = N
        self.deg = deg
        self.fat_policy = fat_policy
        self.n_modes = len(self.modes)
        self.n_triads = len(self.triads)

        for (i_sum, i_p, i_q) in self.triads:
            m_sum = self.modes[i_sum][0]
            m_p = self.modes[i_p][0]
            m_q = self.modes[i_q][0]
            if m_sum != m_p + m_q:
                raise ValueError(
                    f"triad ({i_sum},{i_p},{i_q}): sum mode's zonal wavenumber "
                    f"m={m_sum} != m_p+m_q={m_p}+{m_q}={m_p + m_q}")

        for (m, n, alpha) in self.modes:
            if n < m:
                raise ValueError(
                    f"mode (m={m}, n={n}, alpha={alpha}): n < m is out of range "
                    f"for the associated-Legendre/Hough expansion (norm_Pmn "
                    f"requires n >= m) -- norm_Hough would not raise on this, it "
                    f"silently returns a zero-padded/degenerate eigenvector "
                    f"instead (found 2026-08-13: two such 'candidate gravity "
                    f"modes' turned out to be bit-identical to an already-"
                    f"present RH mode, corrupting a precession-resonance search "
                    f"that used them without ever detecting the duplication).")

        self._uvh = []
        self.omega = np.empty(self.n_modes)
        self.symmetric = []
        self.labels = []
        for (m, n, alpha) in self.modes:
            raw = norm_Hough(m, n, alpha, gamma, N, deg)
            eigen = raw[-1]
            uvh = raw[:-3]
            self._uvh.append(uvh)
            self.omega[len(self.labels)] = eigen
            self.symmetric.append(symetry(m, n, alpha))
            self.labels.append(_internal_label(m, n, alpha))

        self.alpha = np.empty((self.n_triads, 3), dtype=complex)
        self.delta = np.empty(self.n_triads)
        self.S = np.empty(self.n_triads, dtype=complex)
        self.fat = np.empty(self.n_triads)

        for t, (i_sum, i_p, i_q) in enumerate(self.triads):
            m_p, n_p, alpha_p = self.modes[i_p]
            m_q, n_q, alpha_q = self.modes[i_q]
            m_s, n_s, alpha_s = self.modes[i_sum]
            uvh_p, uvh_q, uvh_s = self._uvh[i_p], self._uvh[i_q], self._uvh[i_sum]

            if fat_policy == 'symmetry':
                fat = -1.0 if (self.symmetric[i_p] and self.symmetric[i_q]
                                and self.symmetric[i_sum]) else 1.0
            elif fat_policy == 'always':
                fat = -1.0
            else:
                fat = 1.0
            self.fat[t] = fat

            inner_p = inner_product(uvh_p, m_p, uvh_q, m_q, uvh_s, m_s, deg, True)
            inner_q = inner_product(uvh_q, m_q, uvh_p, m_p, uvh_s, m_s, deg, True)
            inner_s = inner_product(uvh_s, m_s, uvh_p, m_p, uvh_q, m_q, deg, False)

            self.alpha[t, 0] = fat * gamma * inner_p
            self.alpha[t, 1] = fat * gamma * inner_q
            self.alpha[t, 2] = fat * gamma * inner_s

            self.delta[t] = -self.omega[i_sum] + self.omega[i_q] + self.omega[i_p]
            self.S[t] = -fat * S_abc(uvh_p, m_p, uvh_q, m_q, uvh_s, m_s, deg)

        self.residual = (self.alpha[:, 0] + self.alpha[:, 1] - self.alpha[:, 2]) \
            + self.delta * self.S
        bad = np.abs(self.residual) > 1e-6
        if np.any(bad):
            warnings.warn(
                f"WaveSet: energy-conservation residual not ~0 for "
                f"triad(s) {np.nonzero(bad)[0].tolist()} -- "
                f"{self.residual[bad]}. This indicates a physically "
                f"inconsistent triad definition or a numerical-precision "
                f"issue (not merely non-conservation of the wave-set "
                f"total energy, which is expected -- see module docstring).")

    def f(self, AMP):
        """RHS of the amplitude ODEs, ``dAMP/dt``.

        Parameters
        ----------
        AMP : ndarray, shape (..., n_modes)
            Complex amplitudes. Mode index is the **last** axis; any
            number of leading batch dimensions is supported (e.g. for
            integrating an entire parameter-sweep grid in one ``RK44``
            call) -- every operation below is elementwise/broadcasting
            over those leading axes.

        Returns
        -------
        ndarray, same shape as ``AMP``
        """
        AMP = np.asarray(AMP)
        dA = -1j * self.omega * AMP
        for t, (i_sum, i_p, i_q) in enumerate(self.triads):
            A_p = AMP[..., i_p]
            A_q = AMP[..., i_q]
            A_s = AMP[..., i_sum]
            alpha_p, alpha_q, alpha_s = self.alpha[t]
            dA[..., i_p] = dA[..., i_p] + 1j * alpha_p * np.conj(A_q) * A_s
            dA[..., i_q] = dA[..., i_q] + 1j * alpha_q * np.conj(A_p) * A_s
            dA[..., i_sum] = dA[..., i_sum] + 1j * alpha_s * A_p * A_q
        return dA

    def energy(self, AMP):
        """Quadratic and cubic energy, per the same convention as
        ``rsw_sphere.dynamics.dynamic_triads.Energy_0`` (one constituent
        triad's cubic term reduces to exactly ``Energy_0``'s formula).

        Parameters
        ----------
        AMP : ndarray, shape (..., n_modes)

        Returns
        -------
        E2, E3 : ndarray, shape AMP.shape[:-1]
        """
        AMP = np.asarray(AMP)
        E2 = np.sum(np.real(AMP * np.conj(AMP)), axis=-1)
        E3 = np.zeros(AMP.shape[:-1])
        for t, (i_sum, i_p, i_q) in enumerate(self.triads):
            A_p = AMP[..., i_p]
            A_q = AMP[..., i_q]
            A_s = AMP[..., i_sum]
            E3 = E3 + 2 * np.real(np.conj(A_s) * A_p * A_q) * self.S[t]
        return E2, E3

    def amplitudes_from_velocities(self, velocities, h_e, g=9.8):
        """Real initial amplitudes producing the given per-mode initial
        zonal velocities, via ``velocity_to_amplitude`` (paper ``eq:
        Azonal``) -- the same ``/2``-corrected conversion used throughout
        the (retired) §2.2 triad toolchain, centralized here so every §3
        module (``wave_set_dynamics``, ``wave_set_pmeasure``,
        ``wave_set_periods``) shares one implementation.

        Parameters
        ----------
        velocities : array_like, shape (..., n_modes)
            Zonal velocities in m/s, mode index last. Any number of
            leading batch dimensions is supported (e.g. an entire
            parameter-sweep grid), broadcasting the same way ``f`` does.
        h_e : float
            Equivalent height, m.
        g : float, optional
            Default ``9.8``.

        Returns
        -------
        ndarray, shape (..., n_modes), complex
        """
        velocities = np.asarray(velocities, dtype=float)
        A0 = np.empty(velocities.shape, dtype=complex)
        for j in range(self.n_modes):
            A0[..., j] = velocity_to_amplitude(velocities[..., j], self._uvh[j][0], h_e, g=g)
        return A0

    def sub_triad_local_indices(self, i):
        """Index triple ``(i_p, i_q, i_sum)`` for constituent triad ``i``,
        in the order ``TRIAD.__init__`` expects (sum mode last).
        """
        i_sum, i_p, i_q = self.triads[i]
        return i_p, i_q, i_sum

    def sub_triad_modes(self, i):
        """The 3 ``(m, n, alpha)`` triples for constituent triad ``i``,
        **sum mode last** -- directly unpackable into ``TRIAD.__init__``'s
        flat positional args via ``sum((), sub_triad_modes(i))`` or see
        ``sub_triad``.
        """
        i_p, i_q, i_sum = self.sub_triad_local_indices(i)
        return self.modes[i_p], self.modes[i_q], self.modes[i_sum]

    def sub_triad(self, i):
        """Build the independent-reference ``TRIAD`` for constituent triad
        ``i`` (same ``gamma``/``N``/``deg`` as this wave set, sum mode
        placed last). Used by ``rsw_sphere/utilities/check_wave_set_physics.py`` (C1)
        and by ``rsw_sphere.plotting.wave_set_dynamics`` to plot a
        sub-triad's own trajectory in the same units as the parent set.

        Returns
        -------
        TRIAD
        """
        (m_p, n_p, a_p), (m_q, n_q, a_q), (m_s, n_s, a_s) = self.sub_triad_modes(i)
        return TRIAD(self.gamma, m_p, n_p, a_p, m_q, n_q, a_q, m_s, n_s, a_s,
                      self.N, self.deg)


if __name__ == "__main__":
    from rsw_sphere.physics import gamma_from_he

    eps, gamma = gamma_from_he(10000)

    # RH(4,5)+RH(1,2)+RH(3,4): reproduces Triad B (triad_rossby_only_non_resonant)
    # from the §2.2 registry, as a 1-triad WaveSet -- sum mode is RH(4,5)? No:
    # m=4 = m_b(1) + m_c(3)? 1+3=4, so sum mode is index 0 (m=4).
    modes = [(4, 5, 3), (1, 2, 3), (3, 4, 3)]
    triads = [(0, 1, 2)]  # sum=RH(4,5), members=RH(1,2)+RH(3,4)
    ws = WaveSet(gamma, modes, triads, N=10, deg=300)
    print(f"WaveSet: {ws.n_modes} modes, {ws.n_triads} triad(s)")
    print(f"  omega = {ws.omega}")
    print(f"  alpha[0] = {ws.alpha[0]}")
    print(f"  delta[0] = {ws.delta[0]:.6e}")
    print(f"  residual[0] = {ws.residual[0]:.3e} (expect ~0)")

    T = ws.sub_triad(0)
    print(f"  TRIAD.coef_ABC = {T.coef_ABC} (expect == alpha[0,0])")
    print(f"  TRIAD.mismatch = {T.mismatch:.6e} (expect == delta[0])")
