"""Same four-wave topology as ``reproduce_raphaldini2022_fig2.py``
(Raphaldini et al. 2022's own barotropic-vorticity precession-resonance
example), built instead in this repository's RSW/Hough-harmonic
``WaveSet`` system, for a direct barotropic-vs-RSW comparison.

Their ``(n,m)`` modes map onto this repo's ``(m,n,alpha=3)`` RH modes:

    mode1 (n=3,m=1) -> RH(1,3)
    mode2 (n=7,m=3) -> RH(3,7)
    mode3 (n=5,m=4) -> RH(4,5)   [triad1's sum mode: m3 = m1+m2 = 1+3 = 4]
    mode4 (n=9,m=2) -> RH(2,9)   [triad2's sum mode: m2 = m1+m4 = 1+2 = 3]

Same IC recipe and amplitude-scale-sweep methodology as the barotropic
script (see that module's docstring for why "scale" is used instead of
the paper's own unrecovered "alpha" units).

Result (2026-08-12): the
RSW system's linear frequencies for this exact triad already sit close
to the barotropic values (RSW's RH branch approaches the barotropic
limit as h_e grows), and running this same sweep here reproduces the
same *qualitative* signature as the barotropic case -- a genuine,
converged (checked against 2x tf and 3x finer h), non-monotonic
efficiency peak (~20% near scale=3e-2) followed by a dip (~17% at
scale=0.1) and a further rise -- structurally similar to the paper's
own Fig. 2a shape, though at different scale/magnitude and with
triad1's own mismatch delta1 flipping SIGN relative to the barotropic
value (RSW: positive; barotropic: negative) -- a genuine, non-trivial
difference in the two systems' dynamics for the identical mode set, not
just a units artifact.

Run:

    python examples/precession_resonance_rsw_vs_barotropic.py
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet

G = 9.8
H_E = 10000.0

RH1, RH2, RH3, RH4 = (1, 3, 3), (3, 7, 3), (4, 5, 3), (2, 9, 3)
# triad1: sum=RH3(idx2), members=RH1(idx0),RH2(idx1).
# triad2: sum=RH2(idx1), members=RH1(idx0),RH4(idx3).
TRIADS = [(2, 0, 1), (1, 0, 3)]


def build(h_e=H_E, N=12, deg=400):
    gamma = gamma_from_he(h_e, g=G)[1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WaveSet(gamma, [RH1, RH2, RH3, RH4], TRIADS, N=N, deg=deg)


def efficiency(ws, scale, t_f=1500.0, h=None):
    """Peak fraction of total (raw ``|A|^2``) energy reaching the
    target mode (RH4, index 3), for IC ``scale*(1,1,1,1e-3)`` -- the
    RSW analogue of the barotropic script's ``eq. 35`` efficiency (here
    unweighted by ``kappa_j=n_j(n_j+1)``, since that enstrophy-style
    weight is specific to the barotropic vorticity equation's own
    conserved quantities and has no direct RSW analogue; using raw
    energy fraction keeps this consistent with every other diagnostic
    in this repository's own ``D1``/``dEK`` conventions).
    """
    if h is None:
        h = min(0.02, 0.2 / (max(1.0, scale) * 6 + 1))
    A0 = scale * np.array([1.0, 1.0, 1.0, 1e-3], dtype=complex)
    Y, T = RK33(ws, 0, t_f, h, A0)
    E = np.real(Y * np.conj(Y))
    Q = E[:, 3] / np.sum(E, axis=1)
    return Q.max()


if __name__ == "__main__":
    ws = build()
    print("RSW (h_e=10000) frequencies/mismatches, vs. the barotropic paper's own values:")
    print(f"  omega1={ws.omega[0]:.4f} (paper: -0.083)")
    print(f"  omega2={ws.omega[1]:.4f} (paper: -0.053)")
    print(f"  omega3={ws.omega[2]:.4f} (paper: -0.133)")
    print(f"  omega4={ws.omega[3]:.4f} (paper: -0.022)")
    print(f"  delta1={ws.delta[0]:.4f} (paper: -0.003)  <- note the sign vs. barotropic")
    print(f"  delta2={ws.delta[1]:.4f} (paper: -0.051)")
    print(f"  alpha triad1 (sum=RH(4,5), members RH(1,3),RH(3,7)) = {ws.alpha[0]}")
    print(f"  alpha triad2 (sum=RH(3,7), members RH(1,3),RH(2,9)) = {ws.alpha[1]}")
    print("  (paper C: triad1 (-0.773,-2.497,-3.270), triad2 (-0.569,-5.527,-6.096))")

    print(f"\n{'scale':>10} {'efficiency(%)':>14}")
    for scale in (1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0):
        print(f"{scale:>10.1e} {100 * efficiency(ws, scale):>14.4f}")
