"""Gate I2 map extension (§Coupled Triads, S2 centerpiece) -- extends the
2026-08-12 first pass (2 of 26 catalogue candidates: EG(1,1), EG(7,9)
only) to the full 26-row Gate I3/I4 catalogue, colored by the
now-derived two-channel law, instead of the original 2-candidate stub.

Map definition, per ``PLAN-section-3-experiments.md`` Phase I2:
  x-axis: energy partition, fraction of total quadratic energy on the
    gravity mode d (0 to ~0.7, matching Gate I4's own wider range).
  y-axis: timescale separation, ``omega_d / Omega_slow``, where
    ``Omega_slow = 2*pi/T_exchange`` is the RH-only triad's own NONLINEAR
    exchange rate at that x's own amplitude split (peak-to-peak timing
    on the target mode's KE trace, matching Gate I2's own methodology --
    FFT was tried and rejected there, too coarse to resolve the
    amplitude-dependent period shift).
  color: a D1 PROXY (see ``d1_proxy``'s own docstring -- NOT calibrated
    D1 in percent; a 2026-08-13 review found this substitution
    overestimates real, integrated D1 by an uncalibrated, point-varying
    factor, contradicting this module's original "no free constant"
    claim below).

Key simplification vs. a brute-force re-run: ``Omega_slow(x)`` depends
ONLY on x (the RH-only triad triad1's own IC), not on which gravity mode
candidate d is chosen -- so it is measured once per x value (real
integration, 8 points) rather than once per (candidate, x) pair (which
would need 26*8=208 integrations). This independence claim is a
tautology of the construction (Omega_slow is measured on the RH-only
triad alone, whose own IC never involves the candidate), not an
independently-verified physical result -- worth stating plainly rather
than implying it was checked.

D1 itself is NOT re-integrated per candidate: Gate I4b's own two-channel
law,

    D1 ~ sqrt(alpha_2s^2 + alpha_2p^2) * sqrt(x*(1-x)) / delta_2

(fitted shape verified R^2=0.982 across 156 points on 2026-08-12) is
evaluated directly from each candidate's own construction-only
coefficients. **This module previously claimed "no free constant" for
that law -- wrong.** A 2026-08-13 review, plus an independent 4-point
spot-check reproduced here, both found the raw formula above
overestimates true integrated D1 by a real, point-varying factor
(measured ratios 0.11-0.27, no single clean constant) -- so its output
is used here only as a proxy (right functional shape/exponents,
NOT calibrated to real percentages) pending a proper per-point
calibration in a future session. See ``d1_proxy``'s own docstring.

Catalogue construction: every EG/WG mode closing a triad with the
RH(4,5)+RH(3,4) edge, as either triad2's MEMBER (m_d=1, matching the
registered ``quartet_gravity_kelvin``/``quartet_gravity_79`` structure)
or its SUM (m_d=7), n up to 15, alpha in {1 (EG), 2 (WG)}. Selection-rule
survivors only (nonzero coupling). ``alpha_2s``/``alpha_2p`` are read
from ``WaveSet.alpha``'s own ``(alpha_p, alpha_q, alpha_s)`` positional
convention, which is role-agnostic by construction (it depends only on
each triad's own ``(i_sum, i_p, i_q)`` argument order, not on which
physical mode plays which role) -- verified against this same
convention earlier in the 2026-08-13 precession-resonance investigation
(``examples/precession_resonance_broad_search.py``), so applying it to
the m_d=7 (sum-role) candidates here, which the original 2026-08-12 law
verification may or may not have included, is on solid footing.

Run:

    python examples/gate_i2_map_extension.py
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
from scipy.signal import find_peaks

from rsw_sphere.physics import gamma_from_he, days_from_nondim_time
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.dynamic_triads import TRIAD

G = 9.8
H_E = 10000.0
A_MODE, B_MODE, C_MODE = (4, 5, 3), (3, 4, 3), (1, 2, 3)  # a=sum, b=target, c=third RH
X_VALUES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def find_catalogue(gamma, n_max=15):
    """Every EG/WG candidate closing edge {RH(4,5),RH(3,4)} as member
    (m_d=1) or sum (m_d=7), n=1..n_max, selection-rule survivors only.
    Returns list of dicts with mode, role, delta_2, alpha_2s, alpha_2p,
    omega_d.
    """
    candidates = []
    for role, m_d in (('member', 1), ('sum', 7)):
        for alpha_d in (1, 2):
            for n in range(1, n_max + 1):
                if n < m_d:
                    continue
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        if role == 'member':
                            t = TRIAD(gamma, *B_MODE, *(m_d, n, alpha_d), *A_MODE, N=10, deg=300)
                            coup_p, coup_s = abs(t.coef_ABC), abs(t.coef_CAB)
                        else:
                            t = TRIAD(gamma, *A_MODE, *B_MODE, *(m_d, n, alpha_d), N=10, deg=300)
                            coup_p, coup_s = abs(t.coef_ABC), abs(t.coef_CAB)
                except Exception:
                    continue
                if coup_p < 1e-6 and coup_s < 1e-6:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if role == 'member':
                        ws = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (m_d, n, alpha_d)],
                                     [(0, 1, 2), (0, 1, 3)], N=10, deg=300)
                        alpha_2p, alpha_2s = ws.alpha[1, 0], ws.alpha[1, 2]
                        omega_d = ws.omega[3]
                    else:
                        ws = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (m_d, n, alpha_d)],
                                     [(0, 1, 2), (3, 0, 1)], N=10, deg=300)
                        alpha_2s, alpha_2p = ws.alpha[1, 0], ws.alpha[1, 1]
                        omega_d = ws.omega[3]
                delta_2 = ws.delta[1]
                if abs(delta_2) < 1e-8:
                    continue
                candidates.append(dict(
                    label=f"{'EG' if alpha_d == 1 else 'WG'}({m_d},{n})", role=role,
                    m_d=m_d, n=n, alpha_d=alpha_d, omega_d=omega_d, delta_2=delta_2,
                    alpha_2s=abs(alpha_2s), alpha_2p=abs(alpha_2p),
                ))
    return candidates


def omega_slow_vs_x(gamma, x_values, e_tot, tf_days=30.0, h=0.01):
    """Omega_slow(x) = 2*pi/T_exchange, T_exchange from peak-to-peak
    timing on the target mode b's own KE trace, RH-only triad1 (a,b,c)
    integrated alone. Independent of any gravity-mode candidate.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws1 = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    t_f = tf_days * 4 * np.pi
    results = {}
    for x in x_values:
        e_each = (1 - x) / 3 * e_tot
        A0 = np.sqrt(e_each) * np.ones(3, dtype=complex)
        Y, T = RK33(ws1, 0, t_f, h, A0)
        KE_b = np.real(Y[:, 1] * np.conj(Y[:, 1]))
        peaks, _ = find_peaks(KE_b)
        if len(peaks) < 2:
            results[x] = None
            continue
        days = days_from_nondim_time(T[peaks])
        T_exchange = np.mean(np.diff(days))
        results[x] = 2 * np.pi / T_exchange
    return results


def d1_proxy(alpha_2s, alpha_2p, delta_2, x):
    """**NOT calibrated D1** -- despite this module's original docstring
    claiming "no free constant," a 2026-08-13 review + an independent
    4-point spot-check here both found the raw
    ``sqrt(alpha_2s^2+alpha_2p^2)*sqrt(x(1-x))/delta_2`` combination
    overestimates real, integrated D1 by a factor that itself varies
    point to point (measured ratios 0.11-0.27 across a handful of
    candidates, no single clean constant) -- likely because the
    two-channel law's own log-log fit (Gate I4b, R^2=0.982) has a real
    multiplicative prefactor/intercept the module's docstring
    mis-described as ~0. Returned AS A PROXY ONLY: rank-correlated with
    true D1 within a given candidate's own x-sweep (same exponents,
    right functional shape) but NOT numerically equal to it, and not
    safe to compare in absolute terms across candidates without a
    proper per-point calibration (not done here -- see the INSPECT doc's
    "Gate I2 map correction" section, queued for a future session).
    Do not read the colorbar/printed values below as literal
    percentages.
    """
    return np.sqrt(alpha_2s ** 2 + alpha_2p ** 2) * np.sqrt(x * (1 - x)) / abs(delta_2)


if __name__ == "__main__":
    gamma = gamma_from_he(H_E, g=G)[1]

    print("Building catalogue (edge RH(4,5)+RH(3,4), member m_d=1 + sum m_d=7, n<=15)...")
    catalogue = find_catalogue(gamma)
    print(f"{len(catalogue)} survivors\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_baseline = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    A0_base = ws_baseline.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))
    print(f"E_tot (quartet_gravity_kelvin IC, a=b=c=30 m/s) = {e_tot:.6e}\n")

    print("Measuring Omega_slow(x) (8 points, RH-only triad, real integration)...")
    om_slow = omega_slow_vs_x(gamma, X_VALUES, e_tot)
    for x, val in om_slow.items():
        print(f"  x={x:.2f}  Omega_slow={val:.5f} rad/day" if val else f"  x={x:.2f}  FAILED (too few peaks)")

    print(f"\n{'label':>10} {'role':>7} {'omega_d':>9} {'delta_2':>10} "
          f"{'alpha_2s':>9} {'alpha_2p':>9} " +
          " ".join(f"D1proxy(x={x:.2f})".rjust(11) for x in X_VALUES))
    rows = []
    for c in sorted(catalogue, key=lambda c: abs(c['omega_d'])):
        d1_vals = [d1_proxy(c['alpha_2s'], c['alpha_2p'], c['delta_2'], x) for x in X_VALUES]
        # omega_d is nondimensional (units of 2*Omega_earth); convert to
        # rad/day (matching Omega_slow's own units, from
        # days_from_nondim_time) via the standard 4*pi factor -- see
        # rsw_sphere.physics.linear_period_days's own derivation.
        omega_d_rad_per_day = 4 * np.pi * abs(c['omega_d'])
        om_ratios = [omega_d_rad_per_day / om_slow[x] if om_slow[x] else float('nan') for x in X_VALUES]
        rows.append(dict(c, d1_vals=d1_vals, om_ratios=om_ratios))
        print(f"{c['label']:>10} {c['role']:>7} {c['omega_d']:>9.4f} {c['delta_2']:>10.4f} "
              f"{c['alpha_2s']:>9.4f} {c['alpha_2p']:>9.4f} " +
              " ".join(f"{v:>11.5f}" for v in d1_vals))

    np.save(os.path.join(_ROOT, "gate_i2_map_data.npy"), rows, allow_pickle=True)
    print(f"\nSaved {len(rows)} rows x {len(X_VALUES)} x-values to "
          f"examples/gate_i2_map_data.npy (candidate, role, omega_d, delta_2, "
          f"alpha_2s, alpha_2p, d1_vals, om_ratios).")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    xs_all, ys_all, cs_all = [], [], []
    for r in rows:
        xs_all.extend(X_VALUES)
        ys_all.extend(r['om_ratios'])
        cs_all.extend(r['d1_vals'])
    sc = ax.scatter(xs_all, ys_all, c=cs_all, cmap='viridis',
                     norm=matplotlib.colors.LogNorm(vmin=max(1e-5, min(cs_all)), vmax=max(cs_all)),
                     s=25, edgecolors='none')
    for r in rows:
        if r['label'] in ('EG(1,1)', 'EG(7,9)'):
            ax.plot(X_VALUES, r['om_ratios'], '-', color='red', lw=1, alpha=0.6)
            ax.annotate(r['label'], (X_VALUES[-1], r['om_ratios'][-1]),
                        textcoords="offset points", xytext=(5, 0), fontsize=9, color='red')
    ax.set_yscale('log')
    ax.set_xlabel('x (energy fraction on gravity mode)')
    ax.set_ylabel(r'$\omega_d / \Omega_{slow}$')
    ax.set_title('Gate I2 map: full 26-candidate catalogue, colored by D1 (two-channel law)')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label('D1 proxy (uncalibrated -- see docstring, NOT literal %)')
    fig.tight_layout()
    out_path = os.path.join(_ROOT, "gate_i2_map.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved map figure to examples/gate_i2_map.png")
