"""Gate I4/I4b scaling law (S3, "the mechanism") -- gives the
2026-08-12 investigation's own two-channel law a reproducible script for
the first time.

Background: the law

    F2 ~ sqrt(alpha_2s^2 + alpha_2p^2) * sqrt(x*(1-x)) / delta_2

(derived by two-timescale averaging for a_d(0)!=0, i.e. the gravity mode
d starts with its OWN share x of the total energy) was fit and
verified previously (R^2=0.982, 156 points)
but the fit itself was never saved as a script -- only prose. This
module redoes that fit from scratch, against the paper's own §3.1
diagnostic $\\mathcal{F}_2$ (eq: F2, relative RMS amplitude error of the
target mode b=RH(3,4), full quartet vs. RH-only sub-triad) rather than
whatever ad hoc "D1" the original investigation session used -- so the
number quoted in the paper is traceable to the same formula the paper
itself defines, not a re-derived proxy.

Target mode is b=RH(3,4) (index 1 in the [a,b,c,d] mode ordering below),
matching Gate I5's own headline diagnostic and the fact that RH(3,4) is
the mode with BOTH channels: alpha_2s (indirect, via the shared/pump
mode a=RH(4,5)) and alpha_2p (direct, triad2's own coupling into b) --
exactly why the two-channel (not single-channel) law was needed (Gate
I4b's own WG(1,3) anomaly, resolved by adding alpha_2p).

Reuses ``find_catalogue`` from ``gate_i2_map_extension.py`` (26
selection-rule survivors on edge RH(4,5)+RH(3,4), n<=15) rather than
duplicating it -- both this script and that one now read from the same
catalogue-construction code, per the plan's own "factor into a shared
helper" note (importing the function directly serves that purpose
without a separate module).

This same integration pass is deliberately reused for Phase 1's Gate I2
map calibration (see ``gate_i2_map_recalibrate.py``): the map's
uncalibrated ``d1_proxy`` colorbar is replaced by this script's own
saved, real F2 values rather than integrating the 26x8 grid a second
time.

Run:

    python examples/gate_i4_scaling_law.py
"""
import os
import sys
import time
import warnings

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.integrators import RK33
from rsw_sphere.dynamics.wave_sets import WaveSet
from gate_i2_map_extension import find_catalogue, A_MODE, B_MODE, C_MODE, G, H_E

X_VALUES = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
I_B = 1  # target mode b = RH(3,4) in the [a,b,c,d] ordering below


def build_full_and_sub(gamma, cand):
    """One WaveSet for the full quartet (a,b,c,d) and one for the
    RH-only sub-triad (a,b,c) -- built once per candidate, reused across
    every x (only the initial condition changes with x).
    """
    m_d, n, alpha_d = cand['m_d'], cand['n'], cand['alpha_d']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cand['role'] == 'member':
            ws_full = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (m_d, n, alpha_d)],
                               [(0, 1, 2), (0, 1, 3)], N=10, deg=300)
        else:
            ws_full = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (m_d, n, alpha_d)],
                               [(0, 1, 2), (3, 0, 1)], N=10, deg=300)
        ws_sub = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    return ws_full, ws_sub


def measure_F2(ws_full, ws_sub, x, e_tot, tf_days, h=0.01):
    """eq: F2 -- relative RMS amplitude error of target mode b, full
    quartet vs. RH-only sub-triad, over [0, t_f]. IC: RH triad (a,b,c)
    share (1-x)*e_tot equally (real amplitudes, zonal-jet convention,
    eq: Azonal); candidate mode d gets its own share x*e_tot, also real
    and in phase with the others (a_d(0) != 0 -- Gate I0's "energy
    partition" scenario, not the drop-mode scenario Gate I5 uses).
    """
    e_rh_each = (1 - x) / 3 * e_tot
    A0_sub = np.sqrt(e_rh_each) * np.ones(3, dtype=complex)
    A0_full = np.concatenate([A0_sub, [np.sqrt(x * e_tot)]]).astype(complex)
    t_f = tf_days * 4 * np.pi
    Yf, Tf = RK33(ws_full, 0, t_f, h, A0_full)
    Ys, Ts = RK33(ws_sub, 0, t_f, h, A0_sub)
    assert np.allclose(Tf, Ts)
    amp_full = np.abs(Yf[:, I_B])
    amp_sub = np.abs(Ys[:, I_B])
    return np.sqrt(np.mean((amp_full - amp_sub) ** 2)) / np.sqrt(np.mean(amp_sub ** 2))


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

    # --- Horizon choice (PLAN-section-3.3.md Decision 1 / Phase 0):
    # F2 grows with tf rather than saturating quickly (checked directly
    # below at EG(1,1), the slowest-ripple candidate: 15.9% at tf=20d,
    # 17.2% at tf=40d, 17.2% at tf=80d) -- the SAME phase-decorrelation
    # growth Gate I5 already found and documented for the drop-mode
    # scenario (S3.1's own diagnostics text: "F2 is always reported
    # together with its own tf; it is not, in general, tf-independent").
    # Rather than chase asymptotic convergence per candidate (expensive:
    # the 26x8 grid at tf=80d would take ~30 min), this script fixes
    # tf_days=20 -- matching quartet_gravity_kelvin's OWN registered
    # horizon, so the catalogue's F2 values are directly comparable to
    # S3.3.1's headline number -- and reports the law as horizon-
    # qualified, consistent with the paper's own stated discipline, not
    # as an asymptotic constant. A robustness subset at tf_days=40 (run
    # after the main grid, below) checks that the fitted LAW's shape
    # (exponents, R^2) is not an artifact of that specific choice, even
    # though the absolute F2 values do shift with tf.
    print("--- Horizon check (EG(1,1), x=0.3) ---")
    cand_eg11 = next(c for c in catalogue if c['label'] == 'EG(1,1)')
    ws_full, ws_sub = build_full_and_sub(gamma, cand_eg11)
    for tf_check in (20.0, 40.0, 80.0):
        t0 = time.time()
        f2 = measure_F2(ws_full, ws_sub, 0.3, e_tot, tf_check)
        print(f"  tf_days={tf_check:5.1f}  F2={f2 * 100:.4f}%  ({time.time() - t0:.2f}s)")

    TF_DAYS = 20.0

    # --- Full grid: 26 candidates x 8 energy fractions, tf_days=20 ---
    print(f"\n--- Full grid (26x{len(X_VALUES)}={26 * len(X_VALUES)} points, tf_days={TF_DAYS}) ---")
    t0 = time.time()
    rows = []
    for i, cand in enumerate(catalogue):
        wf, ws = build_full_and_sub(gamma, cand)
        f2_vals = [measure_F2(wf, ws, x, e_tot, TF_DAYS) for x in X_VALUES]
        rows.append(dict(cand, f2_vals=f2_vals))
        print(f"  [{i + 1:2d}/{len(catalogue)}] {cand['label']:>10} role={cand['role']:>6} "
              f"delta_2={cand['delta_2']:9.5f} F2(x)=" +
              " ".join(f"{v * 100:6.2f}%" for v in f2_vals))
    dt = time.time() - t0
    print(f"Full grid done in {dt:.1f}s ({dt / (26 * len(X_VALUES)) * 1000:.0f}ms/point)")

    np.save(os.path.join(_ROOT, "gate_i4_scaling_law_data.npy"), rows, allow_pickle=True)
    print(f"Saved {len(rows)} rows x {len(X_VALUES)} x-values to "
          f"examples/gate_i4_scaling_law_data.npy")

    # --- Power-law fit: log(F2) ~ log(sqrt(a2s^2+a2p^2)) + log(sqrt(x(1-x))) - log(delta_2) ---
    log_F2, log_coup, log_env, log_delta = [], [], [], []
    for r in rows:
        coup = np.hypot(r['alpha_2s'], r['alpha_2p'])
        for x, f2 in zip(X_VALUES, r['f2_vals']):
            if f2 <= 0:
                continue
            log_F2.append(np.log(f2))
            log_coup.append(np.log(coup))
            log_env.append(np.log(np.sqrt(x * (1 - x))))
            log_delta.append(np.log(abs(r['delta_2'])))
    Xmat = np.column_stack([log_coup, log_env, log_delta, np.ones(len(log_F2))])
    yvec = np.array(log_F2)
    coef, res, rank, sv = np.linalg.lstsq(Xmat, yvec, rcond=None)
    pred = Xmat @ coef
    ss_res = np.sum((yvec - pred) ** 2)
    ss_tot = np.sum((yvec - yvec.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"\n--- Fit: log(F2) = {coef[0]:.3f}*log(coupling) + {coef[1]:.3f}*log(env) "
          f"+ {coef[2]:.3f}*log(delta_2) + {coef[3]:.3f} ---")
    print(f"R^2 = {r2:.4f}  (derived exponents: 1, 1, -1; "
          f"2026-08-12 investigation's own fit: 1.00, 1.06, -1.04, R^2=0.982)")

    # --- Robustness subset at a second horizon (tf_days=40): confirm the
    # law's SHAPE isn't an artifact of tf_days=20 specifically, even
    # though absolute F2 shifts with tf (expected, per the horizon check
    # above). 8 representative candidates (spanning the omega_d range)
    # x all 8 energy fractions = 64 points.
    print("\n--- Robustness subset at tf_days=40 (8 candidates x 8 x-values) ---")
    subset_labels = ['EG(1,1)', 'WG(1,1)', 'EG(1,5)', 'WG(7,7)', 'EG(7,9)',
                      'WG(7,9)', 'EG(1,9)', 'EG(7,13)']
    subset = [c for c in catalogue if c['label'] in subset_labels]
    log_F2b, log_coupb, log_envb, log_deltab = [], [], [], []
    t0 = time.time()
    for cand in subset:
        wf, ws = build_full_and_sub(gamma, cand)
        coup = np.hypot(cand['alpha_2s'], cand['alpha_2p'])
        for x in X_VALUES:
            f2 = measure_F2(wf, ws, x, e_tot, 40.0)
            if f2 <= 0:
                continue
            log_F2b.append(np.log(f2))
            log_coupb.append(np.log(coup))
            log_envb.append(np.log(np.sqrt(x * (1 - x))))
            log_deltab.append(np.log(abs(cand['delta_2'])))
    print(f"  ({time.time() - t0:.1f}s)")
    Xb = np.column_stack([log_coupb, log_envb, log_deltab, np.ones(len(log_F2b))])
    yb = np.array(log_F2b)
    coefb, _, _, _ = np.linalg.lstsq(Xb, yb, rcond=None)
    predb = Xb @ coefb
    r2b = 1 - np.sum((yb - predb) ** 2) / np.sum((yb - yb.mean()) ** 2)
    print(f"  tf=40d subset fit: exponents ({coefb[0]:.3f}, {coefb[1]:.3f}, {coefb[2]:.3f}), R^2={r2b:.4f}")
    print(f"  (compare tf=20d full-grid fit above: ({coef[0]:.3f}, {coef[1]:.3f}, {coef[2]:.3f}), R^2={r2:.4f})")

    # --- Verification figure: measured F2 vs. predicted (the law, fit
    # coefficients above), log-log, EG(1,1)/EG(7,9) highlighted.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    measured = np.exp(yvec) * 100
    predicted = np.exp(pred) * 100
    ax.scatter(predicted, measured, s=18, alpha=0.5, color='steelblue', label='catalogue (26 modes x 8 x)')
    lims = [min(predicted.min(), measured.min()), max(predicted.max(), measured.max())]
    ax.plot(lims, lims, 'k--', lw=1, label='1:1')
    # Highlight EG(1,1)/EG(7,9)
    for label, color in (('EG(1,1)', 'crimson'), ('EG(7,9)', 'darkorange')):
        r = next(rr for rr in rows if rr['label'] == label)
        coup = np.hypot(r['alpha_2s'], r['alpha_2p'])
        pr = [np.exp(coef[0] * np.log(coup) + coef[1] * np.log(np.sqrt(x * (1 - x)))
                      + coef[2] * np.log(abs(r['delta_2'])) + coef[3]) * 100 for x in X_VALUES]
        me = [v * 100 for v in r['f2_vals']]
        ax.scatter(pr, me, s=40, color=color, label=label, zorder=5, edgecolors='k', linewidths=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Predicted $\mathcal{F}_2$ (%) -- fitted law')
    ax.set_ylabel(r'Measured $\mathcal{F}_2$ (%), $t_f=20$d')
    ax.set_title(f'Gate I4b scaling law, $R^2$={r2:.3f}')
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = os.path.join(_ROOT, "gate_i4_scaling_law.png")
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved verification figure to examples/gate_i4_scaling_law.png")
