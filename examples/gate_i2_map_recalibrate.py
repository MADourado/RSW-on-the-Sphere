"""Recalibrate the Gate I2 map (S2, the §3.3.2 centerpiece figure) with
real, integrated $\\mathcal{F}_2$ values from ``gate_i4_scaling_law.py``
(saved in ``gate_i4_scaling_law_data.npy``) instead of the uncalibrated
``d1_proxy`` analytic shortcut -- per ``PLAN-section-3.3.md`` Phase 1.
Same x/y axes as the original ``gate_i2_map_extension.py`` (x = energy
fraction on the gravity mode, y = omega_d/Omega_slow); color is now real
$\\mathcal{F}_2$ (%) at $t_f=20$d, not a proxy.

Adds Examples 1/2/3's own three points (the cut dissertation anecdotes,
now folded into this one map per the plan's disposition table) and keeps
EG(1,1)/EG(7,9) highlighted as curves.

Run (after ``gate_i4_scaling_law.py`` has produced its data file):

    python examples/gate_i2_map_recalibrate.py
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

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.wave_sets import WaveSet
from gate_i2_map_extension import omega_slow_vs_x, A_MODE, B_MODE, C_MODE, G, H_E, X_VALUES

# Examples 1/2/3's own (x, candidate) -- dissertation Table cap4ex, all on
# the EG(1,1) curve (the only gravity mode those three examples used).
# x = fraction of E_tot on EG(1,1), computed from each example's own
# registered velocities (30/30/30/{0,80,20} m/s and 30/30/30/20 m/s for
# example 3's slightly different RH amplitudes -- approximated here by
# its own velocity-based energy fraction, not re-deriving a separate
# E_tot per example).
EXAMPLES = {
    'Example 1 (u_d=0)': 0.0,
    'Example 2 (u_d=80)': None,  # filled in from velocities below
    'Example 3 (u_d=20)': None,
}


if __name__ == "__main__":
    gamma = gamma_from_he(H_E, g=G)[1]
    rows = np.load(os.path.join(_ROOT, "gate_i4_scaling_law_data.npy"), allow_pickle=True)
    print(f"Loaded {len(rows)} candidates from gate_i4_scaling_law_data.npy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_baseline = WaveSet(gamma, [A_MODE, B_MODE, C_MODE], [(0, 1, 2)], N=10, deg=300)
    A0_base = ws_baseline.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    e_tot = np.sum(np.real(A0_base * np.conj(A0_base)))

    print("Measuring Omega_slow(x) (8 points, RH-only triad)...")
    om_slow = omega_slow_vs_x(gamma, X_VALUES, e_tot)

    # Example 2/3's own x: energy fraction on EG(1,1) given its own
    # registered velocity, using the SAME e_tot convention as the rest of
    # this map (a=b=c=30m/s baseline) for direct comparability -- not a
    # per-example E_tot, since the map's x-axis is defined relative to
    # one fixed reference energy scale throughout.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ws_eg11 = WaveSet(gamma, [A_MODE, B_MODE, C_MODE, (1, 1, 1)], [(0, 1, 2), (0, 1, 3)], N=10, deg=300)
    A0_ex2 = ws_eg11.amplitudes_from_velocities([30.0, 30.0, 30.0, 80.0], H_E, g=G)
    A0_ex3 = ws_eg11.amplitudes_from_velocities([40.0, 40.0, 30.0, 20.0], H_E, g=G)
    e_tot_ex2 = np.sum(np.real(A0_ex2 * np.conj(A0_ex2)))
    e_tot_ex3 = np.sum(np.real(A0_ex3 * np.conj(A0_ex3)))
    EXAMPLES['Example 2 (u_d=80)'] = np.abs(A0_ex2[3]) ** 2 / e_tot_ex2
    EXAMPLES['Example 3 (u_d=20)'] = np.abs(A0_ex3[3]) ** 2 / e_tot_ex3
    for lbl, x in EXAMPLES.items():
        print(f"  {lbl}: x={x:.4f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 6))
    xs_all, ys_all, cs_all = [], [], []
    for r in rows:
        om_ratios = [4 * np.pi * abs(r['omega_d']) / om_slow[x] if om_slow[x] else np.nan for x in X_VALUES]
        xs_all.extend(X_VALUES)
        ys_all.extend(om_ratios)
        cs_all.extend([v * 100 for v in r['f2_vals']])

    sc = ax.scatter(xs_all, ys_all, c=cs_all, cmap='viridis',
                     norm=matplotlib.colors.LogNorm(vmin=max(1e-3, min(cs_all)), vmax=max(cs_all)),
                     s=30, edgecolors='none')
    for r in rows:
        if r['label'] in ('EG(1,1)', 'EG(7,9)'):
            om_ratios = [4 * np.pi * abs(r['omega_d']) / om_slow[x] if om_slow[x] else np.nan for x in X_VALUES]
            ax.plot(X_VALUES, om_ratios, '-', color='red' if r['label'] == 'EG(1,1)' else 'darkorange',
                    lw=1.3, alpha=0.8)
            # Label at the LEFT end (x=0.05): the right end is where
            # Example 2's own point/label sits (x=0.64), and the two
            # collided in an earlier draft of this figure.
            off = (-15, -18) if r['label'] == 'EG(1,1)' else (-42, 6)
            ax.annotate(r['label'], (X_VALUES[0], om_ratios[0]),
                        textcoords="offset points", xytext=off, fontsize=9,
                        color='red' if r['label'] == 'EG(1,1)' else 'darkorange')

    # Example points (all on the EG(1,1) curve). Distinct offsets per
    # label -- Examples 1 (x=0) and 3 (x=0.066) sit close together and
    # collided under one shared offset in an earlier draft.
    r_eg11 = next(r for r in rows if r['label'] == 'EG(1,1)')
    y_eg11 = {x: (4 * np.pi * abs(r_eg11['omega_d']) / om_slow[x] if om_slow[x] else np.nan) for x in X_VALUES}
    example_offsets = {
        'Example 1 (u_d=0)': (-10, 12),
        'Example 2 (u_d=80)': (-70, 10),
        'Example 3 (u_d=20)': (10, -14),
    }
    for lbl, x_ex in EXAMPLES.items():
        # Interpolate y at x_ex from the EG(1,1) curve's own 8 points.
        y_ex = np.interp(x_ex, X_VALUES, [y_eg11[x] for x in X_VALUES])
        ax.scatter([x_ex], [y_ex], marker='*', s=180, color='black', zorder=6)
        ax.annotate(lbl, (x_ex, y_ex), textcoords="offset points",
                    xytext=example_offsets[lbl], fontsize=8)

    ax.set_yscale('log')
    ax.set_xlabel('x (energy fraction on the gravity mode)')
    ax.set_ylabel(r'$\omega_d / \Omega_{slow}$ (timescale separation)')
    ax.set_title(r'Energy partition vs. timescale separation, colored by $\mathcal{F}_2$ (%, $t_f=20$d)')
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r'$\mathcal{F}_2$ (%)')
    fig.tight_layout()
    out_path = os.path.join(_ROOT, "gate_i2_map_calibrated.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved calibrated map to examples/gate_i2_map_calibrated.png")
