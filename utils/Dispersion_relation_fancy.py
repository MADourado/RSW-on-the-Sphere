"""Dispersion relation of the normal modes (Hough harmonics) of the linear
rotating shallow-water equations on the sphere.

Run from the command line:

    python utils/Dispersion_relation_fancy.py output.png
    python utils/Dispersion_relation_fancy.py output.png --he 5000
    python -m utils.Dispersion_relation_fancy output.png

or import and call it from another script:

    from utils.Dispersion_relation_fancy import dispersion_relation
    dispersion_relation(h_e=10000, path="output.png")

See README_dispersion.md for the mathematics and a description of the output.
"""
import os
import sys

# Make the repository root importable so that "Hough_Harmonics" resolves
# whether this file is run directly (python utils/Dispersion_relation_fancy.py),
# run as a module (python -m utils.Dispersion_relation_fancy), or imported.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib.pyplot as plt
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Matrix_m0 import matriz_C, matriz_D
from Hough_Harmonics.Eigenvalues_and_eigenvectors.Matrix_system import matriz_A, matriz_B


def dispersion_relation(h_e: float = 10000, path: str = None):
    """Compute and plot the RSW-on-the-sphere dispersion relation.

    Parameters
    ----------
    h_e : float, optional
        Equivalent height (equivalent depth) in metres. Sets Lamb's parameter
        ``eps = 4 a^2 Omega^2 / (g h_e)`` and the gravity-wave speed
        ``c = sqrt(g h_e)``. Default ``10000`` (~barotropic).
    path : str or None, optional
        If given, the figure is saved to this path (PNG, 200 dpi) and the
        figure is closed. If ``None`` (default), the figure is shown
        interactively with ``plt.show()``.

    Returns
    -------
    None
    """
    g     = 9.8
    a     = 6.38e+06
    Omega = 2*np.pi/24/60/60

    eps   = (4*a*a*Omega*Omega)/(g*h_e)
    gamma = 1/np.sqrt(eps)

    N = 3
    M = np.arange(1, 11)
    s = 2*Omega*1e+4   # scale: plot units are 10⁻⁴ rad/s

    eigen_M = []
    for m in M:
        AA = matriz_A(m, gamma, N)
        BB = matriz_B(m, gamma, N)
        eigen_a, _ = np.linalg.eig(AA)
        eigen_b, _ = np.linalg.eig(BB)
        eigen = np.sort(np.concatenate((eigen_a, eigen_b)))
        eigen_M.append(eigen * s)
    eigen_M = np.array(eigen_M)   # shape (10, 18)

    CC = matriz_C(gamma, N)
    DD = matriz_D(gamma, N)
    eigen_c, _ = np.linalg.eig(CC)
    eigen_d, _ = np.linalg.eig(DD)
    eigen0 = np.sort(np.concatenate((np.sqrt(eigen_c), np.sqrt(eigen_d))))[1:7] * s

    M = np.concatenate((np.array([0]), M))   # shape (11,): m = 0..10

    # ── physical reference scales ──────────────────────────────────────────
    c    = np.sqrt(g * h_e)
    f0_p = 2 * Omega / 1e-4           # 2Ω in plot units (~1.45)

    YLIM = (0, 3.5)

    # ── style ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 11,
        'axes.linewidth': 0.8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': False,
    })

    blue  = '#1a5fa8'
    green = '#2ca02c'
    red   = '#c0392b'
    grey  = '0.55'

    fig, ax = plt.subplots(figsize=(6.5, 6.5))   # square

    # ── EIG ────────────────────────────────────────────────────────────────
    n = 12   # Kelvin  (n-m = 0)
    K_kelvin = np.concatenate(([eigen0[n-12]], eigen_M[:, n]))
    ax.plot(M, K_kelvin, '-', lw=1.8, color=blue)

    n = 13   # Mixed Rossby-Gravity eastward  (n-m = 1)
    K_mrg_e = np.concatenate(([eigen0[n-12]], eigen_M[:, n]))
    ax.plot(M, K_mrg_e, '--', lw=1.8, color=green)

    eig_curves = {0: K_kelvin}              # Kelvin is EIG n-m = 0
    for i, n in enumerate(range(14, 18)):   # cols 14..17 -> n-m = 2, 3, 4, 5
        K = np.concatenate(([eigen0[n-12]], eigen_M[:, n]))
        ax.plot(M, K, '-', lw=1.5, color=blue, alpha=0.65)
        eig_curves[i+2] = K

    # ── WIG ────────────────────────────────────────────────────────────────
    # cols 2..5 of the sorted array are WIG n-m = 3, 2, 1, 0 (see Hough_coef)
    wig_curves = {}
    for n in range(2, 6):
        K = np.concatenate(([eigen0[7-n]], -eigen_M[:, n]))
        ax.plot(-M, K, '-', lw=1.5, color=blue, alpha=0.65)
        wig_curves[5 - n] = K            # col 2->nm 3, 3->2, 4->1, 5->0

    # ── RH ─────────────────────────────────────────────────────────────────
    K_mrg_w = np.concatenate(([eigen0[1]], -eigen_M[:, 6]))
    ax.plot(-M, K_mrg_w, '--', lw=1.8, color=green)

    # cols 7..11 of the sorted array are RH n-m = 1, 2, 3, 4, 5 (see Hough_coef)
    rh_curves = {}
    for n in range(7, 12):
        K = np.concatenate(([0], -eigen_M[:, n]))
        ax.plot(-M, K, '-', lw=1.0, color=red)
        rh_curves[n - 6] = K             # col 7->nm 1, ... , 11->nm 5

    # ── asymptotic reference line: short IGW ω ≈ c·k ────────────────────
    m_pos = np.linspace(0.3, 10, 300)
    omega_short = c * m_pos / a / 1e-4
    mask = omega_short <= YLIM[1]
    ax.plot( m_pos[mask], omega_short[mask], ':', lw=1.1, color=grey)
    ax.plot(-m_pos[mask], omega_short[mask], ':', lw=1.1, color=grey)
    m_label = m_pos[mask][-1]
    ax.text(m_label + 0.35, YLIM[1] - 0.15, r'$\omega\approx ck$',
            fontsize=8, color=grey, ha='left', va='top')

    # ── 2Ω horizontal reference (minimum frequency of inertia-gravity waves) ──
    ax.axhline(f0_p, color='0.4', linestyle='--', lw=0.8)
    ax.text(-9.7, f0_p + 0.08, r'$2\Omega$', fontsize=9.5, color='0.35', ha='left')

    # ── helper: x where curve crosses the top of the plot ────────────────
    def x_at_top(x_arr, y_arr, ytop=YLIM[1]):
        x_arr, y_arr = np.asarray(x_arr), np.asarray(y_arr)
        for i in range(len(y_arr) - 1):
            if y_arr[i] <= ytop < y_arr[i+1]:
                t = (ytop - y_arr[i]) / (y_arr[i+1] - y_arr[i])
                return x_arr[i] + t * (x_arr[i+1] - x_arr[i])
        # curve never reaches top — return last x within range
        below = x_arr[y_arr <= ytop]
        return below[-1] if len(below) else None

    # ── n–m labels above the plot (at the top edge of each branch) ────────
    ytop_label = YLIM[1] + 0.08
    for nm, K in eig_curves.items():
        xc = x_at_top(M, K)
        if xc is not None:
            ax.text(xc, ytop_label, f'$n\\!-\\!m={nm}$', fontsize=6.5,
                    color=blue, ha='center', va='bottom', clip_on=False,
                    rotation=70)
    # MRG eastward (green) is EIG n-m = 1
    xc = x_at_top(M, K_mrg_e)
    if xc is not None:
        ax.text(xc, ytop_label, r'$n\!-\!m=1$', fontsize=6.5,
                color=green, ha='center', va='bottom', clip_on=False,
                rotation=70)
    for nm, K in wig_curves.items():
        xc = x_at_top(-M, K)
        if xc is not None:
            ax.text(xc, ytop_label, f'$n\\!-\\!m={nm}$', fontsize=6.5,
                    color=blue, ha='center', va='bottom', clip_on=False,
                    rotation=70)
    for j, nm in enumerate([5, 1]):   # 5 = bottom (lowest |ω|), 1 = top red RH line
        ax.text(-10.15, 0.10 + 0.12 * j, f'$n\\!-\\!m={nm}$',
                fontsize=6.5, color=red, ha='right', va='center', clip_on=False)

    # ── direct curve annotations ──────────────────────────────────────────
    bbox_white = dict(boxstyle='round,pad=0.15', fc='white', ec='none', alpha=0.85)

    # Kelvin
    ax.annotate('Kelvin', xy=(2, K_kelvin[2]),
                xytext=(3.2, K_kelvin[2] - 0.5), fontsize=9, color=blue,
                fontweight='bold', bbox=bbox_white,
                arrowprops=dict(arrowstyle='->', color=blue, lw=0.8))

    # EIG / WIG region labels — free-floating among each fan (no arrows)
    ax.text( 2.7, 3.12, 'East\nInertia-Gravity', fontsize=9, color=blue,
            fontweight='bold', ha='center', va='center',
            multialignment='center', bbox=bbox_white)
    ax.text(-2.7, 3.12, 'West\nInertia-Gravity', fontsize=9, color=blue,
            fontweight='bold', ha='center', va='center',
            multialignment='center', bbox=bbox_white)

    # Mixed Rossby-Gravity (Yanai) — single label pointing to the eastward
    # branch (same physical mode also has a westward branch, RH n-m = 0).
    # Curved arrow so it reaches MRG-E without crossing the Kelvin line.
    ax.annotate('Mixed Rossby-Gravity\n(Yanai)', xy=(4, K_mrg_e[4]),
                xytext=(5.4, 1.45), fontsize=9, color=green, fontweight='bold',
                bbox=bbox_white, ha='center', multialignment='center',
                arrowprops=dict(arrowstyle='->', color=green, lw=0.8,
                                connectionstyle='arc3,rad=0.35'))

    # Rossby-Haurwitz
    rh_mid  = rh_curves[4]
    idx_rh  = np.argmax(rh_mid)
    ax.annotate('Rossby-Haurwitz', xy=(-M[idx_rh], rh_mid[idx_rh]),
                xytext=(-7.5, rh_mid[idx_rh] + 0.35), fontsize=9, color=red,
                fontweight='bold', bbox=bbox_white,
                arrowprops=dict(arrowstyle='->', color=red, lw=0.8))

    # ── Eastward / Westward labels below x-axis ───────────────────────────
    ax.text(-5, -0.28, '← Westward', ha='center', fontsize=9,
            color='0.4', transform=ax.transData, clip_on=False)
    ax.text( 5, -0.28, 'Eastward →', ha='center', fontsize=9,
            color='0.4', transform=ax.transData, clip_on=False)

    # ── axes ──────────────────────────────────────────────────────────────
    ax.set_xlim(-10, 10)
    ax.set_ylim(*YLIM)
    ax.set_xticks(np.arange(-10, 11, 2))
    ax.set_xlabel('Zonal wavenumber $m$', fontsize=12, labelpad=18)
    ax.set_ylabel(r'Frequency $\omega\ (10^{-4}\ \mathrm{rad\ s^{-1}})$', fontsize=12)
    ax.axvline(0, color='k', linewidth=0.6, linestyle=':')

    # ── period axis (right) ────────────────────────────────────────────────
    secax = ax.twinx()
    secax.set_ylim(*YLIM)
    # tick at round-number PERIODS (days) -> avoids the repeated "0.2" that
    # appears when ticking at round frequencies (period varies slowly at high ω)
    period_ticks = np.array([2, 1, 0.75, 0.5, 0.4, 0.3, 0.25])
    freq_at_P = 2*np.pi / (period_ticks * 86400) / 1e-4
    secax.set_yticks(freq_at_P)
    secax.set_yticklabels([f'{p:g}' for p in period_ticks])
    secax.set_ylabel('Period (days)', fontsize=12)
    secax.tick_params(direction='in')

    fig.tight_layout()

    if path:
        fig.savefig(path, dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot the dispersion relation of the normal modes of the "
                    "rotating shallow-water equations on the sphere.")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output image path (e.g. dispersion_exp.png). "
             "If omitted, the figure is shown interactively.")
    parser.add_argument(
        "--he", "--equivalent-height", dest="he", type=float, default=10000,
        help="equivalent height in metres (default: 10000).")
    args = parser.parse_args()

    dispersion_relation(h_e=args.he, path=args.path)
