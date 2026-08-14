"""Regenerate Table cap42/cap43 (Quartet C/D coefficient tables,
JFM-template.tex \\S sec: gravity) in Table cap41's own format --
Mode/Freq/Period/Coeff.1/Coeff.2/Zonal/A0 -- per
paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-section-3.3.md
Phase 2.

`rsw_sphere.plotting.wave_set_table.wave_set_properties` already
provides frequency/period/per-triad coefficients from the registry
(`examples/wave_sets_section_3.yaml`); this script adds the Zonal/A0
columns (via `WaveSet.amplitudes_from_velocities`, the same call Table
cap41's own generation used) and formats both quartets to match Table
cap41's column layout exactly, rather than extending the shared
`wave_set_table` LaTeX formatter (used by other wave sets too) with a
one-off column pair.

Run:

    python examples/regen_gravity_quartet_tables.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_REPO = os.path.dirname(_ROOT)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np

from rsw_sphere.physics import gamma_from_he, G
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
from rsw_sphere.plotting.wave_set_table import wave_set_properties


def build_table(spec, label, tab_label):
    p = wave_set_properties(spec)
    gamma = gamma_from_he(spec.h_e, g=G)[1]
    ws = WaveSet(gamma, list(spec.modes), [spec.triad_indices(i) for i in range(spec.n_triads())],
                 N=10, deg=300)
    A0 = ws.amplitudes_from_velocities(list(spec.velocities), spec.h_e, g=G)

    n_triads = p['triads'].__len__()
    letters = 'abcd'[:spec.n_modes()]
    lines = [
        r'\begin{table}',
        r'    \centering',
        rf'  % python examples/regen_gravity_quartet_tables.py (rsw_sphere.plotting.wave_set_table.wave_set_properties',
        r'    % + WaveSet.amplitudes_from_velocities, same generation discipline as Table \ref{tab: cap41})',
        r'    \begin{tabular}{|c|c|c|c|c|c|c|c|}\hline',
        r'    & Mode  & Freq. & Period (days) & Coeff.$_1$ &  Coeff.$_2$  & Zonal&$A_0$\\\hline',
    ]
    for i in range(spec.n_modes()):
        coef_cells = ' & '.join(
            '$-$' if np.isnan(p['coef'][i, t]) else f"${p['coef'][i, t]:.4f}$"
            for t in range(n_triads))
        lines.append(
            f"         {letters[i]} &  {p['mode_labels'][i]} "
            f"&${p['omega'][i]:.5f}$&${p['period_days'][i]:.3f}$& {coef_cells} "
            f"&{spec.velocities[i]:.0f}&${np.abs(A0[i]):.4f}$  \\\\")
    lines += [
        r'        \hline',
        r'    \end{tabular}',
        rf'    \caption{{{label}}}',
        rf'    \label{{tab: {tab_label}}}',
        r'\end{table}',
    ]
    return '\n'.join(lines)


if __name__ == "__main__":
    specs = load_wave_set_specs()

    print("=" * 20, "Table cap42 (Quartet C, EG(1,1))", "=" * 20)
    tex42 = build_table(
        specs['quartet_gravity_kelvin'],
        "Four-wave configuration containing the Kelvin-like gravity mode EG(1,1) "
        "(Quartet C, Figure \\ref{fig: topology_overview}c): each mode's frequency "
        "$\\omega$, linear period (days), zonal velocity (m/s) and amplitude $A_0$ "
        "(eq. \\ref{eq: Azonal}). Coeff.$_1$ is the RH-only triad (a,b,c); "
        "Coeff.$_2$ is the triad completed by EG(1,1) (a,b,d).",
        "cap42")
    print(tex42)

    print("\n" + "=" * 20, "Table cap43 (Quartet D, EG(7,9))", "=" * 20)
    tex43 = build_table(
        specs['quartet_gravity_79'],
        "Four-wave configuration containing the higher-frequency gravity mode "
        "EG(7,9) (Quartet D, Figure \\ref{fig: topology_overview}d): columns as "
        "in Table \\ref{tab: cap42}. Coeff.$_1$ is the RH-only triad (a,b,c); "
        "Coeff.$_2$ is the triad completed by EG(7,9), where EG(7,9) plays the "
        "SUM role ($m_d=m_a+m_b$), unlike Quartet C's EG(1,1).",
        "cap43")
    print(tex43)

    with open(os.path.join(_ROOT, "gravity_quartet_tables.tex"), 'w') as f:
        f.write(tex42 + "\n\n" + tex43 + "\n")
    print("\nSaved to examples/gravity_quartet_tables.tex")
