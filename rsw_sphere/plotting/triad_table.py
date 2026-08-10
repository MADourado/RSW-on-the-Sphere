"""Master table of resonant-triad properties for §2.2 ("Resonant Triads").

Batch-computes, for each ``TriadSpec`` in the registry (loaded from
``examples/triads_section_2_2.yaml`` via
``rsw_sphere.dynamics.triad_specs.load_triad_specs``): per-mode frequency,
linear period (days) and coupling coefficient; per-triad frequency mismatch
``delta``, the cubic-energy integral ``S_abc``, the pump mode (largest
|coupling coefficient|), and the energy-conservation residual
``(alpha^a_bc + alpha^b_ac - alpha^c_ab) + delta * S_abc``, which must be
~0 for every physically consistent triad -- a free correctness check.

This replaces hand-assembled tables: every number here comes from
``TRIAD`` (``rsw_sphere.dynamics.dynamic_triads``), so the paper/table
numbers cannot silently drift from the code that computes them.

Run from the command line (output written under ``outputs/figures/triads/``
by convention; nothing is written outside ``outputs/`` automatically):

    python rsw_sphere/plotting/triad_table.py outputs/figures/triads/table.tex
    python rsw_sphere/plotting/triad_table.py outputs/figures/triads/table.csv --fmt csv
    python -m rsw_sphere.plotting.triad_table outputs/figures/triads/table.md --fmt markdown

or import and call it from another script:

    from rsw_sphere.plotting.triad_table import triad_properties, triad_table
    from rsw_sphere.dynamics.triad_specs import load_triad_specs
    specs = load_triad_specs()
    props = triad_properties(specs['gravity_catalyst'].modes)
    triad_table(specs, fmt='latex', path='outputs/figures/triads/table.tex')
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rsw_sphere.physics import gamma_from_he, linear_period_days
from rsw_sphere.dynamics.dynamic_triads import TRIAD
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import symetry


def _mode_label(m, n, alpha):
    return {1: 'EIG', 2: 'WIG', 3: 'RH'}[alpha] + f'({m},{n})'


def triad_properties(modes, h_e: float = 10000, N: int = 10, deg: int = 300) -> dict:
    """Compute all properties of a resonant-triad example in one shot.

    Parameters
    ----------
    modes : sequence of 3 (m, n, alpha) int triples
        Mode a, b, c (see ``rsw_sphere.dynamics.triad_specs`` for the
        ``alpha`` convention).
    h_e : float, optional
        Equivalent height in metres. Default ``10000``.
    N : int, optional
        Hough-mode expansion truncation order. Default ``10``.
    deg : int, optional
        Gaussian-quadrature degree. **Must match the ``deg`` used inside
        ``norm_Hough``** -- ``inner_product`` assumes the same quadrature
        grid for all three modes. Default ``300`` (the value used
        throughout ``dynamic_three_waves.triad_evolution``); do **not**
        use ``TRIAD``'s own default of ``60``.

    Returns
    -------
    dict
        Keys: ``modes`` (labels), ``omega`` (3,), ``period_days`` (3,),
        ``coef`` (3,) [coupling coefficients alpha^a_bc, alpha^b_ac,
        alpha^c_ab], ``symmetric`` (3,) [bool], ``delta`` (mismatch),
        ``S_abc``, ``pump_index`` (0/1/2), ``pump_label``, ``residual``
        (energy-conservation residual, should be ~0).
    """
    eps, gamma = gamma_from_he(h_e)

    (m_a, n_a, alpha_a), (m_b, n_b, alpha_b), (m_c, n_c, alpha_c) = modes

    T = TRIAD(gamma, m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c, N, deg)

    omega = np.array([np.real(T.freq_a), np.real(T.freq_b), np.real(T.freq_c)])
    coef = np.array([np.real(T.coef_ABC), np.real(T.coef_BAC), np.real(T.coef_CAB)])
    delta = np.real(T.mismatch)
    S_abc = np.real(T.Sabc)

    # Energy-conservation residual: (alpha^a_bc + alpha^b_ac - alpha^c_ab) + delta*S_abc ~= 0
    residual = (coef[0] + coef[1] - coef[2]) + delta * S_abc

    pump_index = int(np.argmax(np.abs(coef)))
    labels = [_mode_label(*modes[0]), _mode_label(*modes[1]), _mode_label(*modes[2])]

    return {
        'modes': labels,
        'mnalpha': list(modes),
        'omega': omega,
        'period_days': linear_period_days(omega),
        'coef': coef,
        'symmetric': [symetry(*modes[0]), symetry(*modes[1]), symetry(*modes[2])],
        'delta': delta,
        'S_abc': S_abc,
        'pump_index': pump_index,
        'pump_label': labels[pump_index],
        'residual': residual,
        'h_e': h_e,
    }


def _fmt_num(x, sig=6):
    return f'{x:.{sig}g}'


def triad_table(specs, h_e: float = 10000, N: int = 10, deg: int = 300,
                 fmt: str = 'latex', path: str = None) -> str:
    """Build the master table over a dict of ``TriadSpec``.

    Parameters
    ----------
    specs : dict of str -> TriadSpec
        Typically the result of
        ``rsw_sphere.dynamics.triad_specs.load_triad_specs()`` (or a
        subset/union with ``TABLE1_TRIADS``).
    h_e : float, optional
        Equivalent height in metres, used only if a spec does not set its
        own. Default ``10000``.
    N, deg : int, optional
        See ``triad_properties``.
    fmt : {'latex', 'csv', 'markdown'}, optional
        Output format. ``'latex'`` produces a ``booktabs`` table grouped by
        triad (a multicolumn header row with delta/pump mode, then three
        mode rows). Default ``'latex'``.
    path : str or None, optional
        If given, the table text is written to this path. If ``None``
        (default), the table text is printed to stdout.

    Returns
    -------
    str
        The rendered table text (also written to ``path`` if given, or
        printed if ``path`` is ``None``).
    """
    rows = {key: triad_properties(spec.modes, h_e=spec.h_e or h_e, N=N, deg=deg)
            for key, spec in specs.items()}

    if fmt == 'csv':
        lines = ['triad,mode,m,n,alpha,omega,period_days,coef,symmetric,delta,S_abc,pump,residual']
        for key, p in rows.items():
            for i in range(3):
                m, n, alpha = p['mnalpha'][i]
                lines.append(','.join(str(v) for v in [
                    key, p['modes'][i], m, n, alpha,
                    p['omega'][i], p['period_days'][i], p['coef'][i],
                    p['symmetric'][i], p['delta'] if i == 0 else '',
                    p['S_abc'] if i == 0 else '',
                    p['pump_label'] if i == 0 else '',
                    p['residual'] if i == 0 else '',
                ]))
        text = '\n'.join(lines) + '\n'

    elif fmt == 'markdown':
        lines = ['| triad | mode | omega | period (d) | coef | pump | delta | residual |',
                 '|---|---|---|---|---|---|---|---|']
        for key, p in rows.items():
            for i in range(3):
                pump_cell = p['pump_label'] if i == 0 else ''
                delta_cell = _fmt_num(p['delta']) if i == 0 else ''
                residual_cell = _fmt_num(p['residual']) if i == 0 else ''
                lines.append(
                    f"| {key if i == 0 else ''} | {p['modes'][i]} | "
                    f"{_fmt_num(p['omega'][i])} | {_fmt_num(p['period_days'][i])} | "
                    f"{_fmt_num(p['coef'][i])} | {pump_cell} | {delta_cell} | {residual_cell} |"
                )
        text = '\n'.join(lines) + '\n'

    elif fmt == 'latex':
        lines = [
            r'\begin{table}',
            r'\centering',
            r'\begin{tabular}{llrrrr}',
            r'\toprule',
            r'Triad & Mode & $\omega$ & Period (d) & Coeff. & Pump \\',
            r'\midrule',
        ]
        for key, p in rows.items():
            lines.append(
                rf'\multicolumn{{6}}{{l}}{{\textbf{{{key}}}: '
                rf'$\delta={_fmt_num(p["delta"])}$, pump mode {p["pump_label"]}, '
                rf'residual $={_fmt_num(p["residual"])}$}} \\'
            )
            for i in range(3):
                lines.append(
                    f"& {p['modes'][i]} & {_fmt_num(p['omega'][i])} & "
                    f"{_fmt_num(p['period_days'][i])} & {_fmt_num(p['coef'][i])} & "
                    f"{'pump' if i == p['pump_index'] else ''} \\\\"
                )
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'
        lines += [r'\end{tabular}', r'\end{table}']
        text = '\n'.join(lines) + '\n'

    else:
        raise ValueError(f"unknown fmt: {fmt!r} (expected 'latex', 'csv', or 'markdown')")

    if path:
        with open(path, 'w') as f:
            f.write(text)
    else:
        print(text)

    return text


def main():
    import argparse
    from rsw_sphere.dynamics.triad_specs import (
        DEFAULT_SPECS_PATH, TABLE1_TRIADS, load_triad_specs,
    )

    parser = argparse.ArgumentParser(
        description="Generate the master resonant-triad table for §2.2 "
                    "('Resonant Triads') from a triad-registry YAML "
                    "(rsw_sphere.dynamics.triad_specs.load_triad_specs).")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output file path (e.g. outputs/figures/triads/table.tex). "
             "If omitted, the table is printed to stdout.")
    parser.add_argument(
        "--specs", default=DEFAULT_SPECS_PATH,
        help=f"path to the triad-registry YAML (default: {DEFAULT_SPECS_PATH}).")
    parser.add_argument(
        "--fmt", choices=["latex", "csv", "markdown"], default="latex",
        help="output format (default: latex).")
    parser.add_argument(
        "--include-table1", action="store_true",
        help="also include the Table 1 quasi-resonant (zero-coupling) triads.")
    parser.add_argument(
        "--deg", type=int, default=300,
        help="Gaussian-quadrature degree, must match norm_Hough's (default: 300).")
    args = parser.parse_args()

    specs = load_triad_specs(args.specs)
    if args.include_table1:
        specs.update(TABLE1_TRIADS)

    triad_table(specs, fmt=args.fmt, deg=args.deg, path=args.path)


if __name__ == "__main__":
    main()
