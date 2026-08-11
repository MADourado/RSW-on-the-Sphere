"""Master table of resonant-triad properties for §2.2 ("Resonant Triads").

Batch-computes, for each ``TriadSpec`` in the registry (loaded from
``examples/triads_section_2_2.yaml`` via
``rsw_sphere.dynamics.triad_specs.load_triad_specs``): per-mode frequency,
linear period (days) and coupling coefficient; per-triad frequency mismatch
``delta``, the cubic-energy integral ``S_abc``, and the pump mode (largest
|coupling coefficient|). The energy-conservation residual
``(alpha^a_bc + alpha^b_ac - alpha^c_ab) + delta * S_abc`` is also computed
and must be ~0 for every physically consistent triad -- kept as an
internal correctness check (``triad_properties`` warns if it is not ~0)
but not rendered in any output format (paper-review item 4).

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
    props = triad_properties(specs['triad_gravity_with_rossby_catalyst'].modes)
    triad_table(specs, fmt='latex', path='outputs/figures/triads/table.tex')
"""
import os
import sys
import warnings

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rsw_sphere.physics import gamma_from_he, linear_period_days
from rsw_sphere.dynamics.dynamic_triads import TRIAD
from rsw_sphere.hough_harmonics.eigenvalues_and_eigenvectors.eigenvectors import symetry
from rsw_sphere.plotting.labels import _mode_label, _fmt_num


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

    if abs(residual) > 1e-6:
        warnings.warn(
            f"triad_properties: energy-conservation residual {residual:.3g} "
            f"is not ~0 for modes {modes} -- this indicates a physically "
            f"inconsistent triad or a numerical-precision issue, not a "
            f"display concern (the residual itself is no longer rendered "
            f"in the master table, per paper-review item 4, but is kept as "
            f"an internal correctness check)."
        )

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


def _split_name_3lines(key, max_lines=3):
    """Word-wrap an underscore-delimited role key (e.g.
    ``triad_rossby_only_non_resonant``) into up to ``max_lines`` lines for
    display in the master table's leftmost column, by greedily packing
    ``_``-delimited tokens onto each line (balanced by running character
    count, not literally one token per line -- most role keys have more
    than 3 tokens).

    Returns
    -------
    list of str
        Up to ``max_lines`` strings (tokens joined by a space), suitable for
        joining with ``\\\\`` inside ``\\shortstack{...}``.
    """
    # Drop the common "triad_" prefix shared by every registry key -- it
    # carries no information in a table where every row is a triad, so
    # wrapping it would waste one of the 3 lines.
    stripped = key[len('triad_'):] if key.startswith('triad_') else key
    tokens = stripped.split('_')
    target_len = sum(len(t) for t in tokens) / max_lines
    lines, cur, cur_len = [], [], 0
    for tok in tokens:
        if cur and cur_len + len(tok) > target_len and len(lines) < max_lines - 1:
            lines.append(' '.join(cur))
            cur, cur_len = [], 0
        cur.append(tok)
        cur_len += len(tok)
    if cur:
        lines.append(' '.join(cur))
    return lines


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
        triad: the leftmost column (filled only on each triad's first mode
        row) holds the triad's role-key name word-wrapped to 3 lines, its
        "Triad A/B/C/D" ``display_label``, and its mismatch delta; the
        energy-conservation residual is not rendered (still computed and
        checked internally by ``triad_properties``, which warns if it is
        not ~0). Default ``'latex'``.
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
        lines = ['triad,display_label,mode,m,n,alpha,omega,period_days,coef,symmetric,delta,S_abc,pump']
        for key, p in rows.items():
            display_label = specs[key].display_label
            for i in range(3):
                m, n, alpha = p['mnalpha'][i]
                lines.append(','.join(str(v) for v in [
                    key, display_label if i == 0 else '', p['modes'][i], m, n, alpha,
                    p['omega'][i], p['period_days'][i], p['coef'][i],
                    p['symmetric'][i], p['delta'] if i == 0 else '',
                    p['S_abc'] if i == 0 else '',
                    p['pump_label'] if i == 0 else '',
                ]))
        text = '\n'.join(lines) + '\n'

    elif fmt == 'markdown':
        lines = ['| triad | mode | omega (dimensionless) | period (d) | coef | pump | delta |',
                 '|---|---|---|---|---|---|---|']
        for key, p in rows.items():
            display_label = specs[key].display_label
            name_cell = f"{key} ({display_label})" if display_label else key
            for i in range(3):
                pump_cell = p['pump_label'] if i == 0 else ''
                delta_cell = _fmt_num(p['delta']) if i == 0 else ''
                lines.append(
                    f"| {name_cell if i == 0 else ''} | {p['modes'][i]} | "
                    f"{_fmt_num(p['omega'][i])} | {_fmt_num(p['period_days'][i])} | "
                    f"{_fmt_num(p['coef'][i])} | {pump_cell} | {delta_cell} |"
                )
        text = '\n'.join(lines) + '\n'

    elif fmt == 'latex':
        # Column spec matches JFM-template.tex's copy of this table
        # (l|l|c|c|c|c): leftmost column (triad name/label/delta), mode
        # label, frequency, period, coupling coefficient, pump flag.
        # Headers are individually centered via \multicolumn{1}{c}{...}
        # (paper-review item 4); numeric body columns are also centered
        # (not right-aligned) per a later manual revision.
        #
        # Leftmost-column layout (revised 2026-08-11 to match a manual
        # edit made directly in JFM-template.tex): the bold "Triad X" tag
        # sits on its own row above the triad's 3 mode rows (a plain
        # `\textbf{...} \\` line, not a table cell -- LaTeX just leaves the
        # other columns of that row blank); each of the 3 mode rows then
        # carries one line of the wrapped role-key name (up to 2 lines) or
        # the mismatch delta (always the 3rd row), top-aligned rather than
        # vertically centered via \shortstack as in the previous version.
        # No \multirow/\makecell dependency either way.
        lines = [
            r'\begin{table}',
            r'\centering',
            r'\begin{tabular}{l|l|c|c|c|c}',
            r'\toprule',
            r'\multicolumn{1}{c}{Triad} & \multicolumn{1}{c}{Mode} & '
            r'\multicolumn{1}{c}{Frequency ($\omega$)} & '
            r'\multicolumn{1}{c}{Period (days)} & \multicolumn{1}{c}{Coeff.} & '
            r'\multicolumn{1}{c}{Pump} \\',
            r'\midrule',
        ]
        for key, p in rows.items():
            display_label = specs[key].display_label
            name_lines = _split_name_3lines(key, max_lines=2)
            delta_str = rf'$\delta={_fmt_num(p["delta"])}$'
            # Row 0 gets name_lines[0] (if any), row 1 gets name_lines[1]
            # (if any, else the delta if there's only one name line), row 2
            # always gets the delta unless it was already placed on row 1.
            left_cells = ['', '', delta_str]
            for i, line in enumerate(name_lines[:2]):
                left_cells[i] = line
            if len(name_lines) < 2:
                left_cells[1] = delta_str
                left_cells[2] = ''
            if display_label:
                lines.append(rf'\textbf{{{display_label}}} \\')
            for i in range(3):
                indent = ' ' * i
                lines.append(
                    f"{indent}{left_cells[i]} & {p['modes'][i]} & {_fmt_num(p['omega'][i])} & "
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
