"""Properties tables for quartet/quintet ("wave set") examples in the
paper's merged §Coupled Triads section.

Batch-computes, for each ``WaveSetSpec`` in the registry (loaded from
``wave_sets_default.yaml`` via
``rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs``), every mode's
frequency and linear period, and -- since a wave set couples through
*multiple* constituent triads -- one coupling-coefficient column per
triad (``NaN`` where a mode isn't in that triad, matching the
dissertation's own ``tab: cap41``/``cap42``/``cap43`` layout: "Coeff. 1",
"Coeff. 2", ...). Each triad's own mismatch/S/pump mode/energy-
conservation residual is also computed (checked internally, never
rendered) -- but note there is no wave-set-*level* residual (a
quartet/quintet does not conserve energy; see
``rsw_sphere.dynamics.wave_sets``'s module docstring).

Run from the command line (output written under ``outputs/figures/wave_sets/``
by convention):

    python rsw_sphere/plotting/wave_set_table.py outputs/figures/wave_sets/table.tex
    python rsw_sphere/plotting/wave_set_table.py outputs/figures/wave_sets/table.csv --fmt csv

or import and call it from another script:

    from rsw_sphere.plotting.wave_set_table import wave_set_properties, wave_set_table
    from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
    specs = load_wave_set_specs()
    props = wave_set_properties(specs['quartet_rh_preference'])
    wave_set_table(specs, fmt='latex', path='outputs/figures/wave_sets/table.tex')

``wave_set_master_table`` (below) instead combines several wave sets that
share the same triad count (e.g. every quartet) into ONE
``\\begin{table}``, mirroring ``tab: master``'s own hand-merged,
multi-group style rather than one block per wave set -- see
``examples/tables/paper_table02_quartet_master.py``.
"""
import warnings


import numpy as np

from rsw_sphere.physics import gamma_from_he, linear_period_days
from rsw_sphere.dynamics.wave_sets import WaveSet
from rsw_sphere.plotting.labels import _mode_label, _fmt_num


def wave_set_properties(spec, N: int = 10, deg: int = 300) -> dict:
    """Compute all properties of a quartet/quintet example in one shot.

    Parameters
    ----------
    spec : rsw_sphere.dynamics.wave_set_specs.WaveSetSpec
    N, deg : int, optional
        See ``rsw_sphere.dynamics.wave_sets.WaveSet``. Default ``N=10,
        deg=300``, the resolution established as sufficient in section 2.2.

    Returns
    -------
    dict
        ``mode_labels`` (paper-facing EG/WG/RH strings, per mode),
        ``omega``/``period_days`` (per mode, shape ``(n_modes,)``),
        ``coef`` (shape ``(n_modes, n_triads)``, NaN where a mode isn't in
        that triad), ``triads`` (list of per-triad dicts: ``sum_key``,
        ``member_keys``, ``display_label``, ``delta``, ``S``,
        ``pump_key``, ``pump_label``, ``residual``).
    """
    gamma = gamma_from_he(spec.h_e)[1]
    triad_indices = [spec.triad_indices(i) for i in range(spec.n_triads())]
    ws = WaveSet(gamma, spec.modes, triad_indices, N=N, deg=deg)

    mode_labels = [_mode_label(*m) for m in spec.modes]
    coef = np.full((spec.n_modes(), spec.n_triads()), np.nan, dtype=complex)
    triads_out = []
    for t, (i_sum, i_p, i_q) in enumerate(triad_indices):
        coef[i_p, t] = ws.alpha[t, 0]
        coef[i_q, t] = ws.alpha[t, 1]
        coef[i_sum, t] = ws.alpha[t, 2]

        idx_local = [i_p, i_q, i_sum]
        pump_local = int(np.argmax(np.abs(ws.alpha[t])))
        pump_index = idx_local[pump_local]

        if abs(ws.residual[t]) > 1e-6:
            warnings.warn(
                f"wave_set_properties: energy-conservation residual "
                f"{ws.residual[t]:.3g} not ~0 for {spec.key} triad {t} "
                f"({spec.triads[t].sum_key}, {spec.triads[t].member_keys}) "
                f"-- physically inconsistent triad or numerical-precision "
                f"issue, not a display concern.")

        triads_out.append({
            'sum_key': spec.triads[t].sum_key,
            'member_keys': spec.triads[t].member_keys,
            'display_label': spec.triads[t].display_label,
            'delta': np.real(ws.delta[t]),
            'S': np.real(ws.S[t]),
            'pump_index': pump_index,
            'pump_key': spec.mode_keys[pump_index],
            'pump_label': mode_labels[pump_index],
            'residual': np.real(ws.residual[t]),
        })

    return {
        'mode_labels': mode_labels,
        'mode_keys': spec.mode_keys,
        'mnalpha': list(spec.modes),
        'omega': np.real(ws.omega),
        'period_days': linear_period_days(np.real(ws.omega)),
        'coef': np.real(coef),
        'triads': triads_out,
        'h_e': spec.h_e,
    }


def wave_set_table(specs, N: int = 10, deg: int = 300,
                    fmt: str = 'latex', path: str = None) -> str:
    """Build one properties table per wave set in ``specs`` (concatenated),
    matching the dissertation's per-quartet ``tab: cap41``/``cap42``/
    ``cap43`` layout rather than a single combined table (wave sets have
    differing numbers of modes/triads, unlike §2.2's uniformly-3-mode
    triads).

    Parameters
    ----------
    specs : dict of str -> WaveSetSpec
    N, deg : int, optional
        See ``wave_set_properties``.
    fmt : {'latex', 'csv', 'markdown'}, optional
    path : str or None, optional
        If given, written there; else printed to stdout.

    Returns
    -------
    str
    """
    rows = {key: wave_set_properties(spec, N=N, deg=deg) for key, spec in specs.items()}

    if fmt == 'csv':
        lines = ['wave_set,display_label,mode,m,n,alpha,omega,period_days'
                 + ',coef_triad' + ',triad_display,triad_delta,triad_S,triad_pump']
        for key, p in rows.items():
            spec = specs[key]
            n_triads = len(p['triads'])
            for i in range(spec.n_modes()):
                coefs = ';'.join(
                    '' if np.isnan(p['coef'][i, t]) else _fmt_num(p['coef'][i, t])
                    for t in range(n_triads))
                lines.append(','.join(str(v) for v in [
                    key, spec.display_label if i == 0 else '', p['mode_labels'][i],
                    *p['mnalpha'][i], _fmt_num(p['omega'][i]), _fmt_num(p['period_days'][i]),
                    coefs, '', '', '', '',
                ]))
            for t, tri in enumerate(p['triads']):
                lines.append(','.join(str(v) for v in [
                    key, '', f"[triad {t}]", '', '', '', '', '', '',
                    tri['display_label'], _fmt_num(tri['delta']), _fmt_num(tri['S']),
                    tri['pump_label'],
                ]))
        text = '\n'.join(lines) + '\n'

    elif fmt == 'markdown':
        blocks = []
        for key, p in rows.items():
            spec = specs[key]
            n_triads = len(p['triads'])
            coef_headers = ' | '.join(f"Coeff. {t + 1}" for t in range(n_triads))
            lines = [f"### {spec.display_label} (`{key}`)", '',
                     f"| Mode | $\\omega$ | Period (d) | {coef_headers} |",
                     '|---|---|---|' + '---|' * n_triads]
            for i in range(spec.n_modes()):
                coef_cells = ' | '.join(
                    '' if np.isnan(p['coef'][i, t]) else _fmt_num(p['coef'][i, t])
                    for t in range(n_triads))
                lines.append(f"| {p['mode_labels'][i]} | {_fmt_num(p['omega'][i])} | "
                             f"{_fmt_num(p['period_days'][i])} | {coef_cells} |")
            lines.append('')
            for t, tri in enumerate(p['triads']):
                lines.append(f"- Triad {t + 1} ({tri['display_label']}): "
                             f"$\\delta={_fmt_num(tri['delta'])}$, pump={tri['pump_label']}")
            blocks.append('\n'.join(lines))
        text = '\n\n'.join(blocks) + '\n'

    elif fmt == 'latex':
        # One \begin{table}...\end{table} block per wave set. Column count
        # varies with the wave set's own triad count (2 for a quartet, 3
        # for the quintet), so -- unlike §2.2's single uniform master
        # table -- each wave set gets its own table rather than being
        # concatenated into rows of one shared tabular.
        blocks = []
        for key, p in rows.items():
            spec = specs[key]
            n_triads = len(p['triads'])
            col_spec = 'l|c|c' + '|c' * n_triads + '|c'
            coef_headers = ' & '.join(
                rf'\multicolumn{{1}}{{c}}{{Coeff.\ {t + 1}}}' for t in range(n_triads))
            lines = [
                r'\begin{table}',
                r'\centering',
                rf'\begin{{tabular}}{{{col_spec}}}',
                r'\toprule',
                r'\multicolumn{1}{c}{Mode} & \multicolumn{1}{c}{Frequency ($\omega$)} & '
                r'\multicolumn{1}{c}{Period (days)} & ' + coef_headers +
                r' & \multicolumn{1}{c}{Pump} \\',
                r'\midrule',
            ]
            for i in range(spec.n_modes()):
                coef_cells = ' & '.join(
                    '' if np.isnan(p['coef'][i, t]) else f"${_fmt_num(p['coef'][i, t])}$"
                    for t in range(n_triads))
                pump_flags = [t for t, tri in enumerate(p['triads']) if tri['pump_index'] == i]
                pump_cell = ','.join(str(t + 1) for t in pump_flags) if pump_flags else ''
                lines.append(
                    f"{p['mode_labels'][i]} & ${_fmt_num(p['omega'][i])}$ & "
                    f"${_fmt_num(p['period_days'][i])}$ & {coef_cells} & {pump_cell} \\\\")
            lines.append(r'\midrule')
            # Note: joined with ", " not "&" -- a raw "&" inside a
            # \multicolumn{...}{...}{TEXT} argument is a LaTeX alignment-
            # tab error, not a literal ampersand (caught by rendering this
            # table's first draft and finding "Misplaced alignment tab").
            delta_cells = ', '.join(rf'$\delta_{{{t + 1}}}={_fmt_num(tri["delta"])}$'
                                     for t, tri in enumerate(p['triads']))
            lines.append(
                rf'\multicolumn{{{3 + n_triads + 1}}}{{l}}{{\textbf{{{spec.display_label}}}: {delta_cells}}} \\')
            lines += [r'\bottomrule', r'\end{tabular}',
                      rf'\caption{{{spec.label}}}', r'\end{table}']
            blocks.append('\n'.join(lines))
        text = '\n\n'.join(blocks) + '\n'

    else:
        raise ValueError(f"unknown fmt: {fmt!r} (expected 'latex', 'csv', or 'markdown')")

    if path:
        with open(path, 'w') as f:
            f.write(text)
    else:
        print(text)

    return text


def wave_set_master_table(specs, N: int = 10, deg: int = 300,
                           fmt: str = 'latex', path: str = None,
                           caption: str = '', label: str = 'quartet_master') -> str:
    """ONE combined table across several wave sets that all share the same
    triad count (e.g. every quartet here: 2 triads each) -- mirroring
    Table ``tab: master``'s own hand-merged, multi-group style (one
    \\midrule-separated block per wave set, group name + per-triad
    mismatch as a header row) rather than ``wave_set_table``'s one-
    ``\\begin{table}``-per-wave-set layout, which doesn't scale to
    several quartets shown side by side. Reuses ``wave_set_properties``
    unchanged -- this only changes how the same per-wave-set data is
    concatenated/rendered.

    Parameters
    ----------
    specs : dict of str -> WaveSetSpec, in the order groups should appear.
    N, deg : int, optional
        See ``wave_set_properties``.
    fmt : {'latex', 'csv'}, optional
    path : str or None, optional
        If given, written there; else printed to stdout.
    caption, label : str, optional
        LaTeX-only: table caption and ``\\label{tab: <label>}`` name.

    Returns
    -------
    str
    """
    rows = {key: wave_set_properties(spec, N=N, deg=deg) for key, spec in specs.items()}
    n_triads_set = {len(p['triads']) for p in rows.values()}
    if len(n_triads_set) != 1:
        raise ValueError(
            "wave_set_master_table requires every wave set to have the same "
            f"triad count, got {[(k, len(p['triads'])) for k, p in rows.items()]}")
    n_triads = n_triads_set.pop()

    if fmt == 'csv':
        lines = ['wave_set,display_label,mode,m,n,alpha,omega,period_days,coef_triad,pump']
        for key, spec in specs.items():
            p = rows[key]
            for i in range(spec.n_modes()):
                coefs = ';'.join(
                    '' if np.isnan(p['coef'][i, t]) else _fmt_num(p['coef'][i, t])
                    for t in range(n_triads))
                pump_flags = [t for t, tri in enumerate(p['triads']) if tri['pump_index'] == i]
                pump_cell = ';'.join(str(t + 1) for t in pump_flags)
                lines.append(','.join(str(v) for v in [
                    key, spec.display_label if i == 0 else '', p['mode_labels'][i],
                    *p['mnalpha'][i], _fmt_num(p['omega'][i]), _fmt_num(p['period_days'][i]),
                    coefs, pump_cell,
                ]))
        text = '\n'.join(lines) + '\n'

    elif fmt == 'latex':
        col_spec = 'l|c|c' + '|c' * n_triads + '|c'
        coef_headers = ' & '.join(
            rf'\multicolumn{{1}}{{c}}{{Coeff.\ {t + 1}}}' for t in range(n_triads))
        lines = [
            r'\begin{table}',
            r'\centering',
            rf'\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
            r'\multicolumn{1}{c}{Mode} & \multicolumn{1}{c}{Frequency ($\omega$)} & '
            r'\multicolumn{1}{c}{Period (days)} & ' + coef_headers +
            r' & \multicolumn{1}{c}{Pump} \\',
            r'\midrule',
        ]
        for key, spec in specs.items():
            p = rows[key]
            delta_cells = ', '.join(rf'$\delta_{{{t + 1}}}={_fmt_num(tri["delta"])}$'
                                     for t, tri in enumerate(p['triads']))
            lines.append(
                rf'\multicolumn{{{3 + n_triads + 1}}}{{l}}{{\textbf{{{spec.display_label}}}: '
                rf'{delta_cells}}} \\')
            for i in range(spec.n_modes()):
                coef_cells = ' & '.join(
                    '' if np.isnan(p['coef'][i, t]) else f"${_fmt_num(p['coef'][i, t])}$"
                    for t in range(n_triads))
                pump_flags = [t for t, tri in enumerate(p['triads']) if tri['pump_index'] == i]
                pump_cell = ','.join(str(t + 1) for t in pump_flags) if pump_flags else ''
                lines.append(
                    f"{p['mode_labels'][i]} & ${_fmt_num(p['omega'][i])}$ & "
                    f"${_fmt_num(p['period_days'][i])}$ & {coef_cells} & {pump_cell} \\\\")
            lines.append(r'\midrule')
        lines[-1] = r'\bottomrule'
        lines += [r'\end{tabular}',
                  rf'\caption{{{caption}}}',
                  rf'\label{{tab: {label}}}',
                  r'\end{table}']
        text = '\n'.join(lines) + '\n'

    else:
        raise ValueError(f"unknown fmt: {fmt!r} (expected 'latex' or 'csv')")

    if path:
        with open(path, 'w') as f:
            f.write(text)
    else:
        print(text)

    return text


def main():
    import argparse
    from rsw_sphere.dynamics.wave_set_specs import DEFAULT_WAVESETS_PATH, load_wave_set_specs

    parser = argparse.ArgumentParser(
        description="Generate the quartet/quintet properties table(s) for "
                    "§Coupled Triads from the wave-set registry YAML "
                    "(rsw_sphere.dynamics.wave_set_specs.load_wave_set_specs).")
    parser.add_argument(
        "path", nargs="?", default=None,
        help="output file path (e.g. outputs/figures/wave_sets/table.tex). "
             "If omitted, printed to stdout.")
    parser.add_argument(
        "--specs", default=DEFAULT_WAVESETS_PATH,
        help=f"path to the wave-set registry YAML (default: {DEFAULT_WAVESETS_PATH}).")
    parser.add_argument(
        "--wave-set", default=None,
        help="if given, only this one role key from the registry.")
    parser.add_argument(
        "--fmt", choices=["latex", "csv", "markdown"], default="latex",
        help="output format (default: latex).")
    parser.add_argument(
        "--deg", type=int, default=300,
        help="Gaussian-quadrature degree, must match norm_Hough's (default: 300).")
    parser.add_argument(
        "--N", type=int, default=10,
        help="Hough truncation order (default: 10).")
    args = parser.parse_args()

    specs = load_wave_set_specs(args.specs)
    if args.wave_set:
        if args.wave_set not in specs:
            parser.error(f"'{args.wave_set}' not in registry: {', '.join(specs)}")
        specs = {args.wave_set: specs[args.wave_set]}

    wave_set_table(specs, N=args.N, deg=args.deg, fmt=args.fmt, path=args.path)


if __name__ == "__main__":
    main()
