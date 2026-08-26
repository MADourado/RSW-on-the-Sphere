"""Shared mode-labelling and number-formatting helpers for the wave-set
(quartet/quintet) plotting/table modules.

``_mode_label`` implements one rule: the paper's prose consistently uses
**EG/WG** for eastward/westward inertia-gravity modes, not the code's
internal **EIG/WIG** shorthand (``TRIAD.label_a/b/c``) -- so paper-facing
labels must be generated here, not read off ``TRIAD`` directly.

Run as a quick sanity check:

    python -m rsw_sphere.plotting.labels
"""


def _mode_label(m, n, alpha):
    """Paper-facing mode label, e.g. ``RH(4,5)``, ``EG(1,1)``, ``WG(2,4)``.

    Parameters
    ----------
    m, n : int
        Zonal/total wavenumber.
    alpha : int
        Wave family: ``1``=eastward inertia-gravity (EG), ``2``=westward
        inertia-gravity (WG), ``3``=Rossby-Haurwitz (RH).

    Returns
    -------
    str

    Examples
    --------
    >>> _mode_label(4, 5, 3)
    'RH(4,5)'
    >>> _mode_label(1, 1, 1)
    'EG(1,1)'
    >>> _mode_label(2, 4, 2)
    'WG(2,4)'
    """
    return {1: 'EG', 2: 'WG', 3: 'RH'}[alpha] + f'({m},{n})'


def _fmt_num(x, sig=6):
    """Format a float to ``sig`` significant digits for LaTeX/CSV/markdown
    table cells.

    Examples
    --------
    >>> _fmt_num(-0.0994375)
    '-0.0994375'
    >>> _fmt_num(3.21205123, sig=3)
    '3.21'
    """
    return f'{x:.{sig}g}'


if __name__ == "__main__":
    import doctest
    failures, _ = doctest.testmod()
    if failures == 0:
        print("labels doctest OK")
