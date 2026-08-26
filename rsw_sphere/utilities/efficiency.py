"""Efficiency diagnostic, generalized for wave sets that don't conserve
energy exactly (quartets/quintets) -- normalizes by mean total energy
instead of initial total energy, gated on drift.
"""


def wave_set_efficiency(E_target, E_total, drift: float, drift_max: float = 0.1):
    """(E_target.max()-E_target.min()) / E_total.mean(); NaN if drift > drift_max.

    Reduces to the classic triad formula (normalize by E_total(0)) for a
    plain triad, since E_total barely varies there.
    """
    if drift > drift_max:
        return float("nan")
    denom = E_total.mean()
    if denom <= 0:
        return float("nan")
    return (E_target.max() - E_target.min()) / denom
