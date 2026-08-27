"""Efficiency diagnostic, generalized for wave sets that don't conserve
energy exactly (quartets/quintets) -- normalizes by mean total energy
instead of initial total energy, gated on drift.
"""
import math

#: Velocity-sweep range caps by mode family: Rossby (RH, alpha=3) modes are
#: swept up to 100 m/s (jet-stream-strength winds); gravity (EG/WG,
#: alpha=1/2) modes are capped at 50 m/s (realistic Kelvin/inertia-gravity
#: wind-anomaly amplitudes).
RH_VELOCITY_RANGE = (0.0, 100.0)
EG_VELOCITY_RANGE = (0.0, 50.0)


def default_velocity_range(alpha):
    """Mode-family-aware default sweep range (m/s): RH_VELOCITY_RANGE for
    alpha=3 (Rossby-Haurwitz), EG_VELOCITY_RANGE for alpha=1/2 (EIG/WIG).
    """
    return RH_VELOCITY_RANGE if alpha == 3 else EG_VELOCITY_RANGE


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


def efficiency_variation(efficiency_full, efficiency_sub):
    """% change in efficiency, full wave set vs. one reference sub-triad
    -- same formula as the P-measure (`rsw_sphere.utilities.pmeasure`),
    applied to `wave_set_efficiency`'s own drift-gated, mean-total-energy
    -normalized quantity instead of the raw energy variation dEK. This
    answers a different question than P, and its sign can genuinely
    differ: the reference triad has strictly fewer active modes than the
    full wave set, so it has a smaller total-energy budget to share
    (<E_total> smaller for the triad) -- a target's raw swing can grow
    from triad to full wave set (P>0) while its *share* of the now-larger
    budget shrinks (this quantity <0). Both are legitimate; read them
    together, not as two noisy estimates of the same number.

    NaN if either input is NaN (e.g. one run's own drift gate tripped) or
    ``efficiency_sub`` is zero.
    """
    if not math.isfinite(efficiency_full) or not math.isfinite(efficiency_sub) or efficiency_sub == 0:
        return float("nan")
    return 100 * (efficiency_full - efficiency_sub) / efficiency_sub
