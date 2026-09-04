"""Efficiency diagnostic, generalized for wave sets that don't conserve
energy exactly (quartets/quintets) -- normalizes by mean total energy
instead of initial total energy, gated on drift.
"""

#: Reference-too-small-to-normalize-by floor, shared with
#: `rsw_sphere.utilities.pmeasure` (which imports it from here, not the
#: other way, to avoid a circular import -- pmeasure.py already imports
#: `default_velocity_range` from this module). Defined once here since
#: both `wave_set_efficiency`'s own dEK denominator (this module) and
#: efficiency variation's own dEK denominator (`pmeasure.py`) have the
#: identical failure mode: a technically nonzero but too-small reference
#: inflates a percentage-change ratio without saying anything physically
#: meaningful. Deliberately small (1e-4) -- a genuinely small-but-real
#: reference (e.g. an efficiency of ~0.009, order 1e-2) should NOT be
#: suppressed by this guard, only a reference close enough to zero to be
#: numerically degenerate.
MIN_REFERENCE_DEK = 1e-4

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
    plain triad, since E_total barely varies there. This is a
    *standalone* per-configuration efficiency (paper eq. `effgen`) -- a
    meaningful, self-contained number on its own. It is deliberately NOT
    used to build a triad-vs-quartet (or quartet-vs-quintet) comparison:
    see ``rsw_sphere.utilities.pmeasure`` for that (retired
    ``efficiency_variation`` here divided the full/larger configuration's
    own raw swing by *its own* mean total energy and the reference's by
    *its own* -- two different, configuration-dependent denominators, so
    the ratio could drift purely because the larger configuration's own
    energy budget grew, independent of the target's actual dynamical
    response; found 2026-09-03 screening EG(1,n)/WG(1,n) candidates,
    where a fixed driving velocity requires more energy at higher n).
    ``pmeasure``'s comparison instead normalizes both sides by the same
    (reference) energy budget, which cancels out of the ratio entirely --
    algebraically the correct generalization of "efficiency variation" to
    a triad-vs-quartet (or quartet-vs-quintet) comparison, not a
    different diagnostic.
    """
    if drift > drift_max:
        return float("nan")
    denom = E_total.mean()
    if denom <= 0:
        return float("nan")
    return (E_target.max() - E_target.min()) / denom
