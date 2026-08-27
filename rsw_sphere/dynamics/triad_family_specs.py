"""Registry of triad-FAMILY sweeps: one fixed edge, one mode varied across a
family (e.g. RH(3,n) partners), used by ``examples/rh_partner_family.py``.

``alpha`` convention (shared with ``rsw_sphere.hough_harmonics``):
1 = EIG/EG (eastward inertia-gravity), 2 = WIG/WG (westward inertia-gravity),
3 = RH (Rossby-Haurwitz).

Note ``TRIAD.__init__`` (``rsw_sphere.dynamics.dynamic_triads``) takes
**flat positional** arguments ``(gamma, m_a, n_a, alpha_a, m_b, n_b,
alpha_b, m_c, n_c, alpha_c, N, deg)``, not tuples of ``(m, n, alpha)`` -- use
``TriadSpec.flat_modes()`` below to get the calling convention right.

Run as a quick self-check:

    python -m rsw_sphere.dynamics.triad_family_specs
"""
import os
import sys
from dataclasses import dataclass, field

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: Default triad-FAMILY registry YAML.
DEFAULT_FAMILIES_PATH = os.path.join(_ROOT, "examples", "triad_families.yaml")


@dataclass(frozen=True)
class TriadSpec:
    """A single resonant-triad example: three Hough modes plus initial
    zonal velocities.

    Attributes
    ----------
    key : str
        Semantic role identifier.
    modes : tuple of 3 (m, n, alpha) int triples
        Mode a, b, c, in that order (see the ``alpha`` convention above).
    velocities : tuple of 3 float
        Initial zonal velocities (m/s) for modes a, b, c respectively.
    h_e : float
        Equivalent height (equivalent depth), in metres.
    label : str
        Human-readable label for figures/tables.
    display_label : str
        Short display tag.
    """
    key: str
    modes: tuple
    velocities: tuple = (10.0, 10.0, 10.0)
    h_e: float = 10000.0
    label: str = ""
    display_label: str = ""
    settings: dict = field(default_factory=dict)

    def flat_modes(self):
        """Flatten ``modes`` into the positional order expected by
        ``TRIAD.__init__``: ``(m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c,
        n_c, alpha_c)``.
        """
        (m_a, n_a, alpha_a), (m_b, n_b, alpha_b), (m_c, n_c, alpha_c) = self.modes
        return (m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c)


def load_triad_family(key: str, yaml_path: str = DEFAULT_FAMILIES_PATH) -> list:
    """Expand one entry of a triad-FAMILY registry into a list of
    ``TriadSpec``, one per family member.

    Two modes (``fixed_a``/``fixed_c``, TRIAD's own slot convention: slot
    c is the "sum" mode) stay the same across every member; the third
    (``varying``) is swept over an explicit ``n_values`` list, always in
    slot b, held at rest (the target mode, eq. effor). Does **not** itself
    validate the selection rule (``m_sum = m_p + m_q``) or ``n>=m`` --
    every ``TriadSpec`` returned here still needs to pass through
    ``WaveSet``'s validating constructor before use (e.g. via
    ``examples.rh_partner_family.triad_efficiency_point``), the same gate
    every other triad/wave-set construction in this repo goes through.

    Parameters
    ----------
    key : str
        Entry name in the family-registry YAML (e.g. ``rh_partner_family``).
    yaml_path : str, optional
        Path to a YAML file with one entry per family. See
        ``examples/triad_families.yaml`` for the exact schema. Default:
        ``examples/triad_families.yaml`` in the repo root.

    Returns
    -------
    list of TriadSpec
        One per value in ``varying.n_values``, keyed
        ``f"{key}_n{n}"``, each carrying the family's ``settings`` dict
        (``tf_days``, ``h``, ``deg``).
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)
    entry = raw[key]

    fixed_a = entry["fixed_a"]
    fixed_c = entry["fixed_c"]
    varying = entry["varying"]
    h_e = float(entry.get("h_e", 10000.0))
    settings = dict(entry.get("settings", {}))

    specs = []
    for n in varying["n_values"]:
        modes = (
            (fixed_a["m"], fixed_a["n"], fixed_a["alpha"]),
            (varying["m"], n, varying["alpha"]),
            (fixed_c["m"], fixed_c["n"], fixed_c["alpha"]),
        )
        velocities = (float(fixed_a["u"]), 0.0, float(fixed_c["u"]))
        specs.append(TriadSpec(
            key=f"{key}_n{n}", modes=modes, velocities=velocities,
            h_e=h_e, label=f"{entry.get('label', key)} (n={n})",
            settings=settings,
        ))
    return specs


if __name__ == "__main__":
    family = load_triad_family("rh_partner_family")
    assert len(family) == 13, f"self-check FAILED: expected 13 family members, got {len(family)}"
    assert family[0].modes == ((1, 2, 3), (3, 4, 3), (4, 5, 3)), \
        f"self-check FAILED: first member should vary to n=4, got {family[0].modes}"
    assert family[0].settings.get('tf_days') == 30.0, \
        "self-check FAILED: family settings not carried through to TriadSpec"
    for spec in family:
        print(f"{spec.key:20s} modes={spec.modes} velocities={spec.velocities}")
    print("self-check OK: load_triad_family")
