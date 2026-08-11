"""Registry of resonant-triad examples used in the paper/dissertation §2.2
("Resonant Triads").

Single source of truth for the mode triples and initial zonal velocities
that feed the master table (``rsw_sphere.plotting.triad_table``), the
energy-integration figures (``rsw_sphere.plotting.triad_dynamics``) and the
efficiency-sweep figures (``rsw_sphere.plotting.triad_efficiency``), so the
numbers reported in text, tables and figures cannot drift apart from one
another.

The registry itself lives in YAML (``examples/triads_section_2_2.yaml``),
not in this module -- see ``load_triad_specs``. Keys are semantic **role**
labels (``triad_rossby_only_near_resonant``, ``triad_rossby_only_non_resonant``,
``triad_kelvin_rossby_flow``, ``triad_gravity_with_rossby_catalyst``), not
mode numbers, since these triads are reused as building blocks by later
paper sections. Each also has a short ``display_label`` ("Triad A"/"B"/"C"/"D")
used in the master table and figure titles.

``alpha`` convention (shared with ``rsw_sphere.hough_harmonics``):
1 = EIG/EG (eastward inertia-gravity), 2 = WIG/WG (westward inertia-gravity),
3 = RH (Rossby-Haurwitz).

Note ``TRIAD.__init__`` (``rsw_sphere.dynamics.dynamic_triads``) takes
**flat positional** arguments ``(gamma, m_a, n_a, alpha_a, m_b, n_b,
alpha_b, m_c, n_c, alpha_c, N, deg)``, not tuples of ``(m, n, alpha)`` -- use
``TriadSpec.flat_modes()`` below to get the calling convention right.

Run as a quick self-check:

    python -m rsw_sphere.dynamics.triad_specs
"""
import os
import sys
from dataclasses import dataclass

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: Default registry YAML, relative to the repo root.
DEFAULT_SPECS_PATH = os.path.join(_ROOT, "examples", "triads_section_2_2.yaml")


@dataclass(frozen=True)
class TriadSpec:
    """A single resonant-triad example: three Hough modes plus initial
    zonal velocities.

    Attributes
    ----------
    key : str
        Semantic role identifier (dict key in the loaded registry, e.g.
        ``triad_rossby_only_non_resonant``).
    modes : tuple of 3 (m, n, alpha) int triples
        Mode a, b, c, in that order (see the ``alpha`` convention above).
    velocities : tuple of 3 float
        Initial zonal velocities (m/s) for modes a, b, c respectively. Used
        two different ways by the two consumers: ``triad_energy_evolution``
        (``triad_dynamics.py``) takes all three literally as the initial
        condition; ``efficiency_sweep`` (``triad_efficiency.py``) instead
        sweeps two of the three modes over a velocity grid and only uses
        this tuple's entry for whichever mode is held fixed (by default,
        the target mode itself, held at 0 regardless of what is registered
        here -- see ``efficiency_sweep``'s docstring). Don't assume a
        change here silently updates both figure types the same way.
    h_e : float
        Equivalent height (equivalent depth), in metres.
    label : str
        Human-readable label for figures/tables.
    display_label : str
        Short "Triad A"/"B"/"C"/"D" tag for the master table's leftmost
        column and figure titles.
    """
    key: str
    modes: tuple
    velocities: tuple = (10.0, 10.0, 10.0)
    h_e: float = 10000.0
    label: str = ""
    display_label: str = ""

    def flat_modes(self):
        """Flatten ``modes`` into the positional order expected by
        ``TRIAD.__init__``: ``(m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c,
        n_c, alpha_c)``.
        """
        (m_a, n_a, alpha_a), (m_b, n_b, alpha_b), (m_c, n_c, alpha_c) = self.modes
        return (m_a, n_a, alpha_a, m_b, n_b, alpha_b, m_c, n_c, alpha_c)


def load_triad_specs(yaml_path: str = DEFAULT_SPECS_PATH) -> dict:
    """Load the paper's triad registry from a YAML config.

    Parameters
    ----------
    yaml_path : str, optional
        Path to a YAML file with one entry per triad, keyed by a semantic
        role label. Each entry has ``mode_a``/``mode_b``/``mode_c`` blocks
        (each with ``m``, ``n``, ``alpha``, ``u``), plus ``h_e`` and
        ``label``. See ``examples/triads_section_2_2.yaml`` for the exact
        schema. Default: ``examples/triads_section_2_2.yaml`` in the repo
        root.

    Returns
    -------
    dict of str -> TriadSpec
        Keyed by the same role labels as the YAML file.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    specs = {}
    for key, entry in raw.items():
        modes = tuple(
            (entry[mode_key]["m"], entry[mode_key]["n"], entry[mode_key]["alpha"])
            for mode_key in ("mode_a", "mode_b", "mode_c")
        )
        velocities = tuple(
            float(entry[mode_key]["u"]) for mode_key in ("mode_a", "mode_b", "mode_c")
        )
        specs[key] = TriadSpec(
            key=key,
            modes=modes,
            velocities=velocities,
            h_e=float(entry.get("h_e", 10000.0)),
            label=entry.get("label", key),
            display_label=entry.get("display_label", ""),
        )
    return specs

# Table 1's quasi-resonant triads (two EG modes + one RH mode, all with
# coupling coefficients that should come out ~0 by the equatorial-symmetry
# selection rule). Kept separate from the YAML-loaded registry since these
# are a selection-rule check, not energy-transfer examples with velocities.
TABLE1_TRIADS = {
    "table1-1": TriadSpec(
        key="table1-1", modes=((1, 6, 3), (3, 7, 1), (4, 7, 1)),
        label="Table 1 triad 1 (quasi-resonant, zero coupling)",
    ),
    "table1-2": TriadSpec(
        key="table1-2", modes=((3, 12, 3), (9, 15, 1), (12, 15, 1)),
        label="Table 1 triad 2 (quasi-resonant, zero coupling)",
    ),
    "table1-3": TriadSpec(
        key="table1-3", modes=((3, 16, 3), (10, 19, 1), (13, 19, 1)),
        label="Table 1 triad 3 (quasi-resonant, zero coupling)",
    ),
    "table1-4": TriadSpec(
        key="table1-4", modes=((2, 10, 3), (4, 11, 1), (6, 11, 1)),
        label="Table 1 triad 4 (quasi-resonant, zero coupling)",
    ),
}


if __name__ == "__main__":
    for key, spec in {**load_triad_specs(), **TABLE1_TRIADS}.items():
        print(f"{key:20s} modes={spec.modes} flat={spec.flat_modes()}")
