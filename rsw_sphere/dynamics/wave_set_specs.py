"""Registry of quartet/quintet examples used in the paper's merged
§Coupled Triads section (formerly §Multiple Triads / §Inertia-Gravity
Waves / §Five-Wave model).

Mirrors ``rsw_sphere.dynamics.triad_specs`` (the §2.2 triad registry):
YAML is the source of truth (``examples/wave_sets_section_3.yaml``), this
module just loads it into typed, validated objects consumed by
``rsw_sphere.plotting.wave_set_*``.

**Modes are listed explicitly** in each YAML entry, not referenced from
the §2.2 triad registry by role key -- considered and rejected (see
``paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-section-3.md``,
Phase B2): most published quartets' reference triads aren't in the §2.2
registry at all, cross-file coupling would let a §2.2 edit silently move
§3 figures, and explicit listing keeps the ``m_sum = m_p + m_q`` physical
constraint visible and validatable at load time (done below).

Each wave set names its modes with **symbolic keys** (``a``, ``b``, ``c``,
...) matching the paper's own lettering (``eq :4sys1``: mode ``a`` is
always the constituent-triad "sum" mode, ``m_a = m_b + m_c``), and each
triad names its sum mode and two members by those same keys -- so the
YAML reads the same way the paper's equations do.

Run as a quick self-check (validates every registered wave set's
``m_sum = m_p + m_q`` constraint and prints a summary):

    python -m rsw_sphere.dynamics.wave_set_specs
"""
import os
import sys
from dataclasses import dataclass, field

import yaml

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

#: Default registry YAML, relative to the repo root.
DEFAULT_WAVESETS_PATH = os.path.join(_ROOT, "examples", "wave_sets_section_3.yaml")


@dataclass(frozen=True)
class TriadRef:
    """One constituent triad of a ``WaveSetSpec``, named symbolically.

    Attributes
    ----------
    sum_key : str
        Mode key of the "sum" mode (the paper's convention: the mode whose
        wavenumber is the sum of the other two).
    member_keys : tuple of 2 str
        The other two modes' keys, in either order.
    display_label : str
        E.g. "Triad 1"/"Triad 2" -- how the paper refers to a quartet's
        constituent triads (distinct from §2.2's "Triad A"/"B"/"C"/"D",
        which name whole registered *triads*, not a sub-component).
    triad_key : str or None
        Optional, purely documentary link to a role key in
        ``examples/triads_section_2_2.yaml`` when this constituent triad
        happens to also be independently registered there (e.g.
        ``triad_kelvin_rossby_flow``). Not resolved or cross-checked
        automatically -- see the module docstring for why this registry
        doesn't couple to that one at load time.
    """
    sum_key: str
    member_keys: tuple
    display_label: str = ""
    triad_key: str = None


@dataclass(frozen=True)
class WaveSetSpec:
    """A quartet or quintet example: N Hough modes plus initial zonal
    velocities, coupled through one or more constituent triads.

    Attributes
    ----------
    key : str
        Semantic role identifier (dict key in the loaded registry).
    mode_keys : tuple of str
        Symbolic names (``'a'``, ``'b'``, ...), in a fixed order that
        ``modes``/``velocities`` and every index-returning method below
        agree with.
    modes : tuple of (m, n, alpha) int triples
        One per ``mode_keys`` entry, same order.
    velocities : tuple of float
        Initial zonal velocities (m/s), same order as ``mode_keys``.
    triads : tuple of TriadRef
    reference_triad : int
        Index into ``triads`` -- the constituent triad used as the
        P-measure's denominator (the "triad of RH modes" the paper
        compares each quartet's mode variations against) when a mode
        doesn't specify its own (see
        ``rsw_sphere.plotting.wave_set_pmeasure.p_measure``'s per-mode
        ``triad_index``).
    h_e : float
        Equivalent height, m.
    label, display_label : str
        Human-readable label / short tag ("Quartet A"/"Quintet A") for
        figures and tables.
    settings : dict
        Per-wave-set tuned ``tf_days``/``h``/``n_grid`` etc. (the
        ``TRIAD_SETTINGS`` analogue from ``make_section22_figures.py``) --
        stored here rather than hardcoded in a plotting script so the
        registry stays the single source of truth.
    """
    key: str
    mode_keys: tuple
    modes: tuple
    velocities: tuple
    triads: tuple
    reference_triad: int = 0
    h_e: float = 10000.0
    label: str = ""
    display_label: str = ""
    settings: dict = field(default_factory=dict)

    def n_modes(self) -> int:
        return len(self.mode_keys)

    def n_triads(self) -> int:
        return len(self.triads)

    def index(self, mode_key: str) -> int:
        """Integer index of a symbolic mode key into ``modes``/``velocities``."""
        return self.mode_keys.index(mode_key)

    def triad_indices(self, i: int):
        """``(i_sum, i_p, i_q)`` -- integer indices into ``modes`` for
        constituent triad ``i``, in ``WaveSet``'s own constructor order
        (sum mode first).
        """
        t = self.triads[i]
        i_sum = self.index(t.sum_key)
        i_p = self.index(t.member_keys[0])
        i_q = self.index(t.member_keys[1])
        return (i_sum, i_p, i_q)

    def sub_triad_modes(self, i: int):
        """The 3 ``(m, n, alpha)`` triples for constituent triad ``i``,
        **sum mode last** -- ``TRIAD.__init__``'s calling convention.
        """
        t = self.triads[i]
        i_sum, i_p, i_q = self.index(t.sum_key), self.index(t.member_keys[0]), self.index(t.member_keys[1])
        return self.modes[i_p], self.modes[i_q], self.modes[i_sum]

    def sub_triad_velocities(self, i: int):
        """Registered velocities for constituent triad ``i``'s 3 modes,
        **sum mode last** (matches ``sub_triad_modes``'s order).
        """
        t = self.triads[i]
        i_sum, i_p, i_q = self.index(t.sum_key), self.index(t.member_keys[0]), self.index(t.member_keys[1])
        return self.velocities[i_p], self.velocities[i_q], self.velocities[i_sum]

    def sub_triad_flat(self, i: int):
        """9-tuple flattening ``sub_triad_modes(i)``, matching
        ``TriadSpec.flat_modes()``'s / ``TRIAD.__init__``'s calling
        convention: ``(m_p, n_p, alpha_p, m_q, n_q, alpha_q, m_sum, n_sum,
        alpha_sum)``.
        """
        (m_p, n_p, a_p), (m_q, n_q, a_q), (m_s, n_s, a_s) = self.sub_triad_modes(i)
        return (m_p, n_p, a_p, m_q, n_q, a_q, m_s, n_s, a_s)


def load_wave_set_specs(yaml_path: str = DEFAULT_WAVESETS_PATH) -> dict:
    """Load the §Coupled Triads wave-set registry from a YAML config.

    Validates, for every triad of every wave set, that the sum mode's
    zonal wavenumber equals the sum of its two members' -- the physical
    constraint ``WaveSet.__init__`` also checks, but checking it here too
    gives an error that names the *role key* and *triad*, before any
    (much more expensive) Hough-mode computation is attempted.

    Parameters
    ----------
    yaml_path : str, optional
        Default: ``examples/wave_sets_section_3.yaml``.

    Returns
    -------
    dict of str -> WaveSetSpec

    Raises
    ------
    ValueError
        If any triad's sum-mode wavenumber constraint is violated, or a
        mode/triad key is referenced but not defined.
    """
    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    specs = {}
    for key, entry in raw.items():
        mode_keys = tuple(entry["modes"].keys())
        modes = tuple(
            (entry["modes"][mk]["m"], entry["modes"][mk]["n"], entry["modes"][mk]["alpha"])
            for mk in mode_keys
        )
        velocities = tuple(float(entry["modes"][mk]["u"]) for mk in mode_keys)

        triads = []
        for t in entry["triads"]:
            sum_key = t["sum"]
            member_keys = tuple(t["members"])
            if sum_key not in mode_keys:
                raise ValueError(f"{key}: triad sum key {sum_key!r} not in modes {mode_keys}")
            for mk in member_keys:
                if mk not in mode_keys:
                    raise ValueError(f"{key}: triad member key {mk!r} not in modes {mode_keys}")
            triads.append(TriadRef(
                sum_key=sum_key, member_keys=member_keys,
                display_label=t.get("display_label", ""),
                triad_key=t.get("triad_key"),
            ))

        for t in triads:
            m_sum = modes[mode_keys.index(t.sum_key)][0]
            m_p = modes[mode_keys.index(t.member_keys[0])][0]
            m_q = modes[mode_keys.index(t.member_keys[1])][0]
            if m_sum != m_p + m_q:
                raise ValueError(
                    f"{key}: triad (sum={t.sum_key}, members={t.member_keys}): "
                    f"m_sum={m_sum} != m_p+m_q={m_p}+{m_q}={m_p + m_q}")

        specs[key] = WaveSetSpec(
            key=key,
            mode_keys=mode_keys,
            modes=modes,
            velocities=velocities,
            triads=tuple(triads),
            reference_triad=int(entry.get("reference_triad", 0)),
            h_e=float(entry.get("h_e", 10000.0)),
            label=entry.get("label", key),
            display_label=entry.get("display_label", ""),
            settings=dict(entry.get("settings", {})),
        )
    return specs


if __name__ == "__main__":
    specs = load_wave_set_specs()
    print(f"Loaded {len(specs)} wave set(s) from {DEFAULT_WAVESETS_PATH}:\n")
    for key, spec in specs.items():
        print(f"{key} ({spec.display_label}): {spec.n_modes()} modes, {spec.n_triads()} triad(s)")
        for mk, m, u in zip(spec.mode_keys, spec.modes, spec.velocities):
            print(f"  {mk}: (m,n,alpha)={m}, u={u} m/s")
        for i, t in enumerate(spec.triads):
            print(f"  triad {i} ({t.display_label}): sum={t.sum_key}, members={t.member_keys}"
                  + (f" [also in §2.2 registry as {t.triad_key!r}]" if t.triad_key else ""))
        print()
    print("All m_sum = m_p + m_q constraints validated OK.")
