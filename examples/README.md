# Example configs

Named `configs.yaml` variants reproducing specific figures/tables from the
thesis (`docs/Marco_Msc.pdf`), runnable with either driver:

```bash
python run_diagnostics.py --config examples/table_2_1_quasi_resonant.yaml
python run_dynamics.py --config examples/table_2_1_quasi_resonant.yaml
```

| Config | Reproduces | Notes |
|--------|-----------|-------|
| `../configs.yaml` (repo root) | Fig. 2.9 / Table 2.2 triad 2 | The shipped default. |
| `table_2_1_quasi_resonant.yaml` | Table 2.1 | Near-zero coupling coefficient, tiny frequency mismatch — expect almost no energy exchange. Initial velocities are placeholders (not given in the thesis table). |
| `table_2_3_kelvin_rh_energy_flow.yaml` | Table 2.3 / Fig. 2.7 | Energy flow from Rossby-Haurwitz to eastward gravity modes via the Kelvin wave `(1,1,EIG)`. |

Each writes its figures under its own `OUTPUT_PATH` (subfolders of
`outputs/figures/`) so they don't overwrite each other or the default run.

See `docs/code_guide.md` and `.claude/THESIS_FIGURES.md` for the full
thesis-figure-to-code map these were drawn from.

## Triad registry

`triads_section_2_2.yaml` is a different kind of config — not a
`run_diagnostics.py`/`run_dynamics.py` input, but the registry of triads
used by `rsw-triad-table` / `rsw-triad` / `rsw-triad-efficiency` (see
[`docs/triads.md`](../docs/triads.md)).

## Wave-set (quartet/quintet) registry

`wave_sets_section_3.yaml` is the same kind of registry, generalized to
quartets and quintets (coupled multi-triad configurations) — used by
`rsw-waveset-table` / `rsw-waveset` / `rsw-waveset-periods` /
`rsw-waveset-pmeasure`, and by `make_section3_figures.py`'s composite
builder (see [`docs/wave_sets.md`](../docs/wave_sets.md), which also
covers how to test a triad/quartet/quintet that isn't in this file at
all). `check_wave_set_physics.py` is the hard physics gate every new or
edited entry must pass before any figure is trusted.

## Precession resonance (2026-08-12)

Two standalone scripts from the same-day precession-resonance
investigation (see `paper-nonlinear-interactions-SWE-sphere/.claude/
INSPECT-phase-I0-I4-2026-08-12.md` for the full writeup):

- `reproduce_raphaldini2022_fig2.py` — direct, dependency-free
  reproduction of Raphaldini, Peixoto, Teruya, Raupp & Bustamante (2022,
  Phys. Fluids)'s own four-wave barotropic-vorticity efficiency peak,
  using their exact published numbers (not re-derived). Confirms the
  precession-resonance mechanism is real and correctly transcribed
  before comparing against RSW.
- `precession_resonance_rsw_vs_barotropic.py` — the identical mode
  topology, built in this repo's RSW/`WaveSet` system instead, for a
  direct barotropic-vs-RSW comparison. See
  `rsw_sphere.dynamics.dynamical_phase` for the reusable
  libration-vs-rotation diagnostic (the textbook precession-resonance
  signature) this investigation also produced.
- `precession_resonance_phase_diagnostic.py` — applies that
  libration-vs-rotation diagnostic to both scripts above, at their own
  efficiency peak/dip scales plus an off-peak control. **Result: the
  barotropic case shows genuine phase-locking at its efficiency peak
  (Phi1's precession frequency collapses to ~0 rad/day, matching the
  paper's own Fig. 3) while the off-peak control does not (~-0.04
  rad/day, 0.77 net windings) — but RSW's efficiency peak/dip show NO
  such locking** (Phi1 keeps net-rotating at every scale tested, control
  through dip, getting worse — not better — from peak to dip). The
  RSW efficiency non-monotonicity is real (see the other script) but is
  not accompanied by the same phase-locking mechanism that produces it
  in the barotropic case.
- `precession_resonance_broad_search.py` — broadened the search beyond
  the paper's own transplanted topology (an h_e sweep toward the
  barotropic limit, a finer amplitude-scale sweep, and RSW's own native
  near-commensurate candidates from the 2026-08-12 I4e-h edge search),
  then extended further (same day, continued) to test the registry's
  own "Quartet A" (`quartet_rh_preference`) with the same disciplined
  methodology, and to distinguish *which* triad's phase actually locks.
  **Headline result: genuine, converged, MUTUAL two-triad phase-locking
  found in RSW**, in native triads closing the paper's own fixed
  RH-only triad (`RH(4,5)+RH(3,4)+RH(1,2)`) with a Rossby-Haurwitz
  partner on either of two edges -- `WG(3,1)` on `RH(4,5)+RH(1,2)` and
  `WG(4,2)` on `RH(3,4)+RH(1,2)` both show BOTH triads' dynamical phases
  (`Phi1` and `Phi2`) locking together inside a bounded resonance island
  in (driving velocity) parameter space, convergence-verified against
  2x t_f and 3x finer h. **But the registry's own "Quartet A" does NOT
  show genuine mutual locking** -- only its directly-driven triad's
  phase (`Phi2`) locks; the other, undriven triad (`Phi1`) keeps
  rotating throughout (confirming/sharpening the original 2026-08-12
  I4l finding of "partial softening, no full capture"). The apparent
  reason: `WG(3,1)`/`WG(4,2)` were found via a search explicitly
  targeting near-exact LINEAR mismatch commensurability
  (`delta2~k*delta1`, built into the mode selection, independent of
  amplitude) with shared-mode roles preserved across both triads;
  Quartet A's own "1:1" point is an amplitude-tuned coincidence of the
  two triads' *nonlinear* periods instead (its linear mismatches aren't
  close: `delta1=-0.1085` vs `delta2=-0.0400`). A third candidate,
  `WG(5,3)`, satisfies the linear-commensurability condition but
  requires one shared mode to flip role between the two triads, and
  shows the same Quartet-A-like partial (driven-triad-only) result.
  **A 5th, deliberately-constructed test then broke the simple picture**:
  a role-preserved candidate (`EG(1,1)`, on the *original* Gate I0-I4
  catalogue edge) with strong coupling but a mismatch far from any
  integer multiple of `delta1` (`k~-3.3`) shows a **third, distinct**
  pattern -- at high driving, the triad that does NOT contain the swept
  candidate locks, while the triad that DOES contain it spins
  increasingly fast (moving further from resonance as amplitude grows).
  **Net assessment: which triad locks (driven, undriven, or both)
  depends on the specific relationship between the two triads' own
  mismatches and couplings in a way not yet reduced to one clean rule**
  -- the phase-locking mechanism itself is real and reproducible in
  every individual case tested, but a general predictive condition is
  still open. See the paper repo's
  `.claude/INSPECT-phase-I0-I4-2026-08-12.md`, sections "Broadened
  search (2026-08-13)" onward (especially "Fifth test"), for the full
  numbers, convergence checks, and the honest state of the hypothesis.
