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
  near-commensurate candidates from the 2026-08-12 I4e-h edge search).
  **CORRECTED 2026-08-13 (Opus review): the original "genuine MUTUAL
  two-triad phase-locking found in RSW, WG(3,1)/WG(4,2)" headline result
  was FALSE.** Both `WG(3,1)=(3,1,2)` and `WG(4,2)=(4,2,2)` are invalid
  mode specs (`n<m`, out of range for the Legendre/Hough expansion) that
  `WaveSet` silently resolved to duplicates of `RH(3,4)`/`RH(4,5)`
  already present in the base triad -- the "locking" found was one
  triad's own phase compared against itself, not two distinct triads.
  A validity guard was added to `WaveSet.__init__` the same day
  (raises `ValueError` on `n<m`) so this can't recur silently.
  `WG(5,3)` and the `EG(1,1)`/`WG(1,1)` "fifth test" pair are genuine,
  valid modes and stand independently, but the three-pattern hypothesis
  they were framed against (built on the now-invalid WG(3,1)/WG(4,2)
  results) needs new framing if revisited. Full correction and what
  survives: paper repo's `.claude/INSPECT-phase-I0-I4-2026-08-12.md`,
  "CORRECTION (2026-08-13, later same day)" section near the top.

## Gate I2/I5/I6 and the short-gravity/long-Rossby search (2026-08-13)

Continuation of the same-day inspection work above, per
`PLAN-section-3-experiments.md` Phases I2/I5/I6:

- `gate_i2_map_extension.py` — extends the S2 map from 2 to the full
  26-candidate catalogue. Data/figure: `gate_i2_map_data.npy` /
  `gate_i2_map.png`. **Note**: the colorbar is a `d1_proxy`, not
  calibrated D1 in percent — a review found the two-channel law's own
  fit has a real, point-varying multiplicative prefactor (measured
  ratio 0.11-0.27) this module's docstring originally mis-described as
  "no free constant." See `d1_proxy`'s own docstring.
- `gate_i5_headline.py` — the S4 headline number, corrected 2026-08-13
  after a review caught an inverted phase-lag sign and a wrong
  "D1 doesn't saturate" claim (it saturates near 19% by day ~320-400).
  Current, correct headline: **+0.83% period lengthening, +1.0%
  peak-KE difference (1.88e19 J)**, both tf-independent, on the
  registered `quartet_gravity_kelvin`.
- `short_gravity_long_rossby_example.py` — searched for a short-period
  gravity mode demonstrating both amplitude AND frequency effects on a
  Rossby target (2026-08-11 user request). **The originally-reported
  41-45% frequency shift for `WG(7,9)` was a Savitzky-Golay smoothing
  artifact** (the measured period was a monotone function of the
  smoothing window) — window-independent estimators (FFT with
  parabolic interpolation; prominence-filtered peaks) both give <=0.1%.
  The 2026-08-11 request remains unresolved.

See the paper repo's `.claude/INSPECT-phase-I0-I4-2026-08-12.md`,
"CORRECTION (2026-08-13, later same day)" section, for the full story
on both retractions above, what independently re-verified as solid
(the air-density work, the Gate I2 catalogue itself, the two-channel
law's shape), and the punch list for the next session.

## §3.3 rewrite (2026-08-14)

Executes `paper-nonlinear-interactions-SWE-sphere/.claude/PLAN-section-3.3.md`
(the §3.3 "Gravity-Rossby quartets" execution plan). Closes the one real
gap the plan identified — the Gate I4b two-channel law had no saved
script, only prose — and uses that same integration pass to calibrate
the Gate I2 map's previously-uncalibrated `d1_proxy`.

- `gate_i4_scaling_law.py` — direct, brute-force integration (not the
  analytic proxy) of $\mathcal{F}_2$ across the full 26-candidate x 8
  energy-fraction grid (208 points), target mode b=RH(3,4), $t_f=20$d
  (matching `quartet_gravity_kelvin`'s own registered horizon).
  Re-derives and re-verifies the two-channel law
  `F2 ~ sqrt(alpha_2s^2+alpha_2p^2)*sqrt(x(1-x))/delta_2` from scratch:
  **R^2=0.9955** (vs. the 2026-08-12 investigation's own R^2=0.982),
  exponents (1.01, 1.20, -1.05) against the derived (1,1,-1). A
  robustness subset re-measured at $t_f=40$d (8 candidates) gives
  R^2=0.974, exponents (1.03, 0.82, -1.12) — confirms the law's shape is
  not an artifact of the specific horizon chosen, even though the
  absolute value of $\mathcal{F}_2$ itself is (per §3.1's own stated
  diagnostic doctrine). Saves `gate_i4_scaling_law_data.npy` (all 26
  candidates' own coupling/mismatch/measured-$\mathcal{F}_2$ data) and
  the verification figure `gate_i4_scaling_law.png`.
- `gate_i2_map_recalibrate.py` — re-plots the Gate I2 map using the
  above script's real, integrated $\mathcal{F}_2$ values instead of
  `d1_proxy`; adds the three cut dissertation examples (Table cap4ex) as
  labeled points on the EG(1,1) curve. Output:
  `gate_i2_map_calibrated.png`.
- `regen_gravity_quartet_tables.py` — regenerates Table cap42/cap43
  (Quartet C/D coefficient tables) in Table cap41's own format
  (Mode/Freq/Period/Coeff.1/Coeff.2/Zonal/$A_0$), via
  `rsw_sphere.plotting.wave_set_table.wave_set_properties` +
  `WaveSet.amplitudes_from_velocities`.
- `section33_headline_numbers.py` — generalizes `gate_i5_headline.py`'s
  own methodology (period shift + peak-KE difference, both
  $t_f$-independent; $\mathcal{F}_2$ at the registered $t_f$) to BOTH
  Quartet C and Quartet D — the dissertation never computed a headline
  number for Quartet D. Result: Quartet D's effect (period $+0.2\%$,
  peak-KE $+0.1\%$) is an order of magnitude smaller than Quartet C's
  ($+0.8\%$, $+1.0\%$), exactly as the scaling law predicts from
  EG(7,9)'s much larger $|\delta_2|$.

**A real bug caught and fixed while building the map figure**: an
initial `\begin{table}[h]` in the new Appendix C literally rendered the
string `[h]` as visible text in the compiled PDF (`JFM-FLM_Au.cls`
appears not to support an explicit placement specifier the way a
standard `table` environment does) — every other table in this paper
uses a bare `\begin{table}`, so the fix was simply to match that
convention rather than debug the class internals.

All four scripts feed directly into `JFM-template.tex` §3.3, which was
fully rewritten this session (five subsubsections replacing the three-
example anecdotal structure, per the plan's disposition table); compiles
clean (bibtex + 2x pdflatex, 42 pages, zero undefined references or
warnings).

**Independent triple review (2026-08-14, writing/code-correctness/
literature), same discipline §3.2 received**: all three completed.
Code-correctness review re-derived every number in the section from
scratch (not just re-read the scripts) and found **zero mismatches**;
one traceability gap (the precession-residual correlation quoted a
number from an older, superseded fit rather than the fresh one) was
fixed by recomputing it directly against `gate_i4_scaling_law_data.npy`
(0.015, materially the same conclusion). Writing review found 6 real
issues (a factually wrong "pump mode" attribution — RH(3,4), not
RH(4,5), is the pump mode of this triad, per the paper's own existing
Table `master`/Figure `hough_examples` — plus a leaked internal LaTeX
label in a caption, a stray Python-variable-style phrase, a symbol
collision, a missing relative pronoun, and a `\S` vs `Appendix` style
slip), all fixed. Literature review found one real citation
misattribution, inherited from the original (pre-rewrite) dissertation
prose: `rocha2018stimulated` was cited backwards (claimed gravity waves
are dissipated by the Rossby inverse cascade; the source's actual
finding is the reverse — gravity/near-inertial waves are what extracts
energy *from* the balanced flow's own inverse cascade), fixed.

**§3.3.5 "A systematic search for a frequency-shift effect" (2026-08-14,
same day, per user request to "extend the research on fast gravity -
slow Rossby interactions and frequency change" before the review above)
— the placeholder is now filled with a real, converged finding, not a
`\todo{}`.** New scripts:

- `frequency_shift_catalogue_search.py` — Stage 1: screens the full
  26-candidate catalogue at $x=0.3$ using two window-independent period
  estimators (FFT with parabolic peak interpolation; prominence-filtered
  peak-to-peak timing — **never Savitzky-Golay**, the source of the
  2026-08-13 WG(7,9) retraction), applied symmetrically to the full
  quartet and the RH-only sub-triad, flagging any candidate where the
  two estimators disagree by >1 percentage point as unreliable rather
  than trusting either number blindly.
- `frequency_shift_stage2.py` — Stage 2: full $x$-sweep + $t_f$
  convergence check for the candidates Stage 1 flags as real.

**Finding**: the original request's premise (a short-period mode is the
right place to look) is wrong — 24/26 candidates, including every
genuinely short-period one, shift the target's own period by
$\leq0.1\%$ (extends WG(7,9)'s own corrected null result to essentially
the whole catalogue). The two exceptions, EG(1,1) and WG(1,1), are the
two *longest*-period modes in the catalogue but the two with by far the
smallest timescale separation from the RH-only triad's own exchange
rate — exactly the pair already implicated in the amplitude effect
(Gate I2 map). Where measurable, the effect is large and opposite in
sign: EG(1,1)'s target period lengthens $+16.7\%$ at $x=0.5$ (converged
$t_f=60$-$240$d and $h=0.01$-$0.002$; drifts further to $+18.9\%$ by
$t_f=480$d, reported as the shorter-horizon plateau, not asserted as
asymptotic); WG(1,1)'s shortens $-6.2\%$ to $-6.6\%$ over the same
range. Governed by timescale separation, not period length.

**Amplitude effects added alongside frequency, 2026-08-14, same day,
per user follow-up ("check the amplitude effects too, not just
frequency")**: `frequency_amplitude_companion.py` reports $\mathcal{F}_2^a$
(already computed for the full catalogue, reused not recomputed) and a
new $\mathcal{F}_{max}^a$ (signed peak-KE difference, not previously
computed for this catalogue) for EG(1,1)/WG(1,1) at the same $x$ values
as the frequency sweep. **The two channels agree with each other**:
EG(1,1) routes progressively MORE peak kinetic energy into the target
as $x$ grows ($+6.9\%$ to $+27.2\%$ over $x=0.1$-$0.7$) — the same mode
whose period lengthens; WG(1,1) routes progressively LESS ($-0.2\%$ to
$-2.8\%$) — the same mode whose period shortens. $\mathcal{F}_2^a$ alone
(non-negative by construction) cannot distinguish the two directions;
only the signed $\mathcal{F}_{max}^a$ and the frequency shift can, and
both tell the same story. New table in the paper, `tab: freq_amp`.

**Generalized further, 2026-08-14, same day, per user follow-up ("test
also with higher frequency modes... track it" + "change the target and
control mode structure... different experiment of topology/energy
flux" + "always verify both amplitude effects and frequency/period
effects") — new §3.3.6 "Generality: more modes, another target, another
edge".** Three new scripts:

- `catalogue_wide_tracking.py` — extends the frequency-shift + amplitude
  tracking from 2 candidates to the FULL 26-candidate catalogue, and
  from 1 target (RH(3,4)) to BOTH shared modes (RH(4,5) too), getting
  both targets from a single integration per (candidate, x). Confirms
  EG(1,1)/WG(1,1) remain the only two with a real frequency effect, and
  that RH(4,5)/RH(3,4) always share the identical frequency shift
  (expected: period is a property of the coupled triad, not a per-mode
  quantity).
- `alternate_topology_probe.py` — a genuinely new quartet topology: a
  gravity mode closing a triad on Quartet A's own edge RH(4,5)+RH(1,2)
  (§3.2) instead of this section's RH(4,5)+RH(3,4). **First version had
  a real physics bug**, caught by a sanity check before the full run:
  a bare 3-mode "edge triad" with the gravity mode's own IC zeroed is
  NOT a valid gravity-absent baseline (with only one triad,
  `dA_gravity/dt` is driven by `A_a*A_b` regardless of the gravity
  mode's own starting value, so it doesn't stay near zero) — fixed by
  using a proper 4-mode quartet with Quartet A's own registered triad1
  as the genuine RH-only reference, mirroring `gate_i4_scaling_law.py`'s
  structure exactly.

**A second real bug, also caught by a sanity check rather than assumed
away**: `catalogue_wide_tracking.py`'s first version showed $\mathcal{F}_{max}^a$
reading exactly $0.00\%$ for RH(4,5) under several candidates — not
small, exactly zero at every $x$. Diagnosed directly: RH(4,5) is a NET
ENERGY LOSER under these candidates, so its peak KE sits at $t=0$ in
both the full quartet and the sub-triad, identical by construction
regardless of the gravity mode. $\mathcal{F}_{max}^a$ is blind to any
effect on a target that never exceeds its own initial value. Fixed by
adding a range-based diagnostic, $\Delta EK$ (already established in
this paper via eq: Pa, not a new invention) — recomputed for both
scripts and both catalogues. Corrected finding: EG(1,1) DOES
substantially affect RH(4,5) ($+46.7\%$ range change at $x=0.5$,
comparable to its own $+39.1\%$ effect on RH(3,4)) — the
$\mathcal{F}_{max}^a$-based "$0\%$" was a diagnostic blind spot, not a
real null result. The corrected diagnostic also shows amplitude effects
extend further into the catalogue than frequency effects do (e.g.
EG(7,7)/WG(7,9) show real, growing range effects — up to $+17\%$ —
despite $\leq0.1\%$ frequency shift).

**New topology's own finding**: the same qualitative law (smallest
$|\omega_d|$ → largest effect) holds on the new edge, but an order of
magnitude weaker in absolute size (EG(3,3)'s own $+11.4\%$ range effect
vs.\ EG(1,1)'s $+46.7\%$) — the *law* generalizes, its *magnitude*
doesn't. Also surfaced a THIRD finding, this one a confirmation rather
than a bug: RH(1,2) on this edge (an order-of-magnitude weaker coupling
than RH(4,5), Table `cap41`) shows the SAME small-denominator inflation
that originally motivated retiring $\mathcal{P}_a$ (Finding F1,
`PLAN-section-3-experiments.md`) — range-normalized percentages for
RH(1,2) reach triple digits without a proportionally large physical
effect, so it was excluded from the paper's own comparison, with the
reasoning stated explicitly rather than silently dropped.

All numbers in `JFM-template.tex` §3.3.6 independently verified against
these scripts' own saved output before writing (not hand-copied from
memory). Compiles clean, 43 pages.
