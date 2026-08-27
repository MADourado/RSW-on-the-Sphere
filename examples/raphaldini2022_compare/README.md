# raphaldini2022_compare/

Compares this paper's RSW quartets against Raphaldini, Peixoto, Teruya,
Raupp & Bustamante (2022, Phys. Fluids 34)'s own barotropic non-divergent
vorticity equation, on the identical four-wave topology (their (n,m)
modes {(3,1),(7,3),(5,4),(9,2)}, here "Quartet B", registered as
`quartet_rh_borrowed_topology` in `../../wave_sets_default.yaml`). Backs
JFM-template.tex's `sec: quartet_rh_precession`: Table
`precession_comparison` and Figure `borrowed_topology_precession`.

The barotropic model itself (`rsw_sphere.utilities.barotropic_vort_model`)
is a from-scratch reproduction of the paper's own Section III.A numbers
(frequencies, coupling coefficients, eq. 31/35), used verbatim rather
than re-derived from their coupling-coefficient formula (eq. 9) -- that
formula's normalization convention could not be fully pinned down from
the published text, so re-deriving risked an invisible mismatch. Its own
governing equation (quadratic energy, exactly conserved for any spectral
truncation) is genuinely different from this repository's shallow-water/
RSW system (cubic energy, not conserved under truncation), so it has no
`WaveSet`/registry representation of its own -- the RSW side of the
comparison does, and is built from the registry entry above.

## Regenerating the Table/Figure

```bash
python examples/raphaldini2022_compare/precession_comparison.py table
python examples/raphaldini2022_compare/precession_comparison.py figure \
    outputs/figures/wave_sets/quartet_rh_borrowed_topology/precession_cache.npz \
    outputs/figures/wave_sets/quartet_rh_borrowed_topology/borrowed_topology_precession.png
```

`table` prints the Table's own numbers (barotropic vs. RSW: $\delta_1$,
efficiency peak, precession frequency at control/peak scale) next to the
values currently hand-typed into the LaTeX, for comparison. `figure`
follows this repository's compute/plot split: the sweep (precession
frequency + RSW efficiency vs. amplitude scale) is cached to the given
`.npz` path and only recomputed if that file is missing (delete it to
force a recompute); pass `--plot-only` to re-plot from an existing cache
without touching the sweep. Copy the resulting PNG into
`paper-nonlinear-interactions-SWE-sphere/Figures/` to update the paper.

## Other scripts

- `individual_mode_reversal_investigation.py` -- checks for the
  individual-mode phase reversal Raphaldini et al. (2022) report
  (Section III.A: at high driving amplitude, one mode's own raw phase can
  bend enough that its propagation direction flips sign), in the
  barotropic model, in Quartet B (RSW), and in Quartet A (RSW, the
  paper's own native quartet). Backs the live prose claim in
  `sec: quartet_rh_precession` that no such reversal is found.
- `low_frequency_precession_check.py` -- checks whether the
  precession-resonance efficiency peak coincides with elevated
  low-frequency spectral power in the target mode's kinetic energy
  (Raphaldini et al. 2022's eq. 37 diagnostic), in the same three
  settings. Backs the live prose claim that it does, in both quartets.

Both reuse `rsw_sphere.utilities.barotropic_vort_model` for the
barotropic side and `precession_comparison.build_rsw_waveset`/
`rsw_trajectory` for the RSW side -- neither re-derives any physics of
its own.

## Consolidation history

Originally seven scripts, each importing the previous by bare module
name (`reproduce_raphaldini2022_fig2.py` -> `precession_resonance_rsw_vs_
barotropic.py` -> `precession_resonance_phase_diagnostic.py` ->
`borrowed_topology_precession_figure.py`, plus the two investigation
scripts above and `precession_resonance_broad_search.py`). Once the
comparison itself was validated, that four-script chain was consolidated
into `precession_comparison.py`, and the barotropic model was extracted
into `rsw_sphere/utilities/barotropic_vort_model.py` so it has one home
instead of being duplicated across scripts. `precession_resonance_broad_
search.py` (an exploratory search for RSW-side phase-locking at other
amplitudes/equivalent depths/mode candidates, all negative results) was
deleted -- it backed no live paper number or claim.
