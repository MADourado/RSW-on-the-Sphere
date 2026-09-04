# General guide

Read this before starting any work in this repository. It covers how the
project is organized and how to work in it — not the physics/maths (see
`docs/code_guide.md` for the technical, file-by-file map, equations, and
module details) and not the registry/plotting API (see `docs/wave_sets.md`).

## Two repositories

This is **two separate git repositories**, nested but independent:

- **`RSW-on-the-Sphere/`** (this repo) — the code: the Hough-harmonic
  eigenproblem, `WaveSet`/`TRIAD` dynamics, the five root drivers, and
  everything under `rsw_sphere/`.
- **`paper-nonlinear-interactions-SWE-sphere/`** — the paper itself
  (`JFM-template.tex`, `jfm.bib`, figures), its own git history, gitignored
  from the code repo. Treat it as its own repo: check `git status`/`git
  diff` there separately before and after any edit; don't assume a commit
  in one repo has anything to do with the other.

**How they connect:** every paper table/figure that comes from this
codebase is reproduced by one small script in `examples/tables/` or
`examples/figures/` (see "Foldering pattern" below), and the LaTeX itself
carries a `% regenerate: python examples/...` comment right above the
`\begin{table}`/`\begin{figure}` it reproduces, pointing at that script.
When a number changes in the code, find the LaTeX text citing it (grep the
number or the relevant `\label{...}`) and update the prose by hand — the
table-writer scripts print/save a standalone `\begin{table}...\end{table}`
block (`outputs/tables/*.tex`) to copy in, they don't edit
`JFM-template.tex` directly, and none of these scripts touch prose
paragraphs at all.

## Workflow philosophy

**Everything routes through the five root drivers, not new one-off
scripts.** `run_linear_modes.py` (dispersion/Hough plots), `run_dynamics.py`
(integrate a wave set + sub-triads, diagnostics), `run_sweep.py` (1-2 mode
IC sweep + diagnostics), `run_sweep_sets.py` (loop a diagnostic over
candidate-mode variants), `run_mode_search.py` (find candidate modes for a
new triad). Before writing a new `examples/*.py` analysis script, check
whether the question is actually "run one of these five drivers with a
new/edited `wave_sets_default.yaml` entry" — that's true far more often
than it looks. A new physical configuration is a new YAML registry entry,
not a new Python script.

**Config/registry-driven, not hardcoded.** Modes, velocities, `tf_days`,
`h`, sweep ranges — these belong in `wave_sets_default.yaml` (or a
purpose-specific YAML under `examples/`), not as inline Python literals.
Every triad/wave-set construction should go through `WaveSet`'s own
validating constructor (it raises on a selection-rule violation or
`n < m`) even when the actual integration uses `TRIAD` directly — `TRIAD`
itself does not validate, and a mis-ordered triad silently produces a
plausible-looking wrong answer, not an error.

**Don't multiply examples/tests for the same thing.** One general,
parameterized function (arbitrary modes/velocities/target) beats several
near-identical scripts each hardcoded to one specific triad. If the same
non-trivial plotting/analysis block is about to be written a second time,
factor it out (e.g. into `rsw_sphere/plotting/style.py`) instead of
copy-pasting — two occurrences is the trigger, not three.

**Split compute from plotting for anything non-trivial to (re-)run**
(more than a few seconds): a `sweep(...)`-style function that always
writes to a cache (no-op if the cache already exists) and a
`plot_...(...)`-style function that always reads from that cache, never
re-runs the sweep. This is why `run_dynamics.py`'s own trajectories are
cached (`rsw_sphere.dynamics.trajectory_cache`) independently of the
figures built from them — editing a label, color, or title should never
require re-integrating.

**Paired figures from different data sources become LaTeX subfigures, not
one Python script.** When two (or more) figures are shown/discussed
together but come from genuinely different computations (different mode
sets, different sweeps, different caches), keep them as independent
scripts/PNGs and combine them in the LaTeX with `\begin{subfigure}` panels
— don't force one script to produce a multi-panel image just because the
paper shows them side by side. The converse also holds: if every panel
already comes from one shared computation/dataset (e.g. one `WaveSet`
integration split across per-mode panels), one script producing one
multi-subplot PNG is correct — don't split that into separate scripts
just to mirror separate LaTeX subfigures.

## Foldering pattern: figure / table / experiment

Three different things live in three different places under `outputs/`
(gitignored, fully regenerable — never treat anything under it as a
source of truth to hand-edit):

1. **`outputs/trajectories/<topology>/`** — raw integrated trajectories,
   cached by *physical configuration* (modes + initial velocities + `tf`/
   `h`), reusable across any script/experiment that happens to want the
   same physical run. Never re-integrate ad hoc when this cache exists.
2. **`outputs/dynamics/<wave_set_key>/<run_label>/`** — everything ONE
   specific experiment run produces (figures *and* tables together, one
   folder), keyed by the wave set *and* its own initial conditions/`tf`/
   `h` — so two different sweeps or parameter choices over the same wave
   set never collide or overwrite each other. See `docs/code_guide.md`
   "Outputs" for the exact filenames.
3. **`outputs/tables/`, `outputs/figures/`** — the paper-facing outputs
   of `examples/tables/*.py`/`examples/figures/*.py` scripts (below), and
   `run_linear_modes.py`'s own dispersion/Hough plots.

**`examples/tables/paper_table<NN>_<name>.py` /
`examples/figures/paper_figure<NNN>_<name>.py` /
`examples/tables/paper_headline_<name>.py`** — the naming convention for
any script whose entire job is to reproduce one specific paper artifact.
`<NN>`/`<NNN>` + `<name>` cross-reference the LaTeX `\label{tab: ...}`/
`\label{fig: ...}` it reproduces, plus a stable descriptive name (LaTeX
numbering drifts as the paper is edited; the script's own name and module
docstring shouldn't). One script, one artifact — see `examples/README.md`
for the full current list and which `\label{...}` each one covers. A
script that computes something used only in prose (a headline number, not
a table/figure) still goes in `examples/tables/`, named `paper_headline_*`.
When an artifact is dropped from the paper, delete its script rather than
keeping it around unnumbered — git history is the archive, and a script
that still runs but backs nothing invites being cited again by mistake.

## Code conventions

- **Comments: short, and removed on touch.** A phrase or one short
  sentence — even for non-obvious rationale — never a long narrative
  paragraph. When editing a file, delete comments that no longer match
  the code below them in the same edit, rather than layering a new
  comment on top of a stale one.
- **No comment ever references "this session," "the plan," "per Claude,"
  or cites a `.claude/PLAN-*.md`-style internal planning document** — in
  code or in any `docs/*.md`. Those are process artifacts, not part of
  the documented architecture; a reference to one is meaningless to
  anyone (including a future session) without that specific planning
  history. A dated correction note explaining a real bug/fix is fine
  ("sign was inverted here until an independent review caught it") as
  long as it states the technical reason directly, without citing the
  process that found it.
- **Paths are anchored to the repository root, never the working
  directory.** Output locations and the trajectory cache go through
  `rsw_sphere/paths.py` (`REPO_ROOT`, `OUTPUT_ROOT`, `TRAJECTORY_ROOT`,
  `resolve()`); scripts under `examples/` get the root from their
  directory's `_bootstrap.py`. A CWD-relative default silently forks the
  trajectory cache per launch directory, which is how 240 MB of duplicate
  cache once ended up under `examples/figures/`.
- **Validation gates wherever config/user input could silently produce a
  wrong-but-plausible result** — the `WaveSet` `n < m` gate exists because
  an earlier silent failure mode returned a *duplicate* eigenvector for
  an invalid mode, corrupting a full day's results with no error at all.
- **Testing:** run only the targeted subset relevant to what changed
  (`pytest -k "..."` or specific files) — the full `tests/` suite is slow
  enough that it's not a routine step. Run it in full only before a
  release/handoff-style checkpoint, or when explicitly asked (the full
  suite is ~4 minutes). A test
  marked `@pytest.mark.slow` needs a real justification (genuine
  end-to-end coverage nothing cheaper provides) — parameters (`tf_days`,
  `n_grid`, `N`/`deg`) should already be the smallest that still exercise
  the thing being tested.
- **Never commit or push**, in either repo, as part of finishing a task —
  leave changes staged/unstaged for the user to review and commit
  themselves. This applies to delegated/background work too.

## Paper writing style

- **Lean, direct scientific register.** Shorter sentences, minimal
  intensity adjectives/adverbs ("striking," "stark," "genuinely") — state
  the result or number plainly and let it do the work.
- **Rationale/provenance goes in a `%` comment above the text, not in the
  reader-facing prose.** The prose states the corrected result; a comment
  explains why it needed correcting (a bug found, a convergence check
  run), for future reference.
- **New terminology is proposed and discussed before it goes in the
  paper** — don't introduce a new name for a process/quantity unilaterally
  and find out later it's unclear or used inconsistently.
- **Avoid the "— aside —" em-dash case-separation construction.** Use a
  comma, semicolon, colon, or a new sentence instead; keeps prose fluid.
  (The en-dash for a numeric range, e.g. `u\approx83--92`m/s`, is an
  unrelated, still-fine LaTeX convention.)
- **Never fabricate a bibliographic detail** (volume/issue/pages/author
  list) when adding a citation. If a source doesn't supply an exact
  field, keep searching, mark it explicitly unverified, or leave a TODO —
  never fill it in from a "typical journal pattern" guess.

## Where to go next

- `docs/code_guide.md` — file-by-file technical map: what each module
  computes, exact cache/output paths, equations, conventions/gotchas
  (units, non-dimensionalization, mode-index ordering).
- `docs/wave_sets.md` — the `WaveSet`/registry layer: schema, how to add
  or test a new wave-set configuration.
- `examples/README.md` — current catalog of registries and every
  `paper_table*`/`paper_figure*`/`paper_headline*` script with the
  `\label{...}` it reproduces.
