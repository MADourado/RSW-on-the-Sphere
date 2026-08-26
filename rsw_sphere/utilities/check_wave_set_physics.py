"""Physics cross-checks for ``rsw_sphere.dynamics.wave_sets.WaveSet``
(quartets, quintets) against ``rsw_sphere.dynamics.dynamic_triads.TRIAD``
(the independent reference implementation, left untouched).

**HARD GATE for the §Multiple Triads rebuild**: no wave-set plotting module
(``rsw_sphere.plotting.wave_set_*``) is written until this script prints
all-pass.

Checks (see each ``check_c*`` docstring for the precise claim and why it
matters):

    C1  constituent-triad coefficients/mismatch/S match an independent TRIAD
    C2  ...still match with a triad's two member modes swapped
    C3  WaveSet trajectory == TRIAD trajectory for a 1-triad WaveSet
    C4  fat_policy='symmetry' vs 'off': every energy identical at every
        instant (the gauge claim) -- if this fails, STOP, do not pick a
        convention
    C5  per-triad energy-conservation residual ~0 (no wave-set-level
        analogue exists; see wave_sets.py's module docstring)
    C6  energy-drift validity gate: report drift and drift/dEK per mode
        (advisory -- flags loudly, does not hard-fail)
    C7  reproduce the dissertation's hand-typed quartet tables, flag
        discrepancies to NUMBERS-CHECK-section-3.md (advisory)
    C8  batched WaveSet.f/RK44 == looped scalar calls

Every check is a standalone function taking explicit ``(modes, triads,
gamma, N, deg, ...)`` -- never only a registry key -- so it can be run on
an ad-hoc configuration that isn't in any YAML.

Run the built-in default battery (a synthetic triad, quartet and quintet,
not tied to any registry):

    python rsw_sphere/utilities/check_wave_set_physics.py

Run only some checks:

    python rsw_sphere/utilities/check_wave_set_physics.py --check C1,C4

Run against an ad-hoc configuration (4 modes, 2 triads; sum mode listed
first in each ``--triad``):

    python rsw_sphere/utilities/check_wave_set_physics.py \\
        --modes 4,5,3 1,2,3 3,4,3 3,6,3 \\
        --triads 0,1,2 0,1,3

Run against a registered wave set:

    python rsw_sphere/utilities/check_wave_set_physics.py --wave-set quartet_rh_preference
"""
import argparse
import sys

import numpy as np

from rsw_sphere.physics import gamma_from_he
from rsw_sphere.dynamics.dynamic_triads import TRIAD
from rsw_sphere.dynamics.integrators import RK44
from rsw_sphere.dynamics.wave_sets import WaveSet

H_E = 10000.0


# ---------------------------------------------------------------------------
# Synthetic default test cases (not tied to any registry / the paper's
# actual configurations -- those arrive in Phase B via harvest). Each is
# only required to be a *physically valid* wave set (m_sum = m_p + m_q per
# triad) so the checks below have something concrete to run on before the
# registry exists.
# ---------------------------------------------------------------------------

#: 1-triad case: reproduces Triad B (`triad_rossby_only_non_resonant`) from
#: the §2.2 registry -- sum mode RH(4,5) (m=4=1+3), members RH(1,2), RH(3,4).
TRIAD_CASE = dict(
    name="triad (RH(4,5)+RH(1,2)+RH(3,4), matches §2.2 Triad B)",
    modes=[(4, 5, 3), (1, 2, 3), (3, 4, 3)],
    triads=[(0, 1, 2)],
)

#: 2-triad "star" quartet sharing edge {RH(4,5), RH(1,2)}, both private
#: members m=3 -- matches the *topology* of the paper's tab: cap41 quartet
#: (RH(4,5)+RH(1,2) coupled to both RH(3,4) and RH(3,6)); mode values will
#: be reconciled against the real dissertation numbers in Phase B (C7).
QUARTET_CASE = dict(
    name="quartet (RH(4,5)+RH(1,2) x {RH(3,4), RH(3,6)}, star topology)",
    modes=[(4, 5, 3), (1, 2, 3), (3, 4, 3), (3, 6, 3)],
    triads=[(0, 1, 2), (0, 1, 3)],
)

#: 2-triad "chain" quintet sharing a single mode (RH(1,2)) between two
#: otherwise-disjoint triads -- matches the paper's own description of the
#: five-wave topology ("connected by a single mode", §Multiple Triads
#: intro), unlike the star/shared-edge quartet topology above.
QUINTET_CASE = dict(
    name="quintet (chain: RH(4,5)+RH(1,2)+RH(3,4) -- RH(1,2)+RH(3,6)+EG(4,7))",
    modes=[(4, 5, 3), (1, 2, 3), (3, 4, 3), (3, 6, 3), (4, 7, 1)],
    triads=[(0, 1, 2), (4, 1, 3)],
)

DEFAULT_CASES = [TRIAD_CASE, QUARTET_CASE, QUINTET_CASE]


def _build(case, gamma, N, deg, fat_policy='symmetry'):
    return WaveSet(gamma, case["modes"], case["triads"], N=N, deg=deg, fat_policy=fat_policy)


def _synthetic_amplitudes(n_modes, seed=0, scale=0.1):
    """A fixed, reproducible small-amplitude complex IC for integration
    checks -- values are not meant to be physically realistic (see
    ``rsw_sphere.plotting.wave_set_dynamics.velocities_to_amplitudes``,
    Phase C, for that), just non-degenerate (nonzero, distinct phases) so
    the RHS actually exercises every coupling term.
    """
    rng = np.random.default_rng(seed)
    mag = scale * (0.5 + rng.random(n_modes))
    phase = 2 * np.pi * rng.random(n_modes)
    return mag * np.exp(1j * phase)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_c1(case, gamma, N, deg, rtol=1e-12):
    """Each constituent triad's (alpha_p, alpha_q, alpha_sum, delta, S)
    matches an independently-built ``TRIAD`` for the same 3 modes (sum
    mode last). Proves ``WaveSet``'s per-triad coefficient construction is
    identical to the reference implementation, not just "close".
    """
    ws = _build(case, gamma, N, deg)
    problems = []
    for i in range(ws.n_triads):
        T = ws.sub_triad(i)
        checks = [
            ("coef_p vs coef_ABC", ws.alpha[i, 0], T.coef_ABC),
            ("coef_q vs coef_BAC", ws.alpha[i, 1], T.coef_BAC),
            ("coef_sum vs coef_CAB", ws.alpha[i, 2], T.coef_CAB),
            ("delta vs mismatch", ws.delta[i], T.mismatch),
            ("S vs Sabc", ws.S[i], T.Sabc),
        ]
        for label, a, b in checks:
            if not np.isclose(a, b, rtol=rtol, atol=1e-14):
                problems.append(f"triad {i} {label}: {a} != {b}")
    passed = not problems
    detail = "OK" if passed else "; ".join(problems)
    return passed, detail


def check_c2(case, gamma, N, deg, rtol=1e-12):
    """Swapping a triad's two member modes (p<->q) leaves each physical
    mode's own coefficient unchanged (just relabeled p->q or q->p), and
    leaves the sum mode's coefficient, delta and S unchanged. This is the
    claim that ``inner_product``/``S_abc`` are symmetric in the member
    slots -- verified here, not assumed (see ``wave_sets.py`` module
    docstring).
    """
    problems = []
    for i_sum, i_p, i_q in case["triads"]:
        swapped_triads = [t if t != (i_sum, i_p, i_q) else (i_sum, i_q, i_p)
                           for t in case["triads"]]
        ws1 = WaveSet(gamma, case["modes"], case["triads"], N=N, deg=deg)
        ws2 = WaveSet(gamma, case["modes"], swapped_triads, N=N, deg=deg)
        t = case["triads"].index((i_sum, i_p, i_q))
        checks = [
            ("mode p coef (p-slot in ws1 vs q-slot in ws2)", ws1.alpha[t, 0], ws2.alpha[t, 1]),
            ("mode q coef (q-slot in ws1 vs p-slot in ws2)", ws1.alpha[t, 1], ws2.alpha[t, 0]),
            ("sum coef", ws1.alpha[t, 2], ws2.alpha[t, 2]),
            ("delta", ws1.delta[t], ws2.delta[t]),
            ("S", ws1.S[t], ws2.S[t]),
        ]
        for label, a, b in checks:
            if not np.isclose(a, b, rtol=rtol, atol=1e-14):
                problems.append(f"triad(sum={i_sum},p={i_p},q={i_q}) {label}: {a} != {b}")
    passed = not problems
    detail = "OK" if passed else "; ".join(problems)
    return passed, detail


def check_c3(case, gamma, N, deg, t_f=20.0, h=0.01, rtol=1e-10):
    """For a 1-triad ``WaveSet``, the full trajectory (all time points)
    equals ``RK44(TRIAD, ...)`` from the same initial amplitudes. This is
    the end-to-end proof that a wave set's numerator and a TRIAD's
    denominator (as used by the P-measure, Phase C) share a convention --
    C1 alone only proves the coefficients agree at t=0.
    """
    if len(case["triads"]) != 1:
        return True, "skipped (only meaningful for a 1-triad case)"
    ws = _build(case, gamma, N, deg)
    T = ws.sub_triad(0)
    A0_local = _synthetic_amplitudes(3, seed=1)  # order: p, q, sum (TRIAD's a,b,c)
    i_p, i_q, i_sum = ws.sub_triad_local_indices(0)
    A0_ws = np.empty(3, dtype=complex)
    A0_ws[i_p], A0_ws[i_q], A0_ws[i_sum] = A0_local
    Y_ws, T_ws = RK44(ws, 0, t_f, h, A0_ws)
    Y_T, T_T = RK44(T, 0, t_f, h, A0_local)
    Y_ws_reordered = np.stack([Y_ws[:, i_p], Y_ws[:, i_q], Y_ws[:, i_sum]], axis=-1)
    if not np.allclose(Y_ws_reordered, Y_T, rtol=rtol, atol=1e-12):
        max_diff = np.max(np.abs(Y_ws_reordered - Y_T))
        return False, f"trajectories diverge, max abs diff = {max_diff:.3e}"
    return True, f"OK (max abs diff = {np.max(np.abs(Y_ws_reordered - Y_T)):.3e})"


def _find_gauge_signs(case, fat_from, fat_to):
    """Brute-force (n_modes <= ~20, so 2**n_modes is trivial) search for a
    per-mode sign pattern ``s in {+1,-1}^n_modes`` such that, for every
    triad, ``s_p * s_q * s_sum == fat_to[t] / fat_from[t]``.

    This exists (see ``wave_sets.py`` module docstring's gauge argument)
    whenever every triad whose fat must flip has at least one member mode
    not shared with another triad needing a different flip -- true for
    both the star (shared-edge) and chain (single-shared-mode) topologies
    used in this paper, but not asserted in general. Returns ``None`` if
    no solution is found in the brute-force search (which would itself
    contradict the gauge claim and must be investigated, not ignored).
    """
    n = len(case["modes"])
    triads = case["triads"]
    target = [fat_to[t] / fat_from[t] for t in range(len(triads))]
    for bits in range(2 ** n):
        s = np.array([1 if (bits >> j) & 1 == 0 else -1 for j in range(n)])
        ok = True
        for t, (i_sum, i_p, i_q) in enumerate(triads):
            if s[i_p] * s[i_q] * s[i_sum] != target[t]:
                ok = False
                break
        if ok:
            return s
    return None


def check_c4(case, gamma, N, deg, t_f=20.0, h=0.01, rtol=1e-10):
    """``fat_policy='symmetry'`` vs ``'off'`` are related by a per-mode
    amplitude sign gauge (``A_j -> s_j A_j``): find that gauge explicitly
    (``_find_gauge_signs``), apply it to the initial condition, and check
    that the two systems' trajectories then agree exactly (not just in
    magnitude) at **every timestep** -- the precise, checkable form of the
    gauge-invariance claim in the ``wave_sets.py`` module docstring.

    Comparing the two systems from the *same, ungauged* initial condition
    (an earlier version of this check did that) is **not** a valid test:
    a sign-convention change that isn't compensated in the initial data is
    a genuinely different initial-value problem, so of course it diverges
    -- that failure mode was caught by running this corrected check and
    seeing it now pass where the naive version failed.

    **If no gauge exists, or the gauged trajectories disagree: STOP and
    escalate -- do not pick a convention.**
    """
    ws_sym = _build(case, gamma, N, deg, fat_policy='symmetry')
    ws_off = _build(case, gamma, N, deg, fat_policy='off')

    s = _find_gauge_signs(case, ws_sym.fat, ws_off.fat)
    if s is None:
        return False, "no per-mode sign gauge reproduces fat_policy='off' from 'symmetry' -- the two are NOT equivalent conventions for this topology"

    A0_sym = _synthetic_amplitudes(ws_sym.n_modes, seed=2)
    A0_off = s * A0_sym

    Y_sym, _ = RK44(ws_sym, 0, t_f, h, A0_sym)
    Y_off, _ = RK44(ws_off, 0, t_f, h, A0_off)

    Y_off_ungauged = Y_off / s  # should reproduce Y_sym exactly if the gauge is correct
    if not np.allclose(Y_off_ungauged, Y_sym, rtol=rtol, atol=1e-12):
        max_diff = np.max(np.abs(Y_off_ungauged - Y_sym))
        return False, (f"gauge found ({s.tolist()}) but trajectories disagree after "
                        f"un-gauging, max abs diff = {max_diff:.3e}")

    E2_sym, E3_sym = ws_sym.energy(Y_sym)
    E2_off, E3_off = ws_off.energy(Y_off)
    if not (np.allclose(E2_sym, E2_off, rtol=rtol) and np.allclose(E3_sym, E3_off, rtol=rtol, atol=1e-12)):
        return False, "E2(t)/E3(t) differ between the gauge-equivalent systems"

    return True, (f"OK -- gauge s={s.tolist()} makes fat_policy='off' trajectories "
                   f"exactly reproduce 'symmetry' trajectories at every timestep "
                   f"(max diff {np.max(np.abs(Y_off_ungauged - Y_sym)):.3e}); "
                   f"E2(t)/E3(t) identical either way")


def check_c5(case, gamma, N, deg, tol=1e-6):
    """Per-constituent-triad energy-conservation residual ~0 (already
    computed and warned on inside ``WaveSet.__init__`` -- this check just
    asserts it explicitly). **No wave-set-level analogue exists** (the
    quartet truncation does not conserve energy in general); do not add
    one here.
    """
    ws = _build(case, gamma, N, deg)
    bad = np.abs(ws.residual) > tol
    if np.any(bad):
        return False, f"residuals exceed {tol}: {ws.residual[bad]}"
    return True, f"OK (max |residual| = {np.max(np.abs(ws.residual)):.3e})"


def check_c6(case, gamma, N, deg, t_f=40.0, h=0.01):
    """Advisory energy-drift validity gate (never hard-fails): report
    ``drift = max_t|E_T(t) - E_T(0)| / |E_T(0)|`` and, per mode,
    ``drift / dEK`` where ``dEK = max_t|A_j|^2 - min_t|A_j|^2``. If
    ``drift/dEK`` is not small for a mode, any P-measure or ``ΔEK``-derived
    quantity for that mode is measuring truncation error, not physics --
    flag loudly rather than silently proceed to figure generation.
    """
    ws = _build(case, gamma, N, deg)
    A0 = _synthetic_amplitudes(ws.n_modes, seed=3)
    Y, T = RK44(ws, 0, t_f, h, A0)
    E2, E3 = ws.energy(Y)
    E_T = E2 + E3
    drift = np.max(np.abs(E_T - E_T[0])) / np.abs(E_T[0])

    per_mode = np.real(Y * np.conj(Y))
    dEK = per_mode.max(axis=0) - per_mode.min(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(dEK > 1e-14, drift * np.abs(E_T[0]) / np.maximum(dEK, 1e-300), np.inf)

    lines = [f"drift = {drift:.3e} (relative to E_T(0))"]
    flagged = False
    for j in range(ws.n_modes):
        flag = " <-- LARGE, P/ΔEK unreliable for this mode" if ratio[j] > 0.2 else ""
        if flag:
            flagged = True
        lines.append(f"  mode {j} ({ws.labels[j]}): dEK={dEK[j]:.3e}, drift/dEK={ratio[j]:.3e}{flag}")
    detail = "\n".join(lines)
    return (not flagged), detail  # reported as advisory in main(), never blocks


def check_c7(specs=None):
    """Advisory: reproduce published (dissertation/paper) quartet tables
    from fresh code and report discrepancies. Placeholder until Phase B
    (harvest) supplies the actual mode tuples and hand-typed values to
    compare against -- flags itself as not-yet-runnable rather than
    silently passing.
    """
    return True, "skipped (Phase B harvest not yet done -- no published values registered to compare against)"


def check_c8(case, gamma, N, deg, rtol=1e-12):
    """Batched ``WaveSet.f`` (called once on a stacked array of several
    initial conditions) agrees with calling ``f`` separately on each
    initial condition in a Python loop -- proves the broadcasting
    implementation used for whole-grid sweeps (Phase C) is equivalent to
    the naive per-point computation, not just fast.
    """
    ws = _build(case, gamma, N, deg)
    batch = np.stack([_synthetic_amplitudes(ws.n_modes, seed=s) for s in range(5)], axis=0)
    f_batched = ws.f(batch)
    f_looped = np.stack([ws.f(batch[k]) for k in range(batch.shape[0])], axis=0)
    if not np.allclose(f_batched, f_looped, rtol=rtol, atol=1e-14):
        max_diff = np.max(np.abs(f_batched - f_looped))
        return False, f"batched vs looped f() differ, max abs diff = {max_diff:.3e}"

    # Also check a short batched RK44 integration matches per-row scalar runs.
    Y_batched, _ = RK44(ws, 0, 2.0, 0.01, batch)
    Y_looped = np.stack([RK44(ws, 0, 2.0, 0.01, batch[k])[0] for k in range(batch.shape[0])], axis=1)
    if not np.allclose(Y_batched, Y_looped, rtol=rtol, atol=1e-12):
        max_diff = np.max(np.abs(Y_batched - Y_looped))
        return False, f"batched vs looped RK44 differ, max abs diff = {max_diff:.3e}"
    return True, "OK (batched f() and RK44 match looped scalar calls exactly)"


CHECKS = {
    "C1": ("per-triad coefficients vs independent TRIAD", check_c1, True),
    "C2": ("member-swap symmetry", check_c2, True),
    "C3": ("1-triad WaveSet trajectory == TRIAD trajectory", check_c3, True),
    "C4": ("fat_policy gauge invariance", check_c4, True),
    "C5": ("per-triad energy-conservation residual", check_c5, True),
    "C6": ("energy-drift validity (advisory)", check_c6, False),
    "C7": ("reproduce published tables (advisory)", lambda *a, **k: check_c7(), False),
    "C8": ("batched vs scalar f()/RK44", check_c8, True),
}


def run_all(cases, N, deg, which=None):
    gamma = gamma_from_he(H_E)[1]
    which = which or list(CHECKS.keys())
    any_hard_fail = False
    print(f"gamma_from_he({H_E}) -> gamma={gamma:.8f}, N={N}, deg={deg}\n")
    for case in cases:
        print(f"=== {case['name']} ===")
        for key in which:
            if key not in CHECKS:
                print(f"  {key}: unknown check, skipping")
                continue
            desc, fn, is_hard = CHECKS[key]
            try:
                passed, detail = fn(case, gamma, N, deg)
            except Exception as e:
                passed, detail = False, f"raised {type(e).__name__}: {e}"
            status = "PASS" if passed else ("FAIL" if is_hard else "ADVISORY-FLAG")
            print(f"  [{status}] {key} ({desc})")
            for line in detail.splitlines():
                print(f"      {line}")
            if is_hard and not passed:
                any_hard_fail = True
        print()
    return not any_hard_fail


def _parse_mode(s):
    m, n, alpha = (int(x) for x in s.split(","))
    return (m, n, alpha)


def _parse_triad(s):
    i_sum, i_p, i_q = (int(x) for x in s.split(","))
    return (i_sum, i_p, i_q)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", nargs="+", type=_parse_mode, default=None,
                         help='ad-hoc configuration: modes as "m,n,alpha" '
                              '(space-separated). Requires --triads too.')
    parser.add_argument("--triads", nargs="+", type=_parse_triad, default=None,
                         help='ad-hoc configuration: triads as "i_sum,i_p,i_q" '
                              '(space-separated indices into --modes).')
    parser.add_argument("--wave-set", default=None,
                         help="role key from the wave-set registry YAML "
                              "(Phase B; not yet available).")
    parser.add_argument("--check", default=None,
                         help="comma-separated subset of checks to run "
                              f"(default: all of {','.join(CHECKS)}).")
    parser.add_argument("--N", type=int, default=10, help="Hough truncation order.")
    parser.add_argument("--deg", type=int, default=300, help="quadrature degree.")
    args = parser.parse_args()

    which = args.check.split(",") if args.check else None

    if args.wave_set:
        try:
            from rsw_sphere.dynamics.wave_set_specs import load_wave_set_specs
        except ImportError:
            parser.error("--wave-set requires Phase B's wave_set_specs.py, not yet built.")
        specs = load_wave_set_specs()
        if args.wave_set not in specs:
            parser.error(f"'{args.wave_set}' not in registry: {', '.join(specs)}")
        spec = specs[args.wave_set]
        cases = [dict(name=args.wave_set, modes=spec.modes,
                      triads=[spec.triad_indices(i) for i in range(len(spec.triads))])]
    elif args.modes and args.triads:
        cases = [dict(name="ad-hoc configuration", modes=args.modes, triads=args.triads)]
    elif args.modes or args.triads:
        parser.error("--modes and --triads must be given together.")
    else:
        cases = DEFAULT_CASES

    ok = run_all(cases, args.N, args.deg, which=which)
    if ok:
        print("ALL HARD CHECKS PASS.")
        sys.exit(0)
    else:
        print("HARD CHECK(S) FAILED -- do not proceed to Phase B/C.")
        sys.exit(1)


if __name__ == "__main__":
    main()
