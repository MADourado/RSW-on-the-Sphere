"""Shared cache-key hashing for expensive 2D parameter sweeps (triad
efficiency maps, and -- planned -- wave-set P-measure/period-difference
maps).

``cache_key_hash`` was defined in ``triad_efficiency.py``; moved here so the
wave-set sweep modules (``rsw_sphere.plotting.wave_set_pmeasure`` etc.) can
reuse the same hashing discipline without re-deriving it. **The payload
tuple and its ordering must not change** -- the ``.npz`` caches already on
disk under ``outputs/figures/triads/`` are keyed by this function's exact
output for triad sweeps; changing the payload silently invalidates them
(the same failure mode as the stale-NaN bug this function was written to
fix in the first place).

Run as a quick sanity check:

    python -m rsw_sphere.plotting.sweeps
"""
import hashlib


def wave_set_cache_key_hash(modes, triads, h_e, swept_indices, fixed_velocities,
                             target_indices, reference_triad, n_grid, t_f, h,
                             N=10, deg=300):
    """Short hash of every parameter that changes a wave-set 2D sweep's
    result (``rsw_sphere.plotting.wave_set_pmeasure.p_measure_sweep``),
    for a ``.npz`` cache filename that auto-invalidates on parameter
    change. Sibling of ``cache_key_hash`` (the triad-sweep version) --
    kept as a **separate** function rather than overloading that one,
    since its own payload/ordering is pinned to the triad caches already
    on disk (see its docstring) and must not change shape.

    Returns
    -------
    str
        8 hex characters.

    Examples
    --------
    >>> h1 = wave_set_cache_key_hash([(4,5,3),(3,4,3),(1,2,3),(1,1,1)],
    ...     [(0,1,2)], 10000, (2,3), {0: 30.0, 1: 30.0}, [0, 1], 0, 8, 100.0, 0.01)
    >>> h2 = wave_set_cache_key_hash([(4,5,3),(3,4,3),(1,2,3),(1,1,1)],
    ...     [(0,1,2)], 10000, (2,3), {0: 30.0, 1: 30.0}, [0, 1], 0, 8, 100.0, 0.01)
    >>> h1 == h2 and len(h1) == 8
    True
    """
    payload = repr((
        tuple(tuple(m) for m in modes), tuple(tuple(t) for t in triads), h_e,
        tuple(swept_indices), tuple(sorted(fixed_velocities.items())),
        tuple(target_indices), reference_triad, n_grid, t_f, h, N, deg))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def cache_key_hash(modes, h_e, target, fixed_index, fixed_velocity,
                    u1_range, u2_range, n_grid, t_f, h, N=10, deg=300):
    """Short hash of every parameter that changes a 2D sweep's result, for
    building a ``.npz`` cache filename that auto-invalidates when a sweep
    parameter changes.

    Fixes the stale-cache bug logged in the §2.2 plan's "Known issues"
    section: caching by triad name alone silently serves an old sweep's
    result after any of ``target``/``fixed_index``/velocity ranges/
    ``n_grid``/``t_f``/``h`` change (this happened in practice -- e.g. a
    ``gravity_catalyst`` cache computed before the ``E_0==0`` guard was
    added kept a stale NaN baked into a shipped figure).

    Returns
    -------
    str
        8 hex characters, stable across runs for identical parameters
        (``hashlib.sha1`` of the repr of every argument).

    Examples
    --------
    >>> h1 = cache_key_hash([(4,5,3),(1,2,3),(3,10,3)], 10000, 0, 0, 0.0,
    ...                     (0.0,100.0), (0.0,100.0), 40, 100.0, 0.001)
    >>> h2 = cache_key_hash([(4,5,3),(1,2,3),(3,10,3)], 10000, 0, 0, 0.0,
    ...                     (0.0,100.0), (0.0,100.0), 40, 100.0, 0.001)
    >>> h1 == h2 and len(h1) == 8
    True
    """
    payload = repr((tuple(tuple(m) for m in modes), h_e, target, fixed_index,
                     fixed_velocity, tuple(u1_range), tuple(u2_range),
                     n_grid, t_f, h, N, deg))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


if __name__ == "__main__":
    import doctest
    failures, _ = doctest.testmod()
    if failures == 0:
        print("sweeps doctest OK")
