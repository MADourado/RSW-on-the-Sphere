"""Raw ODE-trajectory caching for ``WaveSet``/``RK44`` runs.

Every sweep-style script in this repository used to
re-integrate from scratch on every run -- no ``.npz`` cache anywhere in the
repo stores the raw ``Y(t)`` solution itself, only summary/sweep arrays
derived from it. That makes re-analysis (e.g. checking a different
diagnostic on an already-run trajectory) always pay the integration cost
again. This module is the one piece with no existing analogue elsewhere in
the repo (unlike ``rsw_sphere.plotting.sweeps``' 2D-sweep hashing, which
this reuses the *convention* of but not the function itself, since a
trajectory cache key is shaped differently from a 2D-sweep cache key).

**Cache layout, 2026-08-25: grouped by topology, named by initial
condition.** ``<output_root>/<topology>/<ic_label>_tf<days>_h<h>_<hash8>.npz``,
where ``topology`` is auto-derived from the wave set's own mode count
(``triads``/``quartets``/``quintets``/``n<k>modes``) and ``ic_label`` is
built from every mode's own label + velocity (``ic_label()`` below),
canonically sorted so the same physical configuration always produces the
same filename. Previously grouped by a caller-supplied ``wave_set_key``
string instead (e.g. ``quartet_rh_preference``, or a hand-invented shared
tag like ``quartet_b_rsw`` used by three different scripts specifically
to make them share a cache namespace) with a caller-crafted ``run_label``
that usually only named the *swept* mode, not every mode's own IC. The
new scheme makes that manual convention unnecessary: any two call sites
that build the same modes at the same velocities land in the same cache
entry automatically, which is the actual point of caching trajectories in
the first place (reuse across scripts, not just re-runs of one script).

Run as a quick self-check (verifies a cache hit returns byte-identical
``Y``/``T`` without re-integrating):

    python -m rsw_sphere.dynamics.trajectory_cache
"""
import hashlib
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from rsw_sphere.dynamics.integrators import RK44

#: Cache subfolder per wave-set size -- extend this, not a hardcoded
#: "triads"/"quartets" branch, if a larger topology (e.g. a 6-mode
#: double-star) is ever registered.
_TOPOLOGY_FOLDERS = {3: "triads", 4: "quartets", 5: "quintets"}

_FAMILY_SLUG = {1: "eg", 2: "wg", 3: "rh"}


def _mode_slug(m, n, alpha):
    """Filesystem-safe, lowercase mode tag, e.g. ``rh45``, ``eg11`` --
    the ``ic_label``/cache-filename analogue of
    ``rsw_sphere.plotting.labels._mode_label``'s paper-facing ``RH(4,5)``.
    """
    return f"{_FAMILY_SLUG[alpha]}{m}{n}"


def topology_folder(n_modes: int) -> str:
    """Cache subfolder name for a wave set of ``n_modes`` modes."""
    return _TOPOLOGY_FOLDERS.get(n_modes, f"n{n_modes}modes")


def ic_label(modes, velocities) -> str:
    """Canonical, readable label encoding every mode's own family/
    wavenumber and initial zonal velocity, e.g. ``eg11_0.00-rh12_50.00-
    rh24_30.00-rh34_15.00``. Sorted by mode tuple (not registration
    order), so the same physical configuration produces the same label
    regardless of which order a caller happened to list its modes in --
    see the module docstring's "reusability" rationale.
    """
    pairs = sorted(zip(modes, velocities), key=lambda mv: mv[0])
    return "-".join(f"{_mode_slug(*m)}_{v:.2f}" for m, v in pairs)


def _trajectory_cache_key_hash(modes, triads, gamma, N, deg, A0, t_f, h):
    """8-hex-character hash of every parameter that changes a ``WaveSet``
    trajectory's numerical result: modes, triads, ``gamma`` (equivalent to
    ``h_e``, which ``WaveSet`` does not store directly), truncation
    (``N``, ``deg``), initial amplitudes ``A0`` (equivalent to the
    velocities that produced them), ``t_f``, ``h``. Same ``sha1(...)[:8]``
    convention as ``rsw_sphere.plotting.sweeps.cache_key_hash`` /
    ``wave_set_cache_key_hash``.
    """
    A0 = np.asarray(A0)
    payload = repr((
        tuple(tuple(m) for m in modes), tuple(tuple(t) for t in triads),
        round(float(gamma), 12), N, deg,
        tuple(complex(a) for a in A0.ravel()), float(t_f), float(h)))
    return hashlib.sha1(payload.encode()).hexdigest()[:8]


def run_and_cache(ws, A0, t_f, h, velocities=None, output_root="outputs/trajectories", label=None):
    """Cache-or-compute a raw ``WaveSet`` trajectory.

    Parameters
    ----------
    ws : rsw_sphere.dynamics.wave_sets.WaveSet
        Already-built wave set (modes/triads/gamma/N/deg all read from it
        for the cache key, and ``n_modes`` for the cache subfolder --
        see ``topology_folder``).
    A0 : ndarray
        Initial complex amplitudes (``ws.amplitudes_from_velocities(...)``).
    t_f, h : float
        Nondimensional integration horizon / step (``RK44``'s own args).
    velocities : sequence of float or None, optional
        Initial zonal velocities (m/s), one per mode, in ``ws.modes``'s
        own order -- the same array ``A0`` was built from. Used to build
        a readable, canonical filename (``ic_label``) so the SAME
        physical configuration lands in the same cache entry regardless
        of which script built it (module docstring). Required unless
        ``label`` is given explicitly.
    output_root : str, optional
        Default ``"outputs/trajectories"``.
    label : str or None, optional
        Explicit override for the readable filename part (before the
        hash suffix). Default: ``ic_label(ws.modes, velocities)`` plus
        ``t_f``/``h`` (as integration days / step, for a human browsing
        the cache folder -- the hash alone already covers correctness).

    Returns
    -------
    Y : ndarray, shape (n_times, n_modes), complex
    T : ndarray, shape (n_times,)
    path : str
        The ``.npz`` cache path used (whether just read or just written).
    """
    key_hash = _trajectory_cache_key_hash(ws.modes, ws.triads, ws.gamma, ws.N, ws.deg, A0, t_f, h)
    if label is None:
        if velocities is None:
            raise ValueError("run_and_cache needs `velocities` to build a readable cache "
                              "label, or an explicit `label` override")
        tf_days = t_f / (4 * np.pi)
        label = f"{ic_label(ws.modes, velocities)}_tf{tf_days:.0f}_h{h:g}"
    out_dir = os.path.join(output_root, topology_folder(ws.n_modes))
    path = os.path.join(out_dir, f"{label}_{key_hash}.npz")

    if os.path.exists(path):
        data = np.load(path)
        return data['Y'], data['T'], path

    Y, T = RK44(ws, 0, t_f, h, A0)

    os.makedirs(out_dir, exist_ok=True)
    np.savez(path, Y=Y, T=T, omega=ws.omega, delta=ws.delta, alpha=ws.alpha,
              modes=np.array(ws.modes), A0=np.asarray(A0), t_f=t_f, h=h)
    return Y, T, path


if __name__ == "__main__":
    import shutil
    import time

    from rsw_sphere.physics import gamma_from_he
    from rsw_sphere.dynamics.wave_sets import WaveSet

    G, H_E = 9.8, 10000.0
    gamma = gamma_from_he(H_E, g=G)[1]
    modes = [(4, 5, 3), (3, 4, 3), (1, 2, 3)]
    velocities = [30.0, 30.0, 30.0]
    ws = WaveSet(gamma, modes, [(0, 1, 2)], N=10, deg=300)
    A0 = ws.amplitudes_from_velocities(velocities, H_E, g=G)
    t_f = 20 * 4 * np.pi

    tmp_root = "outputs/trajectories/_selftest_tmp"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)

    t0 = time.time()
    Y1, T1, path1 = run_and_cache(ws, A0, t_f, 0.01, velocities=velocities, output_root=tmp_root)
    first_call_s = time.time() - t0

    t0 = time.time()
    Y2, T2, path2 = run_and_cache(ws, A0, t_f, 0.01, velocities=velocities, output_root=tmp_root)
    second_call_s = time.time() - t0

    assert path1 == path2, "cache path must be stable across identical calls"
    assert os.path.join(tmp_root, "triads") in path1, "3-mode wave set must land in the 'triads' subfolder"
    assert np.array_equal(T1, T2), "T must be byte-identical on a cache hit"
    assert np.allclose(Y1, Y2), "Y must match on a cache hit"
    assert second_call_s < first_call_s / 3, (
        f"cache hit ({second_call_s:.3f}s) should be far faster than the "
        f"initial integration ({first_call_s:.3f}s) -- caching may not be working")

    shutil.rmtree(tmp_root)
    print(f"trajectory_cache self-check OK: first call {first_call_s:.3f}s, "
          f"cached call {second_call_s:.3f}s")
