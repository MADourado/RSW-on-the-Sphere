"""Raw ODE-trajectory caching for ``WaveSet``/``RK33`` runs.

Every sweep-style script in this repository (``examples/quartet_precession_sweep.py``,
``examples/precession_sweep_figure.py``, the ``wave_set_*`` sweep modules)
re-integrates from scratch on every run -- no ``.npz`` cache anywhere in the
repo stores the raw ``Y(t)`` solution itself, only summary/sweep arrays
derived from it. That makes re-analysis (e.g. checking a different
diagnostic on an already-run trajectory) always pay the integration cost
again. This module is the one piece with no existing analogue elsewhere in
the repo (unlike ``rsw_sphere.plotting.sweeps``' 2D-sweep hashing, which
this reuses the *convention* of but not the function itself, since a
trajectory cache key is shaped differently from a 2D-sweep cache key).

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

from rsw_sphere.dynamics.integrators import RK33


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


def run_and_cache(ws, A0, t_f, h, wave_set_key, run_label, output_root="outputs/trajectories"):
    """Cache-or-compute a raw ``WaveSet`` trajectory.

    Parameters
    ----------
    ws : rsw_sphere.dynamics.wave_sets.WaveSet
        Already-built wave set (modes/triads/gamma/N/deg all read from it
        for the cache key).
    A0 : ndarray
        Initial complex amplitudes (``ws.amplitudes_from_velocities(...)``).
    t_f, h : float
        Nondimensional integration horizon / step (``RK33``'s own args).
    wave_set_key : str
        Registry role key (e.g. ``quartet_rh_preference``) -- names the
        cache subfolder. Use any short, filesystem-safe tag for an ad-hoc
        (non-registry) wave set.
    run_label : str
        Human-readable tag for the varying parameter(s) of this specific
        run (e.g. ``"d85.00_tf150_h0.01"``) -- the hash alone would be
        opaque; this keeps the cache directory browsable.
    output_root : str, optional
        Default ``"outputs/trajectories"``.

    Returns
    -------
    Y : ndarray, shape (n_times, n_modes), complex
    T : ndarray, shape (n_times,)
    path : str
        The ``.npz`` cache path used (whether just read or just written).
    """
    key_hash = _trajectory_cache_key_hash(ws.modes, ws.triads, ws.gamma, ws.N, ws.deg, A0, t_f, h)
    out_dir = os.path.join(output_root, wave_set_key)
    path = os.path.join(out_dir, f"{run_label}_{key_hash}.npz")

    if os.path.exists(path):
        data = np.load(path)
        return data['Y'], data['T'], path

    Y, T = RK33(ws, 0, t_f, h, A0)

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
    ws = WaveSet(gamma, modes, [(0, 1, 2)], N=10, deg=300)
    A0 = ws.amplitudes_from_velocities([30.0, 30.0, 30.0], H_E, g=G)
    t_f = 20 * 4 * np.pi

    tmp_root = "outputs/trajectories/_selftest_tmp"
    if os.path.exists(tmp_root):
        shutil.rmtree(tmp_root)

    t0 = time.time()
    Y1, T1, path1 = run_and_cache(ws, A0, t_f, 0.01, "_selftest_tmp", "check", output_root=tmp_root)
    first_call_s = time.time() - t0

    t0 = time.time()
    Y2, T2, path2 = run_and_cache(ws, A0, t_f, 0.01, "_selftest_tmp", "check", output_root=tmp_root)
    second_call_s = time.time() - t0

    assert path1 == path2, "cache path must be stable across identical calls"
    assert np.array_equal(T1, T2), "T must be byte-identical on a cache hit"
    assert np.allclose(Y1, Y2), "Y must match on a cache hit"
    assert second_call_s < first_call_s / 3, (
        f"cache hit ({second_call_s:.3f}s) should be far faster than the "
        f"initial integration ({first_call_s:.3f}s) -- caching may not be working")

    shutil.rmtree(tmp_root)
    print(f"trajectory_cache self-check OK: first call {first_call_s:.3f}s, "
          f"cached call {second_call_s:.3f}s")
