"""Repository-anchored filesystem paths.

Every default output location is derived from the repository root rather
than the current working directory, so a script produces the same files
whether it is run from the repo root, from ``examples/figures/``, or from
anywhere else.
"""
import os

#: Repository root (the directory holding ``pyproject.toml``).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Default registry of wave sets, read by all five drivers.
DEFAULT_WAVESETS_PATH = os.path.join(REPO_ROOT, "wave_sets_default.yaml")

#: Root of every generated artifact (gitignored, fully regenerable).
OUTPUT_ROOT = os.path.join(REPO_ROOT, "outputs")

#: Raw integrated trajectories, shared across scripts
#: (``rsw_sphere.dynamics.trajectory_cache``).
TRAJECTORY_ROOT = os.path.join(OUTPUT_ROOT, "trajectories")


def resolve(path: str) -> str:
    """Absolute path for ``path``, resolving a relative one against the
    repository root instead of the current working directory."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)
