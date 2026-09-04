"""Puts the repository root on ``sys.path`` so ``rsw_sphere`` imports when a
script in this directory is run directly from a checkout that has not been
``pip install -e .``-ed.

Every script here opens with ``import _bootstrap``; ``_bootstrap.ROOT`` is
the repository root, for anchoring output paths. Identical copies live in
the sibling ``examples/*/`` script directories -- Python only ever puts the
running script's OWN directory on ``sys.path``, so a single shared copy one
level up would not be importable.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
