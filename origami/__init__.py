"""Origami x T-Rex package. Implements REDESIGN_PLAN.md.

Puts the T-Rex submodule on sys.path so upstream modules (e.g. ``utils.lerobot_common``,
``qwen_vla``, ``scripts.test``) import the same way upstream's own scripts do, per
REDESIGN_PLAN.md §13.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_ROOT = _REPO_ROOT / "T-Rex"

# A plain `git clone` (without --recurse-submodules) still creates an empty T-Rex/
# placeholder directory, so `_TREX_ROOT.is_dir()` alone doesn't detect an uninitialized
# submodule -- it would silently add the empty dir to sys.path and defer the failure to
# whatever imports `utils.lerobot_common`/`qwen_vla` later, as a confusing
# "ModuleNotFoundError: No module named 'utils'" with no mention of T-Rex/ at all.
_TREX_MARKER = _TREX_ROOT / "utils" / "lerobot_common.py"
if not _TREX_MARKER.is_file():
    raise ImportError(
        f"T-Rex/ submodule looks uninitialized (missing {_TREX_MARKER}). "
        "Run `git submodule update --init --recursive` from the repo root, then retry."
    )

_trex_str = str(_TREX_ROOT)
if _trex_str not in sys.path:
    sys.path.insert(0, _trex_str)
