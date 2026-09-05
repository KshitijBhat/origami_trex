"""Origami x T-Rex package. Implements REDESIGN_PLAN.md.

Puts the T-Rex submodule on sys.path so upstream modules (e.g. ``utils.lerobot_common``,
``qwen_vla``, ``scripts.test``) import the same way upstream's own scripts do, per
REDESIGN_PLAN.md §13.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_ROOT = _REPO_ROOT / "T-Rex"

if _TREX_ROOT.is_dir():
    _trex_str = str(_TREX_ROOT)
    if _trex_str not in sys.path:
        sys.path.insert(0, _trex_str)
