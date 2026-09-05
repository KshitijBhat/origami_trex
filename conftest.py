"""Ensures T-Rex/ is on sys.path (via origami/__init__.py) before any test imports
upstream modules such as ``utils.lerobot_common`` or ``qwen_vla``.
"""

import origami  # noqa: F401
