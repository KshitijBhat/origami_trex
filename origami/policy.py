"""Checkpoint -> eef-62 chunk (slow/fast). REDESIGN_PLAN.md §9.4, §13, §12 step 13.

Thin wrapper around T-Rex's own ``scripts.test.{model_load,CascadedServer}`` -- never
reimplements the cascaded slow/fast flow-matching inference. ``Policy.slow_and_fast``/``fast``
call ``CascadedServer._run_slow``/``_run_fast`` directly with ``PIL.Image`` objects and raw
numpy arrays, bypassing ``predict()``'s PNG encode/decode round trip that the ZMQ wire protocol
needs but an in-process deploy server (``serve_zenoh.py``) does not.

Everything about how the checkpoint wants to be fed -- the training instruction, per-view
image size, the cascaded step schedule, the tactile delay curriculum offsets, the VQ-VAE
window -- is sourced from the checkpoint's own ``training_args.json`` (§7.2 delta 4), never
re-derived or hardcoded, mirroring ``eval_offline.py::build_eval_args``.

Real-checkpoint gap (recorded in PROGRESS.md, not fixed here): ``sept9_ckpt``'s
``training_args.json["locked_config"]`` holds only ``{"digest": ...}``, not the full
``LockedConfig`` array fields (``train_origami.py``'s ``save_checkpoint`` only ever wrote the
digest). ``resolve_locked_config`` below recovers the full values from an external
``meta/origami_prep.json`` (a prep root's, e.g. the val root's) and asserts its digest matches
the checkpoint's recorded one -- the same G1c cross-check the plan calls for, just sourced from
wherever the full values still live rather than from the checkpoint itself. Likewise
``training_args.json["instruction"]`` is recorded as ``null`` on this checkpoint;
``resolve_instruction`` falls back to ``origami.constants.INSTRUCTION`` (the value actually used
at prep time, per ``meta/origami_prep.json["instruction"]``) with a one-time warning.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TREX_DIR = _REPO_ROOT / "T-Rex"
if str(_TREX_DIR) not in sys.path:
    sys.path.insert(0, str(_TREX_DIR))

import origami.trex_patch as _trex_patch  # noqa: E402

_trex_patch.apply()

from origami.constants import INSTRUCTION  # noqa: E402
from origami.kinematics import LockedConfig  # noqa: E402

REQUIRED_ACTION_DIM = 62
REQUIRED_ACTION_CHUNK = 16


def load_training_args(checkpoint_path: str | Path) -> dict:
    with open(Path(checkpoint_path) / "training_args.json") as f:
        return json.load(f)


def resolve_instruction(ta: dict) -> str:
    """§5.7/G14: the string the checkpoint was trained on. Falls back to
    ``origami.constants.INSTRUCTION`` (with a warning) when the checkpoint's own
    ``training_args.json["instruction"]`` is missing/null -- a real gap on at least one
    checkpoint produced by this project (see module docstring)."""
    instruction = ta.get("instruction")
    if not instruction:
        logger.warning(
            "training_args.json has no recorded instruction; falling back to "
            "origami.constants.INSTRUCTION. Verify this matches what the checkpoint was "
            "actually trained on (check the prep root's meta/origami_prep.json['instruction'])."
        )
        return INSTRUCTION
    return instruction


def resolve_locked_config(ta: dict, locked_config_source: str | Path | None) -> LockedConfig:
    """§3.2/G1c: the ``LockedConfig`` the checkpoint's ``state62``/action frame was built in.

    Tries the checkpoint's own ``training_args.json["locked_config"]`` first (the intended
    path, per REDESIGN_PLAN.md §9.4 item 3). If it only carries a digest (no array fields --
    the real gap this checkpoint has), falls back to ``locked_config_source``: either a JSON
    file whose top level *is* a ``LockedConfig`` dict, or a ``meta/origami_prep.json`` (or its
    parent dataset root) carrying one under a ``"locked_config"`` key. Either way, the
    recovered config's digest is asserted to match the checkpoint's recorded digest -- the
    actual G1c cross-check, just sourced from wherever the full values still live.
    """
    lc = ta.get("locked_config") or {}
    if {"lower_body", "neck", "left_hand", "right_hand"} <= set(lc):
        return LockedConfig(
            lower_body=np.asarray(lc["lower_body"], dtype=np.float64),
            neck=np.asarray(lc["neck"], dtype=np.float64),
            left_hand=np.asarray(lc["left_hand"], dtype=np.float64),
            right_hand=np.asarray(lc["right_hand"], dtype=np.float64),
        )

    expected_digest = lc.get("digest")
    if locked_config_source is None:
        raise ValueError(
            "checkpoint's training_args.json['locked_config'] has no array fields (only "
            f"{sorted(lc)}) -- pass --locked-config-source pointing at a meta/origami_prep.json "
            "(or dataset root containing one) that has the full LockedConfig."
        )
    source_path = Path(locked_config_source)
    if source_path.is_dir():
        source_path = source_path / "meta" / "origami_prep.json"
    with open(source_path) as f:
        payload = json.load(f)
    raw = payload.get("locked_config", payload)  # allow a bare LockedConfig JSON too
    if expected_digest is not None and raw.get("digest") != expected_digest:
        raise ValueError(
            f"LockedConfig digest mismatch (G1c): checkpoint recorded {expected_digest!r}, "
            f"{source_path} has {raw.get('digest')!r}"
        )
    return LockedConfig(
        lower_body=np.asarray(raw["lower_body"], dtype=np.float64),
        neck=np.asarray(raw["neck"], dtype=np.float64),
        left_hand=np.asarray(raw["left_hand"], dtype=np.float64),
        right_hand=np.asarray(raw["right_hand"], dtype=np.float64),
    )


def build_model_load_args(
    checkpoint_path: str | Path, ta: dict, cuda: str = "0", disable_tactile: int = 0
) -> SimpleNamespace:
    args = SimpleNamespace(**ta)
    args.checkpoint_path = str(checkpoint_path)
    args.base_model_path = ""
    args.stats_path = ""
    args.dataset_name = ""
    args.cuda = cuda
    args.image_size = None  # Policy resizes per-view itself (§4.5-B); avoid double-resize.
    args.disable_tactile = int(disable_tactile)
    return args


def _view_size_wh(image_size, key: str) -> tuple[int, int]:
    """Returns a PIL ``(W, H)`` size for view ``key`` in {"head","wrist_left","wrist_right"}.

    ``image_size`` is either a single ``[W, H]`` (upstream, one size for every view) or the
    per-view dict this project's ``train_origami.py`` records (§4.5-B): ``{"shared", "head",
    "wrist_left", "wrist_right"}``, each ``[W, H]``.
    """
    if isinstance(image_size, dict):
        wh = image_size.get(key, image_size.get("shared"))
    else:
        wh = image_size
    if wh is None:
        raise ValueError(f"no image_size recorded for view {key!r}: {image_size!r}")
    return int(wh[0]), int(wh[1])


class Policy:
    """Wraps ``scripts.test.CascadedServer`` for direct in-process slow/fast inference.

    Does not know about kinematics/retargeting (that is ``retarget.py``/``serve_zenoh.py``'s
    job) -- this class only turns (images, tactile, optional 62-D state) into a denormalized
    ``[16, 62]`` eef-62 action chunk, exactly the CascadedServer contract.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        cuda: str = "0",
        disable_tactile: int = 0,
        locked_config_source: str | Path | None = None,
    ):
        from scripts.test import CascadedServer, model_load

        self.checkpoint_path = str(checkpoint_path)
        ta = load_training_args(checkpoint_path)
        assert int(ta["action_dim"]) == REQUIRED_ACTION_DIM, (
            f"expected action_dim=={REQUIRED_ACTION_DIM}, checkpoint has {ta['action_dim']}"
        )
        assert int(ta["action_chunk"]) == REQUIRED_ACTION_CHUNK, (
            f"expected action_chunk=={REQUIRED_ACTION_CHUNK}, checkpoint has {ta['action_chunk']}"
        )

        self.ta = ta
        self.instruction = resolve_instruction(ta)
        self.locked_config = resolve_locked_config(ta, locked_config_source)
        self.image_size = ta.get("image_size")
        self.cascaded_total_steps = int(ta["cascaded_total_steps"])
        self.cascaded_split_step = int(ta["cascaded_split_step"])
        self.tactile_delay_offsets = list(ta.get("tactile_delay_offsets", [0]))
        self.vqvae_window = int(ta.get("vqvae_window", 16))
        self.use_robot_state = bool(ta.get("use_robot_state", 0))
        self.disable_tactile = bool(disable_tactile)

        args = build_model_load_args(checkpoint_path, ta, cuda=cuda, disable_tactile=disable_tactile)
        model, processor, statistic = model_load(args)
        # ``CascadedServer.predict()`` normally does this .to(device) itself (test.py:680);
        # we call ``_run_slow``/``_run_fast`` directly and skip ``predict()``, so it's on us.
        model = model.to(f"cuda:{cuda}" if str(cuda) != "cpu" else "cpu").eval()
        self.args = args
        self.model = model
        self.processor = processor
        self.statistic = statistic
        self.server = CascadedServer(args, model, processor, statistic)

    def _resize(self, image, key: str):
        from PIL import Image

        return image.resize(_view_size_wh(self.image_size, key), Image.LANCZOS)

    def reset(self) -> None:
        """Clears all per-episode CascadedServer state (cached_kv/x_split/f6 buffer/etc.)."""
        s = self.server
        s.cached_kv = None
        s.x_split = None
        s.tau_split = None
        s.position_ids = None
        s.attention_mask = None
        s.n_action_in_cache = 0
        s.chunk_id = -1
        s.last_actions = None
        s.f6_buffer = []

    def slow_and_fast(
        self,
        images: dict,
        tactile_f6_window,
        tactile_deform,
        state62: np.ndarray | None = None,
        instruction: str | None = None,
    ) -> np.ndarray:
        """One slow tick (re-encode vision, cache KV at tau_split) + one fast tick
        (tactile-expert flow continuation). Returns the denormalized ``[16, 62]`` chunk.

        ``images``: dict with keys "head", "wrist_left", "wrist_right" -> PIL.Image RGB.
        ``tactile_f6_window``: ``[W, 10, 6]`` raw float array (or ``[10, 6]`` single frame).
        ``tactile_deform``: ``[10, 240, 240]`` float array in [0, 1].
        ``state62``: ``[62]`` raw (un-normalized) eef-9d+22 state, or None if
        ``use_robot_state==0``.
        """
        slow_img = [self._resize(images["head"], "head")]
        fast_imgs = [
            self._resize(images["wrist_right"], "wrist_right"),
            self._resize(images["wrist_left"], "wrist_left"),
        ]
        self.server._run_slow(
            instruction if instruction is not None else self.instruction,
            slow_img,
            fast_imgs,
            tactile_f6_input=tactile_f6_window,
            tactile_deform_input=tactile_deform,
            state_fast=state62 if self.use_robot_state else None,
        )
        actions, _ = self.server._run_fast(tactile_f6_window, tactile_deform)
        return np.asarray(actions, dtype=np.float32)

    def fast(self, tactile_f6_window, tactile_deform) -> np.ndarray:
        """One fast tick against the cached slow-tick snapshot. Raises if no slow tick has
        run yet (mirrors ``CascadedServer._run_fast``'s own precondition)."""
        actions, _ = self.server._run_fast(tactile_f6_window, tactile_deform)
        return np.asarray(actions, dtype=np.float32)
