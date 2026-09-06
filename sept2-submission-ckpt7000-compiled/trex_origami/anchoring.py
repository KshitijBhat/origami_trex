"""Per-dimension action anchoring — what the 65 predicted numbers are *relative to*.

The first fine-tune made all 65 dims a delta from `observation.state[t]`.  Two
things went wrong with that, both visible in `checkpoint-2-8000`'s metrics:

  * **The arms could not beat `hold_state`.**  With targets
    `action[t+k] - state[t]`, the constant part of every target is the
    command-minus-state tracking offset (~0.74 deg), and that offset is not
    observable from any model input — it lives in the *command* history.  A
    policy that predicts the motion perfectly still pays it.
  * **The fingers inherited that offset as noise.**  q01/q99 over near-static
    delta dims spans little more than the tracking jitter, so normalisation
    stretched that jitter to the full [-1, 1] range and the flow head's residual
    sampling noise denormalised back into degrees of finger wobble.

The paper anchors per *group* instead (§5.1, and the released
`...deltabase_axis_eef...` stats): arms are relative, hands are absolute joint
angles.  This module is the single definition of that rule, shared by the prep
(`trex_origami.prepare`), the loader (`qwen_vla.origami_dataset`), every eval
script and the serving path, so none of them can drift from the others.

Two modes:

    "state"    every dim is `action[t+k] - state[t]`.  What attempt 2 trained.
               Kept so old prepped datasets and old checkpoints still evaluate.
    "hybrid"   arms (dims 0-6, 29-35) are `action[t+k] - action[t-1]`, i.e. a
               delta from the *previous command*; hands and the torso/neck
               motor block are absolute radians.

The rule is recorded per-dim in `meta/dataset.json` (`action_anchor`, 65
strings) rather than re-derived from a mode name, so a dataset prepared by an
older or newer prep still describes itself and consumers read it instead of
assuming.

Reconstruction is the same one line everywhere:

    absolute_chunk = build_anchor(state, prev_command, spec)[..., None, :] + predicted

with the anchor being `state[t]` (mode "state"), `action[t-1]` (arms in mode
"hybrid") or `0` (absolute dims).  At deployment `action[t-1]` is the policy's
own last emitted command, which it always has; nothing here needs information
the robot does not have.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .seasons import ACTION_DIM, JOINT_GROUPS, JOINT_NAMES

#: Per-dim anchor kinds.  A dim's target is `action[t+k] - anchor`, where the
#: anchor is the measured state, the previous command, or zero.
ANCHOR_STATE = "state"
ANCHOR_PREV_COMMAND = "prev_command"
ANCHOR_ABSOLUTE = "absolute"

ANCHOR_KINDS = (ANCHOR_STATE, ANCHOR_PREV_COMMAND, ANCHOR_ABSOLUTE)

#: Named anchoring policies selectable at prep time (`--anchor-mode`).
ANCHOR_MODES = ("state", "hybrid")

#: Joint groups that stay *relative* under "hybrid".  The arms were delta-trained
#: in T-Rex's own midtrain corpus, so that part of the trunk transfers; the hands
#: were trained on absolute finger poses and the motor block barely moves.
DELTA_GROUPS: Tuple[str, ...] = ("left_arm", "right_arm")

#: What a dataset written before this module existed used.
LEGACY_MODE = "state"


def anchor_spec(mode: str = "hybrid") -> Tuple[str, ...]:
    """The 65 per-dim anchor kinds implied by a named mode."""
    if mode not in ANCHOR_MODES:
        raise ValueError(f"unknown anchor mode {mode!r}; expected one of {ANCHOR_MODES}")
    if mode == "state":
        return tuple([ANCHOR_STATE] * ACTION_DIM)
    spec = [ANCHOR_ABSOLUTE] * ACTION_DIM
    for name, lo, hi in JOINT_GROUPS:
        if name in DELTA_GROUPS:
            for i in range(lo, hi):
                spec[i] = ANCHOR_PREV_COMMAND
    return tuple(spec)


def validate_spec(spec: Sequence[str]) -> Tuple[str, ...]:
    if len(spec) != ACTION_DIM:
        raise ValueError(f"action_anchor has {len(spec)} entries, expected {ACTION_DIM}")
    bad = sorted({k for k in spec if k not in ANCHOR_KINDS})
    if bad:
        raise ValueError(f"unknown anchor kind(s) {bad}; expected {ANCHOR_KINDS}")
    return tuple(spec)


def spec_from_meta(meta: Optional[dict]) -> Tuple[str, ...]:
    """The anchoring rule a prepared dataset (or a saved checkpoint) declares.

    Accepts `meta/dataset.json` (top-level `action_anchor`, or `config.anchor_mode`)
    and a checkpoint's `training_args.json` (top-level `action_anchor`).  A file
    that declares neither predates hybrid anchoring, so it gets the legacy
    all-delta-from-state rule — which is exactly what it was prepared with.
    """
    if not meta:
        return anchor_spec(LEGACY_MODE)
    explicit = meta.get("action_anchor")
    if explicit:
        return validate_spec(explicit)
    mode = (meta.get("config", {}) or {}).get("anchor_mode") or meta.get("anchor_mode")
    return anchor_spec(mode) if mode else anchor_spec(LEGACY_MODE)


def mode_of(spec: Sequence[str]) -> str:
    """Reverse lookup: the mode name whose spec matches, or "custom"."""
    for mode in ANCHOR_MODES:
        if tuple(spec) == anchor_spec(mode):
            return mode
    return "custom"


def masks(spec: Sequence[str]) -> Dict[str, np.ndarray]:
    """Boolean [65] selector per anchor kind."""
    arr = np.asarray(spec, dtype=object)
    return {kind: (arr == kind) for kind in ANCHOR_KINDS}


def build_anchor(state: np.ndarray, prev_command: Optional[np.ndarray],
                 spec: Sequence[str]) -> np.ndarray:
    """The per-dim value each predicted number is measured from.

    `state` and `prev_command` are [..., 65]; the result has the same shape.
    `prev_command` may be None only when the spec asks for none — otherwise the
    caller would be silently substituting the state and moving the whole action
    space by the tracking offset.
    """
    state = np.asarray(state)
    sel = masks(spec)
    anchor = np.zeros_like(state, dtype=np.float64)
    if sel[ANCHOR_STATE].any():
        anchor[..., sel[ANCHOR_STATE]] = state[..., sel[ANCHOR_STATE]]
    if sel[ANCHOR_PREV_COMMAND].any():
        if prev_command is None:
            raise ValueError(
                "this dataset anchors some dims to the previous command, but no "
                "prev_command was supplied — reconstruction would be off by the "
                "command-minus-state tracking offset on every arm joint")
        prev_command = np.asarray(prev_command)
        anchor[..., sel[ANCHOR_PREV_COMMAND]] = prev_command[..., sel[ANCHOR_PREV_COMMAND]]
    # ANCHOR_ABSOLUTE dims stay 0: the prediction *is* the joint angle.
    return anchor


def to_absolute(chunk: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """[..., T, 65] target-space chunk + [..., 65] anchor -> absolute radians."""
    return np.asarray(chunk) + np.asarray(anchor)[..., None, :]


def to_target(absolute: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Inverse of `to_absolute` — what the model is trained to emit."""
    return np.asarray(absolute) - np.asarray(anchor)[..., None, :]


def describe(spec: Sequence[str]) -> str:
    """One line per joint group, for startup logs."""
    parts = []
    for name, lo, hi in JOINT_GROUPS:
        kinds = sorted(set(spec[lo:hi]))
        parts.append(f"{name}[{lo}:{hi}]={'+'.join(kinds)}")
    return f"{mode_of(spec)}: " + "  ".join(parts)


def group_of(dim: int) -> str:
    for name, lo, hi in JOINT_GROUPS:
        if lo <= dim < hi:
            return name
    raise IndexError(dim)


def anchor_dim_names(spec: Sequence[str], kind: str) -> Tuple[str, ...]:
    return tuple(JOINT_NAMES[i] for i, k in enumerate(spec) if k == kind)
