#!/usr/bin/env python3
"""T-Rex origami-zenoh-v1 policy server. TeamPolicy is the only adapter;
everything below OrigamiZenohServer is the unmodified public protocol
boundary from policy_server_template.py.
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import math
import os
import signal
import threading
import time
import uuid
from collections.abc import Mapping
from typing import Any

import msgpack
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor
import zenoh

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds

TRANSPORT_VERSION = "origami-zenoh-v1"
SEMANTIC_VERSION = "origami-v1"
ACTION_DIM = 65
# The organizer's own docs say prompt "may be empty" and is free-form task
# instruction text (robot_io_spec.md/competition_participant_complete_guide.md
# §4.4) -- but every prompt string this model actually sees changes the
# tokenized sequence length fed into forward_flow_action_partial, and
# torch.compile needs a full ~80-290s recompile for each distinct length it
# hasn't seen before (confirmed directly this session: three different
# synthetic prompts each forced their own graph, one of which crashed the
# read-only production container on first contact). Rather than pick an
# arbitrary fixed string, this is the ACTUAL training-time value -- confirmed
# directly against origami_trex's main branch (not the modified inference
# branch): trex_origami/seasons.py's INSTRUCTION constant, propagated by
# prepare.py into every episode's metadata and read back by
# qwen_vla/origami_dataset.py's _task_text() for every training sample, with
# zero per-episode variation (also matches the raw LeRobot dataset's own
# "task" field directly, e.g. season_POC22061_.../meta/tasks.parquet). Using
# this exact string, rather than a made-up instruction, avoids a genuine
# train/inference distribution shift on top of closing the compile-graph risk:
# every query uses this single prompt regardless of what the caller sends.
FIXED_PROMPT = "north ces task"
IMAGE_SHAPE = (224, 224, 3)
REQUIRED_IMAGE_SPECS = {
    "observation/image/head_left": IMAGE_SHAPE,
    "observation/image/head_right": IMAGE_SHAPE,
    "observation/image/wrist_left": IMAGE_SHAPE,
    "observation/image/wrist_right": IMAGE_SHAPE,
    "observation/image/tactile_deform": (480, 1200, 3),
}
OPTIONAL_IMAGE_SPECS = {
    "observation/image/tactile_raw": (480, 1600, 3),
}
VECTOR_SPECS = {
    "observation/state": (ACTION_DIM,),
    "observation/state/joint_torque": (ACTION_DIM,),
    "observation/tactile": (60,),
}
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def _hand_joint_names(side: str) -> tuple[str, ...]:
    return (
        f"{side}_thumb_CMC_FE", f"{side}_thumb_CMC_AA", f"{side}_thumb_MCP_FE",
        f"{side}_thumb_MCP_AA", f"{side}_thumb_IP",
        f"{side}_index_MCP_FE", f"{side}_index_MCP_AA", f"{side}_index_PIP", f"{side}_index_DIP",
        f"{side}_middle_MCP_FE", f"{side}_middle_MCP_AA", f"{side}_middle_PIP", f"{side}_middle_DIP",
        f"{side}_ring_MCP_FE", f"{side}_ring_MCP_AA", f"{side}_ring_PIP", f"{side}_ring_DIP",
        f"{side}_pinky_CMC", f"{side}_pinky_MCP_FE", f"{side}_pinky_MCP_AA",
        f"{side}_pinky_PIP", f"{side}_pinky_DIP",
    )


JOINT_NAMES = (
    tuple(f"left_arm_joint_{i}" for i in range(1, 8))
    + _hand_joint_names("left")
    + tuple(f"right_arm_joint_{i}" for i in range(1, 8))
    + _hand_joint_names("right")
    + ("lower_body_joint_1", "lower_body_joint_2", "lower_body_joint_3",
       "lower_body_joint_4", "lower_body_joint_5", "neck_joint_1", "neck_joint_2")
)

if len(JOINT_NAMES) != ACTION_DIM or len(set(JOINT_NAMES)) != ACTION_DIM:
    raise RuntimeError("joint contract must contain 65 unique names")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsupported numpy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {b"__ndarray__": True, b"data": array.tobytes(),
                b"dtype": array.dtype.str, b"shape": array.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _mapping_value(value: Mapping[Any, Any], key: str) -> Any:
    return value[key] if key in value else value.get(key.encode())


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _mapping_value(value, "__ndarray__") is True:
        data = _mapping_value(value, "data")
        shape = _mapping_value(value, "shape")
        dtype = np.dtype(_mapping_value(value, "dtype"))
        if (not isinstance(data, bytes) or not isinstance(shape, (list, tuple))
                or len(shape) > 8 or dtype.kind in {"O", "V", "c"} or dtype.hasobject):
            raise ValueError("invalid numpy array payload")
        normalized_shape = tuple(int(d) for d in shape)
        if any(d < 0 for d in normalized_shape):
            raise ValueError("invalid numpy array shape")
        expected_size = math.prod(normalized_shape) * dtype.itemsize
        if expected_size > MAX_PAYLOAD_BYTES or len(data) != expected_size:
            raise ValueError("numpy array payload size does not match shape")
        return np.frombuffer(data, dtype=dtype).reshape(normalized_shape)
    if _mapping_value(value, "__npgeneric__") is True:
        return np.dtype(_mapping_value(value, "dtype")).type(_mapping_value(value, "data"))
    return value


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("response exceeds 64 MiB")
    return payload


def unpack_payload(value: Any) -> Any:
    payload = value.to_bytes() if hasattr(value, "to_bytes") else bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("request exceeds 64 MiB")
    return msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False,
                           strict_map_key=False, max_bin_len=MAX_PAYLOAD_BYTES,
                           max_array_len=1_000_000, max_map_len=10_000, max_str_len=1_000_000)


# ── T-Rex adapter helpers ────────────────────────────────────────────────────

def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2.0 * (values - vmin) / (vmax - vmin + 1e-8) - 1.0, -1.0, 1.0), values)


def _denormalize(norm_values, mask, vmin, vmax):
    return np.where(mask, 0.5 * (norm_values + 1.0) * (vmax - vmin) + vmin, norm_values)


def _clamp_frozen(delta, mask):
    """Zero the predicted delta on dims the training stats marked frozen
    (normalization passthrough dims -- see stats_data.json's action.mask)."""
    if mask is None:
        return delta
    dims = np.where(~np.asarray(mask, dtype=bool))[0]
    if dims.size == 0:
        return delta
    out = np.array(delta, copy=True)
    out[..., dims] = 0.0
    return out


N_FINGERS, F6_PER_FINGER = 10, 6
DEFORM_TILE, DEFORM_ROWS, DEFORM_COLS = 240, 2, 5


def _split_deform_strip(arr: np.ndarray) -> np.ndarray:
    """[480, 1200] grayscale -> [10, 240, 240], left thumb..little then right."""
    t = arr.reshape(DEFORM_ROWS, DEFORM_TILE, DEFORM_COLS, DEFORM_TILE)
    return t.transpose(0, 2, 1, 3).reshape(N_FINGERS, DEFORM_TILE, DEFORM_TILE)


def _model_load(checkpoint_path: str, action_dim: int, action_chunk: int, device: torch.device):
    ta_path = os.path.join(checkpoint_path, "training_args.json")
    with open(ta_path) as f:
        ta = json.load(f)

    class Args:
        pass
    args = Args()
    required_keys = ["use_robot_state", "use_tactile_vec", "use_tactile_deform",
                     "tactile_intermediate_size", "n_flare_tokens_per_frame", "n_flare_steps",
                     "flare_layer_index", "use_tactile_code", "vqvae_codebook_size",
                     "use_tactile_vqvae", "cascaded_total_steps", "cascaded_split_step",
                     "vqvae_config", "action_chunk"]
    missing = [k for k in required_keys if k not in ta]
    if missing:
        raise KeyError(f"{ta_path} is missing required key(s) {missing} -- refusing to "
                       f"guess a default that may not match this checkpoint's training config")

    class Args:
        pass
    args = Args()
    args.action_dim, args.action_chunk = action_dim, action_chunk
    args.use_robot_state = int(ta["use_robot_state"])
    args.use_tactile_vec = int(ta["use_tactile_vec"])
    args.use_tactile_deform = int(ta["use_tactile_deform"])
    for key in ["tactile_intermediate_size", "n_flare_tokens_per_frame", "n_flare_steps",
               "flare_layer_index", "use_tactile_code", "vqvae_codebook_size",
               "use_tactile_vqvae", "cascaded_total_steps", "cascaded_split_step"]:
        setattr(args, key, ta[key])
    args.vqvae_config = ta["vqvae_config"]

    trained_chunk = ta["action_chunk"]
    if trained_chunk != action_chunk:
        raise ValueError(f"action_horizon ({action_chunk}) does not match this "
                        f"checkpoint's trained action_chunk ({trained_chunk})")

    proc_dir = os.path.join(checkpoint_path, "processor")
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)

    cfg_path = os.path.join(checkpoint_path, "config.json")
    with open(cfg_path) as f:
        full_cfg = json.load(f)
    for k in ("image_token_id", "model_type"):
        if k not in full_cfg:
            raise KeyError(f"{cfg_path} is missing required key {k!r} -- refusing to "
                           f"guess a default that may not match this checkpoint")
    image_token_id = full_cfg["image_token_id"]
    model_type = full_cfg["model_type"]
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    vl_config = Qwen3VLConfig(**{k: v for k, v in full_cfg.items()
                                 if k not in ("architectures", "transformers_version")})

    model = Qwen3VLVLAModel(
        config=vl_config.text_config, action_dim=args.action_dim, action_chunk=args.action_chunk,
        use_tactile_deform=bool(args.use_tactile_deform), use_robot_state=bool(args.use_robot_state),
        image_token_id=image_token_id,
        tactile_intermediate_size=args.tactile_intermediate_size or None,
        n_flare_tokens_per_frame=args.n_flare_tokens_per_frame, n_flare_steps=args.n_flare_steps,
        flare_layer_index=args.flare_layer_index,
        use_tactile_code=bool(args.use_tactile_code), vqvae_codebook_size=args.vqvae_codebook_size,
        use_tactile_vqvae=bool(args.use_tactile_vqvae), vqvae_config=args.vqvae_config,
    )

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel, Qwen3VLModel
    vis_cfg = Qwen3VLVisionConfig(**{k: v for k, v in full_cfg.get("vision_config", {}).items()
                                     if k != "model_type"})
    model.visual = Qwen3VLVisionModel(vis_cfg)

    class _RopeStub:
        def __init__(self, cfg):
            self.config = cfg
        def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
            return Qwen3VLModel.get_rope_index(
                self, input_ids=input_ids, image_grid_thw=image_grid_thw,
                attention_mask=attention_mask)
    object.__setattr__(model, "_rope_index_fn", _RopeStub(vl_config).get_rope_index)

    sd = torch.load(os.path.join(checkpoint_path, "model.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    logging.info("model.pt load_state_dict missing=%d unexpected=%d",
                 len(missing), len(unexpected))
    if missing:
        logging.info("  missing (first 10): %s", missing[:10])
    if unexpected:
        logging.info("  unexpected (first 10): %s", unexpected[:10])
    model = model.to(torch.bfloat16)
    if getattr(model, "tactile_vqvae", None) is not None:
        model.tactile_vqvae.float().eval()
        model.tacf6_vqvae_min = model.tacf6_vqvae_min.float()
        model.tacf6_vqvae_max = model.tacf6_vqvae_max.float()
    model = model.to(device).eval()

    stats_path = os.path.join(checkpoint_path, "stats_data.json")
    with open(stats_path) as f:
        stats_raw = json.load(f)
    ds = next(iter(stats_raw))

    def _arr(key, sub):
        return np.array(stats_raw[ds][key][sub])

    statistic = {
        "action_mask": _arr("action", "mask"), "action_min": _arr("action", "q01"),
        "action_max": _arr("action", "q99"), "tacf6_mask": _arr("tactile_f6", "mask"),
        "tacf6_min": _arr("tactile_f6", "q01"), "tacf6_max": _arr("tactile_f6", "q99"),
        "state_mask": _arr("state", "mask"), "state_min": _arr("state", "q01"),
        "state_max": _arr("state", "q99"),
    }
    logging.info(
        "model_load checkpoint=%s use_robot_state=%d use_tactile_vec=%d "
        "use_tactile_deform=%d use_tactile_vqvae=%d use_tactile_code=%d "
        "cascaded_total_steps=%d cascaded_split_step=%d action_chunk=%d action_dim=%d",
        checkpoint_path, args.use_robot_state, args.use_tactile_vec,
        args.use_tactile_deform, args.use_tactile_vqvae, args.use_tactile_code,
        args.cascaded_total_steps, args.cascaded_split_step, args.action_chunk, args.action_dim,
    )
    return model, processor, statistic, args


def make_warmup_observation() -> dict[str, Any]:
    """Synthetic observation matching production shapes/dtypes exactly
    (batch=1, 224x224x3 head/wrist images, 480x1200x3 tactile deform grid, 60
    float32 tactile channels, 65-dim state). Used for two purposes that must
    stay in lockstep: (1) TeamPolicy's own startup warmup below, so the real
    compile cost lands before the server declares READY rather than on the
    first live query; (2) compile_warmup.py, the build-time script that bakes
    a torch.compile cache into the image (imports this exact function rather
    than duplicating it, so the two can't drift out of shape-sync)."""
    rng = np.random.default_rng(0)
    return {
        "observation/image/head_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image/head_right": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_left": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/image/wrist_right": rng.integers(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": rng.uniform(-0.25, 0.25, ACTION_DIM).astype(np.float32),
        "observation/state/joint_torque": np.zeros(ACTION_DIM, dtype=np.float32),
        "observation/tactile": rng.uniform(0, 1, 60).astype(np.float32),
        "observation/image/tactile_deform": rng.integers(0, 255, (480, 1200, 3), dtype=np.uint8),
        "prompt": FIXED_PROMPT,  # ignored at inference time; see FIXED_PROMPT above
    }


def make_zero_tactile_warmup_observation() -> dict[str, Any]:
    """A second synthetic observation, same shapes as make_warmup_observation()
    but all-zero tactile/joint_torque/tactile_deform plus deterministic
    gradient-pattern images -- this exact pattern matches the public
    protocol validator's own synthetic data (sharpa_north_ces_lite_sdk's
    check_zenoh_policy.py / make_synthetic_observation()). Confirmed directly
    (isolated in-process test, 31 fresh compile_worker subprocesses observed)
    that all-zero tactile input hits a THIRD distinct compiled graph beyond
    the ones make_warmup_observation() and varied-random content produce --
    exercising it here too, during startup, means the live validator's first
    query lands on a graph this process has already traced once instead of
    tracing it for the first time on that query (compile_warmup.py bakes the
    matching kernel cache for this same pattern at build time; see there)."""
    rows = np.arange(224, dtype=np.uint8)[:, None]
    cols = np.arange(224, dtype=np.uint8)[None, :]
    base = np.empty((224, 224, 3), dtype=np.uint8)
    base[..., 0] = rows
    base[..., 1] = cols
    base[..., 2] = rows ^ cols
    return {
        "observation/image/head_left": np.ascontiguousarray(base),
        "observation/image/head_right": np.ascontiguousarray(np.roll(base, 11, axis=0)),
        "observation/image/wrist_left": np.ascontiguousarray(np.roll(base, 17, axis=1)),
        "observation/image/wrist_right": np.ascontiguousarray(np.flip(base, axis=1)),
        "observation/state": np.linspace(-0.25, 0.25, ACTION_DIM, dtype=np.float32),
        "observation/state/joint_torque": np.zeros(ACTION_DIM, dtype=np.float32),
        "observation/tactile": np.zeros(60, dtype=np.float32),
        "observation/image/tactile_deform": np.zeros((480, 1200, 3), dtype=np.uint8),
        "prompt": FIXED_PROMPT,  # value is irrelevant -- overridden at inference
    }


class TeamPolicy:
    """T-Rex (Qwen3-VL MoT cascaded flow-matching) adapter.

    Images: slow = observation/image/head_left only (head_right is unused --
    this checkpoint's slow expert takes a single head image, see
    trex_origami/seasons.py's VIDEO_KEYS). fast = [wrist_right, wrist_left],
    in that order. tactile_deform is a [480,1200,3] 2x5 fingertip grid, split
    into 10 [240,240] tiles.

    The model output is a delta from observation/state, not absolute --
    infer() adds it back before returning.

    Cadence: every infer() must return a complete action chunk, but the
    tactile expert (fast tick) only produces one by continuing a cached
    slow-tick anchor (forward_flow_action_partial). The first call, and every
    full_refresh_every-th call after, reruns the full slow+fast pass and
    refreshes the anchor; the rest reuse it with fresh tactile.
    """

    def __init__(self, action_horizon: int) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint_path = os.environ.get("TREX_CKPT_PATH", "/app/checkpoints/model")
        self.model, self.processor, self.statistic, self.margs = _model_load(
            checkpoint_path, ACTION_DIM, action_horizon, self.device)
        self.action_horizon = action_horizon
        self.full_refresh_every = int(os.environ.get("TREX_FULL_REFRESH_EVERY", str(action_horizon)))

        # All tensor shapes here are fixed by design (fixed action horizon,
        # fixed image/state/tactile shapes -- see REQUIRED_IMAGE_SPECS/
        # VECTOR_SPECS). Without this, Dynamo's "automatic dynamic shapes"
        # heuristic treats the first real (non-warmup) call as evidence that
        # some value might vary, and silently recompiles a second, dynamic-
        # shape graph for it -- observed directly: the warmup call compiles
        # and caches cleanly, then the very first live query still triggers
        # a brand-new ~80s compile for a second graph, which crashes under
        # --read-only (no writable+exec location exists for a fresh Triton
        # kernel under the mandated tmpfs profile). Since shapes never
        # actually vary here, disabling automatic promotion keeps every call
        # on the one static graph warmup already compiled.
        torch._dynamo.config.automatic_dynamic_shapes = False

        # torch.compile targets the bound methods directly, not self.model's
        # forward()/__call__ -- these two are the only entry points the server
        # ever calls (see _run_slow_and_fast / _run_fast). Compiling self.model
        # itself would be a no-op here: torch.compile only instruments
        # forward()/__call__, and OptimizedModule.__getattr__ delegates any
        # other attribute (including these two) straight to the uncompiled
        # original module. torch.compile itself is lazy -- wrapping here does
        # not yet trigger compilation, only the warmup call below (still
        # inside __init__, before the Zenoh session opens or queryables are
        # declared) does. Per participant_zenoh_submission.md's own guidance:
        # "Do not rely on unlimited first-request compilation time. Load and
        # warm the model during startup where possible."
        #
        # Plain default compile for both -- mode="reduce-overhead" (CUDA
        # graphs) was tried and gave a real speedup in isolation, but CUDA
        # graphs are captured against a specific stream tied to the
        # capturing thread; replaying from a different thread deadlocks
        # rather than crashing (confirmed: 0% GPU util, process resident,
        # no exception, infer request just times out). That's fatal here
        # since OrigamiZenohServer's Zenoh queryable callback runs on
        # Zenoh's own internal thread, not the thread that ran __init__'s
        # warmup. Plain default compile has no static-buffer/stream
        # affinity, so it's immune to this regardless of which thread calls
        # it; confirmed working at 18.8ms (5.37x) in isolation.
        self.model.tactile_flow_continue = torch.compile(
            self.model.tactile_flow_continue, mode="default")
        self.model.forward_flow_action_partial = torch.compile(
            self.model.forward_flow_action_partial, mode="default")

        self.use_embedded_vqvae = bool(
            getattr(self.model, "use_tactile_vqvae", False)
            and getattr(self.model, "tactile_vqvae", None) is not None)
        self.vqvae_window = (int(self.model.tactile_vqvae.cfg.window)
                            if self.use_embedded_vqvae else 16)

        self._reset_episode_state()
        logging.info(
            "TeamPolicy ready device=%s action_horizon=%d full_refresh_every=%d "
            "use_embedded_vqvae=%s vqvae_window=%d flow_compile_mode=default "
            "flow_compile_targets=tactile_flow_continue,forward_flow_action_partial",
            self.device, self.action_horizon, self.full_refresh_every,
            self.use_embedded_vqvae, self.vqvae_window,
        )
        self._warmup()

    def _warmup(self) -> None:
        """Trigger both flow functions' compile now, during __init__ -- this
        runs before OrigamiZenohServer opens the Zenoh session or declares
        any queryable, so the one-time compile cost lands in the startup
        window rather than on the first live query. reset() afterward clears
        the warmup's own tick/cache state so the real first episode starts
        clean; it does not re-trigger compilation (torch.compile wraps the
        method once, for the process's lifetime, independent of episodes)."""
        logging.info(
            "warmup starting -- compiling tactile_flow_continue and "
            "forward_flow_action_partial (both default mode); measured ~80-100s "
            "combined on an RTX 5090 when the baked compile-cache isn't reused "
            "(see Dockerfile's compile-cache COPY note), this is a one-time cost "
            "for the process's lifetime"
        )
        t_start = time.monotonic()
        self.reset()
        self.infer(make_warmup_observation())
        self.reset()
        self.infer(make_zero_tactile_warmup_observation())
        self.reset()
        logging.info("warmup complete elapsed_ms=%.1f", (time.monotonic() - t_start) * 1000.0)

    def _reset_episode_state(self) -> None:
        self.cached_kv = None
        self.x_split = None
        self.tau_split = None
        self.position_ids = None
        self.attention_mask = None
        self.n_action_in_cache = 0
        self.tick = 0
        self.f6_history: collections.deque = collections.deque(maxlen=self.vqvae_window)

    def reset(self) -> None:
        logging.info("policy reset episode_ticks_completed=%d", self.tick)
        self._reset_episode_state()

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        return Image.fromarray(image)

    def _f6_history_window(self, tactile: np.ndarray) -> torch.Tensor:
        f6 = tactile.reshape(N_FINGERS, F6_PER_FINGER)
        if not self.f6_history:
            for _ in range(self.vqvae_window):
                self.f6_history.append(f6)
        else:
            self.f6_history.append(f6)
        arr = np.stack(self.f6_history, axis=0)  # [W, 10, 6]
        return torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).to(self.device)

    def _run_slow_and_fast(self, observation: dict[str, Any]) -> np.ndarray:
        model, statistic, device = self.model, self.statistic, self.device
        t_start = time.monotonic()

        slow_images = [self._to_pil(observation["observation/image/head_left"])]
        fast_images = [self._to_pil(observation["observation/image/wrist_right"]),
                       self._to_pil(observation["observation/image/wrist_left"])]
        content = [{"type": "image"} for _ in slow_images]
        content.append({"type": "text", "text": FIXED_PROMPT})
        content += [{"type": "image"} for _ in fast_images]
        text = self.processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inp = self.processor(text=text, images=slow_images + fast_images,
                             return_tensors="pt", padding=False)

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = inp.pixel_values.to(device, dtype=torch.bfloat16)
        image_grid_thw = inp.image_grid_thw.to(device)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        merge = getattr(model.visual, "spatial_merge_size",
                        getattr(self.processor.image_processor, "merge_size", 2))
        n_slow_tokens = int(image_grid_thw[0, 0] * (image_grid_thw[0, 1] // merge)
                           * (image_grid_thw[0, 2] // merge))
        slow_embeds, fast_embeds = split_slow_fast_embeds(
            inputs_embeds, input_ids, model.image_token_id, n_slow_tokens)

        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]
        t_vision = time.monotonic() - t_start

        state = np.asarray(observation["observation/state"], dtype=np.float32)
        norm_state = _normalize(state, statistic["state_mask"], statistic["state_min"],
                                statistic["state_max"])
        state_embeds = model.state_embedder(
            torch.tensor(norm_state, dtype=torch.bfloat16, device=device).unsqueeze(0)
        ).unsqueeze(1)

        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(position_ids, model.n_flare_tokens)

        noise = torch.randn(1, self.action_horizon, ACTION_DIM, dtype=torch.bfloat16, device=device)
        x_split, cached_kv, n_action_in_cache, tau_split = model.forward_flow_action_partial(
            inputs_embeds=slow_embeds, position_ids=position_ids, attention_mask=attention_mask,
            noise=noise, state_embeds=state_embeds, fast_embeds=fast_embeds,
            num_steps_total=self.margs.cascaded_total_steps,
            split_step=self.margs.cascaded_split_step, refresh_clean_kv=True,
        )
        self.x_split, self.tau_split = x_split, tau_split
        self.cached_kv, self.position_ids = cached_kv, position_ids
        self.attention_mask, self.n_action_in_cache = attention_mask, n_action_in_cache
        t_slow = time.monotonic() - t_start - t_vision
        logging.info("slow_pass tick=%d vision_ms=%.1f slow_ms=%.1f n_action_in_cache=%d",
                     self.tick, t_vision * 1000.0, t_slow * 1000.0, n_action_in_cache)

        return self._run_fast(observation, state)

    def _run_fast(self, observation: dict[str, Any], state: np.ndarray) -> np.ndarray:
        model, statistic, device = self.model, self.statistic, self.device
        t_start = time.monotonic()

        tactile = np.asarray(observation["observation/tactile"], dtype=np.float32)
        norm_tactile = _normalize(tactile.reshape(-1), statistic["tacf6_mask"],
                                  statistic["tacf6_min"], statistic["tacf6_max"])
        tac_f6 = torch.tensor(norm_tactile.reshape(-1, 6), dtype=torch.bfloat16,
                              device=device).unsqueeze(0)

        deform_gray = observation["observation/image/tactile_deform"][..., 0].astype(np.float32) / 255.0
        deform_tiles = _split_deform_strip(deform_gray)  # [10, 240, 240]
        tac_deform = torch.from_numpy(deform_tiles).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)

        tac_hist = self._f6_history_window(tactile) if self.use_embedded_vqvae else None

        refined = model.tactile_flow_continue(
            cached_kv=self.cached_kv, latent_position_ids=self.position_ids,
            n_action_in_cache=self.n_action_in_cache, x_split=self.x_split,
            tau_split=self.tau_split, attention_mask=self.attention_mask,
            tactile_f6=tac_f6, tactile_deform=tac_deform, tactile_f6_history=tac_hist,
            num_steps_total=self.margs.cascaded_total_steps,
            split_step=self.margs.cascaded_split_step,
        )
        delta = _clamp_frozen(
            _denormalize(refined[0].float().cpu().numpy(), statistic["action_mask"],
                        statistic["action_min"], statistic["action_max"]),
            statistic["action_mask"])
        actions = (state[None, :] + delta).astype(np.float32)
        logging.info("fast_pass tick=%d fast_ms=%.1f actions_shape=%s",
                     self.tick, (time.monotonic() - t_start) * 1000.0, actions.shape)
        return actions

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        t_start = time.monotonic()
        due_for_refresh = self.cached_kv is None or self.tick % self.full_refresh_every == 0
        if due_for_refresh:
            optional_present = [k for k in OPTIONAL_IMAGE_SPECS if k in observation]
            logging.info("infer tick=%d observation_optional_keys=%s", self.tick, optional_present)
            actions = self._run_slow_and_fast(observation)
        else:
            state = np.asarray(observation["observation/state"], dtype=np.float32)
            actions = self._run_fast(observation, state)
        if torch.cuda.is_available():
            mem_alloc_mb = torch.cuda.memory_allocated(self.device) / 1e6
            mem_reserved_mb = torch.cuda.memory_reserved(self.device) / 1e6
        else:
            mem_alloc_mb = mem_reserved_mb = 0.0
        logging.info("infer tick=%d refresh=%s total_ms=%.1f mem_alloc_mb=%.1f mem_reserved_mb=%.1f",
                     self.tick, due_for_refresh, (time.monotonic() - t_start) * 1000.0,
                     mem_alloc_mb, mem_reserved_mb)
        self.tick += 1
        return actions


class OrigamiZenohServer:
    def __init__(
        self,
        policy: TeamPolicy,
        *,
        endpoint: str,
        session_id: str,
        action_horizon: int,
        execution_mode: str = "async",
    ) -> None:
        if action_horizon < 1 or action_horizon > 1024:
            raise ValueError("action_horizon must be in [1, 1024]")
        if execution_mode not in {"sync", "async"}:
            raise ValueError("execution_mode must be 'sync' or 'async'")
        self.policy = policy
        self.endpoint = endpoint
        self.session_id = session_id
        self.action_horizon = action_horizon
        self.execution_mode = execution_mode
        self._policy_lock = threading.Lock()
        self._stop = threading.Event()
        self._session: Any | None = None
        self._queryables: list[Any] = []
        self.metadata = {
            "protocol_version": SEMANTIC_VERSION,
            "action_dim": ACTION_DIM,
            "action_horizon": action_horizon,
            "action_type": "absolute_joint_position",
            "action_units": "radians",
            "joint_names": JOINT_NAMES,
            "execution_mode": execution_mode,
        }

    def serve_forever(self) -> None:
        config = zenoh.Config()
        config.insert_json5("mode", json.dumps("client"))
        config.insert_json5("connect/endpoints", json.dumps([self.endpoint]))
        config.insert_json5("scouting/multicast/enabled", "false")
        config.insert_json5("transport/shared_memory/enabled", "false")
        self._session = zenoh.open(config)
        self._queryables = [
            self._session.declare_queryable(
                f"{TRANSPORT_VERSION}/{operation}", self._handle_query, complete=True)
            for operation in ("metadata", "reset", "infer")
        ]
        signal.signal(signal.SIGTERM, lambda *_: self._stop.set())
        signal.signal(signal.SIGINT, lambda *_: self._stop.set())
        logging.info("READY transport=%s endpoint=%s horizon=%d",
                    TRANSPORT_VERSION, self.endpoint, self.action_horizon)
        self._stop.wait()
        for queryable in self._queryables:
            queryable.undeclare()
        self._session.close()

    def _handle_query(self, query: Any) -> None:
        operation = str(query.key_expr).rsplit("/", 1)[-1]
        request: Any = None
        try:
            request = unpack_payload(query.payload)
            response = self.process(operation, request)
        except Exception as exc:
            error_id = uuid.uuid4().hex
            logging.error("request failed operation=%s error_id=%s type=%s",
                         operation, error_id, type(exc).__name__, exc_info=True)
            public_message = f"request failed; error_id={error_id}"
            if not isinstance(request, Mapping):
                query.reply_err(pack_payload({"error": {
                    "code": "INVALID_REQUEST", "message": public_message, "retryable": False}}),
                    encoding="application/msgpack")
                return
            response = self._envelope(operation, request)
            response["error"] = {
                "code": "INFERENCE_FAILED" if operation == "infer" else "INVALID_REQUEST",
                "message": public_message, "retryable": False,
            }
        query.reply(str(query.key_expr), pack_payload(response), encoding="application/msgpack")

    def process(self, operation: str, request: Any) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("request must be a MessagePack map")
        response = self._envelope(operation, request)
        if request.get("protocol_version") != TRANSPORT_VERSION:
            raise ValueError("invalid protocol_version")
        if request.get("operation") != operation:
            raise ValueError("operation does not match queryable key")
        if request.get("session_id") != self.session_id:
            raise ValueError("session_id does not match assigned session")
        if not isinstance(request.get("request_id"), str) or not request["request_id"]:
            raise ValueError("request_id must be a non-empty string")

        if operation == "metadata":
            logging.info("metadata query request_id=%s", request.get("request_id"))
            response["metadata"] = self.metadata
            return response
        if operation == "reset":
            logging.info("reset query request_id=%s", request.get("request_id"))
            with self._policy_lock:
                self.policy.reset()
            response["ok"] = True
            return response
        if operation != "infer":
            raise ValueError(f"unsupported operation: {operation}")

        observation = request.get("observation")
        self._validate_observation(observation)
        started = time.monotonic()
        with self._policy_lock:
            actions = self.policy.infer(dict(observation))
        actions = np.asarray(actions)
        expected_shape = (self.action_horizon, ACTION_DIM)
        if actions.dtype != np.float32 or actions.shape != expected_shape:
            raise ValueError(f"policy actions must be float32{expected_shape}, "
                            f"got {actions.dtype}{actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("policy actions contain NaN or Inf")
        response["actions"] = np.ascontiguousarray(actions)
        response["server_timing"] = {"infer_ms": (time.monotonic() - started) * 1000.0}
        return response

    def _envelope(self, operation: str, request: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "protocol_version": TRANSPORT_VERSION,
            "operation": operation,
            "request_id": request.get("request_id"),
            "session_id": self.session_id,
        }

    @staticmethod
    def _validate_observation(observation: Any) -> None:
        if not isinstance(observation, Mapping):
            raise ValueError("infer request must contain an observation map")
        required = {*REQUIRED_IMAGE_SPECS, *VECTOR_SPECS, "prompt"}
        allowed = required | set(OPTIONAL_IMAGE_SPECS)
        if not required.issubset(observation) or not set(observation).issubset(allowed):
            raise ValueError("observation keys do not match the public full schema")
        for key, shape in {**REQUIRED_IMAGE_SPECS, **OPTIONAL_IMAGE_SPECS}.items():
            if key not in observation:
                continue
            image = observation.get(key)
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
                    or image.shape != shape):
                raise ValueError(f"{key} must be uint8{shape}")
        for key, shape in VECTOR_SPECS.items():
            vector = observation.get(key)
            if (not isinstance(vector, np.ndarray) or vector.dtype != np.float32
                    or vector.shape != shape or not np.isfinite(vector).all()):
                raise ValueError(f"{key} must be finite float32{shape}")
        if not isinstance(observation.get("prompt"), str):
            raise ValueError("prompt must be a string")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default=os.environ.get("ORIGAMI_ZENOH_ENDPOINT"))
    parser.add_argument("--session-id", default=os.environ.get("ORIGAMI_SESSION_ID"))
    parser.add_argument("--action-horizon", type=int,
                        default=int(os.environ.get("ORIGAMI_ACTION_HORIZON", "25")))
    parser.add_argument("--execution-mode", choices=("sync", "async"),
                        default=os.environ.get("EXECUTION_MODE", "async"))
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    if not args.endpoint:
        raise SystemExit("--endpoint or ORIGAMI_ZENOH_ENDPOINT is required")
    if not args.session_id:
        raise SystemExit("--session-id or ORIGAMI_SESSION_ID is required")
    logging.info(
        "startup flags endpoint=%s session_id=%s action_horizon=%d execution_mode=%s "
        "TREX_CKPT_PATH=%s TREX_FULL_REFRESH_EVERY=%s TREX_PROFILE_STEPS=%s",
        args.endpoint, args.session_id, args.action_horizon, args.execution_mode,
        os.environ.get("TREX_CKPT_PATH", "/app/checkpoints/model (default)"),
        os.environ.get("TREX_FULL_REFRESH_EVERY", f"{args.action_horizon} (default=action_horizon)"),
        os.environ.get("TREX_PROFILE_STEPS", "0 (default, off)"),
    )
    policy = TeamPolicy(args.action_horizon)
    server = OrigamiZenohServer(
        policy, endpoint=args.endpoint, session_id=args.session_id,
        action_horizon=args.action_horizon, execution_mode=args.execution_mode)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
