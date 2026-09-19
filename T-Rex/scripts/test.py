"""
Real-world ZeroMQ inference server for the Qwen3-VL MoT VLA model
with cascaded flow matching and flare visual prediction tokens.

Stateful slow/fast protocol:
  slow request → forward_flow_action_partial → cache (latent + action) KV
                 at τ=τ_split, return [] (no usable action without a fast tick).
  fast request → tactile_flow_continue on cached KV with fresh tactile,
                 returning the final action chunk directly (no Â + Δa add).
  slow_and_fast → run both in sequence (typical at chunk start).

The client orchestrates cadence (e.g. slow every 16 robot steps, fast at
offsets 0, 4, 8, 12).  ZMQ REP is single-threaded so a fast request
arriving mid-slow naturally waits until slow finishes — the "if a
refinement has not finished, wait until it finishes" guarantee.

What comes back on the wire
---------------------------
`actions` is float32[T, 65] of **absolute joint angles**, which is the
competition's contract, and is also exactly what the all-absolute model
predicts (normalized) -- reconstruction is just denormalize + hold the
frozen dims at the measured state (`_clamp_frozen_absolute`), no per-dim
anchor addition. `--action_output delta` remains as a legacy escape hatch
for the pre-all-absolute all-delta-from-state checkpoints this server used
to also serve: it subtracts the measured state back off before replying,
reproducing the old "client adds observation/state" behaviour.
"""

import os
import sys
import threading

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse
import json
import io
import pickle
import time
import traceback

import numpy as np
import torch
from PIL import Image
import zmq
from transformers import AutoProcessor
from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds

#: The evaluator streams commands at 30 Hz; used to work out which element of
#: the last emitted chunk is "the command for the frame just before now".
CONTROL_HZ = 30.0


def _normalize(values, mask, vmin, vmax):
    return np.where(
        mask,
        np.clip(2.0 * (values - vmin) / (vmax - vmin + 1e-8) - 1.0, -1.0, 1.0),
        values,
    )

def _denormalize(norm_values, mask, vmin, vmax):
    return np.where(
        mask,
        0.5 * (norm_values + 1.0) * (vmax - vmin) + vmin,
        norm_values,
    )


def _clamp_frozen_absolute(absolute, mask, state):
    """Hold the measured position on the dims the training stats marked frozen.

    A masked-off dim is normalisation *passthrough*: `_normalize` left it in raw
    units, so the flow head's job there is to cancel its own unit-variance input
    noise against a target of ~0, and whatever it fails to cancel reaches the
    wire as raw radians.  On the origami split that is the torso
    `lower_body_joint_1/2` -- constant to ~4e-4 rad across every season, but
    +-0.3 rad out of the policy, which alone blows the motor group's 0.06 rad
    step-jump budget on essentially every chunk.

    Commanding `state[j]` for the whole chunk is exactly what the teleoperator
    did there. All dims are absolute under the all-absolute target, so this
    holds the *value*, not "zero the delta" -- zeroing would command 0 rad, a
    full-travel move, not holding. Driven by the mask, so a checkpoint whose
    stats say the torso moves clamps nothing.
    """
    if mask is None or state is None:
        return absolute
    dims = np.where(~np.asarray(mask, dtype=bool))[0]
    if dims.size == 0:
        return absolute
    out = np.array(absolute, copy=True)
    out[..., dims] = np.asarray(state, dtype=out.dtype)[dims]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Build model from config.json
# ─────────────────────────────────────────────────────────────────────────────

def _build_qwen3vl_from_config(config_path, args):
    with open(config_path) as f:
        full_cfg = json.load(f)

    image_token_id = full_cfg.get("image_token_id", 151655)
    model_type = full_cfg.get("model_type", "qwen2_vl")

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        vl_config = Qwen3VLConfig(**{k: v for k, v in full_cfg.items()
                                     if k not in ("architectures", "transformers_version")})
        text_config = vl_config.text_config
    except Exception:
        from transformers import AutoConfig
        vl_config = AutoConfig.from_pretrained(
            os.path.dirname(config_path), trust_remote_code=True)
        text_config = getattr(vl_config, "text_config", vl_config)

    tac_isize = getattr(args, "tactile_intermediate_size", 0)
    tac_isize = tac_isize if tac_isize > 0 else None
    n_flare_tpf = getattr(args, "n_flare_tokens_per_frame", 0)
    n_flare_steps = getattr(args, "n_flare_steps", 0)

    model = Qwen3VLVLAModel(
        config             = text_config,
        action_dim         = args.action_dim,
        action_chunk       = args.action_chunk,
        use_tactile_deform = bool(args.use_tactile_deform),
        use_robot_state    = bool(args.use_robot_state),
        image_token_id     = image_token_id,
        tactile_intermediate_size = tac_isize,
        n_flare_tokens_per_frame = n_flare_tpf,
        n_flare_steps            = n_flare_steps,
        use_tactile_code         = bool(getattr(args, "use_tactile_code", 0)),
        vqvae_codebook_size      = getattr(args, "vqvae_codebook_size", 64),
        use_tactile_vqvae        = bool(getattr(args, "use_tactile_vqvae", 0)),
        vqvae_config             = getattr(args, "vqvae_config", None),
    )

    vis_cfg_dict = full_cfg.get("vision_config", {})
    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            vis_cfg = Qwen3VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items()
                                             if k != "model_type"})
            model.visual = Qwen3VLVisionModel(vis_cfg)
        else:
            from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionModel
            vis_cfg = Qwen2VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items()
                                             if k != "model_type"})
            model.visual = Qwen2VLVisionModel(vis_cfg)
        print(f"  Visual tower created from config")
    except Exception as e:
        print(f"  Warning: visual tower creation failed: {e}")
        model.visual = None

    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _VLModel
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as _VLModel

        class _RopeStub:
            def __init__(self, cfg):
                self.config = cfg
            def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
                return _VLModel.get_rope_index(
                    self, input_ids=input_ids,
                    image_grid_thw=image_grid_thw, attention_mask=attention_mask)

        object.__setattr__(model, '_rope_index_fn', _RopeStub(vl_config).get_rope_index)
        print("  M-RoPE helper ready.")
    except Exception as e:
        print(f"  Warning: rope index setup failed ({e}).")

    return model


def _has_hf_weights(path):
    import glob as _glob
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        if _glob.glob(os.path.join(path, pattern)):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def model_load(args):
    ckpt = args.checkpoint_path

    ta_path = os.path.join(ckpt, "training_args.json")
    ta = {}
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
        for key, default in [("tactile_intermediate_size", 0),
                             ("n_flare_tokens_per_frame", 0),
                             ("n_flare_steps", 0),
                             ("use_tactile_code", 0),
                             ("vqvae_codebook_size", 64),
                             ("use_tactile_vqvae", 0),
                             ("cascaded_total_steps", 10),
                             ("cascaded_split_step", 6),
                             ("instruction", "")]:
            saved = ta.get(key, default)
            cli_val = getattr(args, key, default)
            if saved and cli_val == default:
                setattr(args, key, saved)
                print(f"Auto-detected {key}={saved} from training_args.json")
        # vqvae_config is a dict — restore it verbatim so the embedded VQ-VAE
        # submodule is rebuilt with the right architecture before weights load.
        if ta.get("vqvae_config") is not None and getattr(args, "vqvae_config", None) is None:
            args.vqvae_config = ta["vqvae_config"]

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None
    n_flare_tpf = getattr(args, "n_flare_tokens_per_frame", 0)
    n_flare_steps = getattr(args, "n_flare_steps", 0)

    proc_dir = os.path.join(ckpt, "processor")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"processor/ not found in checkpoint: {ckpt}")
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    print(f"Processor loaded from: {proc_dir}")

    base_model_path = getattr(args, "base_model_path", "")
    ckpt_config = os.path.join(ckpt, "config.json")

    if base_model_path and os.path.isdir(base_model_path) and _has_hf_weights(base_model_path):
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path=base_model_path,
            action_dim=args.action_dim, action_chunk=args.action_chunk,
            use_tactile_deform=bool(args.use_tactile_deform),
            use_robot_state=bool(args.use_robot_state),
            torch_dtype=torch.bfloat16,
            tactile_intermediate_size=tac_isize,
            n_flare_tokens_per_frame=n_flare_tpf,
            n_flare_steps=n_flare_steps,
            use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
            vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
            use_tactile_vqvae=bool(getattr(args, "use_tactile_vqvae", 0)),
            vqvae_config=getattr(args, "vqvae_config", None),
        )
    elif os.path.exists(ckpt_config):
        model = _build_qwen3vl_from_config(ckpt_config, args)
    else:
        pretrained_path = None
        if ta:
            mp = ta.get("model_path", "")
            if mp and os.path.isdir(mp) and _has_hf_weights(mp):
                pretrained_path = mp
        if pretrained_path is None:
            raise FileNotFoundError(f"Cannot reconstruct model from {ckpt}")
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path=pretrained_path,
            action_dim=args.action_dim, action_chunk=args.action_chunk,
            use_tactile_deform=bool(args.use_tactile_deform),
            use_robot_state=bool(args.use_robot_state),
            torch_dtype=torch.bfloat16,
            tactile_intermediate_size=tac_isize,
            n_flare_tokens_per_frame=n_flare_tpf,
            n_flare_steps=n_flare_steps,
            use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
            vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
            use_tactile_vqvae=bool(getattr(args, "use_tactile_vqvae", 0)),
            vqvae_config=getattr(args, "vqvae_config", None),
        )

    ckpt_file = os.path.join(ckpt, "model.pt")
    sd = torch.load(ckpt_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 10): {missing[:10]}")
    model = model.to(torch.bfloat16)

    # Keep the embedded VQ-VAE + its F6 stats in fp32 so on-the-fly codes match
    # the standalone tokenizer (the bf16 cast above would otherwise downcast the
    # codebook and normalization buffers).
    if getattr(model, "tactile_vqvae", None) is not None:
        model.tactile_vqvae.float().eval()
        model.tacf6_vqvae_min = model.tacf6_vqvae_min.float()
        model.tacf6_vqvae_max = model.tacf6_vqvae_max.float()

    n_flare_total = n_flare_tpf * n_flare_steps
    if n_flare_total > 0:
        print(f"Flare prediction: {n_flare_steps} steps × {n_flare_tpf} tok/frame = {n_flare_total} total tokens")

    stats_path = args.stats_path or ""
    if not stats_path:
        candidate = os.path.join(ckpt, "stats_data.json")
        if os.path.exists(candidate):
            stats_path = candidate
    if not stats_path or not os.path.exists(stats_path):
        raise FileNotFoundError("Cannot find stats JSON.")

    with open(stats_path) as f:
        stats_raw = json.load(f)
    ds = args.dataset_name if args.dataset_name and args.dataset_name in stats_raw \
         else next(iter(stats_raw))

    def _arr(key, sub):
        return np.array(stats_raw[ds][key][sub])

    statistic = {
        "action_mask": _arr("action", "mask"),
        "action_min":  _arr("action", "q01"),
        "action_max":  _arr("action", "q99"),
        "tacf6_mask":  _arr("tactile_f6", "mask"),
        "tacf6_min":   _arr("tactile_f6", "q01"),
        "tacf6_max":   _arr("tactile_f6", "q99"),
    }
    if args.use_robot_state:
        statistic["state_mask"] = _arr("state", "mask")
        statistic["state_min"]  = _arr("state", "q01")
        statistic["state_max"]  = _arr("state", "q99")

    frozen = np.where(~np.asarray(statistic["action_mask"], dtype=bool))[0]
    if frozen.size:
        print(f"[serve] frozen action dims {frozen.tolist()} -> held at state[j] "
              f"(normalisation passthrough dims; see _clamp_frozen_absolute)")

    return model, processor, statistic


# ─────────────────────────────────────────────────────────────────────────────
# Tactile encoding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_tactile_f6(tactile_f6_input, statistic, device):
    """Encode F6 for the single-frame tacf6_embedder.

    Accepts either:
      * `[n_fingers, 6]`           — single current frame (legacy clients).
      * `[T, n_fingers, 6]`        — dense rolling window (new clients send
                                     the full VQ-VAE window).  We take the
                                     last frame here for the per-frame
                                     embedder; the full window is consumed
                                     separately by `_push_f6_and_encode`.
    """
    if tactile_f6_input is None:
        return None
    arr = np.asarray(tactile_f6_input, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[-1]                    # most recent frame
    tacf6 = arr.reshape(-1)
    norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                            statistic["tacf6_min"], statistic["tacf6_max"])
    return (torch.tensor(norm_tacf6.reshape(-1, 6), dtype=torch.bfloat16)
            .unsqueeze(0).to(device))


def _encode_tactile_deform(tactile_deform_input, device):
    if tactile_deform_input is None:
        return None
    arr = np.array(tactile_deform_input, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3:
        return (torch.tensor(arr).unsqueeze(0).unsqueeze(2)
                .to(device, dtype=torch.bfloat16))
    elif arr.ndim == 4:
        return (torch.tensor(arr).unsqueeze(0)
                .to(device, dtype=torch.bfloat16))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Cascaded slow/fast inference
# ─────────────────────────────────────────────────────────────────────────────

class CascadedServer:
    """
    Stateful server for cascaded flow-matching inference.

    Holds the slow-tick snapshot (cached latent + action KV at τ_split, the
    partially-denoised intermediate state x_split, encoded fast cameras +
    state, latent position ids) between requests so a subsequent fast tick
    can run the tactile expert without re-encoding the visual tower.

    A single `lock` serializes all model calls so:
      • only one inference runs on the GPU at a time, and
      • a fast request that arrives while a slow inference is still in flight
        blocks until the slow one finishes ("wait until refinement finishes"
        semantic).  ZMQ REP is already single-threaded; the lock is belt-and-
        braces for any future move to a multi-threaded transport.
    """

    def __init__(self, args, model, processor, statistic):
        self.args      = args
        self.model     = model
        self.processor = processor
        self.statistic = statistic
        self.device    = f"cuda:{args.cuda}"
        self.lock      = threading.Lock()

        # Ablation: when True, skip the cascaded split entirely and let the
        # action expert integrate the full τ ∈ [0, 1] flow alone.  The tactile
        # expert is never invoked; fast ticks return the cached full-flow
        # chunk unchanged.
        self.disable_tactile = bool(getattr(args, "disable_tactile", 0))

        # Slow-tick snapshot
        self.cached_kv          = None
        self.x_split            = None             # [B, n_chunk, action_dim] bf16,
                                                   # action-expert intermediate at τ=τ_split
        self.tau_split          = None
        self.position_ids       = None
        self.attention_mask     = None
        self.n_action_in_cache  = 0
        self.chunk_id           = -1               # incremented per slow
        self.last_actions       = None             # cached absolute chunk for
                                                   # disable_tactile fast-tick passthrough

        # ── cross-timestep memory (dev/memory Part C) ──
        # Raw-content buffers, not pre-computed KV: RoPE bakes each row's
        # position into its cached K before it's ever stored (confirmed by
        # reading Qwen3VLAttentionMoT.forward -- apply_rotary_pos_emb_1d runs
        # BEFORE cache.update()), so a KV snapshot computed once can't be
        # "re-shifted" for a later, differently-timed tick without recompute.
        # build_memory_kv_slow/_fast already recompute fresh from raw inputs
        # every call for exactly this reason -- these buffers just hold what
        # to feed them, mirroring what OrigamiDataset stores per episode.
        self.memory_slow_seconds = sorted(
            (float(s) for s in str(getattr(args, "memory_slow_seconds", "") or "").split(",") if s.strip()),
            reverse=True)                              # oldest target first
        self.memory_fast          = int(getattr(args, "memory_fast", 0))
        self.memory_rope_stride_slow = float(getattr(args, "memory_rope_stride_slow", 32.0))
        self.memory_rope_stride_fast = float(getattr(args, "memory_rope_stride_fast", 8.0))
        self.memory_buffer_margin_sec = float(getattr(args, "memory_buffer_margin_sec", 2.0))
        # memory_buf_slow: List[(timestamp, PIL.Image head, str task_text)]
        self.memory_buf_slow: list = []
        # memory_buf_fast: List[(timestamp, [wrist_right, wrist_left], action_raw_np)]
        # tacf6_hist is intentionally NOT captured here -- see MEMORY_DESIGN.md
        # "Part C known limitation" (avoids a rolling-F6-buffer double-mutation
        # risk between this snapshot and _run_fast's own encode of the same tick).
        self.memory_buf_fast: list = []
        # Wrist images from the most recent slow tick -- the real wire
        # protocol never sends fresh wrist images on a fast-mode request
        # (confirmed: eval_trex_async.py's fast_payload carries only
        # tactile_f6/tactile_deform), so a fast tick has no new image to
        # remember. Reusing the last-seen images (paired with THAT tick's
        # freshly refined action) still captures real fast-tick-rate
        # temporal density for the part that actually changes that fast --
        # the action -- instead of only updating this buffer at slow-tick
        # rate, which was silently ~6x coarser than what training's
        # per-parquet-row memory_fast represents (dev/memory, found while
        # auditing the architecture, not caught at build time).
        self._last_fast_images = None
        if self.memory_slow_seconds:
            print(f">>> slow memory enabled: targets {self.memory_slow_seconds}s back, "
                  f"rope_stride={self.memory_rope_stride_slow}")
        if self.memory_fast > 0:
            print(f">>> fast memory enabled: window={self.memory_fast} ticks, "
                  f"rope_stride={self.memory_rope_stride_fast} (tactile not "
                  f"included in server-side fast memory yet)")

        # ── action output ──
        # All-absolute target: the model's (denormalized) output IS the
        # absolute chunk directly, see _reconstruct. `--action_output delta`
        # is a legacy escape hatch for pre-all-absolute all-delta-from-state
        # checkpoints -- it subtracts the measured state back off before
        # replying, reproducing the old "client adds observation/state"
        # behaviour, for whoever still serves one of those.
        self.action_output = str(getattr(args, "action_output", "absolute"))
        # Last emitted absolute chunk + when it was emitted -- kept for
        # _prev_command, the server's best estimate of "what did we just
        # tell the robot to do", used by the cross-timestep memory buffer's
        # fast-tier action_abs proxy (dev/memory, MEMORY_DESIGN.md). No
        # longer feeds an anchor reconstruction (there isn't one anymore).
        self.last_chunk         = None
        self.last_chunk_time    = 0.0
        self.seed_state         = None             # state at the last slow tick

        # ── phase conditioning ──
        # `--phase-mode progress` bakes "(fold k of N)" into the training prompt
        # from the sample's fraction through its episode -- which needs the
        # episode's total length, a number the robot does not have.  Online we
        # approximate it from elapsed wall-clock against the training set's
        # median episode duration (`median_episode_frames` in the prep's
        # dataset.json).  It is the same prompt for most of the attempt and it
        # degrades gracefully: a slow attempt just reaches the later folds late.
        # ── the language prompt ──
        # The policy trained on one constant string (--instruction, recorded in
        # training_args.json).  The robot sends its own `prompt` field, which the
        # competition spec says "may be empty" and the kit's example fills with
        # "fold the plane" -- neither is what the weights saw.  Serving the
        # client's text would put an unseen token span in front of every
        # observation, so the trained string wins unless asked otherwise.
        self.instruction = str(getattr(args, "instruction", "") or "")
        self.use_client_prompt = bool(getattr(args, "use_client_prompt", 0))
        if self.instruction and not self.use_client_prompt:
            print(f"Prompt pinned to the trained instruction: {self.instruction!r}")
        elif not self.instruction:
            print("WARNING: checkpoint records no instruction; using the client's "
                  "prompt verbatim, which may not be what the policy trained on.")

        self.phase_mode = str(getattr(args, "phase_mode", "none") or "none")
        self.phase_seconds = float(getattr(args, "phase_episode_seconds", 0.0))
        self.n_phases = int(getattr(args, "n_phases", 6))
        self.episode_start = None
        if self.phase_mode == "progress" and self.phase_seconds <= 0:
            raise SystemExit(
                "--phase_mode progress needs --phase_episode_seconds: the fold "
                "index the policy trained against is a fraction of the episode, "
                "and online that fraction can only come from elapsed time over an "
                "expected duration. Use median_episode_frames / 30 from the "
                "training split's meta/dataset.json.")

        # VQ-VAE tactile-code encoder.  Two modes:
        #   • embedded  — the model carries `tactile_vqvae`; the server only
        #                 builds the raw F6 history window and the model encodes
        #                 it internally (codes never leave the model).
        #   • external  — legacy: load a standalone VQ-VAE here and feed
        #                 pre-computed codes to the model.
        # Both keep a rolling F6 buffer so each fast tick sees the historical
        # `window` frames — same alignment as offline JSON encoding.
        self.vqvae_model    = None
        self.vqvae_stats    = None
        self.vqvae_window   = 16
        self.f6_buffer: list = []                  # list of [10, 6] np arrays
        self.use_embedded_vqvae = bool(
            getattr(model, "use_tactile_vqvae", False)
            and getattr(model, "tactile_vqvae", None) is not None)
        if self.use_embedded_vqvae:
            self.vqvae_window = int(model.tactile_vqvae.cfg.window)
            print(f">>> embedded VQ-VAE in model — F6 encoded on-the-fly "
                  f"(K={model.tactile_vqvae.cfg.codebook_size}, W={self.vqvae_window})")
        elif bool(getattr(args, "use_tactile_code", 0)):
            from tactile_vqvae.models.tactile_vqvae import (
                TactileVQVAE, TactileVQVAEConfig)
            from tactile_vqvae.data.stats import TacF6Stats
            blob = torch.load(args.vqvae_ckpt, map_location="cpu",
                              weights_only=False)
            cfg = TactileVQVAEConfig.from_dict(blob["config"])
            self.vqvae_model = TactileVQVAE(cfg)
            self.vqvae_model.load_state_dict(blob["model_state"])
            self.vqvae_model.eval().to(self.device)
            self.vqvae_stats  = TacF6Stats.from_dict(blob["stats"])
            self.vqvae_window = int(cfg.window)
            print(f">>> external VQ-VAE loaded for tactile codes "
                  f"(K={cfg.codebook_size}, W={self.vqvae_window})")

    def _rolling_f6_window(self, tactile_f6_input):
        """Build the raw [window, 10, 6] F6 history from either a dense client
        window ([T,10,6]) or a single frame ([10,6]) via the rolling buffer."""
        if tactile_f6_input is None:
            return None
        arr = np.asarray(tactile_f6_input, dtype=np.float32)
        w = self.vqvae_window
        if arr.ndim == 3:
            if arr.shape[0] >= w:
                arr = arr[-w:]
            else:
                head = np.repeat(arr[:1], w - arr.shape[0], axis=0)
                arr = np.concatenate([head, arr], axis=0)
        else:
            f6 = arr.reshape(10, 6)
            self.f6_buffer.append(f6)
            if len(self.f6_buffer) > w:
                self.f6_buffer = self.f6_buffer[-w:]
            if len(self.f6_buffer) < w:
                head = [self.f6_buffer[0]] * (w - len(self.f6_buffer))
                arr = np.stack(head + self.f6_buffer, axis=0)
            else:
                arr = np.stack(self.f6_buffer, axis=0)
        return arr                                  # [W, 10, 6] raw

    def _f6_history_window(self, tactile_f6_input):
        """Raw F6 history tensor [1, window, 10, 6] for the embedded VQ-VAE."""
        arr = self._rolling_f6_window(tactile_f6_input)
        if arr is None:
            return None
        return (torch.from_numpy(arr.astype(np.float32))
                .unsqueeze(0).to(self.device))

    def _push_f6_and_encode(self, tactile_f6_input):
        """Encode an F6 history window into per-hand VQ-VAE codes.

        Accepted inputs:
          * `[10, 6]`         — single current frame.  We append it to a
                                server-side rolling buffer and encode the
                                last `window` frames.  Used by legacy clients
                                that don't track tactile history themselves.
          * `[T, 10, 6]`      — dense rolling window from a client that
                                already maintains the F6 history at fetch-
                                rate (e.g. eval_trex_async.py with the F6_HISTORY
                                deque).  We use the window directly — this
                                matches VQ-VAE training-time temporal density.

        Returns `[1, K]` int64 codes (K=2 hand or K=10 finger) on `self.device`,
        or None when VQ-VAE is disabled / input is missing.
        """
        if self.vqvae_model is None or tactile_f6_input is None:
            return None
        arr = self._rolling_f6_window(tactile_f6_input)        # [W, 10, 6] raw
        arr_n = self.vqvae_stats.normalize(arr).astype(np.float32, copy=False)

        is_per_finger = getattr(self.vqvae_model.cfg, "granularity", "hand") == "finger"
        n_fingers = int(getattr(self.vqvae_model.cfg, "n_fingers", 5)) if is_per_finger else 1

        if is_per_finger:
            codes = np.zeros((2, n_fingers), dtype=np.int64)     # [2, 5]
        else:
            codes = np.zeros(2, dtype=np.int64)                  # [2]

        for hand in (0, 1):
            wh = arr_n[:, hand * 5: (hand + 1) * 5, :]           # [W, 5, 6]
            t = torch.from_numpy(wh).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = self.vqvae_model.encode(t).cpu().numpy()   # [1] or [1, 5]
            if is_per_finger:
                codes[hand] = idx.reshape(-1)
            else:
                codes[hand] = int(idx.item())

        # Flatten and add batch dim → [1, 2] (hand) or [1, 10] (finger).
        flat = codes.reshape(-1)
        return torch.tensor(flat, dtype=torch.long,
                            device=self.device).unsqueeze(0)

    # -- internal: phase-conditioned prompt --------------------------------
    def _task_text(self, task_description):
        """The prompt as the policy saw it in training.

        Mirrors `qwen_vla.origami_dataset._task_text`; the only difference is
        where the progress fraction comes from (elapsed time here, the sample's
        row index there).
        """
        if self.instruction and not self.use_client_prompt:
            task_description = self.instruction
        if self.phase_mode != "progress":
            return task_description
        if self.episode_start is None:
            self.episode_start = time.time()
        phase = min(1.0, max(0.0, (time.time() - self.episode_start) / self.phase_seconds))
        k = min(self.n_phases, 1 + int(phase * self.n_phases))
        return f"{task_description} (fold {k} of {self.n_phases})"

    def reset_episode(self):
        """Start a new fold attempt: clears the phase clock and the last-command chain."""
        self.episode_start = None
        self.last_chunk = None
        self.last_chunk_time = 0.0
        # Memory must not leak across attempts, exactly like the training-side
        # index never lets a lookback window cross an episode boundary.
        self.memory_buf_slow = []
        self.memory_buf_fast = []

    # -- internal: cross-timestep memory (dev/memory Part C) --------------
    def _remember_slow(self, now, head_image, task_description):
        """Record this slow tick's own content so a LATER tick can use it as
        memory. Must only be called with content from a tick already fully
        processed -- never the tick currently selecting memory for itself."""
        if not self.memory_slow_seconds:
            return
        self.memory_buf_slow.append((now, head_image.copy(), task_description))
        cutoff = now - (max(self.memory_slow_seconds) + self.memory_buffer_margin_sec)
        self.memory_buf_slow = [e for e in self.memory_buf_slow if e[0] >= cutoff]

    def _remember_fast(self, now, fast_images, action_raw):
        """Record this slow tick's fast-content snapshot (wrist images +
        best-known executed action) for a later tick's fast memory."""
        if self.memory_fast <= 0:
            return
        self.memory_buf_fast.append(
            (now, [img.copy() for img in fast_images], np.asarray(action_raw, dtype=np.float64)))
        if len(self.memory_buf_fast) > self.memory_fast:
            self.memory_buf_fast = self.memory_buf_fast[-self.memory_fast:]

    def _select_memory_slow_rows(self, now):
        """For each configured lookback target, the buffered entry whose
        timestamp is nearest `now - target` (buffered entries aren't
        uniformly spaced -- inference ticks aren't uniform -- so this is a
        nearest-timestamp snap, not a fixed-position lookup; see Context in
        MEMORY_DESIGN.md). Returns [] (a true no-op) when memory is off or
        the buffer is still empty (e.g. the first slow tick of an episode)."""
        if not self.memory_slow_seconds or not self.memory_buf_slow:
            return []
        rows = []
        for dt_nominal in self.memory_slow_seconds:            # oldest target first
            target_ts = now - dt_nominal
            ts, img, task = min(self.memory_buf_slow, key=lambda e: abs(e[0] - target_ts))
            content = [{"type": "image"}, {"type": "text", "text": task}]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True)
            inp = self.processor(text=text, images=[img], return_tensors="pt", padding=False)
            rows.append({
                "input_ids": inp.input_ids.to(self.device),
                "attention_mask": inp.attention_mask.to(self.device),
                "pixel_values": (inp.pixel_values.to(self.device, dtype=torch.bfloat16)
                                if getattr(inp, "pixel_values", None) is not None else None),
                "image_grid_thw": (inp.image_grid_thw.to(self.device)
                                   if getattr(inp, "image_grid_thw", None) is not None else None),
                "dt_actual": torch.tensor([now - ts], dtype=torch.float32, device=self.device),
            })
        return rows

    def _select_memory_fast_rows(self):
        """The last `memory_fast` buffered fast-tick snapshots, oldest first
        -- a linear window, not exponential, matching training (see Context
        in MEMORY_DESIGN.md: fast ticks are close enough together that
        count ~= time). Returns [] when memory is off or nothing buffered
        yet (e.g. before the first slow tick of an episode)."""
        if self.memory_fast <= 0 or not self.memory_buf_fast:
            return []
        statistic = self.statistic
        rows = []
        for ts, fast_images, action_raw in self.memory_buf_fast:
            content = [{"type": "image"} for _ in fast_images]
            text = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False,
                add_generation_prompt=True)
            inp = self.processor(text=text, images=fast_images,
                                 return_tensors="pt", padding=False)
            action_norm = _normalize(action_raw, statistic["action_mask"],
                                     statistic["action_min"], statistic["action_max"])
            rows.append({
                "input_ids": inp.input_ids.to(self.device),
                "attention_mask": inp.attention_mask.to(self.device),
                "pixel_values": (inp.pixel_values.to(self.device, dtype=torch.bfloat16)
                                if getattr(inp, "pixel_values", None) is not None else None),
                "image_grid_thw": (inp.image_grid_thw.to(self.device)
                                   if getattr(inp, "image_grid_thw", None) is not None else None),
                "action_abs": torch.tensor(action_norm, dtype=torch.float32,
                                           device=self.device).unsqueeze(0),
            })
        return rows

    def _build_memory_kv(self, now):
        """Combined slow+fast memory cache for this tick, or None (a true
        no-op) when both tiers are off/empty -- forward_flow_action_full/
        _partial's memory_kv=None path is then byte-identical to current
        behavior."""
        memory_kv = None
        slow_rows = self._select_memory_slow_rows(now)
        if slow_rows:
            memory_kv = self.model.build_memory_kv_slow(
                slow_rows, rope_stride=self.memory_rope_stride_slow)
        fast_rows = self._select_memory_fast_rows()
        if fast_rows:
            memory_kv = self.model.build_memory_kv_fast(
                fast_rows, past_kv=memory_kv, rope_stride=self.memory_rope_stride_fast)
        return memory_kv

    # -- internal: last-command tracking (feeds the memory buffer) --------
    def _prev_command(self, state):
        """Best estimate of the executed action for the frame just before now.

        The robot streams our last chunk at 30 Hz, so the element it is on is
        set by elapsed wall-clock rather than by anything in the request.
        Index `k-1` is the frame before the one now being replaced. Before
        any chunk has been emitted, fall back to the measured state.

        Used as the cross-timestep memory buffer's fast-tier action_abs
        proxy (dev/memory, see MEMORY_DESIGN.md) -- no longer feeds an
        anchor reconstruction (all-absolute has none).
        """
        if self.last_chunk is None:
            return state
        elapsed = max(0.0, time.time() - self.last_chunk_time)
        k = int(round(elapsed * CONTROL_HZ)) - 1
        k = max(0, min(k, self.last_chunk.shape[0] - 1))
        return self.last_chunk[k]

    def _reconstruct(self, normalised_chunk, state):
        """Normalised model output -> the absolute [T, 65] the wire contract wants.

        All-absolute target: the model's output IS the absolute chunk
        directly once denormalized -- no per-dim anchor addition.
        """
        statistic = self.statistic
        absolute = _denormalize(np.asarray(normalised_chunk, dtype=np.float64),
                                statistic["action_mask"],
                                statistic["action_min"], statistic["action_max"])
        absolute = _clamp_frozen_absolute(absolute, statistic["action_mask"], state)
        self.last_chunk = absolute
        self.last_chunk_time = time.time()
        if self.action_output == "delta":
            # Legacy escape hatch for pre-all-absolute checkpoints; the
            # client adds the state back.
            return absolute - np.asarray(state, dtype=np.float64)[None, :]
        return absolute

    # -- internal: build slow embeddings, run action-only flow, cache state --
    def _run_slow(
        self, task_description, slow_images, fast_images,
        tactile_f6_input=None, tactile_deform_input=None, state_fast=None,
    ):
        args, model, processor, statistic = (
            self.args, self.model, self.processor, self.statistic)
        device = self.device
        now = time.time()
        # Built from buffered PRIOR ticks only -- this tick's own content is
        # recorded into the buffers further down, after this call.
        memory_kv = self._build_memory_kv(now)

        if args.image_size:
            _sz = tuple(args.image_size)
            slow_images = [img.resize(_sz, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(_sz, Image.LANCZOS) for img in fast_images]

        # The chunk is conditioned on this tick's observation, so it's pinned
        # here and reused (self.seed_state) by every fast tick that continues
        # it -- used for the frozen-dim hold in _reconstruct
        # (_clamp_frozen_absolute) and the memory buffer's action_abs proxy
        # (_prev_command).
        state = (np.asarray(state_fast, dtype=np.float64)
                 if state_fast is not None else None)
        if state is None:
            frozen = np.where(~np.asarray(statistic["action_mask"], dtype=bool))[0]
            if frozen.size:
                print(f"[serve] WARNING: slow request carries no state_fast -- "
                      f"frozen action dims {frozen.tolist()} will NOT be held at "
                      f"the measured state this tick (see _clamp_frozen_absolute).")
        self.seed_state = state

        state_embeds = None
        if args.use_robot_state and state_fast is not None:
            norm_state = _normalize(
                np.array(state_fast, dtype=np.float32),
                statistic["state_mask"], statistic["state_min"], statistic["state_max"])
            state_vec = torch.tensor(norm_state, dtype=torch.bfloat16).unsqueeze(0).to(device)
            state_embeds = model.state_embedder(state_vec).unsqueeze(1)

        n_slow = len(slow_images)
        all_pil = slow_images + fast_images
        content = [{"type": "image"} for _ in slow_images]
        content.append({"type": "text", "text": task_description})
        content += [{"type": "image"} for _ in fast_images]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=all_pil if all_pil else None,
                        return_tensors="pt", padding=False)

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                        if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw)

        fast_embeds = None
        if image_grid_thw is not None and fast_images:
            merge = getattr(model.visual, "spatial_merge_size",
                            getattr(processor.image_processor, "merge_size", 2))
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in image_grid_thw[:n_slow])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, input_ids,
                model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds

        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(
                position_ids, model.n_flare_tokens)

        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)

        # Record this tick's own content for LATER ticks' memory -- after
        # memory_kv above was already built from prior content, so this tick
        # never becomes its own memory.
        if slow_images:
            self._remember_slow(now, slow_images[0], task_description)
        if fast_images and state is not None:
            self._remember_fast(now, fast_images, self._prev_command(state))
            self._last_fast_images = fast_images

        if self.disable_tactile:
            # Action-expert-only ablation: integrate the full τ ∈ [0, 1] flow
            # and return the resulting chunk directly.  No tactile expert is
            # invoked; subsequent fast ticks reuse the same chunk.
            full_chunk = model.forward_flow_action_full(
                inputs_embeds=slow_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise=noise,
                state_embeds=state_embeds,
                fast_embeds=fast_embeds,
                num_steps=args.cascaded_total_steps,
                memory_kv=memory_kv,
            )
            self.x_split           = None
            self.tau_split         = None
            self.cached_kv         = None
            self.position_ids      = position_ids
            self.attention_mask    = attention_mask
            self.n_action_in_cache = 0
            self.chunk_id         += 1
            a_full = self._reconstruct(full_chunk[0].float().cpu().numpy(),
                                       self.seed_state)
            self.last_actions = list(a_full)
            return self.last_actions, self.chunk_id

        x_split, cached_kv, n_action_in_cache, tau_split = (
            model.forward_flow_action_partial(
                inputs_embeds=slow_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                noise=noise,
                state_embeds=state_embeds,
                fast_embeds=fast_embeds,
                num_steps_total=args.cascaded_total_steps,
                split_step=args.cascaded_split_step,
                refresh_clean_kv=True,
                memory_kv=memory_kv,
            ))
        self.x_split    = x_split           # action expert's intermediate at τ=τ_split
        self.tau_split  = tau_split
        self.cached_kv         = cached_kv
        self.position_ids      = position_ids
        self.attention_mask    = attention_mask
        self.n_action_in_cache = n_action_in_cache
        self.chunk_id         += 1

        # Cascaded mode produces no usable action from the slow tick alone;
        # caller should always send 'slow_and_fast' so a_refined comes back.
        return [], self.chunk_id

    # -- internal: run tactile expert flow continuation on cached state --
    def _run_fast(self, tactile_f6_input=None, tactile_deform_input=None):
        args, model, statistic = self.args, self.model, self.statistic
        device = self.device

        if self.disable_tactile:
            # Tactile expert is disabled — fast ticks just replay the chunk
            # computed by the last slow tick.
            if self.last_actions is None:
                raise RuntimeError(
                    "fast request received before any slow request — server "
                    "has no cached action.  Send mode='slow_and_fast' first.")
            return self.last_actions, self.chunk_id

        if self.cached_kv is None or self.x_split is None:
            raise RuntimeError(
                "fast request received before any slow request — server has "
                "no cached state. Send mode='slow' or 'slow_and_fast' first.")

        tac_f6_tensor     = _encode_tactile_f6(
            tactile_f6_input if args.use_tactile_vec else None,
            statistic, device)
        tac_deform_tensor = _encode_tactile_deform(
            tactile_deform_input if args.use_tactile_deform else None, device)
        # Embedded VQ-VAE: hand the model the raw F6 history and let it encode.
        # External / legacy: encode here and pass pre-computed codes.
        if self.use_embedded_vqvae:
            tac_codes_tensor   = None
            tac_hist_tensor    = self._f6_history_window(tactile_f6_input)
        else:
            tac_codes_tensor   = self._push_f6_and_encode(tactile_f6_input)
            tac_hist_tensor    = None

        # Continue the action expert's flow with the tactile expert from
        # x_split → τ=0; the result IS the clean action (no Â + Δa add).
        refined = model.tactile_flow_continue(
            cached_kv          = self.cached_kv,
            latent_position_ids= self.position_ids,
            n_action_in_cache  = self.n_action_in_cache,
            x_split            = self.x_split,
            tau_split          = self.tau_split,
            tactile_f6         = tac_f6_tensor,
            tactile_deform     = tac_deform_tensor,
            tactile_codes      = tac_codes_tensor,
            tactile_f6_history = tac_hist_tensor,
            num_steps_total    = args.cascaded_total_steps,
            split_step         = args.cascaded_split_step,
        )
        a_refined = self._reconstruct(refined[0].float().cpu().numpy(),
                                      self.seed_state)

        # Fast-tick-rate fast-memory capture. Wrist images are reused from
        # the last slow tick (nothing fresher exists over the wire -- see
        # __init__'s note on self._last_fast_images), but the action
        # component genuinely is fresh: _reconstruct just wrote this tick's
        # own refined chunk into self.last_chunk, so _prev_command called
        # right now (elapsed~=0) returns THIS tick's command, not the
        # previous one. This is what actually fixes the rate mismatch --
        # action density now matches fast-tick cadence even though image
        # density is capped at slow-tick cadence by the wire protocol.
        if self._last_fast_images:
            self._remember_fast(time.time(), self._last_fast_images,
                                self._prev_command(self.seed_state))

        return list(a_refined), self.chunk_id

    def predict(self, mode, payload):
        """Top-level dispatch.  Returns dict suitable for pickling back to
        the client.  Any exception inside a mode's body is propagated to the
        caller, which logs and replies with status='error'."""
        slow_img = (Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
                    if "image_head" in payload else None)
        fast_list = []
        if "image_wrist_right" in payload:
            fast_list.append(Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB"))
        if "image_wrist_left" in payload:
            fast_list.append(Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB"))

        tac_f6     = payload.get("tactile_f6")
        tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform"))
        state_fast = payload.get("state_fast")
        # A client that folds several planes in one session must say so, or the
        # phase clock and the previous-command chain carry over from the last
        # attempt.  Absent the flag the first slow request starts the clock.
        if payload.get("reset_episode"):
            self.reset_episode()
        task_desc  = self._task_text(payload.get("task_description", ""))

        with self.lock, torch.inference_mode():
            self.model = self.model.to(self.device).eval()
            t0 = time.time()
            if mode == "slow":
                if slow_img is None:
                    raise ValueError("slow request requires image_head")
                actions, cid = self._run_slow(
                    task_desc, [slow_img], fast_list,
                    tac_f6, tac_deform, state_fast)
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": "success", "mode": "slow",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            elif mode == "fast":
                actions, cid = self._run_fast(tac_f6, tac_deform)
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": "success", "mode": "fast",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            elif mode == "slow_and_fast":
                if slow_img is None:
                    raise ValueError("slow_and_fast request requires image_head")
                self._run_slow(task_desc, [slow_img], fast_list,
                               tac_f6, tac_deform, state_fast)
                actions, cid = self._run_fast(tac_f6, tac_deform)
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": "success", "mode": "slow_and_fast",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            else:
                raise ValueError(f"unknown mode: {mode}")


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic = model_load(args)
    print("Model loaded successfully!")

    # Warm-up (use 2 fast images for bimanual / dual-arm tasks)
    print("Warming up model...")
    dummy_slow  = [Image.new("RGB", (224, 224), color="black")]
    n_fast_cams = 2 if args.action_dim > 31 else 1
    dummy_fast  = [Image.new("RGB", (224, 224), color="black") for _ in range(n_fast_cams)]
    # Always sent, even with --use_robot_state 0: the reconstruction needs the
    # state for the frozen-dim hold (_clamp_frozen_absolute), not just the encoder.
    dummy_state = np.zeros(args.action_dim, dtype=np.float32)
    dummy_f6    = np.zeros((5, 6), dtype=np.float32) if args.use_tactile_vec else None
    dummy_deform = np.zeros((5, 240, 240), dtype=np.float32) if args.use_tactile_deform else None

    server = CascadedServer(args, model, processor, statistic)
    # Warm-up: run one slow_and_fast and discard
    dummy_payload = {
        "image_head":         _pil_to_bytes(dummy_slow[0]),
        "image_wrist_right":  _pil_to_bytes(dummy_fast[0]),
        "task_description":   "dummy task",
        "tactile_f6":         dummy_f6,
        "tactile_deform":     dummy_deform,
        "state_fast":         dummy_state,
    }
    if len(dummy_fast) > 1:
        dummy_payload["image_wrist_left"] = _pil_to_bytes(dummy_fast[1])
    result = server.predict("slow_and_fast", dummy_payload)
    print(f"Warm-up output shape: "
          f"{np.array(result['actions']).shape} ({server.action_output} radians), "
          f"latency {result['latency_ms']:.1f} ms")

    # ZMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"VLA Server listening on port {args.port} "
          f"(cascaded slow/fast, single-threaded REP)")

    step_counter = 0
    n_slow = n_fast = 0
    while True:
        try:
            payload = pickle.loads(socket.recv())

            # Default to slow_and_fast for first request; clients can override
            # with mode='slow' or mode='fast'.
            mode = payload.get("mode", "slow_and_fast")
            result = server.predict(mode, payload)
            if mode == "fast":
                n_fast += 1
            else:
                n_slow += 1

            socket.send(pickle.dumps(result))
            step_counter += 1
            if step_counter % 10 == 0:
                print(f"Processed {step_counter} requests "
                      f"(slow={n_slow}, fast={n_fast}, "
                      f"chunk_id={server.chunk_id}). "
                      f"Task: {payload.get('task_description', '')}")

        except Exception as e:
            traceback.print_exc()
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-world ZMQ server (with flare prediction)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--n_flare_steps", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--instruction", type=str, default="",
                        help="language prompt to serve; auto-detected from the "
                             "checkpoint's training_args.json")
    parser.add_argument("--use_client_prompt", type=int, default=0,
                        help="1 = pass the caller's prompt through untouched "
                             "instead of the string the policy trained on")
    parser.add_argument("--phase_mode", choices=["none", "progress"], default="none",
                        help="must match the prep the checkpoint trained on. "
                             "`progress` appends '(fold k of N)' to the prompt.")
    parser.add_argument("--phase_episode_seconds", type=float, default=0.0,
                        help="expected attempt duration, used only with "
                             "--phase_mode progress. Take it from the training "
                             "split's median_episode_frames / 30.")
    parser.add_argument("--n_phases", type=int, default=6,
                        help="folds in the target figure; matches the prep's n_phases")
    parser.add_argument("--action_output", choices=["absolute", "delta"],
                        default="absolute",
                        help="`absolute` is the competition wire contract: float32"
                             "[T, 65] joint angles, the all-absolute model's "
                             "denormalized output directly (see _reconstruct). "
                             "`delta` returns action - state for the client to add "
                             "-- a legacy escape hatch for pre-all-absolute "
                             "all-delta-from-state checkpoints.")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # Cascaded flow matching schedule (auto-detected from training_args.json
    # when available).  The client sends payloads with mode='slow' once per
    # action chunk and mode='fast' multiple times within the chunk window;
    # the first request must be 'slow' or 'slow_and_fast'.
    parser.add_argument("--cascaded_total_steps", type=int, default=10)
    parser.add_argument("--cascaded_split_step",  type=int, default=6)

    # Ablation: action-expert-only inference (no tactile expert ever invoked).
    # The action expert integrates the full τ ∈ [0, 1] flow for
    # `cascaded_total_steps` Euler steps; fast ticks return the cached chunk.
    parser.add_argument("--disable_tactile", type=int, default=0,
                        help="1: skip the cascaded split.  Action expert "
                             "integrates the full flow alone; tactile expert "
                             "is never called.  Useful for ablating the "
                             "tactile-expert contribution at test time.")

    # VQ-VAE tactile code tokens (fast-path only).  When 0 (default) the model
    # graph and behavior are identical to the pre-feature version — flip the
    # flag to revert.  When 1, --vqvae_ckpt must be a TactileVQVAE latest.pt.
    # Server maintains a rolling 16-frame F6 buffer and encodes per-hand on
    # each fast tick.
    parser.add_argument("--use_tactile_code", type=int, default=0,
                        help="1: server-side VQ-VAE encodes a rolling F6 "
                             "window into 2 codes per fast tick.")
    parser.add_argument("--vqvae_codebook_size", type=int, default=64,
                        help="Codebook size of the VQ-VAE that produces the codes.")
    parser.add_argument("--vqvae_ckpt", type=str, default="",
                        help="Path to TactileVQVAE checkpoint (latest.pt). "
                             "Required when --use_tactile_code 1.")

    # Cross-timestep memory (dev/memory Part C). Default off (empty/0), byte-
    # identical to current behavior -- see T-Rex/qwen_vla/MEMORY_DESIGN.md.
    parser.add_argument("--memory_slow_seconds", type=str, default="",
                        help="Comma-separated exponential lookback targets in "
                             "seconds, e.g. '0.25,0.5,1.0,5.0'. Empty (default): "
                             "no slow memory. Must match what the checkpoint "
                             "was trained with.")
    parser.add_argument("--memory_fast", type=int, default=0,
                        help="Linear fast-memory window (past wrist+action "
                             "ticks). 0 (default): no fast memory. Must match "
                             "what the checkpoint was trained with.")
    parser.add_argument("--memory_rope_stride_slow", type=float, default=32.0,
                        help="RoPE position units per real second for slow "
                             "memory (Qwen3VLVLAModel.build_memory_kv_slow). "
                             "Untuned placeholder -- see MEMORY_DESIGN.md.")
    parser.add_argument("--memory_rope_stride_fast", type=float, default=8.0,
                        help="RoPE position units per fast-memory step "
                             "(Qwen3VLVLAModel.build_memory_kv_fast). Untuned "
                             "placeholder -- see MEMORY_DESIGN.md.")
    parser.add_argument("--memory_buffer_margin_sec", type=float, default=2.0,
                        help="Extra seconds kept beyond max(memory_slow_seconds) "
                             "before a slow-memory buffer entry is evicted, so "
                             "the nearest-timestamp lookup always has a real "
                             "candidate even when ticks run slower than nominal.")

    args = parser.parse_args()
    if bool(args.use_tactile_code) and not args.vqvae_ckpt:
        parser.error("--vqvae_ckpt must be set when --use_tactile_code 1")
    main(args)

