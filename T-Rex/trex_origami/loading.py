"""Checkpoint loader for the T-Rex origami policy (serving side).

This is `scripts/test.py::model_load` lifted verbatim into the package so the
deployment adapter (`trex_origami.policy`) and the submission image can import
it without `scripts/`, `pyzmq` or anything else the ZMQ dev server pulls in.
Keep the two in sync: `scripts/sync_submission_files.sh --check` in the
submission folder diffs this file against the source tree, and the eval
scripts still go through `scripts/test.py` unchanged.
"""
from __future__ import annotations

import glob as _glob_mod
import json
import os
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoProcessor

from qwen_vla import Qwen3VLVLAModel

__all__ = ["model_load", "load_args_from_checkpoint"]


def load_args_from_checkpoint(checkpoint_path: str) -> SimpleNamespace:
    """The `args` namespace `model_load` wants, filled from training_args.json.

    Fields that `model_load` auto-detects from the file are left at their
    defaults so the auto-detection (not the CLI) decides.
    """
    ta_path = os.path.join(checkpoint_path, "training_args.json")
    ta = json.load(open(ta_path)) if os.path.exists(ta_path) else {}
    return SimpleNamespace(
        checkpoint_path=checkpoint_path, base_model_path="", stats_path="", dataset_name="",
        action_dim=int(ta.get("action_dim", 65)), action_chunk=int(ta.get("action_chunk", 25)),
        use_robot_state=int(ta.get("use_robot_state", 1)),
        use_tactile_vec=int(ta.get("use_tactile_vec", 1)),
        use_tactile_deform=int(ta.get("use_tactile_deform", 1)),
        tactile_intermediate_size=0, n_flare_tokens_per_frame=0, n_flare_steps=0,
        use_tactile_code=0, vqvae_codebook_size=64, use_tactile_vqvae=0, vqvae_config=None,
        cascaded_total_steps=10, cascaded_split_step=6)


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
    _glob = _glob_mod
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

    # All 65 outputs are absolute joint radians -- no anchor/delta reconstruction.
    # A handful of near-static dims are held at the measured state instead of
    # predicted; frozen_action_dims/clamp_frozen_absolute (qwen_vla.origami_dataset)
    # derive that list from the norm-stats mask, not from anything recorded here.
    frozen = np.where(~np.asarray(statistic["action_mask"], dtype=bool))[0]
    if frozen.size:
        print(f"[serve] frozen action dims {frozen.tolist()} -> held at state[j] "
              f"(normalisation passthrough dims; see _clamp_frozen_absolute)")

    return model, processor, statistic
