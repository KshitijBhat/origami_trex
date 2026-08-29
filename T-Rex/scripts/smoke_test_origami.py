"""Check a prepared origami-flat dataset against the trainer's batch contract.

Runs on a CPU box with nothing but torch + pyarrow + PIL installed: the Qwen
processor is stubbed and `origami_dataset.py` is loaded by path, so neither
transformers nor torchvision is needed.  The point is to catch a shape or
key mismatch here rather than 20 minutes into a paid A100 session.

    python scripts/smoke_test_origami.py --root <split root>

Checks the batch dict `train.py` consumes (all 24 keys, exact shapes for the
configured chunk/dim), the flow-matching identity the loss is built on per
anchoring group, that the declared anchoring rule actually reconstructs the wire
contract, that anchor augmentation moves only the dims it is allowed to, that
normalised values land in [-1, 1], that the F6 history reaches the model
un-normalised, that state-noise augmentation perturbs rather than scrambles,
and that BlockShuffleSampler emits a true permutation.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import types

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
# `origami_dataset` imports `trex_origami.anchoring`, which is a real package
# import even though the module itself is loaded by path below.
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from trex_origami.anchoring import (ANCHOR_ABSOLUTE, ANCHOR_PREV_COMMAND,
                                    build_anchor, describe as describe_anchor,
                                    masks as anchor_masks, to_absolute)

# The trainer consumes exactly these keys (train.py's loop + run_validation);
# the last three are extras the offline evaluator reads and train.py ignores.
EXPECTED_KEYS = {
    "input_ids", "attention_mask", "pixel_values", "image_grid_thw", "n_slow_images",
    "noisy_actions", "target", "timesteps", "norm_actions", "tactile_f6s",
    "tactile_deforms", "tactile_f6s_delayed", "tactile_deforms_delayed",
    "tactile_codes", "tactile_f6_history", "time_r", "eps_r", "state_raw",
    "flare_pixel_values", "flare_grid_thw",
    "eval_state", "eval_action_raw", "eval_prev_command", "eval_contact",
}


def _load_origami_dataset():
    """Import qwen_vla/origami_dataset.py without triggering the package __init__.

    `qwen_vla/__init__.py` imports the full MoT model, which drags in
    transformers and torchvision — neither is needed to exercise the loader.
    """
    path = os.path.join(_PROJECT_DIR, "qwen_vla", "origami_dataset.py")
    spec = importlib.util.spec_from_file_location("origami_dataset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _StubProcessor:
    """The three things `collate_fn` asks of an AutoProcessor."""

    class tokenizer:
        pad_token_id = 0

    class image_processor:
        pass

    @staticmethod
    def apply_chat_template(messages, tokenize=False, add_generation_prompt=True):
        content = messages[0]["content"]
        n_images = sum(1 for c in content if c["type"] == "image")
        text = next((c.get("text", "") for c in content if c["type"] == "text"), "")
        return "<im>" * n_images + text

    @staticmethod
    def __call__(text=None, images=None, return_tensors=None, padding=False):
        n_images = len(images)
        # 224x224 -> 16x16 merged token grid -> 64 tokens per image.
        n_tokens = 12 + 64 * n_images
        return types.SimpleNamespace(
            input_ids=torch.arange(n_tokens).unsqueeze(0),
            pixel_values=torch.zeros(n_images * 256, 1176),
            image_grid_thw=torch.tensor([[1, 16, 16]] * n_images))


class _Printer:
    @staticmethod
    def print(*args, **kwargs):
        print(*args, **kwargs)


def build_config(args) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        action_dim=args.action_dim, action_chunk=args.action_chunk,
        image_size=[args.image_size, args.image_size],
        use_robot_state=1, use_tactile_vec=1, use_tactile_deform=1,
        use_tactile_vqvae=1, vqvae_window=args.vqvae_window,
        state_noise_mode="joint",
        anchor_noise_mode=args.anchor_noise_mode, anchor_dropout=args.anchor_dropout,
        use_flare=0, flare_loss_weight=0.0, n_flare_steps=0, flare_frame_stride=4,
        phase_mode="", origami_sampler="block",
        origami_pool_groups=4, origami_cache_groups=4,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True)
    parser.add_argument("--action_dim", type=int, default=65)
    parser.add_argument("--action_chunk", type=int, default=25)
    parser.add_argument("--vqvae_window", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--anchor_noise_mode", default="tracking",
                        choices=["none", "tracking"])
    parser.add_argument("--anchor_dropout", type=float, default=0.15)
    args = parser.parse_args(argv)

    module = _load_origami_dataset()
    config = build_config(args)
    dataset = module.OrigamiDataset(config, _StubProcessor(), _Printer(), root=args.root)
    print(f"len={len(dataset)}  episodes={len(dataset.episodes)}  "
          f"row_groups={len(dataset.row_groups)}")

    n = len(dataset)
    items = [dataset[i] for i in (0, n // 3, n - 1, min(7, n - 1))]
    batch = dataset.collate_fn(items)
    b = len(items)
    chunk, dim, window = args.action_chunk, args.action_dim, args.vqvae_window

    missing, extra = EXPECTED_KEYS - set(batch), set(batch) - EXPECTED_KEYS
    assert not missing and not extra, f"key mismatch: missing {missing}, extra {extra}"

    shapes = {
        "noisy_actions": (b, chunk, dim), "target": (b, chunk, dim),
        "norm_actions": (b, chunk, dim), "eps_r": (b, chunk, dim),
        "timesteps": (b,), "time_r": (b,),
        "tactile_f6s": (b, 10, 6), "tactile_deforms": (b, 10, 1, 240, 240),
        "tactile_f6_history": (b, window, 10, 6), "state_raw": (b, dim),
        "eval_state": (b, dim), "eval_action_raw": (b, chunk, dim),
        "eval_prev_command": (b, dim), "eval_contact": (b,),
    }
    for key, want in shapes.items():
        got = tuple(batch[key].shape)
        assert got == want, f"{key}: got {got}, want {want}"
        assert torch.isfinite(batch[key].float()).all(), f"{key} has non-finite values"
    print("batch keys + shapes OK")

    # The loss regresses `target` against the velocity of x_t = t*eps + (1-t)*A
    # with target = eps - A, which rearranges to  x_t = A + t*target.  Check it
    # in that direction: solving for eps instead divides by t, and t is sampled
    # as low as 0.001, so bf16 rounding would be amplified a thousandfold.
    t = batch["timesteps"].float()[:, None, None]
    reconstructed = batch["norm_actions"].float() + t * batch["target"].float()
    residual = (reconstructed - batch["noisy_actions"].float()).abs().max().item()
    assert residual < 5e-2, (
        f"flow-matching target inconsistent with noisy_actions (max |x_t - (A + t*u)| "
        f"= {residual:.4f}); x_t, target and timesteps disagree")
    print(f"flow-matching target consistent (max residual {residual:.2e})")
    per_dim = (reconstructed - batch["noisy_actions"].float()).abs().amax(dim=(0, 1)).numpy()
    for kind in (ANCHOR_PREV_COMMAND, ANCHOR_ABSOLUTE, "state"):
        dims = np.where(anchor_masks(dataset.anchor_spec)[kind])[0]
        if dims.size:
            print(f"    {kind:<13} {dims.size:2d} dims, max residual "
                  f"{per_dim[dims].max():.2e}")

    # ── anchoring: the target the evaluator scores must reconstruct the wire ──
    spec = dataset.anchor_spec
    sel = anchor_masks(spec)
    print(f"anchoring {describe_anchor(spec)}")
    anchor = build_anchor(batch["eval_state"].numpy().astype(np.float64),
                          batch["eval_prev_command"].numpy().astype(np.float64), spec)
    absolute = to_absolute(batch["eval_action_raw"].numpy().astype(np.float64), anchor)
    want_abs = np.stack([item["action_abs"] for item in items]).astype(np.float64)
    err = np.abs(absolute[:, 0] - want_abs).max()
    assert err < 1e-5, (
        f"anchor + eval_action_raw[0] != action_abs (max |diff| {err:.2e} rad): the "
        f"dataset's declared anchoring does not reconstruct the wire contract")
    print(f"anchoring round-trip to absolute radians OK (max |diff| {err:.2e} rad)")
    if sel[ANCHOR_ABSOLUTE].any():
        raw_step0 = batch["eval_action_raw"].numpy()[:, 0][:, sel[ANCHOR_ABSOLUTE]]
        assert np.allclose(raw_step0, want_abs[:, sel[ANCHOR_ABSOLUTE]], atol=1e-5), \
            "absolute-anchored dims must already be joint angles, with no state added"
        print(f"{int(sel[ANCHOR_ABSOLUTE].sum())} absolute dims are joint angles as stored")

    # Anchor augmentation is allowed to move the previous-command dims and
    # nothing else -- if it leaks into the absolute dims it is corrupting labels.
    moved = np.abs(np.stack([item["action"] for item in items])
                   - batch["eval_action_raw"].numpy()).max(axis=(0, 1))
    leaked = np.where(moved[~sel[ANCHOR_PREV_COMMAND]] > 1e-6)[0]
    assert leaked.size == 0, (
        f"anchor augmentation moved {leaked.size} dim(s) that are not anchored to "
        f"the previous command")
    if sel[ANCHOR_PREV_COMMAND].any() and args.anchor_noise_mode != "none":
        print(f"anchor augmentation confined to the {int(sel[ANCHOR_PREV_COMMAND].sum())} "
              f"prev-command dims (max shift {moved.max():.4f} rad)")

    normed = batch["norm_actions"].float().numpy()
    mask = dataset.action_mask
    lo, hi = normed[..., mask].min(), normed[..., mask].max()
    print(f"norm_actions range on masked dims: [{lo:.3f}, {hi:.3f}]")
    assert lo >= -1.01 and hi <= 1.01

    deform = batch["tactile_deforms"].float().numpy()
    print(f"deform range [{deform.min():.3f}, {deform.max():.3f}]  "
          f"contact flags {batch['eval_contact'].tolist()}")
    assert 0.0 <= deform.min() and deform.max() <= 1.0

    # The embedded VQ-VAE applies its own min-max stats inside the model, so a
    # collate that normalised this window would double-scale every tactile code.
    raw = np.stack([item["tacf6_hist"] for item in items])
    assert np.allclose(batch["tactile_f6_history"].float().numpy(), raw), \
        "tactile_f6_history must reach the model un-normalised"
    print("tactile_f6_history passed through raw")

    config.state_noise_mode = "none"
    clean_ds = module.OrigamiDataset(config, _StubProcessor(), _Printer(), root=args.root,
                                     _stats=dataset.stats_data, _quiet=True)
    clean = clean_ds.collate_fn(items)["state_raw"].float().numpy()
    noisy = batch["state_raw"].float().numpy()
    delta = np.abs(noisy - clean).mean()
    print(f"state noise: mean|delta| = {delta:.4f} (normalised units)")
    assert delta < 0.3, "state-noise augmentation is far larger than real servo lag"

    order = list(iter(dataset.make_sampler(seed=0)))
    assert sorted(order) == list(range(len(dataset))), \
        "BlockShuffleSampler is not a permutation of the dataset"
    print(f"BlockShuffleSampler covers all {len(order)} samples exactly once")

    print("\nALL DATALOADER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
