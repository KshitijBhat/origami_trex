#!/usr/bin/env python3
"""Warm-start final_layer's hand rows from an original T-Rex checkpoint.

Implements PLAN_fresh_branch.md §6: fc1 (hidden->hidden) is dimension-count
independent of action_dim, so it's copied wholesale. fc2 (hidden->action_dim)
is copied only on the rows both checkpoints agree mean the same physical
joint -- hands, same Sharpa Wave hardware, same 22-dim order confirmed this
session. Arm and motor rows are left as whatever the target checkpoint
already has (fresh init), since original T-Rex has no matching representation
for either (9D EEF pose vs our 7D raw joints; no motor group at all).

Usage:
    python3 init_final_layer.py \
        --source /path/to/original_trex/model.pt \
        --target /path/to/origami_resume_checkpoint/model.pt \
        --output /path/to/warm_started/model.pt
"""
import argparse
import copy

import torch

# source (original T-Rex, 62D: [L_arm9d, L_hand22, R_arm9d, R_hand22]) hand rows
SOURCE_LEFT_HAND = slice(9, 31)
SOURCE_RIGHT_HAND = slice(40, 62)

# target (origami, 65D: [L_arm7, L_hand22, R_arm7, R_hand22, motor7]) hand rows
TARGET_LEFT_HAND = slice(7, 29)
TARGET_RIGHT_HAND = slice(36, 58)


def warm_start_hands(source_sd: dict, target_sd: dict) -> dict:
    """Return a copy of target_sd with final_layer's hand rows + all of fc1
    overwritten from source_sd. Everything else in target_sd is untouched."""
    out = copy.deepcopy(target_sd)

    for name in ("final_layer.mlp.fc1.weight", "final_layer.mlp.fc1.bias"):
        if name in source_sd and name in out:
            assert source_sd[name].shape == out[name].shape, (
                f"{name}: fc1 shape mismatch {source_sd[name].shape} vs "
                f"{out[name].shape} -- fc1 should be action_dim-independent")
            out[name] = source_sd[name].clone()

    w_name, b_name = "final_layer.mlp.fc2.weight", "final_layer.mlp.fc2.bias"
    if w_name in source_sd and w_name in out:
        for src, dst in ((SOURCE_LEFT_HAND, TARGET_LEFT_HAND), (SOURCE_RIGHT_HAND, TARGET_RIGHT_HAND)):
            out[w_name][dst] = source_sd[w_name][src].clone()
            if b_name in source_sd and b_name in out:
                out[b_name][dst] = source_sd[b_name][src].clone()

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="original T-Rex checkpoint's model.pt")
    ap.add_argument("--target", required=True, help="origami resume checkpoint's model.pt")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    source_sd = torch.load(args.source, map_location="cpu")
    target_sd = torch.load(args.target, map_location="cpu")
    out = warm_start_hands(source_sd, target_sd)
    torch.save(out, args.output)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
