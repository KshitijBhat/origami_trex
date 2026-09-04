#!/usr/bin/env python3
"""Diff two model.pt checkpoints tensor-by-tensor to empirically confirm which
decoder layers/experts were actually frozen during a training run, instead of
trusting the training script's argument defaults. Bit-identical = frozen,
any difference = received gradient updates.

Usage: python3 diff_checkpoint_weights.py <checkpoint_a>/model.pt <checkpoint_b>/model.pt
"""
import re
import sys

import torch

path_a, path_b = sys.argv[1], sys.argv[2]
sd_a = torch.load(path_a, map_location="cpu")
sd_b = torch.load(path_b, map_location="cpu")
sd_a = sd_a.get("model_state", sd_a) if isinstance(sd_a, dict) and "model_state" in sd_a else sd_a
sd_b = sd_b.get("model_state", sd_b) if isinstance(sd_b, dict) and "model_state" in sd_b else sd_b


def suffix(name):
    if "_action" in name:
        return "action"
    if "_tactile" in name:
        return "tactile"
    return "latent"


print(f"comparing {path_a}\n      vs   {path_b}\n")

per_layer = {}
for name in sd_a:
    if name not in sd_b:
        continue
    m = re.match(r"model\.layers\.(\d+)\.", name)
    if not m:
        continue
    li = int(m.group(1))
    same = torch.equal(sd_a[name].float(), sd_b[name].float())
    per_layer.setdefault(li, []).append((name, suffix(name), same))

layers = sorted(per_layer)
n_layers = len(layers)
print(f"{'layer':>6s} {'latent':>16s} {'action':>16s} {'tactile':>16s}")
for li in layers:
    counts = {"latent": [0, 0], "action": [0, 0], "tactile": [0, 0]}
    for _, s, same in per_layer[li]:
        counts[s][0 if same else 1] += 1
    row = " ".join(f"{counts[k][0]:2d} same/{counts[k][1]:2d} diff".rjust(16)
                    for k in ["latent", "action", "tactile"])
    print(f"{li:>6d} {row}")

print("\n--- aggregate: frozen-looking layer range vs the rest (edit lo/hi to match your run) ---")
lo, hi = 0, n_layers - 5  # guess: everything but the last 4 layers
for label, rng in [(f"layers {lo}-{hi}", range(lo, hi + 1)),
                    (f"layers {hi+1}-{n_layers-1}", range(hi + 1, n_layers))]:
    counts = {"latent": [0, 0], "action": [0, 0], "tactile": [0, 0]}
    for li in rng:
        for _, s, same in per_layer.get(li, []):
            counts[s][0 if same else 1] += 1
    print(label, counts)
