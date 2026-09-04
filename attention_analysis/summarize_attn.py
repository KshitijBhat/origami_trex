#!/usr/bin/env python3
"""Numeric summary of saved per-layer/per-query vision-attention npy files:
for each (query, layer), mean/min/max total mass to vision tokens across all
frames, plus a concentration metric (max-patch-share of that mass) so peaked
vs diffuse layers are distinguishable, not just total mass."""
import json
import os
import sys

import numpy as np

out_dir = sys.argv[1]

queries = sorted(d for d in os.listdir(out_dir)
                  if os.path.isdir(os.path.join(out_dir, d)))

summary = {}
for q in queries:
    qdir = os.path.join(out_dir, q)
    layer_dirs = sorted(d for d in os.listdir(qdir) if d.startswith("layer"))
    summary[q] = {}
    for ld in layer_dirs:
        ldir = os.path.join(qdir, ld)
        files = sorted(f for f in os.listdir(ldir) if f.endswith(".npy"))
        if not files:
            continue
        sums, peaks = [], []
        for f in files:
            v = np.load(os.path.join(ldir, f))
            s = float(v.sum())
            sums.append(s)
            peaks.append(float(v.max() / s) if s > 1e-8 else 0.0)
        summary[q][ld] = {
            "mass_mean": round(float(np.mean(sums)), 4),
            "mass_min": round(float(np.min(sums)), 4),
            "mass_max": round(float(np.max(sums)), 4),
            "peak_share_mean": round(float(np.mean(peaks)), 4),
        }

with open(os.path.join(out_dir, "attn_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# also print a compact table: query x [layer00, layer07, layer14, layer21, layer27] mass_mean
sel_layers = ["layer00", "layer07", "layer14", "layer21", "layer27"]
print(f"{'query':30s} " + " ".join(f"{l:>10s}" for l in sel_layers))
for q in queries:
    row = []
    for l in sel_layers:
        v = summary[q].get(l, {}).get("mass_mean")
        row.append(f"{v:.4f}" if v is not None else "  n/a")
    print(f"{q:30s} " + " ".join(f"{r:>10s}" for r in row))
