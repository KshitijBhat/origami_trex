#!/usr/bin/env python3
"""Rebuild GIFs from already-saved per-frame PNGs, skipping ones that already
exist. Standalone -- no model, no forward passes, just PIL. For recovering
from the optimize=True slowdown without re-running the expensive part."""
import argparse
import os
import PIL.Image

ap = argparse.ArgumentParser()
ap.add_argument("--out_dir", required=True)
ap.add_argument("--camera_name", required=True)
args = ap.parse_args()

n_built = 0
n_skipped = 0
for qi in sorted(os.listdir(args.out_dir)):
    qdir = os.path.join(args.out_dir, qi)
    if not os.path.isdir(qdir):
        continue
    for li_name in sorted(os.listdir(qdir)):
        layer_dir = os.path.join(qdir, li_name)
        if not os.path.isdir(layer_dir):
            continue
        gif_path = os.path.join(args.out_dir, f"{args.camera_name}_{qi}_{li_name}.gif")
        if os.path.exists(gif_path):
            n_skipped += 1
            continue
        pngs = sorted(f for f in os.listdir(layer_dir) if f.endswith(".png"))
        if not pngs:
            continue
        frames = [PIL.Image.open(os.path.join(layer_dir, p)).convert("RGB") for p in pngs]
        frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                       duration=150, loop=0, optimize=False)
        n_built += 1
        if n_built % 20 == 0:
            print(f"[rebuild] {n_built} gifs built so far")

print(f"[rebuild] done: {n_built} built, {n_skipped} already existed")
