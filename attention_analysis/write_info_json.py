#!/usr/bin/env python3
"""Write a small info.json into a decoder-sweep output dir, parsed from its
run log. No model/GPU needed -- purely reads the log this clip's
decoder_episode_sweep.py run already produced."""
import json
import os
import re
import sys

name = sys.argv[1]
log_path = f"/workspace/dec_v2_{name}.log"
out_dir = f"/workspace/decoder_sweep/{name}"

with open(log_path) as f:
    log = f.read()

ckpt = re.search(r"loading checkpoint (\S+)", log).group(1)
instr = re.search(r"instruction='(.*?)'\n", log).group(1)
video, fps = re.search(r"extracting frames from (\S+) at ([\d.]+)fps", log).groups()
n_frames = int(re.search(r"(\d+) frames to process", log).group(1))
n_layers = int(re.search(r"n_layers=(\d+)", log).group(1))
matched, total = re.search(r"matched (\d+)/(\d+) words", log).groups()
words = re.findall(r"^\s+'(\w+_[^']+)': (\d+) tokens", log, re.MULTILINE)

info = {
    "clip": name,
    "checkpoint": ckpt,
    "video": video,
    "instruction": instr,
    "fps": float(fps),
    "n_frames": n_frames,
    "n_layers": n_layers,
    "words_matched": f"{matched}/{total}",
    "content_words": {w: int(n) for w, n in words},
}

os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "info.json"), "w") as f:
    json.dump(info, f, indent=2)
print(f"wrote {out_dir}/info.json")
