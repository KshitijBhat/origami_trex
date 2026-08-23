#!/usr/bin/env bash
# One-shot environment + asset setup for a Colab A100 session.
#
#   !bash /content/T-Rex/scripts/colab_setup.sh
#
# Colab sessions are pre-empted, so this is written to be re-runnable: every
# step is skipped when its output already exists.
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/content/T-Rex}"
ASSET_ROOT="${ASSET_ROOT:-/content/assets}"
DATA_ROOT="${DATA_ROOT:-/content/data}"

# Your prepared dataset, pushed by trex_origami/run_prepare.sh.
DATA_REPO="${DATA_REPO:-}"                 # e.g. your-user/origami_flat_pilot
MIDTRAIN_REPO="miniFranka/T-Rex_midtrain_mecka23k_ucb100_vqvae_epoch6"
BASE_REPO="Qwen/Qwen3-VL-2B-Instruct"

mkdir -p "${ASSET_ROOT}" "${DATA_ROOT}"

echo ">>> [1/4] python deps (uv)"
# uv resolves and installs several times faster than pip, which matters on a
# metered A100 session.  Installing into Colab's existing interpreter rather
# than a fresh venv keeps the preinstalled CUDA stack and notebook wiring.
command -v uv >/dev/null 2>&1 || pip install -q uv
UV_PY="$(command -v python)"
uv pip install --python "${UV_PY}" -q \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "${UV_PY}" -q -e "${PROJECT_ROOT}"
# 8-bit Adam halves optimizer state vs torch AdamW and behaves better than
# AdamW's bf16 moments; the trainer selects it with --optim adamw8bit.
uv pip install --python "${UV_PY}" -q bitsandbytes

echo ">>> [2/4] base model: ${BASE_REPO}"
python - <<PY
import os
from huggingface_hub import snapshot_download
target = os.path.join("${ASSET_ROOT}", "Qwen3-VL-2B-Instruct")
if not os.path.exists(os.path.join(target, "config.json")):
    snapshot_download("${BASE_REPO}", local_dir=target)
print("base model at", target)
PY

echo ">>> [3/4] midtrain checkpoint: ${MIDTRAIN_REPO}"
python - <<PY
import os, glob
from huggingface_hub import snapshot_download
target = os.path.join("${ASSET_ROOT}", "trex_midtrain")
if not glob.glob(os.path.join(target, "**", "model.pt"), recursive=True):
    snapshot_download("${MIDTRAIN_REPO}", local_dir=target)
found = glob.glob(os.path.join(target, "**", "model.pt"), recursive=True)
if not found:
    raise SystemExit("no model.pt in the midtrain download")
ckpt = os.path.dirname(found[0])
print("RESUME_CHECKPOINT =", ckpt)
# The embedded VQ-VAE must be there, or --resume_source midtrain silently
# gives us an untrained tactile expert (the exact stage the paper shows
# matters most: 65% -> 45% success without it).
import json
args_path = os.path.join(ckpt, "training_args.json")
if os.path.exists(args_path):
    ta = json.load(open(args_path))
    print("  use_tactile_vqvae:", ta.get("use_tactile_vqvae"),
          "| vqvae_config:", "present" if ta.get("vqvae_config") else "MISSING",
          "| action_dim:", ta.get("action_dim"), "| action_chunk:", ta.get("action_chunk"))
else:
    print("  WARNING: no training_args.json next to model.pt")
PY

echo ">>> [4/4] dataset"
if [ -n "${DATA_REPO}" ]; then
python - <<PY
import os
from huggingface_hub import snapshot_download
target = os.path.join("${DATA_ROOT}", "origami_flat")
if not os.path.exists(os.path.join(target, "train", "meta", "dataset.json")):
    snapshot_download("${DATA_REPO}", repo_type="dataset", local_dir=target, max_workers=8)
print("dataset at", target)
PY
else
    echo "    DATA_REPO unset — upload with:"
    echo "      huggingface-cli upload <user>/origami_flat_pilot <local dir> . --repo-type dataset --private"
fi

echo
echo ">>> ready. Next:"
echo "    bash ${PROJECT_ROOT}/scripts/train_origami.sh"
