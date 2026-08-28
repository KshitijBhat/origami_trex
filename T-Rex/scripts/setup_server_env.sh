#!/usr/bin/env bash
# One-shot environment setup for a persistent cluster server that has NO
# internet access to huggingface.co (unlike scripts/colab_setup.sh, which
# downloads the base model + midtrain checkpoint directly). Instead, this
# unpacks a pre-downloaded bundle -- see notebooks/download_trex_backbones.ipynb
# for the Colab side: it downloads the exact same assets colab_setup.sh would,
# zips them as trex_backbones_ckpts.zip, and saves it to Drive. Get that zip
# onto this server (scp/rsync from wherever you pulled it off Drive) and point
# BACKBONE_ZIP at it.
#
#   BACKBONE_ZIP=/path/to/trex_backbones_ckpts.zip bash scripts/setup_server_env.sh
#
# Re-runnable: every step is skipped when its output already exists.
set -euo pipefail

# --- EDIT THESE for your machine (same convention as run_ori_job.sh's
#     HOME=${HOME:-/home/sr5/sairaj.loke}) ---
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_DIR="${VENV_DIR:-${HOME}/other/venv_trex}"
BACKBONE_DIR="${BACKBONE_DIR:-${HOME}/other/trex_assets}"
BACKBONE_ZIP="${BACKBONE_ZIP:-${HOME}/other/trex_backbones_ckpts.zip}"
DATA_ROOT="${DATA_ROOT:-${HOME}/other/new_data/origami_flat}"

mkdir -p "${BACKBONE_DIR}" "${DATA_ROOT}"

echo ">>> [1/3] venv + python deps"
if [ ! -d "${VENV_DIR}" ]; then
    python3.10 -m venv "${VENV_DIR}"
fi
source "${VENV_DIR}/bin/activate"
command -v uv >/dev/null 2>&1 || pip install -q uv
uv pip install --python "$(command -v python)" -q \
    torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python "$(command -v python)" -q -e "${PROJECT_ROOT}"
uv pip install --python "$(command -v python)" -q bitsandbytes
# only needed if you use --data_format lerobot instead of the origami-flat pipeline:
# uv pip install --python "$(command -v python)" -q -e /path/to/lerobot

echo ">>> [2/3] backbones + checkpoints from local bundle (no download)"
if [ -f "${BACKBONE_DIR}/Qwen3-VL-2B-Instruct/config.json" ] && \
   find "${BACKBONE_DIR}/trex_midtrain" -name model.pt 2>/dev/null | grep -q .; then
    echo "    already unpacked at ${BACKBONE_DIR} -- skipping"
elif [ -f "${BACKBONE_ZIP}" ]; then
    echo "    unzipping ${BACKBONE_ZIP} -> ${BACKBONE_DIR}"
    unzip -q -o "${BACKBONE_ZIP}" -d "${BACKBONE_DIR}"
    # the notebook zips Qwen3-VL-2B-Instruct/ and trex_midtrain/ at the zip
    # root -- if yours nests them one level deeper, flatten here:
    if [ ! -f "${BACKBONE_DIR}/Qwen3-VL-2B-Instruct/config.json" ]; then
        nested="$(find "${BACKBONE_DIR}" -maxdepth 2 -name config.json -path "*Qwen3-VL-2B-Instruct*" | head -1)"
        if [ -n "${nested}" ]; then
            mv "$(dirname "$(dirname "${nested}")")"/* "${BACKBONE_DIR}/" 2>/dev/null || true
        fi
    fi
else
    echo "ERROR: no backbones at ${BACKBONE_DIR} and BACKBONE_ZIP=${BACKBONE_ZIP} not found." >&2
    echo "  Run notebooks/download_trex_backbones.ipynb on Colab, save the zip to Drive," >&2
    echo "  copy it to this server, and re-run with BACKBONE_ZIP=<path>." >&2
    exit 1
fi

python - <<PY
import glob, json, os
target = os.path.join("${BACKBONE_DIR}", "trex_midtrain")
found = glob.glob(os.path.join(target, "**", "model.pt"), recursive=True)
if not found:
    raise SystemExit(f"no model.pt under {target} -- bundle didn't unpack as expected")
ckpt = os.path.dirname(found[0])
print("RESUME_CHECKPOINT =", ckpt)
args_path = os.path.join(ckpt, "training_args.json")
if os.path.exists(args_path):
    ta = json.load(open(args_path))
    print("  use_tactile_vqvae:", ta.get("use_tactile_vqvae"),
          "| vqvae_config:", "present" if ta.get("vqvae_config") else "MISSING -- STOP, this is the 65%->45% ablation",
          "| action_dim:", ta.get("action_dim"), "| action_chunk:", ta.get("action_chunk"))
else:
    print("  WARNING: no training_args.json next to model.pt")
PY

echo ">>> [3/3] dataset"
if [ -d "${DATA_ROOT}/train/meta" ] || [ -d "${DATA_ROOT}/full/train/meta" ] || [ -d "${DATA_ROOT}/pilot/train/meta" ]; then
    echo "    found a prepared dataset under ${DATA_ROOT} -- OK"
else
    echo "    nothing found under ${DATA_ROOT}. This also needs to come from a machine with hub"
    echo "    access (trex_origami/run_prepare.sh streams the raw seasons) -- prep it there, then"
    echo "    rsync/scp the resulting origami_flat/{pilot,dense,full} directory onto this server."
fi

echo
echo ">>> ready. venv: ${VENV_DIR}   backbones: ${BACKBONE_DIR}"
echo ">>> next: bash run_trex_job.sh <NUM_GPUS>   (see cmd_trex.sh for phd run examples)"
