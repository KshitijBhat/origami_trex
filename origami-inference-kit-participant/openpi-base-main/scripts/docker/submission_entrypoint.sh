#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir="${CHECKPOINT_DIR:-/opt/policy/checkpoint}"
config="${POLICY_CONFIG:-pi05_origami_fold_plane}"
host="${POLICY_BIND_HOST:-0.0.0.0}"
port="${PORT:-8000}"
prompt="${PROMPT:-north ces task}"

if [ ! -d "${checkpoint_dir}/params" ] && [ ! -f "${checkpoint_dir}/model.safetensors" ]; then
  echo "[submission][ERROR] ${checkpoint_dir} must contain params/ or model.safetensors" >&2
  exit 2
fi

if [ ! -d "${checkpoint_dir}/assets" ] || ! find "${checkpoint_dir}/assets" -name norm_stats.json -print -quit | grep -q .; then
  echo "[submission][ERROR] ${checkpoint_dir}/assets must contain <asset_id>/norm_stats.json" >&2
  exit 2
fi

echo "[submission] config=${config} checkpoint=${checkpoint_dir} host=${host} port=${port}" >&2
exec /.venv/bin/python /app/scripts/serve_policy.py \
  --host="${host}" \
  --port="${port}" \
  --default-prompt="${prompt}" \
  policy:checkpoint \
  --policy.config="${config}" \
  --policy.dir="${checkpoint_dir}"
