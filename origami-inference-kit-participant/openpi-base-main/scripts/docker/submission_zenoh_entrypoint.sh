#!/usr/bin/env bash
set -euo pipefail

checkpoint_dir="${CHECKPOINT_DIR:-/opt/policy/checkpoint}"
config="${POLICY_CONFIG:-pi05_origami_fold_plane_v2_v0715}"
prompt="${PROMPT:-north ces task}"

: "${ORIGAMI_ZENOH_ENDPOINT:?ORIGAMI_ZENOH_ENDPOINT is required}"
: "${ORIGAMI_SESSION_ID:?ORIGAMI_SESSION_ID is required}"

if [ ! -d "${checkpoint_dir}/params" ] \
  && [ ! -f "${checkpoint_dir}/model.safetensors" ]; then
  echo "[submission][ERROR] ${checkpoint_dir} must contain params/ or model.safetensors" >&2
  exit 2
fi

if [ ! -d "${checkpoint_dir}/assets" ] \
  || ! find "${checkpoint_dir}/assets" -name norm_stats.json -print -quit | grep -q .; then
  echo "[submission][ERROR] ${checkpoint_dir}/assets must contain norm_stats.json" >&2
  exit 2
fi

export HOME="${HOME:-/tmp/origami-home}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/origami-cache}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/origami-jax-cache}"
mkdir -p "$HOME" "$XDG_CACHE_HOME" "$JAX_COMPILATION_CACHE_DIR"

echo "[submission] transport=origami-zenoh-v1 config=${config} checkpoint=${checkpoint_dir}" >&2
exec /.venv/bin/python /app/scripts/serve_policy_zenoh.py \
  --default-prompt="${prompt}" \
  policy:checkpoint \
  --policy.config="${config}" \
  --policy.dir="${checkpoint_dir}"
