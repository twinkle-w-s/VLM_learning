#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE_ROOT="${VLM_STORAGE_ROOT:-/data/${USER}/vlm_learning}"

mkdir -p "${STORAGE_ROOT}"
mkdir -p "${PROJECT_ROOT}/logs"

export VLM_STORAGE_ROOT="${STORAGE_ROOT}"
export HF_HOME="${STORAGE_ROOT}/cache/huggingface"
export HF_DATASETS_CACHE="${STORAGE_ROOT}/datasets/flickr30k"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TORCH_HOME="${STORAGE_ROOT}/cache/torch"

python -u "${PROJECT_ROOT}/scripts/download_assets.py" \
  --storage-root "${STORAGE_ROOT}"
