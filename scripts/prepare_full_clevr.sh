#!/usr/bin/env bash
set -euo pipefail

# 这个脚本在服务器上生成全量 CLEVR 空间数据、配方和 Parquet。

PROJECT_ROOT="${HOME}/projects/VLM_learning"
DATA_ROOT="/data/${USER}/vlm_learning"
OUTPUT_ROOT="${DATA_ROOT}/datasets/clevr"

cd "${PROJECT_ROOT}"

python src/minimind_v_lab/pipeline/prepare_full_clevr.py \
  "${OUTPUT_ROOT}" \
  --val-ratio 0.1 \
  --seed 42 \
  --limit 0
