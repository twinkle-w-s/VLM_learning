#!/usr/bin/env bash
set -euo pipefail

# CLEVR v1 validation: generate answers and save per-example results.
PROJECT_ROOT="${HOME}/projects/VLM_learning"
DATA_ROOT="/data/${USER}/vlm_learning"
EXP_ROOT="${DATA_ROOT}/experiments/clevr_spatial_v1_initial_recipe"
MANIFEST_PATH="${EXP_ROOT}/data/val_image_holdout_5pct.jsonl"
CHECKPOINT_PATH="${DATA_ROOT}/runs/clevr_spatial_v1/projector_warmup/clevr_spatial_v1_projector_warmup_768.pth"
OUTPUT_PATH="${DATA_ROOT}/reports/clevr_spatial_v1/projector_warmup/val_generation.jsonl"

cd "${PROJECT_ROOT}"

python src/minimind_v_lab/evaluate/generate_clevr.py \
  "${MANIFEST_PATH}" \
  "${CHECKPOINT_PATH}" \
  --device cuda:0 \
  --output-path "${OUTPUT_PATH}"

python src/minimind_v_lab/evaluate/analyze_clevr_eval.py \
  "${MANIFEST_PATH}" \
  "${OUTPUT_PATH}"
