#!/usr/bin/env bash
set -euo pipefail

# CLEVR v1 projector warmup: train only the vision projection layer.
MINIMIND_ROOT="${HOME}/projects/minimind-v"
DATA_ROOT="/data/${USER}/vlm_learning"
EXP_ROOT="${DATA_ROOT}/experiments/clevr_spatial_v1_initial_recipe"
DATA_PATH="${EXP_ROOT}/data/train_initial_recipe.parquet"
SAVE_DIR="${DATA_ROOT}/runs/clevr_spatial_v1/projector_warmup"

mkdir -p "${SAVE_DIR}"
cd "${MINIMIND_ROOT}/trainer"

python train_sft_vlm.py \
  --data_path "${DATA_PATH}" \
  --from_weight llm \
  --save_weight clevr_spatial_v1_projector_warmup \
  --save_dir "${SAVE_DIR}" \
  --epochs 1 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --freeze_llm 2 \
  --max_seq_len 512 \
  --num_workers 2 \
  --save_interval 5000 \
  --device cuda:0
