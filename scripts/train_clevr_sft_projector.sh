#!/usr/bin/env bash
set -euo pipefail

# 这个脚本只训练 vision_proj，用于生成后续 SFT 使用的 projector warmup 权重。

MINIMIND_ROOT="${HOME}/projects/minimind-v"
DATA_ROOT="/data/${USER}/vlm_learning"
DATA_PATH="${DATA_ROOT}/datasets/clevr/curated/spatial_initial_recipe_train.parquet"
SAVE_DIR="${DATA_ROOT}/runs/clevr_projector_warmup"

mkdir -p "${SAVE_DIR}"
cd "${MINIMIND_ROOT}/trainer"

python train_sft_vlm.py \
  --data_path "${DATA_PATH}" \
  --from_weight llm \
  --save_weight clevr_projector_warmup \
  --save_dir "${SAVE_DIR}" \
  --epochs 1 \
  --batch_size 16 \
  --learning_rate 1e-4 \
  --freeze_llm 2 \
  --max_seq_len 512 \
  --num_workers 2 \
  --save_interval 5000 \
  --device cuda:0
