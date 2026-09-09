#!/usr/bin/env bash
set -euo pipefail

# 这个脚本使用初始全量配方启动 MiniMind-V 的第一轮 CLEVR SFT。

MINIMIND_ROOT="${HOME}/projects/minimind-v"
DATA_ROOT="/data/${USER}/vlm_learning"
DATA_PATH="${DATA_ROOT}/datasets/clevr/curated/spatial_initial_recipe_train.parquet"
SAVE_DIR="${DATA_ROOT}/runs/clevr_initial_sft"

mkdir -p "${SAVE_DIR}"
cd "${MINIMIND_ROOT}/trainer"

python train_sft_vlm.py \
  --data_path "${DATA_PATH}" \
  --from_weight llm \
  --save_weight clevr_initial_sft \
  --save_dir "${SAVE_DIR}" \
  --epochs 1 \
  --batch_size 4 \
  --learning_rate 5e-6 \
  --freeze_llm 1 \
  --max_seq_len 512 \
  --num_workers 2 \
  --save_interval 5000 \
  --device cuda:0
