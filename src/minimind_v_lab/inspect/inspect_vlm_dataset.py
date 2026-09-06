# 这个脚本用于检查 MiniMind-V 的数据集加载接口、tokenizer 和视觉资源状态。
# 先完成资源预检，避免把模型文件缺失误判为 Parquet 数据格式错误。

import argparse
import inspect
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("parquet_path", type=Path)
parser.add_argument(
    "--minimind-root",
    type=Path,
    default=Path.home() / "projects" / "minimind-v",
)

args = parser.parse_args()

dataset_path = args.parquet_path.expanduser().resolve()
minimind_root = args.minimind_root.expanduser().resolve()

model_root = minimind_root / "model"
tokenizer_root = model_root
vision_root = model_root / "siglip2-base-p32-256-ve"

print("dataset:", dataset_path)
print("minimind root:", minimind_root)
print("tokenizer root:", tokenizer_root)
print("vision root:", vision_root)

if not dataset_path.is_file():
    raise SystemExit(f"dataset not found: {dataset_path}")

if not minimind_root.is_dir():
    raise SystemExit(f"minimind root not found: {minimind_root}")

if vision_root.is_dir():
    print("[OK] vision encoder directory exists")
else:
    print("[WARN] vision encoder directory is missing")

    sys.path.insert(0, str(minimind_root))

from dataset.lm_dataset import VLMDataset

print("VLMDataset module:", inspect.getsourcefile(VLMDataset))
print("VLMDataset constructor:")
print(inspect.signature(VLMDataset.__init__))

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    str(tokenizer_root),
    local_files_only=True,
)

print("tokenizer vocab size:", len(tokenizer))
print("tokenizer pad token:", tokenizer.pad_token)
print("tokenizer eos token:", tokenizer.eos_token)

if not vision_root.is_dir():
    print("preflight finished: vision encoder is required before dataset instantiation")