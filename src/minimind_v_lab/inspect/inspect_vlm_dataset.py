# 这个脚本用于检查 MiniMind-V 的数据集加载接口、tokenizer 和视觉资源状态。
# 先完成资源预检，避免把模型文件缺失误判为 Parquet 数据格式错误。

import argparse
import inspect
import sys
from pathlib import Path
import json
import pyarrow.parquet as pq

parser = argparse.ArgumentParser()
parser.add_argument("parquet_path", type=Path)
parser.add_argument(
    "--minimind-root",
    type=Path,
    default=Path.home() / "projects" / "minimind-v",
)

parser.add_argument(
    "--raw-only",
    action="store_true",
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

table = pq.read_table(
    dataset_path,
    columns=["conversations", "image_bytes"],
)

print("parquet rows:", table.num_rows)
print("parquet columns:", table.column_names)
######这里确认关键列是否######


row = table.slice(0, 1).to_pylist()[0]

conversations = row["conversations"]
if isinstance(conversations, str):
    conversations = json.loads(conversations)

print("conversation type:", type(conversations).__name__)
for turn in conversations:
    print(
        "role:",
        turn["role"],
        "content:",
        turn["content"],
    )

image_bytes = row["image_bytes"]

if isinstance(image_bytes, list):
    print("number of images:", len(image_bytes))
    print("first image bytes:", len(image_bytes[0]))
else:
    print("image bytes:", len(image_bytes))

if args.raw_only:
    raise SystemExit("raw parquet inspection finished")



# if vision_root.is_dir():
#     print("[OK] vision encoder directory exists")
# else:
#     print("[WARN] vision encoder directory is missing")

# sys.path.insert(0, str(minimind_root))

# from dataset.lm_dataset import VLMDataset

# print("VLMDataset module:", inspect.getsourcefile(VLMDataset))
# print("VLMDataset constructor:")
# print(inspect.signature(VLMDataset.__init__))

# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained(
#     str(tokenizer_root),
#     local_files_only=True,
# )

# print("tokenizer vocab size:", len(tokenizer))
# print("tokenizer pad token:", tokenizer.pad_token)
# print("tokenizer eos token:", tokenizer.eos_token)

# from transformers import SiglipImageProcessor

# processor = SiglipImageProcessor.from_pretrained(
#     str(vision_root),
#     local_files_only=True,
# )

# print("processor:", type(processor).__name__)

# dataset = VLMDataset(
#     str(dataset_path),
#     tokenizer,
#     preprocess=processor,
# )

# print("dataset size:", len(dataset))

# input_ids, labels, image_data = dataset[0]

# print("input_ids shape:", tuple(input_ids.shape))
# print("labels shape:", tuple(labels.shape))
# print("image data type:", type(image_data).__name__)

# if hasattr(image_data, "items"):
#     for name, tensor in image_data.items():
#         print(f"image_data[{name}] shape:", tuple(tensor.shape))
# else:
#     print("image_data shape:", tuple(image_data.shape))