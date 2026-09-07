# 这个脚本用于检查 MiniMind-V 的数据集加载接口、tokenizer 和视觉资源状态。
# 先完成资源预检，避免把模型文件缺失误判为 Parquet 数据格式错误。
import torch
import argparse
import inspect
import sys
from pathlib import Path
import json
import pyarrow.parquet as pq

#导入路线
parser = argparse.ArgumentParser()
parser.add_argument("parquet_path", type=Path)
parser.add_argument(
    "--minimind-root",
    type=Path,
    default=Path.home() / "projects" / "minimind-v",
)
parser.add_argument(
    "--weight",
    default="llm",
)

parser.add_argument(
    "--hidden-size",
    type=int,
    default=768,
)
#默认找llm_768
parser.add_argument(
    "--num-hidden-layers",
    type=int,
    default=8,
)

parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)

parser.add_argument(
    "--forward",
    action="store_true",
)

args = parser.parse_args()

dataset_path = args.parquet_path.expanduser().resolve()
minimind_root = args.minimind_root.expanduser().resolve()

model_root = minimind_root / "model"
tokenizer_root = model_root

vision_root = model_root / "siglip2-base-p32-256-ve"
weight_root = minimind_root / "out"
weight_path = weight_root / (
    f"{args.weight}_{args.hidden_size}.pth"
)
sys.path.insert(0, str(minimind_root))

from dataset.lm_dataset import VLMDataset
from trainer.trainer_utils import vlm_collate_fn
from transformers import AutoTokenizer
from transformers import SiglipImageProcessor
from model.model_vlm import MiniMindVLM, VLMConfig

#如果数据或模型路径不存在，报错
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


######这里确认关键列是否正确######


# row = table.slice(0, 1).to_pylist()[0]

# conversations = row["conversations"]
# if isinstance(conversations, str):
#     conversations = json.loads(conversations)

# print("conversation type:", type(conversations).__name__)
# for turn in conversations:
#     print(
#         "role:",
#         turn["role"],
#         "content:",
#         turn["content"],
#     )

# image_bytes = row["image_bytes"]


required_vision_files = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",#这是参数本体
]

# for filename in required_vision_files:
#     path = vision_root / filename
#     print(f"vision file {filename}:", path.is_file())

#检查权重是否正常
# if weight_path.is_file():
#     size_mb = weight_path.stat().st_size / 1024 / 1024
#     print(f"[OK] base weight exists: {size_mb:.1f} MB")
# else:
#     print("[WARN] base weight is missing")
#     print(
#         "next model-forward step requires:",
#         weight_path,
#     )



#################加载tokenizer 和processor ##############

tokenizer = AutoTokenizer.from_pretrained(
    str(tokenizer_root),
    local_files_only=True,
)

processor = SiglipImageProcessor.from_pretrained(
    str(vision_root),
    local_files_only=True,
)

dataset = VLMDataset(
    str(dataset_path),
    tokenizer,
    preprocess=processor,
)


input_ids, labels, image_data = dataset[0]
# supervised_mask = labels != -100

# print("supervised token count:", int(supervised_mask.sum()))
# print("ignored token count:", int((~supervised_mask).sum()))

# answer_token_ids = input_ids[supervised_mask].tolist()

# print(
#     "supervised text:",
#     repr(tokenizer.decode(answer_token_ids)),
# )

# image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
# image_pad_count = int((input_ids == image_pad_id).sum())

# print("image pad token id:", image_pad_id)
# print("image pad token count:", image_pad_count)




# batch_input_ids, batch_labels, batch_images = vlm_collate_fn(
#     [dataset[0], dataset[1]],
# )#调用批处理函数，把两个样本拼成batch，在这里既dataset[0], dataset[1]

# print("batch input_ids shape:", tuple(batch_input_ids.shape))
# print("batch labels shape:", tuple(batch_labels.shape))

# if hasattr(batch_images, "items"):
#     for name, tensor in batch_images.items():
#         print(f"batch_images[{name}] shape:", tuple(tensor.shape))#如果他是张量，则直接输出shape
# else:
#     print("batch images shape:", tuple(batch_images.shape))


################forward#######################
if not args.forward:
    raise SystemExit(
        "dataset inspection finished; "
        "add --forward to run one model forward pass"
    )
device = torch.device(args.device)

if device.type == "cuda" and not torch.cuda.is_available():
    raise SystemExit(f"CUDA is unavailable: {device}")

config = VLMConfig(
    hidden_size=args.hidden_size,
    num_hidden_layers=args.num_hidden_layers,
)

model = MiniMindVLM(
    config,
    vision_model_path=str(vision_root),
)


state_dict = torch.load(
    weight_path,
    map_location="cpu",
)#导入权重

missing_keys, unexpected_keys = model.load_state_dict(
    state_dict,
    strict=False,
)

print("missing key count:", len(missing_keys))
print("unexpected key count:", len(unexpected_keys))

model = model.to(device)
model.eval()

one_input_ids = input_ids.unsqueeze(0).to(device)
one_labels = labels.unsqueeze(0).to(device)

one_images = {
    name: tensor.unsqueeze(0).to(device)
    for name, tensor in image_data.items()
}
#构造一条前向传播数据
with torch.inference_mode():
    output = model(
        one_input_ids,
        labels=one_labels,
        pixel_values=one_images,
    )

print("forward logits shape:", tuple(output.logits.shape))
print("forward loss:", float(output.loss))
print("forward aux loss:", float(output.aux_loss))