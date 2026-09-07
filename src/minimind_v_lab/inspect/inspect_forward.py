# 这个脚本用于检查 MiniMind-V 的数据集加载接口、tokenizer 和视觉资源状态。
# 检查使用一条数据，对视觉encoder做一次参数更新的实验
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

parser.add_argument(
    "--train-step",
    action="store_true",
)

parser.add_argument(
    "--freeze-llm",
    type=int,
    choices=[0, 1, 2],
    default=2,#只训练最小vision
)

parser.add_argument(
    "--learning-rate",
    type=float,
    default=1e-4,
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
from trainer.trainer_utils import vlm_collate_fn,init_vlm_model
from transformers import AutoTokenizer
from transformers import SiglipImageProcessor
from model.model_vlm import VLMConfig

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



required_vision_files = [
    "config.json",
    "preprocessor_config.json",
    "model.safetensors",#这是参数本体
]




#################加载tokenizer 和processor ##############

# tokenizer = AutoTokenizer.from_pretrained(
#     str(tokenizer_root),
#     local_files_only=True,
# )

# processor = SiglipImageProcessor.from_pretrained(
#     str(vision_root),
#     local_files_only=True,
# )

# dataset = VLMDataset(
#     str(dataset_path),
#     tokenizer,
#     preprocess=processor,
# )


input_ids, labels, image_data = dataset[0]


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


model, _, _ = init_vlm_model(
    config,
    from_weight=args.weight,
    tokenizer_path=str(tokenizer_root),
    vision_model_path=str(vision_root),
    save_dir=str(weight_root),
    device=str(device),
    freeze_llm=args.freeze_llm,
)

model = model.to(device)
trainable = [
    (name, parameter)
    for name, parameter in model.named_parameters()
    if parameter.requires_grad
]#列表中包含二元组

trainable_count = sum(
    parameter.numel()
    for _, parameter in trainable
)

print("trainable tensors:", len(trainable))
print("trainable parameters (M):", trainable_count / 1e6)
print("first trainable name:", trainable[0][0])
#统计可学习参数


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


#根据输入参数判断是否训练
if not args.train_step:
    raise SystemExit(
        "forward inspection finished; "
        "add --train-step to run one optimizer update"
    )

model.train()

optimizer=torch.optim.AdamW(
    [parameter for _, parameter in trainable],
    lr=args.learning_rate,
)

first_name,first_parameter=trainable[0]
before_update=first_parameter.detach().clone()#复制一份第一个参数

optimizer.zero_grad(set_to_none=True)

train_output = model(
    one_input_ids,
    labels=one_labels,
    pixel_values=one_images,
)

train_output.loss.backward()
optimizer.step()#更新参数

###############检查变化##############
gradient_norm_sq = sum(
    parameter.grad.detach().float().pow(2).sum()
    for _, parameter in trainable
    if parameter.grad is not None
)

mean_update = (
    first_parameter.detach() - before_update
).abs().mean()

print("train loss:", float(train_output.loss))
print("gradient norm:", float(torch.sqrt(gradient_norm_sq)))
print(f"mean update for {first_name}:", float(mean_update))

