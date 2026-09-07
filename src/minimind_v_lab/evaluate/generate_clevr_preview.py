# 这个脚本加载微调后的 MiniMind-V，在 CLEVR 验证集生成答案预览。

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer

def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


parser = argparse.ArgumentParser()

parser.add_argument("manifest_path", type=Path)
parser.add_argument("checkpoint_path", type=Path)

parser.add_argument(
    "--minimind-root",
    type=Path,
    default=Path.home() / "projects" / "minimind-v",
)

parser.add_argument(
    "--image-root",
    type=Path,
    default=(
        Path("/data")
        / os.environ["USER"]
        / "vlm_learning"
        / "CLEVR_v1.0"
        / "images"
        / "train"
    ),
)

parser.add_argument("--limit", type=int, default=5)
parser.add_argument("--max-new-tokens", type=int, default=12)
parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
)

args = parser.parse_args()

manifest_path = args.manifest_path.expanduser().resolve()
checkpoint_path = args.checkpoint_path.expanduser().resolve()
minimind_root = args.minimind_root.expanduser().resolve()

if not manifest_path.is_file():
    raise SystemExit(f"manifest not found: {manifest_path}")

if not checkpoint_path.is_file():
    raise SystemExit(f"checkpoint not found: {checkpoint_path}")

sys.path.insert(0, str(minimind_root))

from model.model_vlm import MiniMindVLM, VLMConfig

device = torch.device(args.device)
tokenizer = AutoTokenizer.from_pretrained(
    str(minimind_root / "model"),
    local_files_only=True,
)

model = MiniMindVLM(
    VLMConfig(hidden_size=768, num_hidden_layers=8),
    vision_model_path=str(
        minimind_root / "model" / "siglip2-base-p32-256-ve"
    ),
)

state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict, strict=False)

model = model.to(device).eval()


def generate_answer(record: dict) -> str:
    image = Image.open(
        args.image_root / record["image_filename"]
    ).convert("RGB")

    pixel_values = MiniMindVLM.image2tensor(
        image,
        model.processor,
    )

    pixel_values = {
        name: tensor.to(device)
        for name, tensor in pixel_values.items()
    }

    content = (
        record["question"]
        + model.config.image_special_token
        * model.config.image_token_len
    )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)


    with torch.inference_mode():
        generated_ids = model.generate(
            inputs=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=pixel_values,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_ids = generated_ids[
        0,
        inputs["input_ids"].shape[1]:,
    ]

    return tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
    ).strip()


records = read_jsonl(manifest_path)[:args.limit]

for index, record in enumerate(records, start=1):
    prediction = generate_answer(record)

    print(f"\n[{index}] question: {record['question']}")
    print("target:", record["answer"])
    print("prediction:", repr(prediction))