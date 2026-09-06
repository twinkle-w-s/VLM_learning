# 本脚本把 CLEVR manifest 和图像转换为 MiniMind-V Parquet。
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

DATA_ROOT = (
    Path("/data")
    / os.environ["USER"]
    / "vlm_learning"
    / "CLEVR_v1.0"
)#原数据的路径

IMAGE_ROOT = DATA_ROOT / "images" / "train"
#读jsonl的脚本
def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)
#读图像的脚本
def read_image_bytes(image_path: Path) -> bytes:
    with image_path.open("rb") as file:
        return file.read()

def build_conversations(
    question: str,
    answer: str,
) -> str:
    conversations = [
        {
            "role": "user",
            "content": f"{question}<image>",
        },
        {
            "role": "assistant",
            "content": answer,
        },
    ]

    return json.dumps(
        conversations,
        ensure_ascii=False,
    )#构造conversations

def build_row(record: dict) -> dict:
    image_path = IMAGE_ROOT / record["image_filename"]

    return {
        "question_index": record["question_index"],
        "image_index": record["image_index"],
        "image_filename": record["image_filename"],
        "task_type": record["task_type"],
        "answer_type": record["answer_type"],
        "difficulty": record["difficulty"],
        "conversations": build_conversations(
            question=record["question"],
            answer=record["answer"],
        ),
        "image_bytes": read_image_bytes(image_path),
    }#根据image_filename取出
