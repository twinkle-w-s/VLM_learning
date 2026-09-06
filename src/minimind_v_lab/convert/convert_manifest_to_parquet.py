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
    }#根据image_filename取出record中的多个字段,训练时只使用conversation和image


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path",
        type=Path,
    )

    parser.add_argument(
        "output_path",
        type=Path,
    )

    args = parser.parse_args()
    records = list(read_jsonl(args.input_path))

    print("input records:", len(records))

    rows = []

    for index, record in enumerate(records, start=1):
        row = build_row(record)
        rows.append(row)

        if index <= 3:
            print(
                "converted:",
                index,
                record["image_filename"],
            )

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, args.output_path)

    print("rows:", table.num_rows)
    print("columns:", table.column_names)
    print("output:", args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
