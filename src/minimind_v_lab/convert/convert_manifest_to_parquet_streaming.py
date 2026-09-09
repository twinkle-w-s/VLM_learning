# 这个脚本分批把全量 manifest 转成 MiniMind-V Parquet，避免图像字节耗尽内存。

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_IMAGE_ROOT = (
    Path("/data")
    / os.environ["USER"]
    / "vlm_learning"
    / "CLEVR_v1.0"
    / "images"
    / "train"
)

SCHEMA = pa.schema(
    [
        ("question_index", pa.int64()),
        ("image_index", pa.int64()),
        ("image_filename", pa.string()),
        ("task_type", pa.string()),
        ("answer_type", pa.string()),
        ("difficulty", pa.string()),
        ("conversations", pa.string()),
        ("image_bytes", pa.binary()),
    ]
)


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def build_row(record: dict, image_root: Path) -> dict:
    image_path = image_root / record["image_filename"]
    conversations = [
        {
            "role": "user",
            "content": f'{record["question"]}<image>',
        },
        {
            "role": "assistant",
            "content": record["answer"],
        },
    ]

    return {
        "question_index": record["question_index"],
        "image_index": record["image_index"],
        "image_filename": record["image_filename"],
        "task_type": record["task_type"],
        "answer_type": record["answer_type"],
        "difficulty": record["difficulty"],
        "conversations": json.dumps(conversations, ensure_ascii=False),
        "image_bytes": image_path.read_bytes(),
    }


def write_batch(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    written = 0

    with pq.ParquetWriter(
        args.output_path,
        SCHEMA,
        compression="zstd",
    ) as writer:
        for record in read_jsonl(args.input_path):
            rows.append(build_row(record, args.image_root))

            if len(rows) >= args.batch_size:
                write_batch(writer, rows)
                written += len(rows)
                rows.clear()
                print("written:", written)

        if rows:
            write_batch(writer, rows)
            written += len(rows)

    print("rows:", written)
    print("columns:", SCHEMA.names)
    print("output:", args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
