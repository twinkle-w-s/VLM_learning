from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

def main() ->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("train_path", type=Path)
    parser.add_argument("val_path", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    if not 0 < args.val_ratio < 1:
        raise SystemExit("--val-ratio must be between 0 and 1")

    image_names = sorted(
        {
            record["image_filename"]
            for record in read_jsonl(args.input_path)
        }
    )
    random.Random(args.seed).shuffle(image_names)

    split_index = round(len(image_names) * (1 - args.val_ratio))
    train_images = set(image_names[:split_index])
    val_images = set(image_names[split_index:])
    assert not train_images & val_images

    args.train_path.parent.mkdir(parents=True, exist_ok=True)
    args.val_path.parent.mkdir(parents=True, exist_ok=True)

    train_count = 0
    val_count = 0
    with args.train_path.open("w", encoding="utf-8") as train_file:
        with args.val_path.open("w", encoding="utf-8") as val_file:
            for record in read_jsonl(args.input_path):
                serialized = json.dumps(record, ensure_ascii=False) + "\n"
                if record["image_filename"] in train_images:
                    train_file.write(serialized)
                    train_count += 1
                else:
                    val_file.write(serialized)
                    val_count += 1

    print("total records:", train_count + val_count)
    print("train records:", train_count)
    print("val records:", val_count)
    print("train images:", len(train_images))
    print("val images:", len(val_images))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
