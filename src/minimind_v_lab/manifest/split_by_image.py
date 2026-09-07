from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def write_jsonl(records: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
#按照图片划分
def split_by_image(
    records: list[dict],
    val_ratio: float,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    image_names = sorted(
        {record["image_filename"] for record in records}
    )

    random.Random(seed).shuffle(image_names)#打乱图像名，而非打乱问题

    split_index = round(len(image_names) * (1 - val_ratio))
    train_images = set(image_names[:split_index])

    train_records = [
        record
        for record in records
        if record["image_filename"] in train_images
    ]

    val_records = [
        record
        for record in records
        if record["image_filename"] not in train_images
    ]

    train_image_names = {
        record["image_filename"]
        for record in train_records
    }
    val_image_names = {
        record["image_filename"]
        for record in val_records
    }

    assert not train_image_names & val_image_names#同一图像不能被泄露到两个集合

    return train_records, val_records

def main() ->int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("train_path", type=Path)
    parser.add_argument("val_path", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    records = read_jsonl(args.input_path)

    train_records, val_records = split_by_image(
        records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.train_path.parent.mkdir(parents=True, exist_ok=True)
    args.val_path.parent.mkdir(parents=True, exist_ok=True)

    write_jsonl(train_records, args.train_path)
    write_jsonl(val_records, args.val_path)

    print("total records:", len(records))
    print("train records:", len(train_records))
    print("val records:", len(val_records))
    print(
        "train images:",
        len({x["image_filename"] for x in train_records}),
    )
    print(
        "val images:",
        len({x["image_filename"] for x in val_records}),
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())