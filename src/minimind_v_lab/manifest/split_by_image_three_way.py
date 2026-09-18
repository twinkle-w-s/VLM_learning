# 这个脚本按图像把 manifest 切成 train、validation、test 三个集合，避免图像泄漏。

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("train_path", type=Path)
    parser.add_argument("val_path", type=Path)
    parser.add_argument("test_path", type=Path)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.val_ratio <= 0 or args.test_ratio <= 0:
        raise SystemExit("validation and test ratios must be positive")
    if args.val_ratio + args.test_ratio >= 1:
        raise SystemExit("validation + test ratios must be less than 1")

    image_names = sorted({
        record["image_filename"]
        for record in read_jsonl(args.input_path)
    })
    random.Random(args.seed).shuffle(image_names)

    test_count = round(len(image_names) * args.test_ratio)
    val_count = round(len(image_names) * args.val_ratio)
    test_images = set(image_names[:test_count])
    val_images = set(image_names[test_count:test_count + val_count])
    train_images = set(image_names[test_count + val_count:])

    assert not train_images & val_images
    assert not train_images & test_images
    assert not val_images & test_images

    for path in [args.train_path, args.val_path, args.test_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    counts = {"train": 0, "validation": 0, "test": 0}
    with args.train_path.open("w", encoding="utf-8") as train_file:
        with args.val_path.open("w", encoding="utf-8") as val_file:
            with args.test_path.open("w", encoding="utf-8") as test_file:
                for record in read_jsonl(args.input_path):
                    serialized = json.dumps(record, ensure_ascii=False) + "\n"
                    image_name = record["image_filename"]
                    if image_name in train_images:
                        train_file.write(serialized)
                        counts["train"] += 1
                    elif image_name in val_images:
                        val_file.write(serialized)
                        counts["validation"] += 1
                    else:
                        test_file.write(serialized)
                        counts["test"] += 1

    print("total records:", sum(counts.values()))
    print("record counts:", counts)
    print("image counts:", {
        "train": len(train_images),
        "validation": len(val_images),
        "test": len(test_images),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
