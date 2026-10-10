"""CLEVR 数据准备。

抽取候选池、合并稀疏桶、生成固定 probe 和验收报告。
沿用已有 train/val/test 划分，检查图片泄漏与入选图片解码。
不修改原始数据，不执行模型训练。
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

from data_common import (
    read_jsonl,
    reservoir_add,
    resolve_path,
    write_jsonl,
)


def make_clevr_record(
    row: dict,
    split: str,
    image_root: Path,
) -> dict:
    question = row["question"].strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        raise ValueError("CLEVR 出现空问题或答案")

    return {
        "sample_id": (
            f"clevr:CLEVR_train:{row['question_index']}"
        ),
        "source": "clevr",
        "split": split,
        "image_id": row["image_filename"],
        "task_type": row["task_type"],
        "difficulty": row["difficulty"],
        "spatial_relations": row.get(
            "spatial_relations", []
        ),
        "answer": answer,
        "scorable": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    "<image>\n" + question
                    + "\nAnswer with a single word or number."
                ),
            },
            {"role": "assistant", "content": answer},
        ],
        "images": [
            str(image_root / row["image_filename"])
        ],
    }


def select_clevr(cfg: dict):
    paths = cfg["data"]["clevr"]
    image_root = resolve_path(
        paths["image_root"], "clevr.image_root"
    ).resolve()
    limits = {
        "train": cfg["data"]["train_pool_size"]["clevr"],
        "val": cfg["evaluation"]["clevr_val_size"],
        "test": None,
    }
    pools, inputs, image_sets = {}, {}, {}
    seen_ids = set()

    for split_index, split in enumerate(
        ("train", "val", "test")
    ):
        path = resolve_path(paths[split], f"clevr.{split}")
        limit = limits[split]
        if limit is not None and limit <= 0:
            raise ValueError(
                f"{split} 抽样数量必须大于 0"
            )

        rng = random.Random(cfg["seed"] + split_index)
        pool, names, count = [], set(), 0

        for count, row in enumerate(
            read_jsonl(path), start=1
        ):
            item = make_clevr_record(
                row, split, image_root
            )
            if item["sample_id"] in seen_ids:
                raise ValueError(
                    f"源清单出现重复 sample_id: {item['sample_id']}"
                )
            seen_ids.add(item["sample_id"])
            names.add(item["image_id"])
            reservoir_add(
                pool, item, count, limit, rng
            )
            if count % 50000 == 0:
                print(
                    f"CLEVR {split}: scanned={count}",
                    flush=True,
                )

        if not pool or (
            limit is not None and len(pool) < limit
        ):
            raise ValueError(
                f"{split} 样本不足: available={count}, target={limit}"
            )

        pools[split] = pool
        image_sets[split] = names
        inputs[split] = {
            "path": str(path),
            "rows": count,
            "unique_images": len(names),
        }
        print(
            f"CLEVR {split}: selected={len(pool)}, scanned={count}",
            flush=True,
        )

    overlaps = {
        f"{a}-{b}": len(image_sets[a] & image_sets[b])
        for a, b in (
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        )
    }
    if any(overlaps.values()):
        raise ValueError(
            f"CLEVR 原始 split 存在图片泄漏: {overlaps}"
        )

    return pools, {
        "inputs": inputs,
        "image_overlap": overlaps,
    }


def assign_clevr_buckets(pools: dict, cfg: dict) -> dict:
    fields = cfg["data"]["bucket_fields"]
    minimum = cfg["data"]["min_bucket_train"]
    if minimum <= 0:
        raise ValueError("min_bucket_train 必须大于 0")

    def raw_bucket(item):
        return "|".join(
            str(item[field]) for field in fields
        )

    train = pools["train"]
    raw_counts = Counter(
        raw_bucket(item) for item in train
    )
    task_counts = Counter(
        item["task_type"] for item in train
    )
    merged_tasks = {
        item["task_type"]
        for item in train
        if raw_counts[raw_bucket(item)] < minimum
    }

    for records in pools.values():
        for item in records:
            task = item["task_type"]
            raw = raw_bucket(item)
            if task_counts[task] < minimum:
                item["bucket"] = "other"
            elif task in merged_tasks:
                item["bucket"] = f"task:{task}"
            else:
                item["bucket"] = (
                    raw if raw in raw_counts else "other"
                )

    return {
        "fields": fields,
        "merged_tasks": sorted(merged_tasks),
        "train_counts": dict(
            Counter(item["bucket"] for item in train)
        ),
    }


def prepare_clevr(cfg: dict) -> None:
    output = (
        resolve_path(cfg["output_root"], "output_root")
        / "manifests"
        / "clevr"
    )
    if output.exists():
        raise FileExistsError(
            f"输出目录已存在，拒绝覆盖: {output}"
        )

    pools, stats = select_clevr(cfg)
    bucket_stats = assign_clevr_buckets(pools, cfg)
    probe_size = cfg["evaluation"]["probe_size"]["clevr"]
    if not 0 < probe_size <= len(pools["train"]):
        raise ValueError(
            "CLEVR probe 数量必须在 1 和训练池数量之间"
        )
    probe = random.Random(cfg["seed"] + 100).sample(
        pools["train"], probe_size
    )

    image_paths = {
        item["images"][0]
        for records in pools.values()
        for item in records
    }
    for index, image_path in enumerate(
        sorted(image_paths), start=1
    ):
        with Image.open(image_path) as image:
            image.load()
        if index % 1000 == 0:
            print(
                f"CLEVR decoded unique images: {index}",
                flush=True,
            )

    report = {
        "status": "CLEVR_READY_REPLAY_PENDING",
        "seed": cfg["seed"],
        **stats,
        "output_counts": {
            split: len(rows)
            for split, rows in pools.items()
        },
        "probe_count": len(probe),
        "checked_unique_images": len(image_paths),
        "buckets": bucket_stats,
    }

    output.mkdir(parents=True, exist_ok=False)
    for split, records in pools.items():
        write_jsonl(
            output / f"{split}.jsonl", records
        )
    write_jsonl(output / "probe.jsonl", probe)

    with (output / "report.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            report, file, ensure_ascii=False, indent=2
        )
        file.write("\n")

    print(json.dumps(
        report, ensure_ascii=False, indent=2
    ))
    print("CLEVR PREPARE: PASS")
    print("OUTPUT:", output)
    print(
        "REPLAY_PENDING: 尚未准备 15% 图文和 5% 文字回放。"
    )