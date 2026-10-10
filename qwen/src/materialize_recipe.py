"""按配方生成一轮 Qwen SFT 训练数据。

输入：recipe JSON、CLEVR/replay 训练候选池、每轮样本预算。
输出：训练 JSONL、追溯清单、配方快照和验收报告。
仅操作小候选池，不加载模型、不复制图片、不修改验证集。
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from data_common import read_jsonl, resolve_path, write_jsonl


def allocate_counts(total: int, weights: dict) -> dict:
    """按权重分配整数配额，使用最大余数法保证总数准确。"""
    if total < 0 or not weights:
        raise ValueError("数量不能为负，权重不能为空")
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("权重必须是有限非负数")
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        raise ValueError("权重之和必须大于 0")

    expected = {
        key: total * value / weight_sum
        for key, value in weights.items()
    }
    counts = {
        key: math.floor(value)
        for key, value in expected.items()
    }
    order = sorted(
        weights,
        key=lambda key: (-(expected[key] - counts[key]), key),
    )
    for key in order[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def sample_with_cap(
    rows: list,
    count: int,
    cap: int,
    rng,
) -> list:
    """逐遍洗牌抽样，限制一个样本在本轮最多出现 cap 次。"""
    if cap < 1 or count < 0 or count > len(rows) * cap:
        raise ValueError(
            f"采样容量不足或参数非法: requested={count}, "
            f"pool={len(rows)}, cap={cap}"
        )

    selected = []
    while len(selected) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        remaining = count - len(selected)
        selected.extend(cycle[:remaining])
    return selected


def load_train_pool(path: Path, source: str) -> list:
    """读取小候选池，拒绝来源错误、非训练记录和重复 ID。"""
    rows = list(read_jsonl(path))
    ids = set()

    for item in rows:
        if item["source"] != source or item["split"] != "train":
            raise ValueError(
                f"{path}: 来源不匹配或混入非训练样本"
            )
        if item["sample_id"] in ids:
            raise ValueError(
                f"{path}: 候选池存在重复 sample_id"
            )
        ids.add(item["sample_id"])

        if not item.get("messages") or not item.get("bucket"):
            raise ValueError(
                f"{path}: 缺少 messages 或 bucket"
            )
    return rows


def build_training_rows(recipe: dict, total: int):
    """按来源和 CLEVR 桶两层配额抽样，并验收实际结果。"""
    ratios = recipe["source_ratios"]
    sources = {"clevr", "replay_vl", "replay_text"}

    if set(ratios) != sources:
        raise ValueError(
            "配方必须包含 clevr/replay_vl/replay_text"
        )
    if any(not math.isfinite(v) or v < 0 for v in ratios.values()):
        raise ValueError("来源比例必须是有限非负数")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-8):
        raise ValueError("来源比例之和必须为 1")

    for source, value in ratios.items():
        lower, upper = recipe["source_bounds"][source]
        if not lower <= value <= upper:
            raise ValueError(
                f"{source}: 比例超出配方边界"
            )

    cap = recipe["max_repeats_per_sample"]
    quotas = allocate_counts(total, ratios)
    rng = random.Random(
        recipe["seed"] + 1000 * recipe["round"]
    )
    selected, pool_sizes = [], {}
    bucket_quota = {}

    for source in sorted(sources):
        count = quotas[source]
        if count == 0:
            pool_sizes[source] = 0
            continue

        path = resolve_path(
            recipe["pool_paths"][source],
            f"pool_paths.{source}",
        )
        rows = load_train_pool(path, source)
        pool_sizes[source] = len(rows)

        if source != "clevr":
            selected.extend(
                sample_with_cap(rows, count, cap, rng)
            )
            continue

        buckets = defaultdict(list)
        for item in rows:
            buckets[item["bucket"]].append(item)

        weights = recipe.get("clevr_bucket_weights")
        if weights is None:
            weights = {
                key: len(items)
                for key, items in buckets.items()
            }
        if set(weights) != set(buckets):
            raise ValueError(
                "CLEVR 桶权重与训练池的桶不一致"
            )

        bucket_quota = allocate_counts(count, weights)
        for bucket in sorted(buckets):
            selected.extend(
                sample_with_cap(
                    buckets[bucket],
                    bucket_quota[bucket],
                    cap,
                    rng,
                )
            )

    rng.shuffle(selected)
    repeats = Counter(
        item["sample_id"] for item in selected
    )
    actual = Counter(
        item["source"] for item in selected
    )

    if len(selected) != total or any(
        actual[source] != count
        for source, count in quotas.items()
    ):
        raise ValueError("实际来源数量与配方不一致")
    if max(repeats.values(), default=0) > cap:
        raise ValueError("超过单样本重复上限")

    image_paths = {
        path
        for item in selected
        for path in item.get("images", [])
    }
    for path in image_paths:
        if not Path(path).is_file():
            raise FileNotFoundError(
                f"训练图片不存在: {path}"
            )

    report = {
        "total_rows": len(selected),
        "pool_sizes": pool_sizes,
        "source_counts": {
            key: actual[key]
            for key in sorted(sources)
        },
        "effective_source_ratios": {
            key: actual[key] / total
            for key in sorted(sources)
        },
        "clevr_bucket_counts": bucket_quota,
        "unique_samples": len(repeats),
        "max_actual_repeats": max(
            repeats.values(), default=0
        ),
        "repeat_histogram": dict(
            sorted(Counter(repeats.values()).items())
        ),
        "unique_images": len(image_paths),
    }
    return selected, report


def materialize(
    cfg: dict,
    recipe_path: Path,
    total: int,
) -> None:
    """读取配方，生成训练文件与追溯文件，拒绝覆盖已有输出。"""
    if total <= 0:
        raise ValueError("每轮样本数量必须大于 0")

    with recipe_path.open(encoding="utf-8") as file:
        recipe = json.load(file)
    if recipe["seed"] != cfg["seed"] or recipe["round"] < 1:
        raise ValueError(
            "配方 seed 不一致或 round 非法"
        )

    output = (
        resolve_path(
            cfg["output_root"], "output_root"
        ).resolve()
        / "rounds"
        / f"round{recipe['round']:02d}"
        / "data"
    )
    if output.exists():
        raise FileExistsError(
            f"输出目录已存在，拒绝覆盖: {output}"
        )

    selected, report = build_training_rows(
        recipe, total
    )
    trainer_rows = []
    for item in selected:
        row = {"messages": item["messages"]}
        if item.get("images"):
            row["images"] = item["images"]
        trainer_rows.append(row)

    report.update({
        "status": "RECIPE_READY",
        "round": recipe["round"],
        "seed": recipe["seed"],
        "recipe_path": str(recipe_path.resolve()),
        "outer_feedback_enabled": (
            recipe["outer_feedback_enabled"]
        ),
        "train_path": str(output / "train.jsonl"),
    })

    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(
        output / "train.jsonl", trainer_rows
    )
    write_jsonl(
        output / "train_manifest.jsonl", selected
    )

    for name, content in (
        ("recipe_used.json", recipe),
        ("report.json", report),
    ):
        with (output / name).open(
            "x", encoding="utf-8"
        ) as file:
            json.dump(
                content,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    print(json.dumps(
        report, ensure_ascii=False, indent=2
    ))
    print("MATERIALIZE RECIPE: PASS")
    print("OUTPUT:", output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--num-samples", type=int)
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    root = resolve_path(
        cfg["output_root"], "output_root"
    )
    recipe_path = (
        args.recipe
        if args.recipe is not None
        else root
        / "manifests"
        / "replay"
        / "recipe_initial.json"
    )
    total = (
        args.num_samples
        if args.num_samples is not None
        else cfg["train"]["steps_per_round"]
        * cfg["train"]["global_batch_size"]
    )
    materialize(cfg, recipe_path, total)


if __name__ == "__main__":
    main()