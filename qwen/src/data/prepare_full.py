"""全量 CLEVR 首轮数据准备。

默认只检查配置并计算预算。
--execute 准备 CLEVR/replay，物化首轮并验证 CLEVR 全覆盖。
不加载模型、不训练；拒绝覆盖已有实验目录。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from qwen.src.common.data_common import read_jsonl, resolve_path
from qwen.src.data.materialize_recipe import (
    allocate_counts,
    materialize,
)
from qwen.src.data.prepare_clevr import prepare_clevr
from qwen.src.data.prepare_replay import prepare_replay


def plan_budget(train_rows: int, ratio: float, batch: int):
    if train_rows <= 0 or not 0 < ratio <= 1 or batch <= 0:
        raise ValueError("训练数量、CLEVR 比例或 batch 非法")

    total = math.ceil(train_rows / ratio / batch) * batch
    return total, total // batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/full.yaml"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    if (
        cfg["data"]["train_pool_size"]["clevr"] is not None
        or cfg["evaluation"]["clevr_val_size"] is not None
    ):
        raise ValueError("全量入口要求 CLEVR train/val 数量为 null")
    if cfg["prompt_policy"] != "comparison_yes_no_v1":
        raise ValueError("全量实验统一使用修正后的 comparison 提示")

    root = resolve_path(cfg["output_root"], "output_root")
    if root.exists():
        raise FileExistsError(
            f"拒绝覆盖已有全量实验：{root}。"
            "若之前准备中断，请使用新的 output_root。"
        )

    input_counts = {}
    for split in ("train", "val", "test"):
        path = resolve_path(
            cfg["data"]["clevr"][split], f"clevr.{split}"
        )
        input_counts[split] = sum(
            1 for _ in read_jsonl(path)
        )
        if input_counts[split] == 0:
            raise ValueError(f"CLEVR {split} 为空")

    total, steps = plan_budget(
        input_counts["train"],
        cfg["recipe"]["source_ratios"]["clevr"],
        int(cfg["train"]["global_batch_size"]),
    )
    quotas = allocate_counts(
        total, cfg["recipe"]["source_ratios"]
    )
    if quotas["clevr"] < input_counts["train"]:
        raise ValueError("整数配额不足以覆盖全部 CLEVR train")

    cap = int(cfg["recipe"]["max_repeats_per_sample"])
    for source in ("replay_vl", "replay_text"):
        capacity = (
            cfg["data"]["train_pool_size"][source] * cap
        )
        if quotas[source] > capacity:
            raise ValueError(
                f"{source} 候选池容量不足："
                f"quota={quotas[source]}, capacity={capacity}"
            )

    # 文字不足时，图文池也必须承受转移后的全部 replay。
    fallback_vl_quota = total - quotas["clevr"]
    if fallback_vl_quota > (
        cfg["data"]["train_pool_size"]["replay_vl"] * cap
    ):
        raise ValueError("图文池容量不足以承接文字 fallback")

    cfg["train"]["steps_per_round"] = steps
    cfg["evaluation"]["clevr_val_size"] = input_counts["val"]

    plan = {
        "input_counts": input_counts,
        "total_rows": total,
        "optimizer_steps": steps,
        "global_batch_size": cfg["train"]["global_batch_size"],
        "nominal_source_counts": quotas,
        "clevr_full_coverage_required": True,
        "prompt_policy": cfg["prompt_policy"],
        "output_root": str(root),
        "training": False,
    }
    print(
        json.dumps(plan, ensure_ascii=False, indent=2),
        flush=True,
    )

    if not args.execute:
        print("PREVIEW ONLY：未准备数据、未训练。")
        return

    prepare_clevr(cfg)
    prepare_replay(cfg)

    recipe_path = (
        root / "manifests/replay/recipe_initial.json"
    )
    with recipe_path.open(encoding="utf-8") as file:
        recipe = json.load(file)

    effective_quotas = allocate_counts(
        total, recipe["source_ratios"]
    )
    for source, quota in effective_quotas.items():
        if quota == 0:
            continue
        path = resolve_path(
            recipe["pool_paths"][source], source
        )
        available = sum(1 for _ in read_jsonl(path))
        if quota > available * cap:
            raise ValueError(
                f"{source} 实际容量不足："
                f"quota={quota}, pool={available}, cap={cap}"
            )

    materialize(cfg, recipe_path, total)

    manifest = (
        root / "rounds/round01/data/train_manifest.jsonl"
    )
    covered = {
        row["sample_id"]
        for row in read_jsonl(manifest)
        if row["source"] == "clevr"
    }
    if len(covered) != input_counts["train"]:
        raise ValueError(
            "首轮未覆盖全部 CLEVR train："
            f"{len(covered)}/{input_counts['train']}"
        )

    plan.update({
        "status": "FULL_DATA_READY",
        "effective_source_counts": effective_quotas,
        "covered_unique_clevr": len(covered),
    })

    runtime_config = root / "full_runtime.yaml"
    with runtime_config.open("x", encoding="utf-8") as file:
        yaml.safe_dump(
            cfg, file,
            allow_unicode=True,
            sort_keys=False,
        )
    with (root / "full_plan.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            plan, file, ensure_ascii=False, indent=2
        )

    print("FULL DATA PREPARE: PASS")
    print("RUNTIME CONFIG:", runtime_config)
    print("本入口没有启动训练。")


if __name__ == "__main__":
    main()