"""启动一轮 comparison 提示修正后的探索性 SFT。

读取已物化的 round01，不重新抽样，只修正 CLEVR comparison 的回答要求。
数据、配置和 checkpoint 放到独立实验目录，保留旧数据与 baseline。
默认只预览；--execute 才写出新数据并调用已有 train_sft.py。
不启动第二轮，不更新飞轮，不自动评估。
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import yaml

from data_common import read_jsonl, resolve_path, write_jsonl
from train_sft import build_command


OLD_SUFFIX = "Answer with a single word or number."
NEW_SUFFIX = (
    "Answer with exactly yes or no. "
    "Do not output a number or explanation."
)


def rewrite_record(item: dict) -> tuple[dict, bool]:
    """只根据任务标签改 comparison，不读取答案来选择提示。"""
    row = copy.deepcopy(item)
    changed = (
        row["source"] == "clevr"
        and row["task_type"] == "comparison"
    )

    if changed:
        message = row["messages"][0]
        if (
            message["role"] != "user"
            or not message["content"].endswith(OLD_SUFFIX)
        ):
            raise ValueError(
                f"comparison 提示格式异常：{row['sample_id']}"
            )
        message["content"] = (
            message["content"][:-len(OLD_SUFFIX)]
            + NEW_SUFFIX
        )

    return row, changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    root = resolve_path(
        cfg["output_root"], "output_root"
    ).resolve()
    parent_data = root / "rounds" / "round01" / "data"
    experiment = (
        root / "experiments" / "comparison_prompt_v1"
    )

    if experiment.exists():
        raise FileExistsError(
            f"拒绝覆盖已有实验：{experiment}"
        )

    with (parent_data / "report.json").open(
        encoding="utf-8"
    ) as file:
        report = json.load(file)

    if (
        report["status"] != "RECIPE_READY"
        or report["round"] != 1
        or report["seed"] != cfg["seed"]
    ):
        raise ValueError(
            "需要已验收、seed 一致的 round01 数据"
        )

    # 使用带 source/task_type 标签的追溯清单。
    rows = list(
        read_jsonl(parent_data / "train_manifest.jsonl")
    )
    if len(rows) != report["total_rows"]:
        raise ValueError("追溯清单行数与报告不一致")
    if (
        dict(Counter(row["source"] for row in rows))
        != report["source_counts"]
    ):
        raise ValueError("来源数量与报告不一致")

    rewritten = []
    changed_count = 0
    for item in rows:
        row, changed = rewrite_record(item)
        rewritten.append(row)
        changed_count += int(changed)

    if changed_count == 0:
        raise ValueError(
            "没有找到 comparison 训练样本"
        )

    # 只改变输出根目录；模型、预算、配比均不改变。
    run_cfg = copy.deepcopy(cfg)
    run_cfg["output_root"] = str(experiment)
    _, summary = build_command(run_cfg, "round1")

    if len(rows) != summary["planned_sample_exposures"]:
        raise ValueError(
            "行数与 steps × global_batch 不一致"
        )

    summary.update({
        "experiment_type": "EXPLORATORY_SFT",
        "parent_data_dir": str(parent_data),
        "dataset_rows": len(rows),
        "nominal_epochs": (
            summary["planned_sample_exposures"]
            / len(rows)
        ),
        "source_counts": report["source_counts"],
        "prompt_policy": "comparison_yes_no_v1",
        "comparison_rows_changed": changed_count,
        "new_comparison_suffix": NEW_SUFFIX,
    })
    print(
        json.dumps(
            summary, ensure_ascii=False, indent=2
        ),
        flush=True,
    )

    if not args.execute:
        print(
            "PREVIEW ONLY：未写出数据，"
            "未加载模型，未开始训练。"
        )
        return

    data_dir = (
        experiment / "rounds" / "round01" / "data"
    )
    data_dir.mkdir(parents=True, exist_ok=False)

    # 训练文件只保留 ms-swift 需要的字段。
    trainer_rows = []
    for item in rewritten:
        row = {"messages": item["messages"]}
        if item.get("images"):
            row["images"] = item["images"]
        trainer_rows.append(row)

    write_jsonl(
        data_dir / "train.jsonl", trainer_rows
    )
    write_jsonl(
        data_dir / "train_manifest.jsonl", rewritten
    )

    report.update({
        "train_path": str(data_dir / "train.jsonl"),
        "experiment_type": "EXPLORATORY_SFT",
        "parent_data_dir": str(parent_data),
        "prompt_policy": "comparison_yes_no_v1",
        "comparison_rows_changed": changed_count,
        "new_comparison_suffix": NEW_SUFFIX,
    })
    with (data_dir / "report.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            report, file, ensure_ascii=False, indent=2
        )
        file.write("\n")

    # 保存原配方快照：来源和桶配额没有变化。
    with (parent_data / "recipe_used.json").open(
        encoding="utf-8"
    ) as file:
        recipe = json.load(file)
    with (data_dir / "recipe_used.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            recipe, file, ensure_ascii=False, indent=2
        )
        file.write("\n")

    # 此配置用于本次训练启动，评估路径明天再统一。
    run_config_path = experiment / "mvp.yaml"
    with run_config_path.open(
        "x", encoding="utf-8"
    ) as file:
        yaml.safe_dump(
            run_cfg,
            file,
            allow_unicode=True,
            sort_keys=False,
        )

    # 复用已通过 smoke 的启动入口和环境检查。
    launcher = (
        Path(__file__).resolve()
        .with_name("train_sft.py")
    )
    subprocess.run([
        sys.executable,
        str(launcher),
        "--config", str(run_config_path),
        "--mode", "round1",
        "--execute",
    ], check=True)

    print(
        "EXPLORATORY ROUND1 TRAINING PROCESS FINISHED",
        flush=True,
    )


if __name__ == "__main__":
    main()