"""MMStar 固定数据版本、选择题提示、解析及训练前后对比。

纯 CPU；无外部 judge。无法解析的回答计错，不从分母中移除。
delta_pp=after-base，仅描述本次观测，不判定统计显著性或更新配方。
"""

from __future__ import annotations

import re
from collections import defaultdict


DATASET = "Lin-Chen/MMStar"
REVISION = "bc98d668301da7b14f648724866e57302778ab27"
FILENAME = "mmstar.parquet"
EXPECTED_ROWS = 1500
PROMPT_POLICY = "mmstar_letter_v1"
PARSER_POLICY = "single_letter_optional_brackets_period_v1"
METRIC = "mmstar_letter_accuracy_v1"


def make_prompt(question: str) -> str:
    # 官方单图数据的文本占位符不是新的图片；只向 Swift 提交一张图。
    question = question.replace("<image 1>", "the image").replace("<image>", "the image")
    return (
        "<image>\n" + question + "\n"
        "Answer with only the option letter (A, B, C, or D). Do not include any explanation."
    )


def parse_letter(prediction: str):
    """仅接收独立字母，允许括号/方括号和结尾句号；不从解释里猜字母。"""
    match = re.fullmatch(r"(?:([A-D])|\(([A-D])\)|\[([A-D])\])[.。]?", prediction.strip(), re.IGNORECASE)
    return next(value.upper() for value in match.groups() if value) if match else None


def summarize(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("MMStar 预测为空")

    def score(group):
        correct = sum(row["correct"] for row in group)
        return {
            "n": len(group), "correct": correct, "accuracy": correct / len(group),
            "invalid_predictions": sum(row["parsed_prediction"] is None for row in group),
        }

    categories, subcategories = defaultdict(list), defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row)
        subcategories[(row["category"], row["l2_category"])].append(row)
    by_subcategory = {}
    for (category, subcategory), group in sorted(subcategories.items()):
        by_subcategory.setdefault(category, {})[subcategory] = score(group)
    return {
        **score(rows),
        "empty_predictions": sum(not row["prediction"].strip() for row in rows),
        "length_limited_predictions": sum(row["finish_reason"] == "length" for row in rows),
        "by_category": {key: score(group) for key, group in sorted(categories.items())},
        "by_subcategory": by_subcategory,
    }


PROTOCOL_KEYS = (
    "metric", "dataset", "dataset_config", "dataset_split", "dataset_revision",
    "dataset_rows", "data_path", "model_path", "prompt_policy", "parser_policy",
    "max_length", "max_pixels", "max_new_tokens", "temperature", "batch_size",
    "torch_dtype", "attn_impl", "seed",
)


def validate_result(metrics: dict, rows: list[dict]) -> dict:
    protocol = metrics["protocol"]
    if (
        metrics["status"] != "COMPLETE" or protocol["complete_subset"] is not True
        or protocol["dataset_rows"] != EXPECTED_ROWS or len(rows) != EXPECTED_ROWS
        or protocol["evaluated_rows"] != len(rows)
        or protocol["dataset"] != DATASET or protocol["dataset_revision"] != REVISION
        or protocol["metric"] != METRIC or protocol["parser_policy"] != PARSER_POLICY
        or protocol["prompt_policy"] != PROMPT_POLICY
    ):
        raise ValueError("退化对比必须使用同一版本完整 MMStar，不能使用 smoke")
    indexed = {}
    for row in rows:
        sid = row["sample_id"]
        if sid in indexed:
            raise ValueError(f"MMStar sample_id 重复：{sid}")
        parsed = parse_letter(row["prediction"])
        if (
            row["reference"] not in ("A", "B", "C", "D")
            or row["parsed_prediction"] != parsed or type(row["correct"]) is not bool
            or row["correct"] != (parsed == row["reference"])
        ):
            raise ValueError(f"MMStar 正确性记录不一致：{sid}")
        indexed[sid] = row
    for key, value in summarize(rows).items():
        if metrics[key] != value:
            raise ValueError(f"MMStar metrics.{key} 与逐样本记录不一致")
    return indexed


def compare_results(base_metrics, base_rows, after_metrics, after_rows):
    base = validate_result(base_metrics, base_rows)
    after = validate_result(after_metrics, after_rows)
    if base_metrics["protocol"]["checkpoint"] is not None or not after_metrics["protocol"]["checkpoint"]:
        raise ValueError("必须比较原始模型与真实 adapter 的评估结果")
    if any(base_metrics["protocol"][key] != after_metrics["protocol"][key] for key in PROTOCOL_KEYS):
        raise ValueError("MMStar 评估协议不同，不能直接归因于训练")
    if set(base) != set(after):
        raise ValueError("MMStar 对比样本集合不同")
    pairs = []
    for sid in sorted(base):
        before, current = base[sid], after[sid]
        for key in ("question", "image", "reference", "category", "l2_category"):
            if before[key] != current[key]:
                raise ValueError(f"MMStar 样本 {sid} 的 {key} 发生变化")
        pairs.append({
            "sample_id": sid, "category": current["category"], "l2_category": current["l2_category"],
            "reference": current["reference"],
            "base_prediction": before["prediction"], "after_prediction": current["prediction"],
            "base_correct": before["correct"], "after_correct": current["correct"],
            "base_valid": before["parsed_prediction"] is not None,
            "after_valid": current["parsed_prediction"] is not None,
        })

    def score(group):
        n = len(group)
        base_correct = sum(row["base_correct"] for row in group)
        after_correct = sum(row["after_correct"] for row in group)
        return {
            "n": n, "base_accuracy": base_correct / n, "after_accuracy": after_correct / n,
            "delta_pp": (after_correct - base_correct) / n * 100,
            "correct_to_wrong": sum(row["base_correct"] and not row["after_correct"] for row in group),
            "wrong_to_correct": sum(not row["base_correct"] and row["after_correct"] for row in group),
            "base_invalid": sum(not row["base_valid"] for row in group),
            "after_invalid": sum(not row["after_valid"] for row in group),
        }

    categories = defaultdict(list)
    for row in pairs:
        categories[row["category"]].append(row)
    overall = score(pairs)
    return pairs, {
        "status": "COMPARISON_READY", **overall,
        "direction": "observed_decrease" if overall["delta_pp"] < 0 else "observed_increase" if overall["delta_pp"] > 0 else "unchanged",
        "by_category": {key: score(group) for key, group in sorted(categories.items())},
        "feedback_eligible": False, "recipe_updated": False,
        "project_training_overlap_checked": False,
        "base_protocol": base_metrics["protocol"], "after_protocol": after_metrics["protocol"],
    }
