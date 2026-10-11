"""对齐固定 probe，计算 NLL、学习/遗忘事件及来源/桶汇总。

纯 CPU 计算；依赖已落盘的观察值，不加载模型。
delta=current-previous，负值代表 NLL 下降；gain 使用相反符号。
遗忘只统计相邻已观测 checkpoint 的正确→错误，不推断未观测过程。
"""

from __future__ import annotations

import math
from collections import defaultdict

from qwen.src.evaluation.nll_metrics import aggregate_nll


def nll_change(previous, current: float, steps):
    if previous is None:
        return {
            "nll_delta": None, "nll_gain": None,
            "relative_nll_gain": None, "nll_delta_per_optimizer_step": None,
        }
    delta = current - previous
    return {
        "nll_delta": delta,
        "nll_gain": -delta,
        "relative_nll_gain": -delta / previous if previous > 0 else None,
        "nll_delta_per_optimizer_step": delta / steps,
    }


def transition(previous, current):
    if current is None:
        return "not_scored"
    if previous is None:
        return "first_observation"
    if previous and not current:
        return "forgotten"
    if not previous and current:
        return "learned"
    return "stable_correct" if current else "stable_wrong"


def validate_snapshots(snapshots: list[dict]) -> None:
    if not snapshots:
        raise ValueError("观察历史为空")
    first = snapshots[0]
    if first["round"] != 0 or first["cumulative_optimizer_steps"] != 0:
        raise ValueError("第一条必须为 round=0、step=0 的基础模型")
    identities = set(first["nll_records"])
    if not identities:
        raise ValueError("probe 为空")
    clevr_ids = {
        sid for sid, row in first["nll_records"].items()
        if row["source"] == "clevr"
    }
    if not clevr_ids:
        raise ValueError("需要 CLEVR probe 正确性观察")
    tags = set()
    previous_round, previous_step = -1, -1
    for snapshot in snapshots:
        if snapshot["tag"] in tags:
            raise ValueError("snapshot tag 重复")
        tags.add(snapshot["tag"])
        step, round_id = snapshot["cumulative_optimizer_steps"], snapshot["round"]
        if (
            type(step) is not int or type(round_id) is not int
            or round_id != previous_round + 1 or step <= previous_step
        ):
            raise ValueError("轮次必须从 0 连续增加，累计 step 必须递增")
        previous_round, previous_step = round_id, step
        rows = snapshot["nll_records"]
        if set(rows) != identities:
            raise ValueError("跨轮 probe sample_id 集合不一致")
        if set(snapshot["correctness"]) != clevr_ids:
            raise ValueError("正确性必须完整覆盖 CLEVR probe，不能给 replay 强行计 EM")
        if any(type(flag) is not bool for flag in snapshot["correctness"].values()):
            raise ValueError("correct 必须为布尔值")
        for sid, row in rows.items():
            reference = first["nll_records"][sid]
            for key in ("sample_id", "source", "bucket", "difficulty", "answer_token_count"):
                if row.get(key) != reference.get(key):
                    raise ValueError(f"{sid}: 跨轮 {key} 发生变化")
            count = row["answer_token_count"]
            total, mean = row["sum_nll"], row["mean_nll"]
            if (
                row["sample_id"] != sid
                or type(count) is not int or count <= 0
                or isinstance(total, bool) or isinstance(mean, bool)
                or not math.isfinite(total) or total < 0
                or not math.isfinite(mean) or mean < 0
                or not math.isclose(mean, total / count, rel_tol=1e-6, abs_tol=1e-8)
            ):
                raise ValueError(f"{sid}: NLL 记录非法或均值不一致")


def summarize_group(ids, snapshots, sample_results):
    history = []
    for snapshot in snapshots:
        records = [snapshot["nll_records"][sid] for sid in ids]
        entry = aggregate_nll(records)
        flags = [snapshot["correctness"][sid] for sid in ids if sid in snapshot["correctness"]]
        entry.update({
            "tag": snapshot["tag"], "round": snapshot["round"],
            "cumulative_optimizer_steps": snapshot["cumulative_optimizer_steps"],
            "scored_n": len(flags),
            "accuracy": sum(flags) / len(flags) if flags else None,
        })
        history.append(entry)
    last = history[-1]
    previous = history[-2] if len(history) > 1 else None
    steps = last["cumulative_optimizer_steps"] - previous["cumulative_optimizer_steps"] if previous else None
    scored = [sample_results[sid] for sid in ids if sample_results[sid]["forgetting_count"] is not None]
    return {
        "n": len(ids), "history": history,
        "latest": {
            **nll_change(previous["mean_nll"] if previous else None, last["mean_nll"], steps),
            "accuracy_delta": (
                last["accuracy"] - previous["accuracy"]
                if previous and last["accuracy"] is not None else None
            ),
        },
        "scored_n": len(scored),
        "forgetting_events": sum(row["forgetting_count"] for row in scored) if scored else None,
        "learning_events": sum(row["learning_count"] for row in scored) if scored else None,
        "samples_with_forgetting": sum(row["forgetting_count"] > 0 for row in scored) if scored else None,
    }


def build_dynamics(snapshots: list[dict]):
    """返回逐样本动态和分组摘要；不修改输入观察值。"""
    validate_snapshots(snapshots)
    first = snapshots[0]["nll_records"]
    results = {}
    source_ids, bucket_ids = defaultdict(list), defaultdict(list)
    for sid in sorted(first):
        row = first[sid]
        history = []
        correctness = [s["correctness"].get(sid) for s in snapshots]
        scored = sid in snapshots[0]["correctness"]
        for snapshot, correct in zip(snapshots, correctness):
            observed = snapshot["nll_records"][sid]
            history.append({
                "tag": snapshot["tag"], "round": snapshot["round"],
                "cumulative_optimizer_steps": snapshot["cumulative_optimizer_steps"],
                "sum_nll": observed["sum_nll"],
                "answer_token_count": observed["answer_token_count"],
                "mean_nll": observed["mean_nll"], "correct": correct,
            })
        previous = history[-2] if len(history) > 1 else None
        current = history[-1]
        steps = current["cumulative_optimizer_steps"] - previous["cumulative_optimizer_steps"] if previous else None
        results[sid] = {
            "sample_id": sid, "source": row["source"], "bucket": row["bucket"],
            "difficulty": row.get("difficulty"), "observed_snapshots": len(history),
            "nll_history": history, "correctness_history": correctness,
            "latest": nll_change(previous["mean_nll"] if previous else None, current["mean_nll"], steps),
            "baseline_nll_gain": history[0]["mean_nll"] - current["mean_nll"] if previous else None,
            "latest_transition": transition(correctness[-2] if previous else None, correctness[-1]),
            "forgetting_count": sum(a is True and b is False for a, b in zip(correctness, correctness[1:])) if scored else None,
            "learning_count": sum(a is False and b is True for a, b in zip(correctness, correctness[1:])) if scored else None,
        }
        source_ids[row["source"]].append(sid)
        bucket_ids[(row["source"], row["bucket"])].append(sid)
    by_bucket = {}
    for (source, bucket), ids in sorted(bucket_ids.items()):
        by_bucket.setdefault(source, {})[bucket] = summarize_group(ids, snapshots, results)
    summary = {
        "status": "DYNAMICS_READY" if len(snapshots) > 1 else "BASELINE_ONLY",
        "snapshot_count": len(snapshots), "probe_count": len(results),
        "scope": "fixed_train_probe_only",
        "forgetting_scope": "adjacent_observed_checkpoints_only",
        "optimizer_steps_are_exposures": False,
        "overall": summarize_group(sorted(first), snapshots, results),
        "by_source": {source: summarize_group(ids, snapshots, results) for source, ids in sorted(source_ids.items())},
        "by_bucket": by_bucket,
        "recipe_updated": False,
    }
    return list(results.values()), summary
