"""读取已完成的 probe NLL/EM 产物，核对协议与 checkpoint 步数。

只支持每轮 optimizer/scheduler 重置的 step_accounting=round_local_reset。
不从 val、smoke、计划训练步数或尚未保存的 checkpoint 推断动态。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from qwen.src.common.data_common import read_jsonl, resolve_path
from qwen.src.evaluation.evaluate_clevr import normalize_answer
from qwen.src.evaluation.nll_metrics import aggregate_nll


NLL_KEYS = (
    "metric", "token_scope", "aggregation", "prompt_policy", "model_path",
    "max_length", "max_pixels", "batch_size", "torch_dtype", "attn_impl",
)
EM_KEYS = (
    "metric", "prompt_policy", "model_path", "max_length", "max_pixels",
    "max_new_tokens", "temperature", "batch_size", "torch_dtype", "attn_impl",
)


def read_json(path: Path):
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def index_records(path: Path):
    result = {}
    for row in read_jsonl(path):
        sid = row["sample_id"]
        if sid in result:
            raise ValueError(f"{path}: sample_id 重复：{sid}")
        result[sid] = row
    if not result:
        raise ValueError(f"{path}: 结果为空")
    return result


def check_complete(metrics: dict, directory: Path):
    protocol = metrics["protocol"]
    if (
        metrics["status"] != "COMPLETE"
        or protocol.get("subset") != "probe"
        or protocol.get("complete_subset") is not True
    ):
        raise ValueError(f"{directory}: 必须是完整训练 probe 结果，不能使用 val/smoke")
    return protocol


def protocol_values(protocol, keys):
    missing = set(keys) - set(protocol)
    if missing:
        raise ValueError(f"评估协议缺字段：{sorted(missing)}")
    return {key: protocol[key] for key in keys}


def load_history(config: dict) -> list[dict]:
    if config.get("step_accounting") != "round_local_reset":
        raise ValueError("目前只支持每轮重置 optimizer 的 round_local_reset 步数口径")
    entries = config.get("snapshots", [])
    if not entries:
        raise ValueError("请至少配置基础模型 snapshot")
    stages, checkpoints = [], set()
    cumulative_steps = 0
    previous_nll, previous_em, examples = None, None, None
    for index, entry in enumerate(entries):
        if type(entry["round"]) is not int or entry["round"] != index:
            raise ValueError("历史必须按 base=0、round1=1、round2=2 的顺序配置")
        nll_dir = resolve_path(entry["nll_dir"], "nll_dir")
        em_dir = resolve_path(entry["clevr_dir"], "clevr_dir")
        nll_metrics = read_json(nll_dir / "metrics.json")
        em_metrics = read_json(em_dir / "metrics.json")
        nll_protocol = check_complete(nll_metrics, nll_dir)
        em_protocol = check_complete(em_metrics, em_dir)
        nll_values = protocol_values(nll_protocol, NLL_KEYS)
        em_values = protocol_values(em_protocol, EM_KEYS)
        if (
            nll_values["metric"] != "assistant_template_nll_v1"
            or em_values["metric"] != "clevr_normalized_em_v1"
        ):
            raise ValueError("不支持的 NLL/EM 协议")
        if previous_nll is not None and (nll_values != previous_nll or em_values != previous_em):
            raise ValueError("跨轮 NLL 或生成协议变化，不能直接比较")
        previous_nll, previous_em = nll_values, em_values
        for key in ("prompt_policy", "model_path", "max_length", "max_pixels", "torch_dtype", "attn_impl"):
            if nll_protocol[key] != em_protocol[key]:
                raise ValueError(f"同一 checkpoint 的 NLL/EM {key} 不一致")
        checkpoint = nll_protocol["checkpoint"]
        if checkpoint != em_protocol["checkpoint"]:
            raise ValueError("NLL 与 EM 来自不同 checkpoint")
        if index == 0:
            if checkpoint is not None:
                raise ValueError("第一条必须是没有 adapter 的基础模型")
            local_steps = 0
        else:
            if not checkpoint or checkpoint in checkpoints:
                raise ValueError("后续轮次必须来自不同的已保存 adapter")
            state_path = resolve_path(checkpoint, "checkpoint") / "trainer_state.json"
            local_steps = read_json(state_path)["global_step"]
            if type(local_steps) is not int or local_steps <= 0:
                raise ValueError("checkpoint global_step 必须为正整数")
        checkpoints.add(checkpoint)
        cumulative_steps += local_steps
        nll_records = index_records(nll_dir / "samples.jsonl")
        predictions = index_records(em_dir / "predictions.jsonl")
        aggregate = aggregate_nll(list(nll_records.values()))
        for key in ("n", "answer_token_count", "sum_nll", "mean_nll"):
            if not math.isclose(aggregate[key], nll_metrics["overall"][key], rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"NLL 逐样本记录与 metrics.{key} 不一致")
        if len(nll_records) != nll_protocol["evaluated_rows"]:
            raise ValueError("NLL 行数与 protocol 不一致")
        clevr_ids = {sid for sid, row in nll_records.items() if row["source"] == "clevr"}
        if set(predictions) != clevr_ids:
            raise ValueError("EM 样本不是同一套完整 CLEVR probe")
        current_examples, correctness = {}, {}
        for sid, prediction in predictions.items():
            if prediction["bucket"] != nll_records[sid]["bucket"]:
                raise ValueError(f"{sid}: NLL/EM 桶不一致")
            target = normalize_answer(prediction["reference"])
            actual = normalize_answer(prediction["prediction"])
            correct = prediction["correct"]
            if (
                not target or type(correct) is not bool
                or prediction["normalized_reference"] != target
                or prediction["normalized_prediction"] != actual
                or correct != (target == actual)
            ):
                raise ValueError(f"{sid}: 正确性记录与原始预测不一致")
            current_examples[sid] = (
                prediction["question"], prediction["image"], prediction["reference"],
            )
            correctness[sid] = correct
        if examples is not None and current_examples != examples:
            raise ValueError("跨轮 CLEVR 问题、图片或标准答案发生变化")
        examples = current_examples
        count = len(predictions)
        if (
            not count or em_metrics["n"] != count
            or em_protocol["evaluated_rows"] != count
            or em_metrics["correct"] != sum(correctness.values())
            or not math.isclose(em_metrics["accuracy"], sum(correctness.values()) / count, abs_tol=1e-8)
        ):
            raise ValueError("EM 逐样本记录与 metrics 不一致")
        stages.append({
            "tag": entry["tag"], "round": index,
            "checkpoint": checkpoint,
            "checkpoint_local_steps": local_steps,
            "cumulative_optimizer_steps": cumulative_steps,
            "nll_dir": str(nll_dir), "clevr_dir": str(em_dir),
            "nll_protocol": nll_values, "generation_protocol": em_values,
            "nll_records": nll_records, "correctness": correctness,
        })
    return stages
