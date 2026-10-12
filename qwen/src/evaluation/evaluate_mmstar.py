"""用基础 Qwen 或其 LoRA adapter 评估固定 MMStar，支持单卡批量生成。

默认预览；--execute 才加载模型。--limit 为 smoke，不用于正式退化对比。
只输入图片、问题与选项；标准答案只用于计分，结果不反馈训练配方。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import yaml

from qwen.src.common.data_common import read_jsonl, resolve_path
from qwen.src.data.prepare_mmstar import DEFAULT_DATA_ROOT
from qwen.src.evaluation.mmstar_metrics import (
    DATASET, EXPECTED_ROWS, METRIC, PARSER_POLICY, PROMPT_POLICY, REVISION,
    make_prompt, parse_letter, summarize,
)


def load_mmstar(root: Path):
    with (root / "report.json").open(encoding="utf-8") as file:
        report = json.load(file)
    if report["status"] != "MMSTAR_READY" or report["dataset"] != DATASET or report["dataset_revision"] != REVISION:
        raise ValueError("需要固定官方版本的已准备 MMStar")
    rows = list(read_jsonl(root / "manifest.jsonl"))
    if len(rows) != EXPECTED_ROWS or len(rows) != report["rows"]:
        raise ValueError("MMStar 应为完整1500条")
    categories = dict(Counter(row["category"] for row in rows))
    if categories != report["categories"] or len(categories) != 6 or set(categories.values()) != {250}:
        raise ValueError("MMStar 能力类别与准备报告不一致")
    ids = set()
    for row in rows:
        if row["sample_id"] in ids or row["source"] != "mmstar" or row["split"] != "external_eval":
            raise ValueError("MMStar 样本ID/来源/划分不一致")
        ids.add(row["sample_id"])
        if row["answer"] not in ("A", "B", "C", "D") or not row["question"].strip():
            raise ValueError("MMStar 题目或答案非法")
        if not resolve_path(row["image"], "image").is_file():
            raise FileNotFoundError(row["image"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须为正整数")
    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    settings = cfg["evaluation"]["mmstar"]
    if (settings["dataset"], settings["config"], settings["split"]) != (DATASET, "val", "val") or settings["feedback"] is not False:
        raise ValueError("本入口只评估官方 MMStar，且不能开启配方反馈")
    data_root = resolve_path(args.data_root, "data_root")
    model_path = resolve_path(cfg["model"]["name_or_path"], "model")
    output = resolve_path(args.output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有 MMStar 评估：{output}")
    rows = load_mmstar(data_root)
    complete = args.limit is None
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit 超出 MMStar 数量")
        rows = rows[:args.limit]
    checkpoint = resolve_path(args.checkpoint, "checkpoint") if args.checkpoint else None
    if checkpoint:
        for name in ("adapter_config.json", "adapter_model.safetensors"):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)
    protocol = {
        "metric": METRIC, "dataset": DATASET, "dataset_config": "val", "dataset_split": "val",
        "dataset_revision": REVISION, "dataset_rows": EXPECTED_ROWS,
        "data_path": str(data_root / "manifest.jsonl"), "model_path": str(model_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "prompt_policy": PROMPT_POLICY, "parser_policy": PARSER_POLICY,
        "max_length": int(cfg["train"]["max_length"]), "max_pixels": int(cfg["train"]["max_pixels"]),
        "max_new_tokens": int(settings["max_new_tokens"]), "temperature": 0, "batch_size": args.batch_size,
        "torch_dtype": "bfloat16", "attn_impl": "sdpa", "seed": int(cfg["seed"]),
        "evaluated_rows": len(rows), "complete_subset": complete, "output": str(output),
    }
    print(json.dumps(protocol, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("PREVIEW ONLY：未加载模型、未生成答案。")
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible:
        raise ValueError("请指定一张获分配的 GPU")
    if version("ms-swift") != "4.5.3":
        raise RuntimeError("需要 ms-swift==4.5.3")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    import torch
    from swift import InferRequest, RequestConfig, TransformersEngine

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or not torch.cuda.is_bf16_supported():
        raise RuntimeError("需要一张支持 BF16 的可见 GPU")
    torch.manual_seed(protocol["seed"])
    engine = TransformersEngine(
        str(model_path), adapters=[str(checkpoint)] if checkpoint else [], model_type="qwen3_vl",
        torch_dtype=torch.bfloat16, attn_impl="sdpa", device_map={"": "cuda:0"}, max_batch_size=args.batch_size,
    )
    engine.template.max_length = protocol["max_length"]
    engine.template.max_pixels = protocol["max_pixels"]
    engine.template.truncation_strategy = "raise"
    engine.model.eval()
    engine.model.requires_grad_(False)
    request_config = RequestConfig(max_tokens=protocol["max_new_tokens"], temperature=0, stream=False)
    results, started = [], time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("x", encoding="utf-8") as file:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            requests = [InferRequest(
                messages=[{"role": "user", "content": make_prompt(row["question"])}], images=[row["image"]],
            ) for row in batch]
            with torch.inference_mode():
                responses = engine.infer(requests, request_config, use_tqdm=False)
            if len(responses) != len(batch):
                raise RuntimeError("MMStar 响应数量与输入不一致")
            for row, response in zip(batch, responses):
                if isinstance(response, Exception):
                    raise response
                prediction = response.choices[0].message.content
                if not isinstance(prediction, str):
                    raise TypeError("MMStar 预测不是文本")
                parsed = parse_letter(prediction)
                result = {
                    "sample_id": row["sample_id"], "category": row["category"], "l2_category": row["l2_category"],
                    "question": row["question"], "image": row["image"], "reference": row["answer"],
                    "prediction": prediction, "parsed_prediction": parsed, "correct": parsed == row["answer"],
                    "finish_reason": response.choices[0].finish_reason,
                }
                results.append(result)
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
            completed = start + len(batch)
            if completed // 100 != start // 100 or completed == len(rows):
                print(f"MMStar: {completed}/{len(rows)}, ACC={sum(r['correct'] for r in results) / completed:.4f}", flush=True)
    metrics = {
        **summarize(results), "status": "COMPLETE" if complete else "SMOKE_COMPLETE",
        "feedback_eligible": False, "protocol": protocol,
        "project_training_overlap_checked": False,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "environment": {key: version(key) for key in ("torch", "transformers", "ms-swift", "peft")},
    }
    with (output / "metrics.json").open("x", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print("MMSTAR EVALUATION: PASS")


if __name__ == "__main__":
    main()
