"""评估 Qwen3-VL 在固定 CLEVR val 上的生成准确率。

不传 --checkpoint 时评估原始 Instruct 模型；传入则评估基础模型 + LoRA。
默认只预览；--execute 才加载模型并写出预测与分桶指标。
--limit 用于链路 smoke，不作为飞轮反馈。此文件暂不计算 NLL 或 MMStar。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from importlib.metadata import version
from pathlib import Path

import yaml

from data_common import read_jsonl, resolve_path


def normalize_answer(text: str) -> str:
    """不抽取关键词，不丢弃第二行，不把解释性回答判成正确。"""
    return " ".join(
        text.lower().strip().rstrip(".。").split()
    )


def load_val(path: Path, expected_rows: int) -> list[dict]:
    rows = list(read_jsonl(path))

    if len(rows) != expected_rows:
        raise ValueError(
            "固定 val 行数不一致："
            f"actual={len(rows)}, expected={expected_rows}"
        )

    ids = set()
    for row in rows:
        if row["sample_id"] in ids:
            raise ValueError(
                f"重复 sample_id：{row['sample_id']}"
            )
        ids.add(row["sample_id"])

        if row["source"] != "clevr" or row["split"] != "val":
            raise ValueError(
                "本文件只评估准备后的 CLEVR val"
            )
        if not row["scorable"] or not row["bucket"]:
            raise ValueError("缺少可评分标记或 bucket")
        if [m["role"] for m in row["messages"]] != [
            "user", "assistant"
        ]:
            raise ValueError(
                "需要单轮 user/assistant 格式"
            )
        if not normalize_answer(str(row["answer"])):
            raise ValueError("标准答案为空")
        if len(row["images"]) != 1:
            raise ValueError("需要单张 CLEVR 图片")

        image = resolve_path(
            row["images"][0], "val.image"
        )
        if not image.is_file():
            raise FileNotFoundError(image)

    return rows


def summarize(
    results: list[dict],
    min_bucket_val: int,
    full_val: bool,
) -> dict:
    buckets = defaultdict(list)
    for result in results:
        buckets[result["bucket"]].append(
            result["correct"]
        )

    by_bucket = {}
    for bucket, values in sorted(buckets.items()):
        correct = sum(values)
        by_bucket[bucket] = {
            "n": len(values),
            "correct": correct,
            "accuracy": correct / len(values),
            "feedback_eligible": (
                full_val
                and len(values) >= min_bucket_val
            ),
        }

    correct = sum(r["correct"] for r in results)
    return {
        "n": len(results),
        "correct": correct,
        "accuracy": correct / len(results),
        "empty_predictions": sum(
            not r["normalized_prediction"]
            for r in results
        ),
        "length_limited_predictions": sum(
            r["finish_reason"] == "length"
            for r in results
        ),
        "by_bucket": by_bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    root = resolve_path(
        cfg["output_root"], "output_root"
    )
    model_path = resolve_path(
        cfg["model"]["name_or_path"], "model"
    )
    val_path = (
        root / "manifests" / "clevr" / "val.jsonl"
    )
    rows = load_val(
        val_path,
        int(cfg["evaluation"]["clevr_val_size"]),
    )

    # --limit 即使恰好覆盖所有行，也按 smoke 管理。
    full_val = args.limit is None
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError(
                "--limit 必须在 1 和固定 val 行数之间"
            )
        rows = rows[:args.limit]

    checkpoint = (
        resolve_path(args.checkpoint, "checkpoint")
        if args.checkpoint
        else None
    )
    if checkpoint is not None and not args.output:
        raise ValueError(
            "评估 adapter 时请明确填写 --output"
        )
    if checkpoint is not None:
        for name in (
            "adapter_config.json",
            "adapter_model.safetensors",
        ):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(
                    checkpoint / name
                )

    output = (
        resolve_path(args.output, "output")
        if args.output
        else (
            root / "baseline" / "clevr"
            if full_val
            else root / "smoke" / "eval_base_clevr"
        )
    )
    if output.exists():
        raise FileExistsError(
            f"拒绝覆盖已有评估：{output}"
        )

    protocol = {
        "metric": "clevr_normalized_em_v1",
        "model_path": str(model_path),
        "checkpoint": (
            str(checkpoint) if checkpoint else None
        ),
        "val_path": str(val_path),
        "evaluated_rows": len(rows),
        "full_val": full_val,
        "max_length": int(
            cfg["train"]["max_length"]
        ),
        "max_pixels": int(
            cfg["train"]["max_pixels"]
        ),
        "max_new_tokens": int(
            cfg["evaluation"]["qa_max_new_tokens"]
        ),
        "temperature": 0,
        "batch_size": 1,
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "output": str(output),
    }
    print(
        json.dumps(
            protocol, ensure_ascii=False, indent=2
        ),
        flush=True,
    )

    if not args.execute:
        print(
            "PREVIEW ONLY："
            "未加载模型、未写出结果。"
        )
        return

    visible = os.environ.get(
        "CUDA_VISIBLE_DEVICES", ""
    )
    if not visible.strip() or "," in visible:
        raise ValueError(
            "请指定一张获分配的 GPU"
        )
    if version("ms-swift") != "4.5.3":
        raise RuntimeError(
            "本文件按 ms-swift==4.5.3 核对"
        )
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    import torch
    from swift import (
        InferRequest,
        RequestConfig,
        TransformersEngine,
    )

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
    ):
        raise RuntimeError(
            "需要恰好一张可见 CUDA GPU"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "需要支持 BF16 的 GPU"
        )

    torch.manual_seed(int(cfg["seed"]))

    engine = TransformersEngine(
        str(model_path),
        adapters=(
            [str(checkpoint)] if checkpoint else []
        ),
        model_type="qwen3_vl",
        torch_dtype=torch.bfloat16,
        attn_impl="sdpa",
        device_map={"": "cuda:0"},
        max_batch_size=1,
    )
    engine.template.max_length = (
        protocol["max_length"]
    )
    engine.template.max_pixels = (
        protocol["max_pixels"]
    )
    engine.model.eval()
    engine.model.requires_grad_(False)

    request_cfg = RequestConfig(
        max_tokens=protocol["max_new_tokens"],
        temperature=0,
        stream=False,
    )

    results = []
    started = time.perf_counter()
    output.mkdir(parents=True, exist_ok=False)

    # 每条预测及时落盘。
    # 中途异常则退出，不生成完整 metrics。
    with (output / "predictions.jsonl").open(
        "x", encoding="utf-8"
    ) as file:
        for index, row in enumerate(rows, start=1):
            request = InferRequest(
                # 只输入问题，不输入 assistant 标准答案。
                messages=[
                    dict(row["messages"][0])
                ],
                images=[
                    str(
                        resolve_path(
                            row["images"][0],
                            "val.image",
                        )
                    )
                ],
            )

            with torch.inference_mode():
                response = engine.infer(
                    [request], request_cfg
                )[0]

            if isinstance(response, Exception):
                raise response

            prediction = (
                response.choices[0].message.content
            )
            if not isinstance(prediction, str):
                raise TypeError(
                    "生成结果不是文本"
                )

            normalized = normalize_answer(
                prediction
            )
            target = normalize_answer(
                str(row["answer"])
            )

            result = {
                "sample_id": row["sample_id"],
                "bucket": row["bucket"],
                "task_type": row["task_type"],
                "difficulty": row["difficulty"],
                "question": (
                    row["messages"][0]["content"]
                ),
                "image": row["images"][0],
                "reference": str(row["answer"]),
                "prediction": prediction,
                "normalized_prediction": normalized,
                "normalized_reference": target,
                "correct": normalized == target,
                "finish_reason": (
                    response.choices[0].finish_reason
                ),
            }
            results.append(result)

            file.write(
                json.dumps(
                    result, ensure_ascii=False
                ) + "\n"
            )
            file.flush()

            if index % 25 == 0 or index == len(rows):
                correct = sum(
                    r["correct"] for r in results
                )
                print(
                    f"CLEVR val: {index}/{len(rows)}, "
                    f"EM={correct / index:.4f}",
                    flush=True,
                )

    metrics = summarize(
        results,
        int(cfg["flywheel"]["min_bucket_val"]),
        full_val,
    )
    metrics.update({
        "status": (
            "COMPLETE"
            if full_val
            else "SMOKE_COMPLETE"
        ),
        "feedback_eligible": full_val,
        "protocol": protocol,
        "elapsed_seconds": round(
            time.perf_counter() - started, 2
        ),
        "environment": {
            name: version(name)
            for name in (
                "torch",
                "transformers",
                "ms-swift",
                "peft",
            )
        },
    })

    with (output / "metrics.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    print(
        json.dumps(
            metrics, ensure_ascii=False, indent=2
        )
    )
    print("CLEVR EVALUATION: PASS")
    print(
        "PASS 表示评估完成，"
        "不要求模型达到某个准确率。"
    )


if __name__ == "__main__":
    main()