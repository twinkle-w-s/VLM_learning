"""批量计算固定 val/probe 的 assistant-token NLL。

复用 inspect_nll 已验收的编码、掩码和前向。
默认预览；--execute 才加载模型并写结果。
probe 是训练子集，不作为验证集。
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

from qwen.src.common.data_common import read_jsonl, resolve_path
from qwen.src.common.prompt_policy import rewrite_record


def load_records(root: Path, subset: str):
    with (root / "manifests/clevr/report.json").open(
        encoding="utf-8"
    ) as file:
        clevr_report = json.load(file)
    with (root / "manifests/replay/report.json").open(
        encoding="utf-8"
    ) as file:
        replay_report = json.load(file)

    if subset == "val":
        specifications = [
            (
                "clevr", root / "manifests/clevr/val.jsonl",
                clevr_report["output_counts"]["val"],
            ),
            (
                "replay_vl",
                root / "manifests/replay/val_vl.jsonl",
                replay_report["output_counts"]["val_vl"],
            ),
            (
                "replay_text",
                root / "manifests/replay/val_text.jsonl",
                replay_report["output_counts"]["val_text"],
            ),
        ]
        expected_split = "val"
    else:
        specifications = [
            (
                "clevr", root / "manifests/clevr/probe.jsonl",
                clevr_report["probe_count"],
            ),
            (
                "replay_vl",
                root / "manifests/replay/probe_vl.jsonl",
                replay_report["output_counts"]["probe_vl"],
            ),
            (
                "replay_text",
                root / "manifests/replay/probe_text.jsonl",
                replay_report["output_counts"]["probe_text"],
            ),
        ]
        expected_split = "train"

    records, identities, inputs = [], set(), {}
    for source, path, expected in specifications:
        rows = list(read_jsonl(path))
        if len(rows) != expected:
            raise ValueError(
                f"{path}: actual={len(rows)}, expected={expected}"
            )

        for row in rows:
            if (
                row["source"] != source
                or row["split"] != expected_split
            ):
                raise ValueError("来源或 split 与评估协议不一致")
            if row["sample_id"] in identities:
                raise ValueError("评估清单存在重复 sample_id")
            identities.add(row["sample_id"])

        records.extend(rows)
        inputs[source] = {
            "path": str(path),
            "rows": len(rows),
        }

    if not records:
        raise ValueError("评估数据为空")
    return records, inputs


def summarize(records: list[dict]) -> dict:
    from qwen.src.evaluation.nll_metrics import aggregate_nll

    sources, buckets = defaultdict(list), defaultdict(list)
    for row in records:
        sources[row["source"]].append(row)
        buckets[(row["source"], row["bucket"])].append(row)

    by_bucket = {}
    for (source, bucket), rows in sorted(buckets.items()):
        by_bucket.setdefault(source, {})[bucket] = (
            aggregate_nll(rows)
        )

    return {
        "overall": aggregate_nll(records),
        "by_source": {
            source: aggregate_nll(rows)
            for source, rows in sorted(sources.items())
        },
        "by_bucket": by_bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--subset", choices=("val", "probe"), required=True
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    root = resolve_path(cfg["output_root"], "output_root")
    model_path = resolve_path(
        cfg["model"]["name_or_path"], "model"
    )
    rows, inputs = load_records(root, args.subset)

    policy = cfg.get("prompt_policy", "original_v1")
    if policy == "comparison_yes_no_v1":
        rows = [rewrite_record(row)[0] for row in rows]
    elif policy != "original_v1":
        raise ValueError(f"未知提示协议：{policy}")

    complete = args.limit is None
    if args.limit is not None:
        if not 0 < args.limit <= len(rows):
            raise ValueError("--limit 超出评估样本数量")
        rows = rows[:args.limit]

    checkpoint = (
        resolve_path(args.checkpoint, "checkpoint")
        if args.checkpoint else None
    )
    if checkpoint:
        if not args.output:
            raise ValueError("adapter 评估必须指定 --output")
        for name in (
            "adapter_config.json", "adapter_model.safetensors"
        ):
            if not (checkpoint / name).is_file():
                raise FileNotFoundError(checkpoint / name)

    output = (
        resolve_path(args.output, "output")
        if args.output else
        root / "baseline" / f"nll_{args.subset}"
    )
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有评估：{output}")

    protocol = {
        "metric": "assistant_template_nll_v1",
        "token_scope": (
            "assistant body and template-supervised "
            "end markers/newline"
        ),
        "aggregation": "sum_nll / supervised_token_count",
        "subset": args.subset,
        "prompt_policy": policy,
        "model_path": str(model_path),
        "checkpoint": str(checkpoint) if checkpoint else None,
        "inputs": inputs,
        "evaluated_rows": len(rows),
        "complete_subset": complete,
        "max_length": int(cfg["train"]["max_length"]),
        "max_pixels": int(cfg["train"]["max_pixels"]),
        "batch_size": 1,
        "torch_dtype": "bfloat16",
        "attn_impl": "sdpa",
        "output": str(output),
    }
    print(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        flush=True,
    )
    if not args.execute:
        print("PREVIEW ONLY：未加载模型、未写结果。")
        return

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible:
        raise ValueError("请明确指定一张获分配的 GPU")
    if version("ms-swift") != "4.5.3":
        raise RuntimeError("需要 ms-swift==4.5.3")

    import torch
    from swift import TransformersEngine
    from qwen.src.evaluation.inspect_nll import score_one

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("需要一张支持 BF16 的 CUDA GPU")

    engine = TransformersEngine(
        str(model_path),
        adapters=[str(checkpoint)] if checkpoint else [],
        model_type="qwen3_vl",
        torch_dtype=torch.bfloat16,
        attn_impl="sdpa",
        device_map={"": "cuda:0"},
        max_batch_size=1,
    )
    engine.model.eval()
    engine.model.requires_grad_(False)

    template = engine.template
    template.model = engine.model
    template.max_length = protocol["max_length"]
    template.max_pixels = protocol["max_pixels"]
    template.truncation_strategy = "raise"
    template.padding_free = False
    template.set_mode("train")

    started = time.perf_counter()
    results = []
    output.mkdir(parents=True, exist_ok=False)

    with (output / "samples.jsonl").open(
        "x", encoding="utf-8"
    ) as file:
        for index, row in enumerate(rows, start=1):
            print(
                f"NLL {index}/{len(rows)}: {row['sample_id']}",
                flush=True,
            ) if index == 1 else None

            scored = score_one(engine, row)
            result = {
                "sample_id": scored["sample_id"],
                "source": scored["source"],
                "bucket": scored["bucket"],
                "sum_nll": scored["sum_nll"],
                "answer_token_count": scored["answer_token_count"],
                "mean_nll": scored["mean_nll"],
                "model_loss": scored["model_loss"],
                "loss_matches": scored["loss_matches"],
                "difficulty": row.get("difficulty"),
            }
            results.append(result)
            file.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )
            if index % 25 == 0 or index == len(rows):
                file.flush()
                print(
                    f"NLL: {index}/{len(rows)}",
                    flush=True,
                )

    metrics = summarize(results)
    metrics.update({
        "status": "COMPLETE" if complete else "SMOKE_COMPLETE",
        "protocol": protocol,
        "elapsed_seconds": round(
            time.perf_counter() - started, 2
        ),
        "val_feedback_candidate": (
            complete and args.subset == "val"
        ),
        "probe_dynamics_available": (
            complete and args.subset == "probe"
        ),
        "environment": {
            name: version(name)
            for name in (
                "torch", "transformers", "ms-swift", "peft"
            )
        },
    })
    with (output / "metrics.json").open(
        "x", encoding="utf-8"
    ) as file:
        json.dump(
            metrics, file, ensure_ascii=False, indent=2
        )

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("BATCH NLL EVALUATION: PASS")


if __name__ == "__main__":
    main()