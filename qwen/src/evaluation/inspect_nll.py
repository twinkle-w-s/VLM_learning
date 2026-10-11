"""真实 Qwen3-VL 的 assistant-token NLL 小验收。

各来源选择一条固定 val 样本：
检查监督范围，执行无梯度前向，并比较手算 NLL 与模型 loss。
不生成答案、不训练、不写文件；结果不能直接用于飞轮反馈。
NLL 包含模板实际监督的结束符，不是仅答案正文 NLL。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from importlib.metadata import version
from pathlib import Path

import yaml

from qwen.src.common.data_common import read_jsonl, resolve_path
from qwen.src.common.prompt_policy import rewrite_record


def select_examples(root: Path) -> list[dict]:
    clevr_path = root / "manifests/clevr/val.jsonl"
    clevr_rows = list(read_jsonl(clevr_path))
    if not clevr_rows:
        raise ValueError("CLEVR val 为空")

    clevr = next(
        (
            row for row in clevr_rows
            if row["task_type"] == "comparison"
        ),
        clevr_rows[0],
    )
    examples = [clevr]

    replay_root = root / "manifests/replay"
    for name in ("val_vl.jsonl", "val_text.jsonl"):
        row = next(read_jsonl(replay_root / name), None)
        if row is None:
            if name == "val_text.jsonl":
                print("文字 replay 为空，本次跳过。")
                continue
            raise ValueError("图文 replay val 为空")
        examples.append(row)

    return examples


def inspect_labels(batch, tokenizer, model_config, row) -> dict:
    import torch

    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    if input_ids.shape != labels.shape:
        raise ValueError("input_ids 与 labels 形状不一致")
    if labels.ndim != 2 or labels.shape[0] != 1:
        raise ValueError("本验收只支持单样本 batch")

    supervised = labels != -100
    positions = supervised[0].nonzero().flatten()

    if positions.numel() == 0:
        raise ValueError("模板没有生成监督 token")
    if positions[0].item() == 0:
        raise ValueError("首位置不应参与下一 token 监督")
    if not torch.equal(
        labels[supervised], input_ids[supervised]
    ):
        raise ValueError("监督标签与原 token 不一致")
    if supervised[attention_mask == 0].any().item():
        raise ValueError("padding 被错误计入监督")

    image_mask = input_ids == model_config.image_token_id
    if supervised[image_mask].any().item():
        raise ValueError("图片 token 被错误计入监督")

    token_ids = labels[supervised].tolist()
    supervised_text = tokenizer.decode(
        token_ids, skip_special_tokens=False
    )
    body_text = tokenizer.decode(
        token_ids, skip_special_tokens=True
    ).strip()
    reference = row["messages"][-1]["content"].strip()

    if body_text != reference:
        raise ValueError(
            "监督正文与 assistant 答案不一致："
            f"supervised={body_text!r}, "
            f"reference={reference!r}"
        )

    return {
        "sequence_length": int(input_ids.shape[1]),
        "masked_token_count": int(
            (~supervised).sum().item()
        ),
        "image_token_count": int(image_mask.sum().item()),
        "supervised_token_count": len(token_ids),
        "first_supervised_position": int(
            positions[0].item()
        ),
        "supervised_text": supervised_text,
        "assistant_body_matches": True,
    }


def score_one(engine, row: dict) -> dict:
    import torch

    from qwen.src.evaluation.nll_metrics import compute_answer_nll
    from swift.utils import to_device

    row = copy.deepcopy(row)
    if [m["role"] for m in row["messages"]] not in (
        ["user", "assistant"],
        ["system", "user", "assistant"],
    ):
        raise ValueError("需要单轮 user/assistant 样本")

    # 纯文本没有图片；图文样本仍要求存在 images 字段。
    image_paths = (
        []
        if row["source"] == "replay_text"
        else row["images"]
    )

    template = engine.template
    encoded = template.encode({
        "messages": row["messages"],
        "images": [
            str(resolve_path(path, "sample.image"))
            for path in image_paths
        ],
    })
    batch = template.data_collator([encoded])

    mask_info = inspect_labels(
        batch,
        template.tokenizer,
        engine.model.config,
        row,
    )

    # 这些是训练 collator 的辅助字段。
    # 单样本、非 packing、SDPA 前向不需要它们。
    for key in (
        "loss_scale",
        "text_position_ids",
        "cu_seq_lens_q",
        "cu_seq_lens_k",
        "max_length_q",
        "max_length_k",
    ):
        batch.pop(key, None)

    batch = to_device(batch, engine.model.device)

    with torch.inference_mode():
        with template.forward_context(engine.model, batch):
            output = engine.model(
                **batch,
                use_cache=False,
                return_dict=True,
                logits_to_keep=0,
            )

        nll = compute_answer_nll(
            output.logits, batch["labels"]
        )[0]
        model_loss = float(output.loss.float().item())

    if not math.isclose(
        nll["mean_nll"],
        model_loss,
        rel_tol=1e-4,
        abs_tol=1e-4,
    ):
        raise ValueError(
            "NLL 与模型 loss 不一致："
            f"computed={nll['mean_nll']}, "
            f"model={model_loss}"
        )

    return {
        "sample_id": row["sample_id"],
        "source": row["source"],
        "bucket": row["bucket"],
        "mask_check": mask_info,
        **nll,
        "model_loss": model_loss,
        "loss_matches": True,
        "feedback_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    parser.add_argument(
        "--comparison-yes-no", action="store_true"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    root = resolve_path(cfg["output_root"], "output_root")
    model_path = resolve_path(
        cfg["model"]["name_or_path"], "model"
    )
    examples = select_examples(root)

    comparison_yes_no = (
        args.comparison_yes_no
        or cfg.get("prompt_policy") == "comparison_yes_no_v1"
    )
    if comparison_yes_no:
        examples = [
            rewrite_record(row)[0] for row in examples
        ]

    print(json.dumps({
        "model_path": str(model_path),
        "prompt_policy": (
            "comparison_yes_no_v1"
            if comparison_yes_no
            else "original_v1"
        ),
        "samples": [
            {
                "sample_id": row["sample_id"],
                "source": row["source"],
            }
            for row in examples
        ],
        "metric_scope": (
            "assistant supervised tokens including "
            "template-supervised end markers"
        ),
        "training": False,
        "writes_files": False,
    }, ensure_ascii=False, indent=2), flush=True)

    if not args.execute:
        print("PREVIEW ONLY：未加载模型。")
        return


    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible:
        raise ValueError("请明确指定一张获分配的 GPU")
    if version("ms-swift") != "4.5.3":
        raise RuntimeError("本文件按 ms-swift==4.5.3 核对")
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    import torch
    from swift import TransformersEngine

    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("需要一张支持 BF16 的 CUDA GPU")

    engine = TransformersEngine(
        str(model_path),
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
    template.max_length = int(cfg["train"]["max_length"])
    template.max_pixels = int(cfg["train"]["max_pixels"])
    template.truncation_strategy = "raise"
    template.padding_free = False
    template.set_mode("train")

    for row in examples:
        result = score_one(engine, row)
        print(
            json.dumps(
                result, ensure_ascii=False, indent=2
            ),
            flush=True,
        )

    print("REAL MODEL NLL SMOKE: PASS")
    print("仅完成小样本验收；未训练、未写出正式指标。")


if __name__ == "__main__":
    main()
