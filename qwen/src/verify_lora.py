"""验收 Qwen3-VL 的 LLM-only LoRA checkpoint。

默认在 CPU 上检查配置、权重范围、形状和有限值。
--infer 才在一张指定 GPU 上加载基础模型和 adapter，做一条 CLEVR val 问答。
不训练，不加载 optimizer，不修改 checkpoint；单题正确性不作为性能验收。
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import yaml

from data_common import read_jsonl, resolve_path


def inspect_adapter(cfg: dict, checkpoint: Path) -> dict:
    import torch
    from safetensors.torch import load_file

    with (checkpoint / "adapter_config.json").open(
        encoding="utf-8"
    ) as file:
        adapter_cfg = json.load(file)

    model_cfg = cfg["model"]
    if model_cfg["lora_scope"] != "llm_only":
        raise ValueError("本验收文件只适用于 llm_only LoRA")
    if adapter_cfg["peft_type"] != "LORA":
        raise ValueError("checkpoint 不是 LoRA")
    if adapter_cfg["r"] != model_cfg["lora_rank"]:
        raise ValueError("checkpoint 的 rank 与 YAML 不一致")
    if adapter_cfg["lora_alpha"] != model_cfg["lora_alpha"]:
        raise ValueError("checkpoint 的 alpha 与 YAML 不一致")

    weights = load_file(
        str(checkpoint / "adapter_model.safetensors"),
        device="cpu",
    )
    if not weights:
        raise ValueError("adapter 权重为空")

    dtypes = Counter()
    pairs = {}
    nonzero_b = 0

    for name, tensor in weights.items():
        # 当前配置不应包含 visual/aligner 权重。
        if ".language_model." not in name or ".visual." in name:
            raise ValueError(f"发现 LLM 之外的权重：{name}")

        if name.endswith(".lora_A.weight"):
            prefix = name[:-len(".lora_A.weight")]
            side = "A"
        elif name.endswith(".lora_B.weight"):
            prefix = name[:-len(".lora_B.weight")]
            side = "B"
        else:
            raise ValueError(f"发现非 LoRA A/B 权重：{name}")

        if tensor.ndim != 2 or not torch.isfinite(tensor).all().item():
            raise ValueError(f"形状异常或含 NaN/Inf：{name}")

        # A 的形状为 [rank, 输入维度]；
        # B 的形状为 [输出维度, rank]。
        rank_axis = 0 if side == "A" else 1
        if tensor.shape[rank_axis] != adapter_cfg["r"]:
            raise ValueError(f"LoRA rank 维度异常：{name}")

        pairs.setdefault(prefix, set()).add(side)
        dtypes[str(tensor.dtype)] += tensor.numel()

        if side == "B" and torch.count_nonzero(tensor).item() > 0:
            nonzero_b += 1

    if any(sides != {"A", "B"} for sides in pairs.values()):
        raise ValueError("部分 LoRA 模块缺少 A/B 配对")

    summary = {
        "checkpoint": str(checkpoint),
        "base_model_recorded": adapter_cfg.get(
            "base_model_name_or_path"
        ),
        "rank": adapter_cfg["r"],
        "alpha": adapter_cfg["lora_alpha"],
        "tensor_count": len(weights),
        "lora_module_count": len(pairs),
        "parameter_count": sum(
            t.numel() for t in weights.values()
        ),
        "dtype_parameter_counts": dict(dtypes),
        "nonzero_B_modules": nonzero_b,
        "example_keys": list(weights)[:4],
    }
    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )
    print("ADAPTER FILE CHECK: PASS", flush=True)

    if nonzero_b == 0:
        print(
            "WARNING: B 全为零，请检查训练是否真正更新。",
            flush=True,
        )

    return weights


def infer_one(
    cfg: dict,
    checkpoint: Path,
    saved_weights: dict,
) -> None:
    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible:
        raise ValueError(
            "请用 CUDA_VISIBLE_DEVICES 指定一张获分配的 GPU"
        )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("需要恰好一张可见 CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持 BF16")
    if version("ms-swift") != "4.5.3":
        raise RuntimeError("本文件按 ms-swift==4.5.3 核对")

    root = resolve_path(cfg["output_root"], "output_root")
    val_path = root / "manifests" / "clevr" / "val.jsonl"
    row = next(read_jsonl(val_path))

    if [m["role"] for m in row["messages"]] != [
        "user", "assistant"
    ]:
        raise ValueError(
            "本检查要求 CLEVR 单轮 user/assistant 格式"
        )
    if len(row["images"]) != 1:
        raise ValueError("本检查要求一张 CLEVR 图片")

    image_path = resolve_path(row["images"][0], "val.image")
    if not image_path.is_file():
        raise FileNotFoundError(image_path)

    from peft import get_peft_model_state_dict
    from swift import (
        InferRequest,
        RequestConfig,
        TransformersEngine,
    )

    model_path = resolve_path(
        cfg["model"]["name_or_path"], "model"
    )
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    engine = TransformersEngine(
        str(model_path),
        adapters=[str(checkpoint)],
        adapter_names=["default"],
        model_type="qwen3_vl",
        torch_dtype=torch.bfloat16,
        attn_impl="sdpa",
        device_map={"": "cuda:0"},
        max_batch_size=1,
    )
    engine.template.max_length = int(
        cfg["train"]["max_length"]
    )
    engine.template.max_pixels = int(
        cfg["train"]["max_pixels"]
    )

    # PEFT 导出为保存格式的 key，
    # 避免手工猜测 ".default" 的插入位置。
    loaded_weights = get_peft_model_state_dict(
        engine.model,
        adapter_name="default",
        save_embedding_layers=False,
    )

    if set(loaded_weights) != set(saved_weights):
        raise ValueError(
            "重新加载的 LoRA key 集合与 checkpoint 不一致"
        )

    for name, expected in saved_weights.items():
        actual = loaded_weights[name].detach().cpu()
        if actual.shape != expected.shape or not torch.equal(
            actual, expected.to(dtype=actual.dtype)
        ):
            raise ValueError(
                f"重新加载的权重与 checkpoint 不一致：{name}"
            )

    del loaded_weights
    print("ADAPTER RELOAD CHECK: PASS", flush=True)

    # 推理不需要梯度，也不构建 optimizer。
    engine.model.eval()
    engine.model.requires_grad_(False)

    request = InferRequest(
        # 只输入问题，绝不能把标准答案一起输入。
        messages=[dict(row["messages"][0])],
        images=[str(image_path)],
    )
    request_cfg = RequestConfig(
        max_tokens=int(
            cfg["evaluation"]["qa_max_new_tokens"]
        ),
        temperature=0,
        stream=False,
    )

    with torch.inference_mode():
        response = engine.infer([request], request_cfg)[0]

    if isinstance(response, Exception):
        raise response

    prediction = response.choices[0].message.content
    if not isinstance(prediction, str) or not prediction.strip():
        raise ValueError("模型没有生成非空文本")

    print(
        json.dumps(
            {
                "sample_id": row["sample_id"],
                "question": row["messages"][0]["content"],
                "image": str(image_path),
                "reference": row["answer"],
                "prediction": prediction,
                "finish_reason": response.choices[0].finish_reason,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("ONE IMAGE INFERENCE: PASS")
    print(
        "PASS 表示加载与生成正常，"
        "不代表回答正确或性能提升。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--infer", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)

    root = resolve_path(cfg["output_root"], "output_root")
    checkpoint = (
        resolve_path(args.checkpoint, "checkpoint")
        if args.checkpoint
        else (
            root
            / "smoke"
            / "sft"
            / f"checkpoint-{cfg['train']['smoke_steps']}"
        )
    )

    weights = inspect_adapter(cfg, checkpoint)

    if args.infer:
        infer_one(cfg, checkpoint, weights)
    else:
        print(
            "CPU CHECK ONLY："
            "未加载基础模型，未使用 GPU。"
        )


if __name__ == "__main__":
    main()