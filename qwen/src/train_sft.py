"""启动 Qwen3-VL 的单卡 LoRA SFT。

读取 mvp.yaml 和 round01 已验收的数据。
默认只预览命令；--execute 才加载模型并训练。
smoke 与 round1 分开保存，均从原始模型新建 LoRA，不覆盖旧输出。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import yaml

from data_common import read_jsonl, resolve_path


def build_command(cfg: dict, mode: str) -> tuple[list[str], dict]:
    """把项目 YAML 翻译为 ms-swift 4.5.3 的参数。"""
    train, model = cfg["train"], cfg["model"]

    if model["lora_scope"] != "llm_only":
        raise ValueError("本版只实现 llm_only LoRA")
    if not model["freeze_vision"] or not model["freeze_aligner"]:
        raise ValueError("本版要求冻结视觉编码器和 aligner")
    if not train["bf16"]:
        raise ValueError("本版按 RTX 4090 的 BF16 训练实现")

    micro = int(train["micro_batch_size"])
    global_batch = int(train["global_batch_size"])
    if micro <= 0 or global_batch <= 0 or global_batch % micro:
        raise ValueError(
            "单卡 global_batch_size 必须是 micro_batch_size 的正整数倍"
        )
    accumulation = global_batch // micro

    steps = int(
        train["smoke_steps"]
        if mode == "smoke"
        else train["steps_per_round"]
    )
    if steps <= 0:
        raise ValueError("训练步数必须大于 0")

    root = resolve_path(cfg["output_root"], "output_root")
    model_path = resolve_path(
        model["name_or_path"], "model.name_or_path"
    )
    train_path = root / "rounds" / "round01" / "data" / "train.jsonl"
    output = (
        root / "smoke" / "sft"
        if mode == "smoke"
        else root / "rounds" / "round01" / "sft"
    )

    options = {
        # 基础权重冻结，只给语言模型添加 LoRA。
        "model": str(model_path),
        "model_type": "qwen3_vl",
        "dataset": str(train_path),
        "tuner_type": "lora",
        "tuner_backend": "peft",
        "target_modules": "all-linear",
        "freeze_llm": "false",
        "freeze_vit": "true",
        "freeze_aligner": "true",
        "lora_rank": model["lora_rank"],
        "lora_alpha": model["lora_alpha"],
        "lora_dropout": 0.05,
        "lora_bias": "none",

        # 使用 PyTorch SDPA，暂不依赖 flash-attn。
        "torch_dtype": "bfloat16",
        "bf16": "true",
        "fp16": "false",
        "attn_impl": "sdpa",

        # max_steps 是 optimizer 更新次数，不是 micro batch 数。
        "per_device_train_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "learning_rate": train["learning_rate"],
        "max_steps": steps,
        "optim": "adamw_torch",
        "weight_decay": 0.01,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.05,

        # 长度、图片大小与显存控制。
        "max_length": train["max_length"],
        "max_pixels": train["max_pixels"],
        "gradient_checkpointing": str(
            train["gradient_checkpointing"]
        ).lower(),
        "vit_gradient_checkpointing": "false",

        # 只监督 assistant 回答；验证由后续独立评估执行。
        "loss_scale": "default",
        "split_dataset_ratio": 0,
        "eval_strategy": "no",
        "packing": "false",
        "padding_free": "false",

        # strict=true 让超长或坏样本报错，不静默删除或替换。
        "lazy_tokenize": "false",
        "truncation_strategy": "delete",
        "strict": "true",
        "load_from_cache_file": "false",
        "dataset_num_proc": 1,
        "dataloader_num_workers": train["num_workers"],

        # 日志、随机种子与 checkpoint。
        "seed": cfg["seed"],
        "data_seed": cfg["seed"],
        "logging_steps": 1 if mode == "smoke" else 10,
        "save_strategy": "steps",
        "save_steps": (
            steps if mode == "smoke" else min(100, steps)
        ),
        "save_total_limit": 2,
        "report_to": "none",
        "load_args": "false",
        "add_version": "false",
        "output_dir": str(output),
    }

    # 使用当前 Python 环境，不调用其他环境中的 swift。
    command = [sys.executable, "-m", "swift.cli.main", "sft"]
    for key, value in options.items():
        command.extend([f"--{key}", str(value)])

    summary = {
        "mode": mode,
        "world_size": 1,
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "global_batch_size": global_batch,
        "max_steps": steps,
        "planned_sample_exposures": steps * global_batch,
        "model_path": str(model_path),
        "train_path": str(train_path),
        "output_dir": str(output),
    }
    return command, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    parser.add_argument(
        "--mode", choices=["smoke", "round1"], default="smoke"
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    command, summary = build_command(cfg, args.mode)

    # 复用已物化、已验收的数据，不重新处理原始数据。
    train_path = Path(summary["train_path"])
    with (train_path.parent / "report.json").open(
        encoding="utf-8"
    ) as file:
        report = json.load(file)
    if report["status"] != "RECIPE_READY" or report["round"] != 1:
        raise ValueError("需要已验收的 round01 数据")
    if report["seed"] != cfg["seed"]:
        raise ValueError("数据 seed 与训练配置不一致")

    rows = sum(1 for _ in read_jsonl(train_path))
    if rows <= 0 or rows != report["total_rows"]:
        raise ValueError("训练行数与验收报告不一致")
    if (
        args.mode == "round1"
        and rows != summary["planned_sample_exposures"]
    ):
        raise ValueError(
            "round1 数据预算与 steps × global_batch 不一致，"
            "请先检查配方"
        )

    output = Path(summary["output_dir"])
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有输出：{output}")

    summary["dataset_rows"] = rows
    summary["nominal_epochs"] = (
        summary["planned_sample_exposures"] / rows
    )
    print(
        json.dumps(summary, ensure_ascii=False, indent=2),
        flush=True,
    )
    print("COMMAND:", shlex.join(command), flush=True)

    if not args.execute:
        print("PREVIEW ONLY：未加载模型、未创建输出、未启动训练。")
        return

    # 真实执行前要求只暴露一张获分配的 GPU。
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible:
        raise ValueError(
            "请通过 CUDA_VISIBLE_DEVICES 指定一张获分配的 GPU"
        )
    if version("ms-swift") != "4.5.3":
        raise RuntimeError(
            "本启动文件按 ms-swift==4.5.3 核对，请先检查环境"
        )

    # 基础检查不替代模型下载完整性验收。
    model_path = Path(summary["model_path"])
    with (model_path / "model.safetensors.index.json").open(
        encoding="utf-8"
    ) as file:
        index = json.load(file)
    for name in set(index["weight_map"].values()):
        shard = model_path / name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise FileNotFoundError(f"模型分片缺失或为空：{shard}")

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("需要恰好一张可见 CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("当前 GPU 不支持本版要求的 BF16")

    # 不继承旧的多机/多进程启动配置。
    env = os.environ.copy()
    env.pop("NPROC_PER_NODE", None)
    env.pop("NNODES", None)
    env["PYTHONUNBUFFERED"] = "1"

    output.mkdir(parents=True, exist_ok=False)
    with (output / "launch.json").open("x", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "config": cfg,
                "command": command,
                "cuda_visible_devices": visible,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
        file.write("\n")

    # 无 shell；失败不会打印完成信息，也不自动覆盖旧输出重试。
    subprocess.run(command, env=env, check=True)
    print(
        "SFT PROCESS FINISHED：下一步验收 adapter 保存与重新加载。"
    )


if __name__ == "__main__":
    main()
