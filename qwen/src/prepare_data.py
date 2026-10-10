"""Qwen MVP 数据准备命令行入口。

保留 --inspect、--prepare-clevr、--prepare-replay 及原有输出格式。
从项目根目录执行；新代码建议从 data_common 导入公共工具。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

# 兼容此前 materialize_recipe.py 从 prepare_data 导入公共工具的写法。
from data_common import read_jsonl, resolve_path, write_jsonl
from inspect_data import inspect_clevr, inspect_replay
from prepare_clevr import prepare_clevr
from prepare_replay import prepare_replay


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(
        required=True
    )
    mode.add_argument(
        "--inspect", action="store_true"
    )
    mode.add_argument(
        "--prepare-clevr", action="store_true"
    )
    mode.add_argument(
        "--prepare-replay", action="store_true"
    )
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 YAML 字典")

    if args.inspect:
        inspect_clevr(cfg)
        inspect_replay(cfg)
        print(
            "INSPECT COMPLETED — 未写出训练数据。"
        )
    elif args.prepare_clevr:
        prepare_clevr(cfg)
    else:
        prepare_replay(cfg)


if __name__ == "__main__":
    main()