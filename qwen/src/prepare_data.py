"""检查 Qwen MVP 的原始数据；--inspect 不写输出文件。"""

from __future__ import annotations

import argparse
import io
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
import yaml
from PIL import Image


def resolve_path(value, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 尚未填写")
    expanded = os.path.expanduser(os.path.expandvars(value))
    if "${" in expanded:
        raise ValueError(f"{name} 含未展开的环境变量: {expanded}")
    return Path(expanded)

def inspect_clevr(cfg: dict) -> None:
    paths = cfg["data"]["clevr"]
    
    image_root = resolve_path(paths["image_root"], "clevr.image_root")
    required = {"question", "answer", "image_filename", "task_type", "difficulty"}

    for split in ("train", "val", "test"):
        path = resolve_path(paths[split], f"clevr.{split}")
        if split == "test" and not path.exists():
            print(f"WARNING: test 缺失，本次 inspect 跳过: {path}")
            continue
        with path.open(encoding="utf-8") as file:
            record = next(
                (json.loads(line) for line in file if line.strip()), None
            )
        if record is None:
            raise ValueError(f"CLEVR {split} 文件为空: {path}")
        missing = required - record.keys()
        if missing:
            raise ValueError(f"CLEVR {split} 缺字段: {sorted(missing)}")

        image_path = image_root / record["image_filename"]
        with Image.open(image_path) as image:
            image.load()
            image_size = image.size

        print(json.dumps({
            "source": "clevr",
            "split": split,
            "keys": sorted(record),
            "question": record["question"],
            "answer": record["answer"],
            "task_type": record["task_type"],
            "difficulty": record["difficulty"],
            "image_path": str(image_path),
            "image_size": image_size,
        }, ensure_ascii=False))

def summarize_replay_row(row: dict, max_words: int) -> dict:
    conversations = row["conversations"]
    if isinstance(conversations, str):
        conversations = json.loads(conversations)
    if not isinstance(conversations, list) or not conversations:
        raise ValueError("conversations 不是非空列表")

    if any(
        not isinstance(turn, dict)
        or not isinstance(turn.get("role"), str)
        or not isinstance(turn.get("content"), str)
        for turn in conversations
    ):
        raise ValueError("对话缺少字符串 role/content")

    roles = [turn["role"] for turn in conversations]
    has_marker = any(
        "<image>" in turn["content"] or "<|image_pad|>" in turn["content"]
        for turn in conversations
    )
    assistant = next(
        (turn["content"] for turn in conversations if turn["role"] == "assistant"),
        "",
    ).strip()

    raw_images = row["image_bytes"]
    images = raw_images if isinstance(raw_images, list) else [raw_images]
    images = [value for value in images if value is not None]
    placeholder = False
    image_size = None

    if images:
        with Image.open(io.BytesIO(images[0])) as image:
            image.load()
            image_size = image.size
            placeholder = (
                image.size == (8, 8)
                and image.convert("RGB").getextrema() == ((0, 0), (0, 0), (0, 0))
            )

    if len(images) > 1:
        kind = "multi_image"
    elif has_marker and images and not placeholder:
        kind = "image_candidate"
    elif not has_marker and (not images or placeholder):
        kind = "text_candidate"
    else:
        kind = "ambiguous"

    return {
        "kind": kind,
        "roles": roles,
        "single_turn": roles in (
            ["user", "assistant"],
            ["system", "user", "assistant"],
        ),
        "has_image_marker": has_marker,
        "image_count": len(images),
        "image_size": image_size,
        "black_8x8_placeholder": placeholder,
        "question_preview": next(
            (turn["content"] for turn in conversations if turn["role"] == "user"),
            "",
        )[:180],
        "answer_preview": assistant[:180],
        "answer_chars": len(assistant),
        "short_answer_candidate": (
            kind == "image_candidate"
            and bool(assistant)
            and assistant.isascii()
            and len(assistant.split()) <= max_words
        ),
    }

def inspect_replay(cfg: dict) -> None:
    replay = cfg["data"]["replay"]
    path = resolve_path(replay["vl_path"], "replay.vl_path")
    
    if path.suffix.lower() != ".parquet":
        raise ValueError("本轮检查入口只支持方案 A 的 MiniMind-V Parquet")

    parquet = pq.ParquetFile(path)
    print("REPLAY PATH:", path)
    print("REPLAY SCHEMA:")
    print(parquet.schema_arrow)
    print("REPLAY ROWS:", parquet.metadata.num_rows)
    print("REPLAY ROW GROUPS:", parquet.num_row_groups)

    required = {"conversations", "image_bytes"}
    missing = required - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"replay 缺字段: {sorted(missing)}")
    if parquet.num_row_groups == 0:
        raise ValueError("replay 没有 row group")

    groups = sorted({
        index * (parquet.num_row_groups - 1) // 7 for index in range(8)
    })
    print("SAMPLED ROW GROUPS:", groups)

    counts = Counter()
    shown = Counter()
    max_words = cfg["evaluation"]["replay_short_answer_max_words"]

    for group in groups:
        batch = next(parquet.iter_batches(
            batch_size=32,
            row_groups=[group],
            columns=["conversations", "image_bytes"],
        ), None)
        if batch is None:
            continue

        for row_index, row in enumerate(batch.to_pylist()):
            counts["sampled_rows"] += 1
            try:
                summary = summarize_replay_row(row, max_words)
            except (ValueError, TypeError, OSError) as error:
                counts["invalid_rows"] += 1
                if counts["invalid_rows"] <= 2:
                    print("INVALID SAMPLE:", group, row_index, str(error))
                continue

            kind = summary["kind"]
            counts[kind] += 1
            if summary["single_turn"]:
                counts["single_turn_rows"] += 1
            if summary["short_answer_candidate"]:
                counts["short_answer_candidates"] += 1

            if shown[kind] < 2:
                shown[kind] += 1
                print(json.dumps({
                    "row_group": group,
                    "offset_in_group": row_index,
                    **summary,
                }, ensure_ascii=False))

    print("SAMPLE COUNTS:", json.dumps(dict(counts), ensure_ascii=False))
    print("NOTE: 候选分类用于检查，不是最终质量标签或全库分布统计。")
    if replay.get("text_path"):
        print("NOTE: 配置另有 text_path，本次尚未检查该文本文件。")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("qwen/configs/mvp.yaml"))
    parser.add_argument("--inspect", action="store_true")
    args = parser.parse_args()
    if not args.inspect:
        parser.error("本版仅实现 --inspect；尚未实现数据转换")

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 YAML 字典")

    inspect_clevr(cfg)
    inspect_replay(cfg)
    print("INSPECT COMPLETED — 未写出训练数据，尚未执行完整数据验收。")


if __name__ == "__main__":
    main()