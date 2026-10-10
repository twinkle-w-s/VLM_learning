"""Replay 验收与导出。

检查候选数量、组泄漏、图片解码，生成固定 probe。
导出入选图片、候选池、初始配方和验收报告，拒绝覆盖已有输出。
预算内文字不足时把文字份额转给图文。
"""

from __future__ import annotations

import io
import json
import random
from pathlib import Path

from PIL import Image

from data_common import resolve_path, write_jsonl
from replay_sampling import collect_replay


def finalize_replay(
    pools: dict,
    cfg: dict,
    output: Path,
):
    for name, target in (
        (
            "train_vl",
            cfg["data"]["train_pool_size"]["replay_vl"],
        ),
        (
            "val_vl",
            cfg["evaluation"]["replay_vl_val_size"],
        ),
    ):
        if len(pools[name]) < target:
            raise ValueError(
                f"{name} 不足: {len(pools[name])}/{target}"
            )

    use_text = (
        len(pools["train_text"])
        == cfg["data"]["train_pool_size"]["replay_text"]
        and len(pools["val_text"])
        == cfg["evaluation"]["replay_text_val_size"]
    )
    ratios = dict(cfg["recipe"]["source_ratios"])
    bounds = {
        key: list(value)
        for key, value
        in cfg["recipe"]["source_bounds"].items()
    }
    if not use_text:
        pools["train_text"], pools["val_text"] = [], []
        transferred = ratios["replay_text"]
        ratios["replay_vl"] += transferred
        ratios["replay_text"] = 0.0
        bounds["replay_vl"] = [
            value + transferred
            for value in bounds["replay_vl"]
        ]
        bounds["replay_text"] = [0.0, 0.0]

    overlaps = {}
    for modality in ("vl", "text"):
        train_groups = {
            item["group_id"]
            for item in pools[f"train_{modality}"]
        }
        val = list(pools[f"val_{modality}"])
        if modality == "vl":
            val += pools["qa_val_vl"]
        overlaps[modality] = len(
            train_groups
            & {item["group_id"] for item in val}
        )
    if any(overlaps.values()):
        raise ValueError(
            f"replay 内部组泄漏: {overlaps}"
        )

    for index, modality in enumerate(("vl", "text")):
        number = cfg["evaluation"]["probe_size"][
            f"replay_{modality}"
        ]
        if modality == "text" and not use_text:
            number = 0
        if not 0 <= number <= len(
            pools[f"train_{modality}"]
        ):
            raise ValueError(
                f"{modality} probe 数量不合法"
            )

        pools[f"probe_{modality}"] = random.Random(
            cfg["seed"] + 300 + index
        ).sample(
            pools[f"train_{modality}"], number
        )

    blobs = {
        item["image_id"]: item["_image_bytes"]
        for rows in pools.values()
        for item in rows
        if item["_image_bytes"]
    }
    image_paths = {}
    for index, (image_id, blob) in enumerate(
        blobs.items(), start=1
    ):
        with Image.open(io.BytesIO(blob)) as image:
            image.load()
            suffix = {
                "JPEG": ".jpg",
                "PNG": ".png",
                "WEBP": ".webp",
            }.get(image.format)

        if suffix is None:
            raise ValueError("入选图片格式不支持")
        image_paths[image_id] = str(
            output / "images" / (image_id + suffix)
        )
        if index % 1000 == 0:
            print(
                "REPLAY DECODED IMAGES:",
                index,
                flush=True,
            )

    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    for image_id, blob in blobs.items():
        with Path(image_paths[image_id]).open("xb") as file:
            file.write(blob)

    for name, rows in pools.items():
        for item in rows:
            if item["image_id"] is not None:
                item["images"] = [
                    image_paths[item["image_id"]]
                ]
            item.pop("_image_bytes", None)
        write_jsonl(
            output / f"{name}.jsonl", rows
        )

    return (
        use_text,
        ratios,
        bounds,
        overlaps,
        len(blobs),
    )


def prepare_replay(cfg: dict) -> None:
    root = (
        resolve_path(
            cfg["output_root"], "output_root"
        ).resolve()
        / "manifests"
    )
    output = root / "replay"
    if output.exists():
        raise FileExistsError(
            f"输出目录已存在，拒绝覆盖: {output}"
        )

    with (root / "clevr" / "report.json").open(
        encoding="utf-8"
    ) as file:
        clevr_report = json.load(file)
    if clevr_report["seed"] != cfg["seed"]:
        raise ValueError(
            "CLEVR 与 replay seed 不一致"
        )

    pools, stats = collect_replay(cfg)
    use_text, ratios, bounds, overlaps, image_count = (
        finalize_replay(pools, cfg, output)
    )
    reviewed = bool(
        cfg["data"]["replay"].get(
            "short_qa_reviewed", False
        )
    )
    accuracy_available = (
        len(pools["qa_val_vl"])
        >= cfg["flywheel"]["min_source_val"]
    )

    recipe = {
        "version": 1,
        "round": 1,
        "seed": cfg["seed"],
        "source_ratios": ratios,
        "source_bounds": bounds,
        "max_repeats_per_sample": (
            cfg["recipe"]["max_repeats_per_sample"]
        ),
        "pool_paths": {
            "clevr": str(
                root / "clevr" / "train.jsonl"
            ),
            "replay_vl": str(
                output / "train_vl.jsonl"
            ),
            "replay_text": (
                str(output / "train_text.jsonl")
                if use_text else None
            ),
        },
        "clevr_bucket_counts": (
            clevr_report["buckets"]["train_counts"]
        ),
        "outer_feedback_enabled": (
            reviewed and accuracy_available
        ),
    }
    report = {
        "status": "REPLAY_READY",
        **stats,
        "output_counts": {
            name: len(rows)
            for name, rows in pools.items()
        },
        "group_overlap_within_replay": overlaps,
        "unique_exported_images": image_count,
        "text_enabled": use_text,
        "text_fallback_reason": (
            None
            if use_text
            else "insufficient_within_scan_budget"
        ),
        "effective_source_ratios": ratios,
        "effective_source_bounds": bounds,
        "short_qa_reviewed": reviewed,
        "outer_feedback_enabled": (
            recipe["outer_feedback_enabled"]
        ),
        "dedup_scope": (
            "exact bytes for images; "
            "exact normalized dialogue for samples"
        ),
    }

    for name, content in (
        ("recipe_initial.json", recipe),
        ("report.json", report),
    ):
        with (output / name).open(
            "x", encoding="utf-8"
        ) as file:
            json.dump(
                content,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

    print(json.dumps(
        report, ensure_ascii=False, indent=2
    ))
    print("REPLAY PREPARE: PASS")
    print("OUTPUT:", output)
    if not recipe["outer_feedback_enabled"]:
        print(
            "外层准确率反馈暂关闭；内层 CLEVR 飞轮仍可运行。"
        )