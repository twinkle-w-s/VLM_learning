"""检查 Qwen MVP 的原始数据；--inspect 不写输出文件。"""
from __future__ import annotations
import hashlib
import re
from itertools import islice

import random
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



def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def reservoir_add(pool: list, item: dict, seen: int, limit, rng) -> None:
    if limit is None or len(pool) < limit:
        pool.append(item)
    else:
        index = rng.randrange(seen)
        if index < limit:
            pool[index] = item


def make_clevr_record(row: dict, split: str, image_root: Path) -> dict:
    question = row["question"].strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        raise ValueError("CLEVR 出现空问题或答案")
    return {
        "sample_id": f"clevr:CLEVR_train:{row['question_index']}",
        "source": "clevr",
        "split": split,
        "image_id": row["image_filename"],
        "task_type": row["task_type"],
        "difficulty": row["difficulty"],
        "spatial_relations": row.get("spatial_relations", []),
        "answer": answer,
        "scorable": True,
        "messages": [
            {
                "role": "user",
                "content": (
                    "<image>\n" + question
                    + "\nAnswer with a single word or number."
                ),
            },
            {"role": "assistant", "content": answer},
        ],
        "images": [str(image_root / row["image_filename"])],
    }

def select_clevr(cfg: dict):
    paths = cfg["data"]["clevr"]
    image_root = resolve_path(paths["image_root"], "clevr.image_root").resolve()
    limits = {
        "train": cfg["data"]["train_pool_size"]["clevr"],
        "val": cfg["evaluation"]["clevr_val_size"],
        "test": None,
    }
    pools, inputs, image_sets = {}, {}, {}
    seen_ids = set()

    for split_index, split in enumerate(("train", "val", "test")):
        path = resolve_path(paths[split], f"clevr.{split}")
        limit = limits[split]
        if limit is not None and limit <= 0:
            raise ValueError(f"{split} 抽样数量必须大于 0")
        rng = random.Random(cfg["seed"] + split_index)
        pool, names, count = [], set(), 0

        for count, row in enumerate(read_jsonl(path), start=1):
            item = make_clevr_record(row, split, image_root)
            if item["sample_id"] in seen_ids:
                raise ValueError(f"源清单出现重复 sample_id: {item['sample_id']}")
            seen_ids.add(item["sample_id"])
            names.add(item["image_id"])
            reservoir_add(pool, item, count, limit, rng)
            if count % 50000 == 0:
                print(f"CLEVR {split}: scanned={count}", flush=True)

        if not pool or (limit is not None and len(pool) < limit):
            raise ValueError(f"{split} 样本不足: available={count}, target={limit}")
        pools[split] = pool
        image_sets[split] = names
        inputs[split] = {
            "path": str(path),
            "rows": count,
            "unique_images": len(names),
        }
        print(f"CLEVR {split}: selected={len(pool)}, scanned={count}", flush=True)

    overlaps = {
        f"{a}-{b}": len(image_sets[a] & image_sets[b])
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    if any(overlaps.values()):
        raise ValueError(f"CLEVR 原始 split 存在图片泄漏: {overlaps}")
    return pools, {"inputs": inputs, "image_overlap": overlaps}
def assign_clevr_buckets(pools: dict, cfg: dict) -> dict:
    fields = cfg["data"]["bucket_fields"]
    minimum = cfg["data"]["min_bucket_train"]
    if minimum <= 0:
        raise ValueError("min_bucket_train 必须大于 0")

    def raw_bucket(item):
        return "|".join(str(item[field]) for field in fields)

    train = pools["train"]
    raw_counts = Counter(raw_bucket(item) for item in train)
    task_counts = Counter(item["task_type"] for item in train)
    merged_tasks = {
        item["task_type"]
        for item in train
        if raw_counts[raw_bucket(item)] < minimum
    }

    for records in pools.values():
        for item in records:
            task = item["task_type"]
            raw = raw_bucket(item)
            if task_counts[task] < minimum:
                item["bucket"] = "other"
            elif task in merged_tasks:
                item["bucket"] = f"task:{task}"
            else:
                item["bucket"] = raw if raw in raw_counts else "other"

    return {
        "fields": fields,
        "merged_tasks": sorted(merged_tasks),
        "train_counts": dict(Counter(item["bucket"] for item in train)),
    }
def write_jsonl(path: Path, records: list) -> None:
    with path.open("x", encoding="utf-8") as file:
        for item in records:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_clevr(cfg: dict) -> None:
    output = resolve_path(cfg["output_root"], "output_root") / "manifests" / "clevr"
    if output.exists():
        raise FileExistsError(f"输出目录已存在，拒绝覆盖: {output}")

    pools, stats = select_clevr(cfg)
    bucket_stats = assign_clevr_buckets(pools, cfg)
    probe_size = cfg["evaluation"]["probe_size"]["clevr"]
    if not 0 < probe_size <= len(pools["train"]):
        raise ValueError("CLEVR probe 数量必须在 1 和训练池数量之间")
    probe = random.Random(cfg["seed"] + 100).sample(pools["train"], probe_size)

    image_paths = {
        item["images"][0]
        for records in pools.values()
        for item in records
    }
    for index, image_path in enumerate(sorted(image_paths), start=1):
        with Image.open(image_path) as image:
            image.load()
        if index % 1000 == 0:
            print(f"CLEVR decoded unique images: {index}", flush=True)

    report = {
        "status": "CLEVR_READY_REPLAY_PENDING",
        "seed": cfg["seed"],
        **stats,
        "output_counts": {split: len(rows) for split, rows in pools.items()},
        "probe_count": len(probe),
        "checked_unique_images": len(image_paths),
        "buckets": bucket_stats,
    }
    output.mkdir(parents=True, exist_ok=False)
    for split, records in pools.items():
        write_jsonl(output / f"{split}.jsonl", records)
    write_jsonl(output / "probe.jsonl", probe)
    with (output / "report.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("CLEVR PREPARE: PASS")
    print("OUTPUT:", output)
    print("REPLAY_PENDING: 尚未准备 15% 图文和 5% 文字回放。")

#将minifest格式的转化为qwen格式的SFT数据
def replay_hash(value) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def parse_replay_record(row: dict, max_words: int):
    turns = row["conversations"]
    if isinstance(turns, str):
        turns = json.loads(turns)
    if not isinstance(turns, list) or any(
        not isinstance(t, dict) for t in turns
    ):
        return None, "invalid_dialogue"

    roles = [t.get("role") for t in turns]
    if roles not in (
        ["user", "assistant"],
        ["system", "user", "assistant"],
    ):
        return None, "unsupported_dialogue"
    if any(not isinstance(t.get("content"), str) for t in turns):
        return None, "invalid_content"
    if any(
        t.get("tools") or t.get("tool_calls") or t.get("functions")
        for t in turns
    ):
        return None, "tool_sample"

    messages = [
        {
            "role": t["role"],
            "content": t["content"].replace(
                "<|image_pad|>", "<image>"
            ).strip(),
        }
        for t in turns
    ]
    user = next(
        t["content"] for t in messages if t["role"] == "user"
    )
    answer = messages[-1]["content"]
    if not user or not answer or "<image>" in answer:
        return None, "invalid_qa"

    images = row["image_bytes"]
    images = images if isinstance(images, list) else [images]
    images = [blob for blob in images if blob]
    if len(images) > 1:
        return None, "multi_image"

    blob = images[0] if images else None
    placeholder = False
    if blob:
        with Image.open(io.BytesIO(blob)) as image:
            placeholder = (
                image.size == (8, 8)
                and image.convert("RGB").getextrema()
                == ((0, 0), (0, 0), (0, 0))
            )

    marker_count = sum(
        t["content"].count("<image>") for t in messages
    )
    if marker_count == 1 and "<image>" in user and blob and not placeholder:
        source = "replay_vl"
        image_id = replay_hash(blob)
        group_id = "image:" + image_id
    elif marker_count == 0 and (blob is None or placeholder):
        source = "replay_text"
        image_id = None
        prompt = json.dumps(
            messages[:-1], ensure_ascii=False, sort_keys=True
        )
        group_id = "text:" + replay_hash(prompt)
        blob = None
    else:
        return None, "ambiguous_modality"

    payload = json.dumps(
        messages, ensure_ascii=False, sort_keys=True
    )
    short = (
        source == "replay_vl"
        and len(answer.split()) <= max_words
        and re.fullmatch(
            r"(?:[a-z]+(?:[ -][a-z]+){0,2}|\d+(?:\.\d+)?)",
            answer.casefold().strip("."),
        ) is not None
    )
    return {
        "sample_id": "minimind_v:" + replay_hash(
            group_id + "\n" + payload
        ),
        "source": source,
        "group_id": group_id,
        "image_id": image_id,
        "bucket": source,
        "answer": answer,
        "scorable": bool(short),
        "short_answer_candidate": bool(short),
        "messages": messages,
        "_image_bytes": blob,
    }, "accepted"

def iter_replay_rows(parquet, groups):
    for group in groups:
        offset = 0
        for batch in parquet.iter_batches(
            batch_size=128,
            row_groups=[group],
            columns=["conversations", "image_bytes"],
        ):
            for row in batch.to_pylist():
                yield group, offset, row
                offset += 1


def collect_replay(cfg: dict):
    options = cfg["data"]["replay"]
    if options.get("text_path"):
        raise ValueError(
            "本轮仅从 vl_path 提取文字，不读取额外 text_path"
        )

    path = resolve_path(
        options["vl_path"], "replay.vl_path"
    ).resolve()
    parquet = pq.ParquetFile(path)
    missing = {
        "conversations", "image_bytes"
    } - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"replay 缺字段: {sorted(missing)}")

    fraction = options["val_fraction"]
    budget = options["max_scan_rows"]
    if not 0 < fraction < 1 or budget <= 0:
        raise ValueError(
            "val_fraction 必须在 (0,1)，max_scan_rows 必须大于 0"
        )

    targets = {
        "train_vl": cfg["data"]["train_pool_size"]["replay_vl"],
        "val_vl": cfg["evaluation"]["replay_vl_val_size"],
        "qa_val_vl": cfg["evaluation"]["replay_vl_val_size"],
        "train_text": cfg["data"]["train_pool_size"]["replay_text"],
        "val_text": cfg["evaluation"]["replay_text_val_size"],
    }
    if any(n <= 0 for n in targets.values()):
        raise ValueError(
            "replay 候选池和验证集目标数量必须大于 0"
        )

    pools = {name: [] for name in targets}
    rngs = {
        name: random.Random(cfg["seed"] + 200 + index)
        for index, name in enumerate(targets)
    }
    groups = list(range(parquet.num_row_groups))
    random.Random(cfg["seed"] + 201).shuffle(groups)
    counts, seen_ids = Counter(), set()

    for group, offset, raw in islice(
        iter_replay_rows(parquet, groups), budget
    ):
        counts["scanned"] += 1
        try:
            item, reason = parse_replay_record(
                raw,
                cfg["evaluation"]["replay_short_answer_max_words"],
            )
        except (ValueError, TypeError, OSError):
            item, reason = None, "parse_or_image_error"

        if item is not None and item["sample_id"] in seen_ids:
            item, reason = None, "duplicate"
        counts[reason] += 1

        if item is not None:
            seen_ids.add(item["sample_id"])
            item["origin"] = {
                "row_group": group,
                "offset": offset,
            }

            value = int(
                replay_hash(
                    f"{cfg['seed']}:{item['group_id']}"
                )[:16],
                16,
            )
            split = (
                "val" if value / (1 << 64) < fraction
                else "train"
            )
            item["split"] = split
            modality = (
                "vl" if item["source"] == "replay_vl"
                else "text"
            )
            names = [f"{split}_{modality}"]
            if split == "val" and item["short_answer_candidate"]:
                names.append("qa_val_vl")

            for name in names:
                counts[name + "_seen"] += 1
                reservoir_add(
                    pools[name],
                    item,
                    counts[name + "_seen"],
                    targets[name],
                    rngs[name],
                )

        if counts["scanned"] % 25000 == 0:
            print(
                "REPLAY SCAN:", dict(counts), flush=True
            )

    return pools, {
        "input_path": str(path),
        "input_bytes": path.stat().st_size,
        "input_rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "scan_budget": budget,
        "scan_counts": dict(counts),
        "candidate_counts": {
            name: len(rows) for name, rows in pools.items()
        },
    }

def finalize_replay(pools: dict, cfg: dict, output: Path):
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
        for key, value in cfg["recipe"]["source_bounds"].items()
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
        raise ValueError(f"replay 内部组泄漏: {overlaps}")

    for index, modality in enumerate(("vl", "text")):
        number = cfg["evaluation"]["probe_size"][
            f"replay_{modality}"
        ]
        if modality == "text" and not use_text:
            number = 0
        if not 0 <= number <= len(pools[f"train_{modality}"]):
            raise ValueError(f"{modality} probe 数量不合法")

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
                "REPLAY DECODED IMAGES:", index, flush=True
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
        write_jsonl(output / f"{name}.jsonl", rows)

    return use_text, ratios, bounds, overlaps, len(blobs)

def prepare_replay(cfg: dict) -> None:
    root = (
        resolve_path(cfg["output_root"], "output_root").resolve()
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
        raise ValueError("CLEVR 与 replay seed 不一致")

    pools, stats = collect_replay(cfg)
    use_text, ratios, bounds, overlaps, image_count = (
        finalize_replay(pools, cfg, output)
    )
    reviewed = bool(
        cfg["data"]["replay"].get("short_qa_reviewed", False)
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
            "clevr": str(root / "clevr" / "train.jsonl"),
            "replay_vl": str(output / "train_vl.jsonl"),
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
            name: len(rows) for name, rows in pools.items()
        },
        "group_overlap_within_replay": overlaps,
        "unique_exported_images": image_count,
        "text_enabled": use_text,
        "text_fallback_reason": (
            None if use_text
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
                content, file, ensure_ascii=False, indent=2
            )
            file.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("REPLAY PREPARE: PASS")
    print("OUTPUT:", output)
    if not recipe["outer_feedback_enabled"]:
        print(
            "外层准确率反馈暂关闭；内层 CLEVR 飞轮仍可运行。"
        )
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("qwen/configs/mvp.yaml"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--inspect", action="store_true")
    mode.add_argument("--prepare-clevr", action="store_true")
    mode.add_argument("--prepare-replay", action="store_true")
    args = parser.parse_args()

    with args.config.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    if not isinstance(cfg, dict):
        raise ValueError("配置必须是 YAML 字典")

    if args.inspect:
        inspect_clevr(cfg)
        inspect_replay(cfg)
        print("INSPECT COMPLETED — 未写出训练数据。")
    elif args.prepare_clevr:
        prepare_clevr(cfg)
    else:
        prepare_replay(cfg)
if __name__ == "__main__":
    main()