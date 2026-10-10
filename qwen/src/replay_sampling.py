"""Replay 扫描与抽样。

预算内分批读取、精确去重、按组划分 train/val。
采用固定 seed 和蓄水池抽样，保留原始 row group/offset 便于追溯。
本模块只构建内存候选池，不导出图片或训练文件。
"""

from __future__ import annotations

import random
from collections import Counter
from itertools import islice

import pyarrow.parquet as pq

from data_common import reservoir_add, resolve_path
from replay_records import (
    iter_replay_rows,
    parse_replay_record,
    replay_hash,
)


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
        raise ValueError(
            f"replay 缺字段: {sorted(missing)}"
        )

    fraction = options["val_fraction"]
    budget = options["max_scan_rows"]
    if not 0 < fraction < 1 or budget <= 0:
        raise ValueError(
            "val_fraction 必须在 (0,1)，max_scan_rows 必须大于 0"
        )

    targets = {
        "train_vl": (
            cfg["data"]["train_pool_size"]["replay_vl"]
        ),
        "val_vl": (
            cfg["evaluation"]["replay_vl_val_size"]
        ),
        "qa_val_vl": (
            cfg["evaluation"]["replay_vl_val_size"]
        ),
        "train_text": (
            cfg["data"]["train_pool_size"]["replay_text"]
        ),
        "val_text": (
            cfg["evaluation"]["replay_text_val_size"]
        ),
    }
    if any(n <= 0 for n in targets.values()):
        raise ValueError(
            "replay 候选池和验证集目标数量必须大于 0"
        )

    pools = {name: [] for name in targets}
    rngs = {
        name: random.Random(
            cfg["seed"] + 200 + index
        )
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
                cfg["evaluation"][
                    "replay_short_answer_max_words"
                ],
            )
        except (ValueError, TypeError, OSError):
            item, reason = None, "parse_or_image_error"

        if (
            item is not None
            and item["sample_id"] in seen_ids
        ):
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
                "val"
                if value / (1 << 64) < fraction
                else "train"
            )
            item["split"] = split
            modality = (
                "vl"
                if item["source"] == "replay_vl"
                else "text"
            )
            names = [f"{split}_{modality}"]
            if (
                split == "val"
                and item["short_answer_candidate"]
            ):
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
                "REPLAY SCAN:",
                dict(counts),
                flush=True,
            )

    return pools, {
        "input_path": str(path),
        "input_bytes": path.stat().st_size,
        "input_rows": parquet.metadata.num_rows,
        "row_groups": parquet.num_row_groups,
        "scan_budget": budget,
        "scan_counts": dict(counts),
        "candidate_counts": {
            name: len(rows)
            for name, rows in pools.items()
        },
    }