"""Replay 单条记录解析。

负责对话校验、模态识别、样本与分组 ID，并提供分批行迭代。
短答案标记仅是候选，不是已审核的评估标签。
"""

from __future__ import annotations

import hashlib
import io
import json
import re

from PIL import Image


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
    if any(
        not isinstance(t.get("content"), str)
        for t in turns
    ):
        return None, "invalid_content"
    if any(
        t.get("tools")
        or t.get("tool_calls")
        or t.get("functions")
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
        t["content"]
        for t in messages
        if t["role"] == "user"
    )
    answer = messages[-1]["content"]
    if not user or not answer or "<image>" in answer:
        return None, "invalid_qa"

    images = row["image_bytes"]
    images = (
        images if isinstance(images, list) else [images]
    )
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
        t["content"].count("<image>")
        for t in messages
    )
    if (
        marker_count == 1
        and "<image>" in user
        and blob
        and not placeholder
    ):
        source = "replay_vl"
        image_id = replay_hash(blob)
        group_id = "image:" + image_id
    elif marker_count == 0 and (
        blob is None or placeholder
    ):
        source = "replay_text"
        image_id = None
        prompt = json.dumps(
            messages[:-1],
            ensure_ascii=False,
            sort_keys=True,
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