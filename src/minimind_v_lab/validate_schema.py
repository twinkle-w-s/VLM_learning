from __future__ import annotations

import argparse#解析命令行参数
import io
import json
from collections import Counter#统计元素出现次数
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


REQUIRED_COLUMNS = {"conversations", "image_bytes"}

def parse_conversations(raw:Any)->tuple[list[dict[str,Any]] | None, str| None]:
    if isinstance(raw,str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return None, "invalid_conversations_json"
        
    if not isinstance(raw, list) or not raw:
        return None, "conversations_not_nonempty_list"

    for turn in raw:
        if not isinstance(turn, dict):
            return None, "conversation_turn_not_dict"

        if turn.get("role") not in {"system", "user", "assistant"}:
            return None, "invalid_role"

        if not isinstance(turn.get("content"), str):
            return None, "content_not_string"

    return raw, None

def normalize_image_bytes(raw:Any)->list[bytes]:
    if isinstance(raw, (bytes, bytearray, memoryview)):
        return [bytes(raw)]

    if isinstance(raw, list):
        result = []

        for item in raw:
            if isinstance(item, (bytes, bytearray, memoryview)):
                result.append(bytes(item))

        return result

    return []


def validate_row(row: dict[str, Any]) -> list[str]:
    errors = []

    conversations, conversation_error = parse_conversations(
        row.get("conversations")
    )

    if conversation_error:
        errors.append(conversation_error)
    else:
        has_image_token = any(
            "<image>" in turn["content"]
            for turn in conversations
            if turn["role"] != "system"
        )

        if not has_image_token:
            errors.append("missing_image_token")

        roles = [turn["role"] for turn in conversations]

        if "user" not in roles:
            errors.append("missing_user_turn")

        if "assistant" not in roles:
            errors.append("missing_assistant_turn")

    image_blobs = normalize_image_bytes(row.get("image_bytes"))

    if not image_blobs:
        errors.append("missing_image_bytes")

    for image_blob in image_blobs:
        try:
            with Image.open(io.BytesIO(image_blob)) as image:
                image.verify()
        except Exception:
            errors.append("invalid_image_bytes")
            break

    return errors

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate MiniMind-V Parquet data schema."
    )
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument("--max-rows", type=int, default=100)

    args = parser.parse_args()

    metadata = pq.read_metadata(args.parquet_path)
    available_columns = set(metadata.schema.names)
    missing_columns = REQUIRED_COLUMNS - available_columns

    if missing_columns:
        print(
            json.dumps(
                {
                    "valid": False,
                    "missing_columns": sorted(missing_columns),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 2

    table = pq.read_table(
        args.parquet_path,
        columns=sorted(REQUIRED_COLUMNS),
    )

    rows_to_check = min(args.max_rows, table.num_rows)
    error_counter = Counter()
    valid_rows = 0

    for index in range(rows_to_check):
        row = {
            column: table[column][index].as_py()
            for column in REQUIRED_COLUMNS
        }

        errors = validate_row(row)

        if errors:
            error_counter.update(errors)
        else:
            valid_rows += 1

    report = {
        "valid": len(error_counter) == 0,
        "path": str(args.parquet_path),
        "total_rows": table.num_rows,
        "rows_checked": rows_to_check,
        "valid_rows": valid_rows,
        "invalid_rows": rows_to_check - valid_rows,
        "errors": dict(error_counter),
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if valid_rows == rows_to_check else 1


if __name__ == "__main__":
    raise SystemExit(main())