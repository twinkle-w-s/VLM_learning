from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

DATA_ROOT = (
    Path("/data")
    / os.environ["USER"]
    / "vlm_learning"
    / "CLEVR_v1.0"
)

DEFAULT_IMAGE_ROOT = DATA_ROOT / "images" / "train"

REQUIRED_FIELDS = {
    "question_index",
    "image_index",
    "image_filename",
    "question",
    "answer",
    "program",
    "spatial_relations",
    "program_length",
}

def read_jsonl(path:Path):
    with path.open("r",encoding="utf=8") as file:
        for line_number,line in enumerate(file,start=1):
            if not line.strip():
                continue
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                yield line_number,None,"invalid_json"
                continue
            yield line_number,record,None#返回一条记录

def validate_record(
    record: dict,
    image_root: Path,#输入一条记录和对应图片路径
) -> list[str]:
    errors = []

    missing_fields = REQUIRED_FIELDS - set(record)

    if missing_fields:
        errors.append(
            "missing_fields:"
            + ",".join(sorted(missing_fields))
        )
        return errors

    if not isinstance(record["question"], str):
        errors.append("question_not_string")

    if not isinstance(record["answer"], str):
        errors.append("answer_not_string")

    if not isinstance(record["program"], list):
        errors.append("program_not_list")

    if not isinstance(record["spatial_relations"], list):
        errors.append("spatial_relations_not_list")

    image_path = image_root / record["image_filename"]

    if not image_path.exists():
        errors.append("image_not_found")

    return errors#检验record中应该存在的所有字段，不要出现缺失

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=Path)#manifest路径输入
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
    )

    args = parser.parse_args()

    error_counter = Counter()
    question_ids = set()
    checked = 0
    valid = 0


    for line_number, record, parse_error in read_jsonl(#一一验证record
        args.manifest_path
    ):
        checked += 1

        if parse_error:
            error_counter[parse_error] += 1
            continue

        if record["question_index"] in question_ids:
            error_counter["duplicate_question_index"] += 1

        question_ids.add(record["question_index"])

        errors = validate_record(
            record,
            args.image_root,
        )

        if errors:
            error_counter.update(errors)
        else:
            valid += 1

        #输出报告

    report = {
        "manifest_path": str(args.manifest_path),
        "image_root": str(args.image_root),
        "checked": checked,
        "valid": valid,
        "invalid": checked - valid,
        "errors": dict(error_counter),
    }

    print(json.dumps(
        report,
        ensure_ascii=False,
        indent=2,
    ))

    return 0 if not error_counter else 1

if __name__ == "__main__":
    raise SystemExit(main())
        