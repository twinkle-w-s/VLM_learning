from __future__ import annotations

import argparse
import json
from pathlib import Path

def read_manifest(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

def get_terminal_function(program: list[dict]) -> str:
    return program[-1]["function"]#获取program的最后一步，决定问题回答什么

def classify_task(program: list[dict]) -> str:
    terminal_function = get_terminal_function(program)

    if terminal_function.startswith("query_"):
        return terminal_function.replace("query_", "")

    if terminal_function == "count":
        return "count"

    if terminal_function in {
        "greater_than",
        "less_than",
        "equal_integer",
    }:
        return "comparison"

    if terminal_function in {
        "exist",
        "equal_color",
        "equal_shape",
        "equal_material",
        "equal_size",
    }:
        return "boolean_or_attribute"

    return "other"#按照最后一步的任务指令分类

def classify_answer(answer: str) -> str:
    if answer in {"yes", "no"}:
        return "boolean"

    if answer.isdigit():
        return "integer"

    return "attribute"#按照回答类型分类

def classify_difficulty(
    program_length: int,
    relation_count: int,
) -> str:
    if program_length <= 9 and relation_count <= 1:
        return "easy"

    if program_length <= 15 and relation_count <= 2:
        return "medium"

    if program_length <= 20 and relation_count <= 3:
        return "hard"

    return "very_hard"#通过问题长度，暂时确定问题难度


def annotate_record(record: dict) -> dict:
    program = record["program"]
    relation_count = len(record["spatial_relations"])

    record["task_type"] = classify_task(program)
    record["answer_type"] = classify_answer(record["answer"])
    record["difficulty"] = classify_difficulty(
        program_length=record["program_length"],
        relation_count=relation_count,
    )

    return record#返回新的record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)

    args = parser.parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0

    with args.output_path.open("w", encoding="utf-8") as output_file:
        for record in read_manifest(args.input_path):
            annotated = annotate_record(record)

            output_file.write(
                json.dumps(annotated, ensure_ascii=False) + "\n"
            )

            written += 1

    print("written:", written)
    print("output:", args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())