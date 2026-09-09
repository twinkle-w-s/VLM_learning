# 这个脚本根据任务配比构造第一版全量训练配方，并保持样本可追溯。

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


TASK_ORDER = [
    "color",
    "count",
    "material",
    "shape",
    "size",
    "comparison",
    "boolean_or_attribute",
]

# 第一版全量配方保留所有空间任务，并提高首轮表现较弱的 shape/material 占比。
TASK_WEIGHTS = {
    "color": 0.13,
    "count": 0.13,
    "material": 0.14,
    "shape": 0.18,
    "size": 0.10,
    "comparison": 0.14,
    "boolean_or_attribute": 0.18,
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def build_quotas(total: int) -> dict[str, int]:
    quotas = {
        task_type: int(total * TASK_WEIGHTS[task_type])
        for task_type in TASK_ORDER
    }
    remainder = total - sum(quotas.values())
    for task_type in TASK_ORDER[:remainder]:
        quotas[task_type] += 1

    return quotas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="最多保留多少条；0 表示保留全部记录",
    )
    args = parser.parse_args()

    source_counts = Counter(
        record["task_type"]
        for record in read_jsonl(args.input_path)
        if record["task_type"] in TASK_WEIGHTS
    )
    eligible_total = sum(source_counts.values())
    output_total = args.limit if args.limit > 0 else eligible_total
    quotas = build_quotas(output_total)

    missing_tasks = [
        task_type
        for task_type in TASK_ORDER
        if source_counts[task_type] == 0 and quotas[task_type] > 0
    ]
    if missing_tasks:
        raise SystemExit(f"missing task groups: {missing_tasks}")

    base_repeats = {
        task_type: quotas[task_type] // source_counts[task_type]
        for task_type in TASK_ORDER
    }
    extra_remaining = {
        task_type: quotas[task_type] % source_counts[task_type]
        for task_type in TASK_ORDER
    }
    source_remaining = dict(source_counts)
    output_counts = Counter()
    rng = random.Random(args.seed)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as output_file:
        for record in read_jsonl(args.input_path):
            task_type = record["task_type"]
            if task_type not in TASK_WEIGHTS:
                continue

            serialized = json.dumps(record, ensure_ascii=False) + "\n"
            repeat_count = base_repeats[task_type]

            if rng.random() < (
                extra_remaining[task_type]
                / source_remaining[task_type]
            ):
                repeat_count += 1
                extra_remaining[task_type] -= 1

            source_remaining[task_type] -= 1
            for _ in range(repeat_count):
                output_file.write(serialized)
                output_counts[task_type] += 1

    if output_counts != Counter(quotas):
        raise RuntimeError(
            f"recipe count mismatch: output={output_counts}, quotas={quotas}"
        )

    print("eligible records:", eligible_total)
    print("source task counts:", dict(source_counts))
    print("target task counts:", dict(output_counts))
    print("output records:", sum(output_counts.values()))
    print("task weights:", TASK_WEIGHTS)
    print("output:", args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
