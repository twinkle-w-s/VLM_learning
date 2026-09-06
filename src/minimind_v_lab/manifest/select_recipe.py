# 本脚本按照预设数据配方筛选 manifest 样本。
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

def read_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records
#data recipe
RECIPES = {
    "focused": {
        "count",
        "query_color",
        "query_material",
        "query_shape",
        "query_size",
    },
    "all": {
        "count",
        "equal_color",
        "equal_integer",
        "equal_material",
        "equal_shape",
        "equal_size",
        "exist",
        "greater_than",
        "less_than",
        "query_color",
        "query_material",
        "query_shape",
        "query_size",
    },
}

def select_records(
    records: list[dict],
    allowed_endpoints: set[str],
    seed: int,
) -> list[dict]:
    selected = []

    for record in records:
        endpoint = record["program"][-1]["function"]#取问题的最终操作

        if endpoint in allowed_endpoints:
            selected.append(record)

    random.Random(seed).shuffle(selected)

    return selected

def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_path",
        type=Path,
    )#输入manifest

    parser.add_argument(
        "output_path",
        type=Path,
    )#输出manifest

    parser.add_argument(
        "--recipe",
        choices=RECIPES,
        default="focused",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    records = read_jsonl(args.input_path)

    selected = select_records(
        records=records,
        allowed_endpoints=RECIPES[args.recipe],
        seed=args.seed,
    )

    selected = selected[:args.limit]#截断，取前500条

    args.output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in selected:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("recipe:", args.recipe)
    print("input records:", len(records))
    print("selected records:", len(selected))
    print("output:", args.output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
