from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

def read_manifest(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest_path", type=Path)

    args = parser.parse_args()

    endpoint_counter = Counter()#按照最后任务计数
    endpoint_answer_counter = Counter()#按照回答计数
    total = 0

    for record in read_manifest(args.manifest_path):
        total += 1

        program = record["program"]
        endpoint = program[-1]["function"]

        endpoint_counter.update([endpoint])

        answer = record["answer"]
        endpoint_answer_counter.update(
            [(endpoint, answer)]
        )

    print("total:", total)
    print("terminal functions:")

    for function, count in sorted(
        endpoint_counter.items()
    ):
        print(f"  {function}: {count}")

    print("answer examples by terminal function:")

    for function in sorted(endpoint_counter):
        examples = [
            answer
            for (endpoint, answer), count
            in endpoint_answer_counter.items()
            if endpoint == function
        ]

        print(
            f"  {function}:",
            sorted(examples)[:10],
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())