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

    image_counter = Counter()
    relation_counter = Counter()
    answer_counter = Counter()
    program_length_counter = Counter()
    total = 0

    for record in read_manifest(args.manifest_path):
        total += 1

        image_counter.update(
            [record["image_filename"]]
        )#统计图片出现次数

        relation_counter.update(
            record["spatial_relations"]
        )#统计空间关系类型

        answer_counter.update(
            [record["answer"]]
        )#统计答案类别频率

        program_length_counter.update(
            [record["program_length"]]
        )#统计程序长度

    print("total records:", total)
    print("unique images:", len(image_counter))

    print("top images:")

    for image_name, count in image_counter.most_common(10):
        print(f"  {image_name}: {count}")

    print("relation counts:")
    for relation, count in sorted(relation_counter.items()):
        print(f"  {relation}: {count}")

    print("top answers:")
    for answer, count in answer_counter.most_common(10):
        print(f"  {answer}: {count}")

    print("program length distribution:")
    for length, count in sorted(program_length_counter.items()):
        print(f"  {length}: {count}")
    #打印统计的信息

    #接下来增加集中度指标
    if total > 0:
        top_image_count = image_counter.most_common(1)[0][1]
        top_image_ratio = top_image_count / total#出现次数最多的图片对应的问题数 / 总问题数
    else:
        top_image_ratio = 0.0

    print("top image ratio:", round(top_image_ratio, 4))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

    