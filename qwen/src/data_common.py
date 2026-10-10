"""公共数据工具：路径解析、JSONL 读写、蓄水池抽样。

不包含具体数据集逻辑。
写入使用独占模式，拒绝覆盖已有文件。
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def resolve_path(value, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 尚未填写")

    expanded = os.path.expanduser(os.path.expandvars(value))
    if "${" in expanded:
        raise ValueError(f"{name} 含未展开的环境变量: {expanded}")
    return Path(expanded)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def reservoir_add(
    pool: list,
    item: dict,
    seen: int,
    limit,
    rng,
) -> None:
    if limit is None or len(pool) < limit:
        pool.append(item)
    else:
        index = rng.randrange(seen)
        if index < limit:
            pool[index] = item


def write_jsonl(path: Path, records: list) -> None:
    with path.open("x", encoding="utf-8") as file:
        for item in records:
            file.write(
                json.dumps(item, ensure_ascii=False) + "\n"
            )