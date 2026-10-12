"""下载官方 MMStar 单个 Parquet，导出固定评估清单和原图。

默认预览；--execute 才下载约41.8MB并导出，不加载模型、不构造训练数据。
固定官方 commit；不安装 datasets/VLMEvalKit，复用 PyArrow/Pillow/HF Hub。
"""

from __future__ import annotations

import argparse
import io
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from qwen.src.common.data_common import resolve_path, write_jsonl
from qwen.src.evaluation.mmstar_metrics import DATASET, EXPECTED_ROWS, FILENAME, REVISION


DEFAULT_DATA_ROOT = "/data/${USER}/vlm_learning/benchmarks/mmstar"


def export_rows(raw_rows, root: Path, expected_rows: int = EXPECTED_ROWS):
    if len(raw_rows) != expected_rows:
        raise ValueError(f"MMStar 行数不一致：{len(raw_rows)}/{expected_rows}")
    ids, records = set(), []
    images = root / "images"
    images.mkdir(parents=True, exist_ok=False)
    for raw in sorted(raw_rows, key=lambda row: row["index"]):
        index, answer = raw["index"], raw["answer"]
        if type(index) is not int or index < 0 or index in ids:
            raise ValueError("MMStar index 非法或重复")
        ids.add(index)
        if answer not in ("A", "B", "C", "D") or not raw["question"].strip():
            raise ValueError(f"MMStar 题目或答案非法：{index}")
        if not raw["category"] or not raw["l2_category"]:
            raise ValueError(f"MMStar 能力标签缺失：{index}")
        blob = raw["image"]["bytes"]
        if not blob:
            raise ValueError(f"MMStar 缺少内嵌图片：{index}")
        with Image.open(io.BytesIO(blob)) as image:
            image.load()
            suffix = image.format.lower()
        path = images / f"{index:04d}.{suffix}"
        with path.open("xb") as file:
            file.write(blob)
        records.append({
            "sample_id": f"mmstar:{index}", "source": "mmstar", "split": "external_eval",
            "question": raw["question"], "answer": answer, "image": str(path.resolve()),
            "category": raw["category"], "l2_category": raw["l2_category"],
            "meta_info": raw["meta_info"],
        })
    categories = dict(sorted(Counter(row["category"] for row in records).items()))
    if expected_rows == EXPECTED_ROWS and (len(categories) != 6 or set(categories.values()) != {250}):
        raise ValueError("MMStar 应有6个能力类别，每类250题")
    write_jsonl(root / "manifest.jsonl", records)
    report = {
        "status": "MMSTAR_READY", "dataset": DATASET, "dataset_config": "val", "dataset_split": "val",
        "dataset_revision": REVISION, "filename": FILENAME, "rows": len(records),
        "categories": categories, "image_transform": "none; original bytes preserved",
        "training": False, "project_training_overlap_checked": False,
    }
    with (root / "report.json").open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    root = resolve_path(args.data_root, "data_root")
    if any((root / name).exists() for name in ("images", "manifest.jsonl", "report.json")):
        raise FileExistsError(f"保留已有 MMStar 数据，拒绝覆盖：{root}")
    print(json.dumps({
        "dataset": DATASET, "revision": REVISION, "file": FILENAME,
        "expected_rows": EXPECTED_ROWS, "download_size_mb_approx": 41.8, "data_root": str(root),
    }, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("PREVIEW ONLY：未下载数据、未导出图片。")
        return

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATASET, repo_type="dataset", filename=FILENAME, revision=REVISION,
        local_dir=str(root / "raw"),
    )
    parquet = pq.ParquetFile(path)
    required = {"index", "question", "image", "answer", "category", "l2_category", "meta_info"}
    if not required.issubset(parquet.schema_arrow.names):
        raise ValueError("MMStar Parquet 字段与官方版本不一致")
    report = export_rows(parquet.read().to_pylist(), root)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("MMSTAR PREPARE: PASS")


if __name__ == "__main__":
    main()
