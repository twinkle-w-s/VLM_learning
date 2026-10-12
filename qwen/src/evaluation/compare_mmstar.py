"""离线比较基础模型和训练后 adapter 的完整 MMStar 结果。

默认预览；--execute 才保存 comparison.json 与 paired_changes.jsonl。
不加载模型、不占 GPU、不修改训练数据或配方。
"""

from __future__ import annotations

import argparse
import json

from qwen.src.common.data_common import read_jsonl, resolve_path, write_jsonl
from qwen.src.evaluation.mmstar_metrics import compare_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True)
    parser.add_argument("--after-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base_dir = resolve_path(args.base_dir, "base_dir")
    after_dir = resolve_path(args.after_dir, "after_dir")
    output = resolve_path(args.output, "output")
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有退化对比：{output}")
    with (base_dir / "metrics.json").open(encoding="utf-8") as file:
        base_metrics = json.load(file)
    with (after_dir / "metrics.json").open(encoding="utf-8") as file:
        after_metrics = json.load(file)
    pairs, summary = compare_results(
        base_metrics, list(read_jsonl(base_dir / "predictions.jsonl")),
        after_metrics, list(read_jsonl(after_dir / "predictions.jsonl")),
    )
    summary.update({"base_dir": str(base_dir), "after_dir": str(after_dir)})
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("PREVIEW ONLY：已校验结果，未写对比文件。")
        return
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "paired_changes.jsonl", pairs)
    with (output / "comparison.json").open("x", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print("MMSTAR COMPARISON: PASS")


if __name__ == "__main__":
    main()
