"""读取跨轮历史配置，离线生成固定 probe 学习动态。

默认只预览；--execute 保存 sample_dynamics.jsonl、summary.json、history_used.json。
只读原始评估结果与 checkpoint 元数据，不使用 GPU、不训练、不更新配方。
"""

from __future__ import annotations

import argparse
import json

import yaml

from qwen.src.common.data_common import resolve_path, write_jsonl
from qwen.src.learning.artifacts import load_history
from qwen.src.learning.dynamics import build_dynamics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    history_path = resolve_path(args.history, "history")
    output = resolve_path(args.output, "output")
    if not history_path.is_file():
        raise FileNotFoundError(history_path)
    if output.exists():
        raise FileExistsError(f"拒绝覆盖已有学习动态：{output}")
    with history_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    snapshots = load_history(config)
    records, summary = build_dynamics(snapshots)
    used = {
        "step_accounting": config["step_accounting"],
        "history_path": str(history_path),
        "snapshots": [
            {key: value for key, value in stage.items() if key not in ("nll_records", "correctness")}
            for stage in snapshots
        ],
    }
    summary["history"] = used
    print(json.dumps({
        "status": summary["status"], "snapshot_count": len(snapshots),
        "probe_count": len(records), "output": str(output),
        "snapshots": [stage["tag"] for stage in snapshots],
    }, ensure_ascii=False, indent=2), flush=True)
    if not args.execute:
        print("PREVIEW ONLY：已校验历史，未写结果。")
        return
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "sample_dynamics.jsonl", records)
    for name, content in (("summary.json", summary), ("history_used.json", used)):
        with (output / name).open("x", encoding="utf-8") as file:
            json.dump(content, file, ensure_ascii=False, indent=2, allow_nan=False)
            file.write("\n")
    print("LEARNING DYNAMICS: PASS")
    print("OUTPUT:", output)
    if len(snapshots) == 1:
        print("BASELINE_ONLY：尚无跨轮差值；后续追加真实 checkpoint。")


if __name__ == "__main__":
    main()
