"""一次性 comparison 提示词诊断。

复用原始模型 baseline 的样本和推理参数，
仅把末尾回答要求改为明确的 yes/no。
不训练，不修改 baseline，不更新数据配方。
请从服务器项目根目录运行。
"""

import json
import os
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import yaml

sys.path.insert(0, str(Path("qwen/src").resolve()))
from data_common import read_jsonl, resolve_path
from evaluate_clevr import normalize_answer


# 1. 读取原 baseline，确保比较的是同一个模型、同一批题。
with Path("qwen/configs/mvp.yaml").open(
    encoding="utf-8"
) as file:
    cfg = yaml.safe_load(file)

root = resolve_path(cfg["output_root"], "output_root")
baseline = root / "baseline" / "clevr"

with (baseline / "metrics.json").open(
    encoding="utf-8"
) as file:
    metrics = json.load(file)

protocol = metrics["protocol"]
if (
    metrics["status"] != "COMPLETE"
    or protocol["checkpoint"] is not None
):
    raise ValueError("需要完整的原始模型 baseline")

rows = [
    row
    for row in read_jsonl(baseline / "predictions.jsonl")
    if row["task_type"] == "comparison"
]
old_correct = sum(row["correct"] for row in rows)
bucket = metrics["by_bucket"]["task:comparison"]

if (
    len(rows) != bucket["n"]
    or old_correct != bucket["correct"]
):
    raise ValueError(
        "逐样本预测与 baseline 指标不一致"
    )


# 2. 只替换回答要求，不改问题，不把标准答案放入输入。
old_suffix = "Answer with a single word or number."
new_suffix = (
    "Answer with exactly yes or no. "
    "Do not output a number or explanation."
)

if any(
    not row["question"].endswith(old_suffix)
    for row in rows
):
    raise ValueError("原提示词末尾与预期不一致")


# 3. 加载原始模型，不加载任何 adapter。
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not visible.strip() or "," in visible:
    raise ValueError("请指定一张获分配的 GPU")
if version("ms-swift") != "4.5.3":
    raise RuntimeError("需要 ms-swift==4.5.3")

import torch
from swift import (
    InferRequest,
    RequestConfig,
    TransformersEngine,
)

if (
    not torch.cuda.is_available()
    or torch.cuda.device_count() != 1
):
    raise RuntimeError("需要一张可见 CUDA GPU")
if not torch.cuda.is_bf16_supported():
    raise RuntimeError("需要 BF16 支持")

torch.manual_seed(int(cfg["seed"]))

engine = TransformersEngine(
    protocol["model_path"],
    adapters=[],
    model_type="qwen3_vl",
    torch_dtype=torch.bfloat16,
    attn_impl=protocol["attn_impl"],
    device_map={"": "cuda:0"},
    max_batch_size=1,
)
engine.template.max_length = protocol["max_length"]
engine.template.max_pixels = protocol["max_pixels"]
engine.model.eval()
engine.model.requires_grad_(False)

request_cfg = RequestConfig(
    max_tokens=protocol["max_new_tokens"],
    temperature=protocol["temperature"],
    stream=False,
)


# 4. 按原评分规则重新生成，记录逐题正确性变化。
results = []
pairs = Counter()

for index, row in enumerate(rows, start=1):
    question = (
        row["question"][:-len(old_suffix)] + new_suffix
    )
    request = InferRequest(
        messages=[
            {"role": "user", "content": question}
        ],
        images=[
            str(resolve_path(row["image"], "image"))
        ],
    )

    with torch.inference_mode():
        response = engine.infer(
            [request], request_cfg
        )[0]

    if isinstance(response, Exception):
        raise response

    prediction = response.choices[0].message.content
    if not isinstance(prediction, str):
        raise TypeError("生成结果不是文本")

    normalized = normalize_answer(prediction)
    reference = normalize_answer(row["reference"])
    correct = normalized == reference

    results.append({
        "old_correct": row["correct"],
        "new_correct": correct,
        "canonical": normalized in {"yes", "no"},
    })
    pairs[(reference, normalized)] += 1

    if index <= 6:
        print(json.dumps({
            "sample_id": row["sample_id"],
            "reference": row["reference"],
            "old_prediction": row["prediction"],
            "new_prediction": prediction,
        }, ensure_ascii=False))


# 5. 仅输出诊断结果，不覆盖正式 baseline。
new_correct = sum(
    r["new_correct"] for r in results
)

print(json.dumps({
    "status": "DIAGNOSTIC_ONLY",
    "feedback_eligible": False,
    "n": len(rows),
    "old_correct": old_correct,
    "old_accuracy": old_correct / len(rows),
    "new_correct": new_correct,
    "new_accuracy": new_correct / len(rows),
    "new_non_yes_no": sum(
        not r["canonical"] for r in results
    ),
    "wrong_to_correct": sum(
        not r["old_correct"] and r["new_correct"]
        for r in results
    ),
    "correct_to_wrong": sum(
        r["old_correct"] and not r["new_correct"]
        for r in results
    ),
    "new_answer_suffix": new_suffix,
}, ensure_ascii=False, indent=2))

print("新提示下的 标准答案 → 预测:")
for (reference, prediction), count in pairs.most_common(10):
    print(json.dumps({
        "reference": reference,
        "prediction": prediction,
        "n": count,
    }, ensure_ascii=False))