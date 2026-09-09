# 这个脚本分析 CLEVR 评测结果，定位答案偏置和任务类型错误。

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]

def normalize_answer(text: str) -> str:
    text = text.lower().strip()
    text = text.splitlines()[0]
    return re.sub(r"[^\w]+", "", text)
#定义答案空间
ANSWER_GROUPS = {
    "color": {
        "blue", "brown", "cyan", "gray",
        "green", "purple", "red", "yellow",
    },
    "material": {"metal", "rubber"},
    "shape": {"cube", "cylinder", "sphere"},
    "size": {"large", "small"},
    "boolean": {"yes", "no"},
}

def answer_group(answer: str) -> str:
    answer = normalize_answer(answer)

    if answer.isdigit():
        return "integer"

    for group_name, choices in ANSWER_GROUPS.items():
        if answer in choices:
            return group_name

    return "other"#判断预测属于什么空间

parser = argparse.ArgumentParser()
parser.add_argument("train_manifest", type=Path)
parser.add_argument("eval_path", type=Path)

args = parser.parse_args()

train_records = read_jsonl(args.train_manifest)
eval_results = read_jsonl(args.eval_path)

answers_by_task = defaultdict(Counter)

for record in train_records:
    answers_by_task[record["task_type"]].update(
        [record["answer"]]
    )

majority_answer = {
    task_type: counter.most_common(1)[0][0]
    for task_type, counter in answers_by_task.items()
}

total_by_task = Counter()
model_correct_by_task = Counter()
baseline_correct_by_task = Counter()

prediction_counter = Counter()
error_mode_counter = Counter()
group_pair_counter = Counter()


for result in eval_results:
    task_type = result["task_type"]
    target = normalize_answer(result["target"])
    prediction = normalize_answer(result["prediction"])

    total_by_task[task_type] += 1
    prediction_counter[prediction] += 1

    if result["correct"]:
        model_correct_by_task[task_type] += 1

    if majority_answer[task_type] == target:
        baseline_correct_by_task[task_type] += 1

    target_group = answer_group(target)
    prediction_group = answer_group(prediction)
    group_pair_counter[(target_group, prediction_group)] += 1

    if result["correct"]:
        error_mode_counter["correct"] += 1
    elif prediction_group == "other":
        error_mode_counter["invalid_or_extra_text"] += 1
    elif prediction_group == target_group:
        error_mode_counter["wrong_within_answer_space"] += 1
    else:
        error_mode_counter["wrong_answer_space"] += 1

model_correct = sum(model_correct_by_task.values())
baseline_correct = sum(baseline_correct_by_task.values())
total = len(eval_results)

print("total:", total)
print(f"model exact match: {model_correct}/{total}")
print(f"majority baseline: {baseline_correct}/{total}")

print("\nby task type:")
for task_type in sorted(total_by_task):
    count = total_by_task[task_type]
    model_accuracy = model_correct_by_task[task_type] / count
    baseline_accuracy = baseline_correct_by_task[task_type] / count

    print(
        f"  {task_type}: "
        f"model={model_accuracy:.4f}, "
        f"baseline={baseline_accuracy:.4f}, "
        f"majority={majority_answer[task_type]!r}"
    )
print("\ntop predictions:")
for answer, count in prediction_counter.most_common(12):
    print(f"  {answer!r}: {count}")

print("\nerror modes:")
for name, count in error_mode_counter.most_common():
    print(f"  {name}: {count}")

print("\nanswer-group transitions:")
for (target_group, prediction_group), count in sorted(
    group_pair_counter.items(),
    key=lambda item: (-item[1], item[0]),
):
    print(f"  {target_group} -> {prediction_group}: {count}")