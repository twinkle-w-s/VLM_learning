"""统一 CLEVR comparison 回答格式，允许重复应用。

只根据任务标签改提示，不根据答案选择提示；其他任务与 replay 不变。
"""

from __future__ import annotations

import copy


OLD_SUFFIX = "Answer with a single word or number."
NEW_SUFFIX = (
    "Answer with exactly yes or no. "
    "Do not output a number or explanation."
)


def rewrite_record(item: dict) -> tuple[dict, bool]:
    """返回独立记录与本次是否修改提示的标记。"""
    row = copy.deepcopy(item)
    if (
        row["source"] != "clevr"
        or row["task_type"] != "comparison"
    ):
        return row, False

    message = row["messages"][0]
    content = message["content"]
    if message["role"] != "user":
        raise ValueError("comparison 的首条消息必须为 user")
    if content.endswith(NEW_SUFFIX):
        return row, False
    if not content.endswith(OLD_SUFFIX):
        raise ValueError(
            f"comparison 提示格式异常：{row['sample_id']}"
        )

    message["content"] = (
        content[:-len(OLD_SUFFIX)] + NEW_SUFFIX
    )
    return row, True
