"""assistant-token NLL 的基础计算与 CPU 演示。

输入：模型 logits [B, L, V] 和模板生成的 labels [B, L]。
labels=-100 的 prompt/image/pad 不计分；使用 causal shift。
输出每条样本的 NLL 总和、有效 token 数、平均 NLL，以及 token 加权汇总。
只提供度量函数，供评估子包共用；不加载模型、不读取数据、不训练。
"""

from __future__ import annotations

import argparse
import math

def compute_answer_nll(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> list[dict]:
    """每个位置 t 的 logits 预测位置 t+1 的 label。"""
    import torch
    import torch.nn.functional as F

    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError(
            "需要 logits [B,L,V] 与 labels [B,L]"
        )
    if tuple(logits.shape[:2]) != tuple(labels.shape):
        raise ValueError(
            "logits 与 labels 的 batch/序列维度不一致"
        )
    if logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError(
            "需要非空 batch，且序列长度至少为 2"
        )
    if labels.dtype != torch.long:
        raise ValueError("labels 必须为 torch.long")
    if labels.device != logits.device:
        raise ValueError(
            "logits 与 labels 必须在同一设备"
        )

    # 输入 labels 必须是模板的原始标签，不能提前 shift。
    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]
    results = []

    for index in range(logits.shape[0]):
        targets = shifted_labels[index]
        mask = targets != -100
        token_count = int(mask.sum().item())

        if token_count == 0:
            raise ValueError(
                f"batch 第 {index} 条没有有效监督 token"
            )

        # 先选有效 token，再转 FP32。
        # 避免把整个大词表 logits 张量扩为 FP32。
        scores = shifted_logits[index][mask].float()
        losses = F.cross_entropy(
            scores,
            targets[mask],
            reduction="none",
        )

        if not torch.isfinite(losses).all().item():
            raise ValueError(
                f"batch 第 {index} 条 NLL 包含 NaN/Inf"
            )

        sum_nll = float(
            losses.detach().double().sum().item()
        )
        results.append({
            "sum_nll": sum_nll,
            "answer_token_count": token_count,
            "mean_nll": sum_nll / token_count,
        })

    return results


def aggregate_nll(records: list[dict]) -> dict:
    """组 NLL 按有效 token 加权，不直接平均样本 mean_nll。"""
    if not records:
        raise ValueError("不能汇总空记录")

    for record in records:
        count = record["answer_token_count"]
        value = record["sum_nll"]

        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count <= 0
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(
                "NLL 记录的 token 数或总和非法"
            )

    total_nll = math.fsum(
        record["sum_nll"] for record in records
    )
    total_tokens = sum(
        record["answer_token_count"]
        for record in records
    )

    return {
        "n": len(records),
        "sum_nll": total_nll,
        "answer_token_count": total_tokens,
        "mean_nll": total_nll / total_tokens,
    }


def demo() -> None:
    import torch

    # B=2：两条样本。
    # L=6：各 6 个位置。
    # V=5：词表大小为 5。
    labels = torch.tensor([
        [-100, -100, -100, 1, 2, 3],
        [-100, -100, -100, -100, -100, 4],
    ], dtype=torch.long)
    logits = torch.zeros((2, 6, 5))

    # 均匀预测：每个有效 token 的 NLL 应为 ln(5)。
    records = compute_answer_nll(logits, labels)
    assert [
        r["answer_token_count"] for r in records
    ] == [3, 1]

    for record in records:
        assert math.isclose(
            record["mean_nll"],
            math.log(5),
            abs_tol=1e-6,
        )

    # causal shift：
    # logits 的位置 2、3、4 分别预测 labels 的位置 3、4、5。
    confident = torch.zeros_like(logits)
    confident[0, 2, 1] = 12
    confident[0, 3, 2] = 12
    confident[0, 4, 3] = 12
    confident[1, 4, 4] = 12

    confident_records = compute_answer_nll(
        confident, labels
    )
    assert all(
        r["mean_nll"] < 0.001
        for r in confident_records
    )

    # 改动被 mask 排除的位置，不应影响 NLL。
    ignored_changed = confident.clone()
    ignored_changed[:, :2, :] = 100
    assert compute_answer_nll(
        ignored_changed, labels
    ) == confident_records

    # 最后一个 logits 位置没有下一个真实 token，不参与评分。
    ignored_changed[:, -1, :] = 100
    assert compute_answer_nll(
        ignored_changed, labels
    ) == confident_records

    # 按 token 加权：(6+4)/(3+1)=2.5。
    weighted = aggregate_nll([
        {"sum_nll": 6.0, "answer_token_count": 3},
        {"sum_nll": 4.0, "answer_token_count": 1},
    ])
    assert weighted["mean_nll"] == 2.5

    # 不允许把无监督样本当作 loss=0。
    try:
        compute_answer_nll(
            logits,
            torch.full_like(labels, -100),
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "应拒绝没有监督 token 的样本"
        )

    print("ANSWER NLL DEMO: PASS")
    print(
        "uniform_mean_nll:",
        records[0]["mean_nll"],
    )
    print(
        "confident_mean_nll:",
        confident_records[0]["mean_nll"],
    )
    print(
        "weighted_group_mean_nll:",
        weighted["mean_nll"],
    )
    print(
        "CPU ONLY：未加载模型，"
        "未读取训练数据，未训练。"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
