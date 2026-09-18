# src/minimind_v_lab/data_contract/gate_report.py
#
# Shared P0 数据质量门禁组件。
#
# 负责：
# 1. 表示单个质量检查；
# 2. 聚合多个检查；
# 3. 输出 PASS / REVIEW / FAIL；
# 4. 生成结构化 gate_report.json；
# 5. 提供最小 smoke test。
#
# 它不负责读取 CLEVR、GQA 或 Parquet。
# 具体数据读取和统计由后续 validator 完成。
#
# 推荐调用关系：
#
# validator
#   -> 计算 rows、duplicate_rate、decode_failure_rate 等指标
#   -> 调用 require_max / require_min / require_equal
#   -> 得到 GateCheck
#   -> 放进 GateReport
#   -> 写 gate_report.json
#
# 本文件只使用 Python 标准库，兼容 Python 3.8。

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PASS = "PASS"
REVIEW = "REVIEW"
FAIL = "FAIL"

VALID_STATUSES = {
    PASS,
    REVIEW,
    FAIL,
}

STATUS_PRIORITY = {
    PASS: 0,
    REVIEW: 1,
    FAIL: 2,
}


def utc_now_iso() -> str:
    """
    返回 UTC 时间。

    时间只用于报告审计，不参与质量判断。
    """

    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """
    将对象转换为稳定 JSON 字符串。

    主要用于报告落盘和 smoke test。
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@dataclass
class GateCheck:
    """
    表示一个具体检查项。

    示例：

    gate_id:
        G1_SCHEMA_REQUIRED_FIELDS

    observed:
        实际观察到的指标

    threshold:
        允许的阈值

    violations:
        未通过时的具体原因

    artifacts:
        支撑这个检查的报告或数据文件
    """

    gate_id: str
    description: str
    status: str

    observed: Dict[str, Any] = field(default_factory=dict)
    threshold: Dict[str, Any] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    artifacts: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """
        在对象创建时检查状态是否合法。
        """

        if self.status not in VALID_STATUSES:
            raise ValueError(
                "invalid gate status: {}".format(self.status)
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为 JSON 可序列化字典。
        """

        return asdict(self)


@dataclass
class GateReport:
    """
    表示一个阶段或一个数据版本的总体门禁报告。

    status 由 checks 自动聚合：

    任意 FAIL  -> FAIL
    没有 FAIL，但有 REVIEW -> REVIEW
    全部 PASS -> PASS
    """

    gate_id: str
    checks: List[GateCheck] = field(default_factory=list)

    artifacts: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=utc_now_iso)

    @property
    def status(self) -> str:
        """
        根据所有子检查计算总体状态。
        """

        if not self.checks:
            return REVIEW

        highest_status = max(
            (
                check.status
                for check in self.checks
            ),
            key=lambda status: STATUS_PRIORITY[status],
        )

        return highest_status

    @property
    def has_fail(self) -> bool:
        """
        判断报告是否存在 FAIL。
        """

        return self.status == FAIL

    @property
    def needs_review(self) -> bool:
        """
        判断报告是否需要人工或规则复核。

        REVIEW 不一定表示数据不能用，
        但不能直接进入正式 gold 或 serving。
        """

        return self.status == REVIEW

    @property
    def passed(self) -> bool:
        """
        只有总体状态为 PASS 才算通过。
        """

        return self.status == PASS

    def add_check(self, check: GateCheck) -> None:
        """
        向报告中追加一个检查项。
        """

        if not isinstance(check, GateCheck):
            raise TypeError("check must be a GateCheck")

        self.checks.append(check)

    def violations(self) -> List[str]:
        """
        汇总所有检查项的 violation。
        """

        result: List[str] = []

        for check in self.checks:
            for violation in check.violations:
                result.append(
                    "{}: {}".format(
                        check.gate_id,
                        violation,
                    )
                )

        return result

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为最终报告字典。

        status 不由调用方传入，而是由 checks 自动计算。
        """

        return {
            "gate_id": self.gate_id,
            "status": self.status,
            "checks": [
                check.to_dict()
                for check in self.checks
            ],
            "violations": self.violations(),
            "artifacts": list(self.artifacts),
            "metadata": dict(self.metadata),
            "generated_at": self.generated_at,
        }

    def write_json(self, output_path: Path) -> None:
        """
        将报告写入 JSON 文件。

        先写临时文件，再替换正式文件，避免进程中断时留下半个 JSON。
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        temporary_path.write_text(
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        temporary_path.replace(output_path)


def make_check(
    gate_id: str,
    description: str,
    status: str,
    observed: Optional[Dict[str, Any]] = None,
    threshold: Optional[Dict[str, Any]] = None,
    violations: Optional[Iterable[str]] = None,
    artifacts: Optional[Iterable[str]] = None,
) -> GateCheck:
    """
    通用 GateCheck 构造器。

    统一在这里处理 None，避免调用方重复写空字典和空列表。
    """

    return GateCheck(
        gate_id=gate_id,
        description=description,
        status=status,
        observed=dict(observed or {}),
        threshold=dict(threshold or {}),
        violations=list(violations or []),
        artifacts=list(artifacts or []),
    )


def require_max(
    gate_id: str,
    metric_name: str,
    observed_value: float,
    maximum: float,
    description: str,
    artifacts: Optional[Iterable[str]] = None,
) -> GateCheck:
    """
    检查某个指标是否不超过最大值。

    常见用途：

    - 图片解码失败率 <= 0.1%
    - train/val image overlap == 0
    - recipe fraction error <= 1%
    """

    if observed_value <= maximum:
        status = PASS
        violations: List[str] = []
    else:
        status = FAIL
        violations = [
            "{}={} exceeds maximum={}".format(
                metric_name,
                observed_value,
                maximum,
            )
        ]

    return make_check(
        gate_id=gate_id,
        description=description,
        status=status,
        observed={
            metric_name: observed_value,
        },
        threshold={
            "max_{}".format(metric_name): maximum,
        },
        violations=violations,
        artifacts=artifacts,
    )


def require_min(
    gate_id: str,
    metric_name: str,
    observed_value: float,
    minimum: float,
    description: str,
    artifacts: Optional[Iterable[str]] = None,
) -> GateCheck:
    """
    检查某个指标是否达到最小值。

    常见用途：

    - 样本数至少达到某个规模
    - prediction coverage >= 0.99
    - val bucket 最小覆盖率达标
    """

    if observed_value >= minimum:
        status = PASS
        violations: List[str] = []
    else:
        status = FAIL
        violations = [
            "{}={} is below minimum={}".format(
                metric_name,
                observed_value,
                minimum,
            )
        ]

    return make_check(
        gate_id=gate_id,
        description=description,
        status=status,
        observed={
            metric_name: observed_value,
        },
        threshold={
            "min_{}".format(metric_name): minimum,
        },
        violations=violations,
        artifacts=artifacts,
    )


def require_equal(
    gate_id: str,
    metric_name: str,
    observed_value: Any,
    expected_value: Any,
    description: str,
    artifacts: Optional[Iterable[str]] = None,
) -> GateCheck:
    """
    检查某个值是否与期望值完全相等。

    常见用途：

    - train/val/test image overlap == 0
    - schema version 相等
    - 输入输出行数守恒
    """

    if observed_value == expected_value:
        status = PASS
        violations: List[str] = []
    else:
        status = FAIL
        violations = [
            "{}={} does not equal expected={}".format(
                metric_name,
                repr(observed_value),
                repr(expected_value),
            )
        ]

    return make_check(
        gate_id=gate_id,
        description=description,
        status=status,
        observed={
            metric_name: observed_value,
        },
        threshold={
            "expected_{}".format(metric_name): expected_value,
        },
        violations=violations,
        artifacts=artifacts,
    )


def review_if_below(
    gate_id: str,
    metric_name: str,
    observed_value: float,
    review_threshold: float,
    fail_threshold: float,
    description: str,
    artifacts: Optional[Iterable[str]] = None,
) -> GateCheck:
    """
    检查一个指标的分级状态：

    observed >= review_threshold
        PASS

    fail_threshold <= observed < review_threshold
        REVIEW

    observed < fail_threshold
        FAIL

    这个函数适合质量指标存在“灰区”的情况。

    例如：
    - 图片解码率 >= 0.999：PASS
    - 0.995 <= 解码率 < 0.999：REVIEW
    - 解码率 < 0.995：FAIL
    """

    if observed_value >= review_threshold:
        status = PASS
        violations: List[str] = []
    elif observed_value >= fail_threshold:
        status = REVIEW
        violations = [
            "{}={} is in review range [{}, {})".format(
                metric_name,
                observed_value,
                fail_threshold,
                review_threshold,
            )
        ]
    else:
        status = FAIL
        violations = [
            "{}={} is below fail threshold={}".format(
                metric_name,
                observed_value,
                fail_threshold,
            )
        ]

    return make_check(
        gate_id=gate_id,
        description=description,
        status=status,
        observed={
            metric_name: observed_value,
        },
        threshold={
            "review_threshold": review_threshold,
            "fail_threshold": fail_threshold,
        },
        violations=violations,
        artifacts=artifacts,
    )


def build_demo_report() -> GateReport:
    """
    创建一个最小 demo 报告。

    这里故意包含：
    - 一个 PASS；
    - 一个 REVIEW；
    - 一个 FAIL。

    因为 FAIL 的优先级最高，所以总体状态应该是 FAIL。
    """

    report = GateReport(
        gate_id="G0_DEMO",
        artifacts=[
            "tests/fixtures/demo_manifest.jsonl",
        ],
        metadata={
            "dataset_version": "demo_v1",
            "schema_version": "canonical_v1",
        },
    )

    report.add_check(
        require_min(
            gate_id="G0_SAMPLE_COUNT",
            metric_name="row_count",
            observed_value=100,
            minimum=10,
            description="demo 数据至少包含 10 行",
        )
    )

    report.add_check(
        review_if_below(
            gate_id="G0_IMAGE_DECODE_RATE",
            metric_name="image_decode_rate",
            observed_value=0.998,
            review_threshold=0.999,
            fail_threshold=0.995,
            description="图片解码率应达到质量阈值",
        )
    )

    report.add_check(
        require_equal(
            gate_id="G0_IMAGE_OVERLAP",
            metric_name="train_val_image_overlap",
            observed_value=2,
            expected_value=0,
            description="train 和 val 不允许共享图片",
        )
    )

    return report


def run_demo() -> None:
    """
    运行最小 smoke test。

    预期：
    - 总体状态为 FAIL；
    - violations 中包含 train_val_image_overlap；
    - JSON 文件可以成功写入临时目录。
    """

    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        report = build_demo_report()

        output_path = root / "reports" / "gate_report.json"
        report.write_json(output_path)

        assert report.status == FAIL
        assert report.has_fail
        assert not report.passed
        assert output_path.exists()

        payload = json.loads(
            output_path.read_text(encoding="utf-8")
        )

        assert payload["status"] == FAIL
        assert payload["gate_id"] == "G0_DEMO"
        assert len(payload["checks"]) == 3

        print("gate report demo: PASS")
        print("overall_status:", report.status)
        print("report_path:", output_path)

        print("violations:")
        for violation in report.violations():
            print(" -", violation)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Quality gate report utility and smoke test."
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in smoke test",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="reserved for future report generation commands",
    )

    args = parser.parse_args()

    if args.demo:
        run_demo()
        return 0

    parser.error(
        "currently only --demo is implemented"
    )

    return 2


if __name__ == "__main__":
    raise SystemExit(main())