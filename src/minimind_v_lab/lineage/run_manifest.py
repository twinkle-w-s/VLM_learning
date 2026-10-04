# src/minimind_v_lab/lineage/run_manifest.py
#
# 一次数据处理或模型实验的固定运行合同。
#
# 它负责记录：
# 1. 本次运行使用哪个数据版本；
# 2. train / val / test manifest 的身份；
# 3. 使用哪个基础 checkpoint；
# 4. 使用什么冻结策略和训练预算；
# 5. 使用哪些随机种子；
# 6. 使用哪个评估协议；
# 7. 当前 run 的配置、代码和环境身份。
#
# run_manifest 记录“这次实验计划使用什么”。
# job 的动态执行状态由 state_store.py 管理。
# 具体产物的路径和血缘由 artifact_registry.py 管理。
#
# 本文件仅使用 Python 标准库，兼容 Python 3.8。

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


MANIFEST_VERSION = "run_manifest_v1"

VALID_TRACKS = {
    "SHARED",
    "A",
    "B",
}

VALID_FREEZE_MODES = {
    0,
    1,
    2,
}


def utc_now_iso() -> str:
    """
    返回当前 UTC 时间。
    """

    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    """
    生成稳定的 JSON 字符串。

    字段顺序和普通空格不会影响 hash。
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_json(value: Any) -> str:
    """
    对一个 JSON 可序列化对象计算 SHA-256。
    """

    serialized = canonical_json(value)

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def write_json_atomic(
    output_path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    原子写 JSON 文件。

    先写入 .tmp 文件，完整写入后再替换正式文件。
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
            payload,
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)


@dataclass(frozen=True)
class SeedBundle:
    """
    一次实验中使用的所有随机种子。

    只记录一个 seed=42 并不完整，因为：
    - Python random 有自己的随机状态；
    - NumPy 有自己的随机状态；
    - PyTorch CPU/CUDA 有自己的随机状态；
    - DataLoader worker 和采样器也可能独立随机。
    """

    python: int
    numpy: int
    torch: int
    torch_cuda: int
    dataloader_worker_base: int
    distributed_sampler: int
    recipe_sampling: int
    evaluation: int

    def validate(self) -> None:
        """
        检查种子是否为非负整数。
        """

        for field_name, value in asdict(self).items():
            if not isinstance(value, int):
                raise TypeError(
                    "seed {} must be an integer".format(
                        field_name
                    )
                )

            if value < 0:
                raise ValueError(
                    "seed {} must be non-negative".format(
                        field_name
                    )
                )


@dataclass(frozen=True)
class DataSnapshot:
    """
    本次运行使用的数据快照。

    manifest_hash 用于确认：
    当前训练使用的数据清单是否发生变化。

    test 可以为 None：
    例如早期数据 smoke test 可能暂时不运行 test。
    """

    dataset_version: str
    schema_version: str
    tag_version: str
    recipe_version: str

    train_manifest_uri: str
    train_manifest_hash: str

    val_manifest_uri: str
    val_manifest_hash: str

    test_manifest_uri: Optional[str] = None
    test_manifest_hash: Optional[str] = None

    def validate(self) -> None:
        """
        检查数据快照字段。
        """

        required = {
            "dataset_version": self.dataset_version,
            "schema_version": self.schema_version,
            "tag_version": self.tag_version,
            "recipe_version": self.recipe_version,
            "train_manifest_uri": self.train_manifest_uri,
            "train_manifest_hash": self.train_manifest_hash,
            "val_manifest_uri": self.val_manifest_uri,
            "val_manifest_hash": self.val_manifest_hash,
        }

        for field_name, value in required.items():
            if not value:
                raise ValueError(
                    "data snapshot field is empty: {}".format(
                        field_name
                    )
                )

        if (
            self.test_manifest_uri is None
        ) != (
            self.test_manifest_hash is None
        ):
            raise ValueError(
                "test_manifest_uri and test_manifest_hash "
                "must either both exist or both be None"
            )


@dataclass(frozen=True)
class ModelSnapshot:
    """
    本次运行使用的模型起点和训练范围。

    freeze_llm 的当前约定：

    0：
        按 MiniMind-V 当前实现的全量训练方式运行。

    1：
        冻结视觉编码器，训练 LLM 和 Projector。

    2：
        Projector warmup，只训练 Projector。

    具体可训练参数仍需在训练启动前实际统计，
    不能只相信配置数字。
    """

    model_name: str

    base_checkpoint_uri: str
    base_checkpoint_hash: str

    freeze_llm: int

    resume_checkpoint_uri: Optional[str] = None
    resume_checkpoint_hash: Optional[str] = None

    trainable_parameter_count: Optional[int] = None
    total_parameter_count: Optional[int] = None

    def validate(self) -> None:
        """
        检查模型快照字段。
        """

        if not self.model_name:
            raise ValueError("model_name must not be empty")

        if not self.base_checkpoint_uri:
            raise ValueError(
                "base_checkpoint_uri must not be empty"
            )

        if not self.base_checkpoint_hash:
            raise ValueError(
                "base_checkpoint_hash must not be empty"
            )

        if self.freeze_llm not in VALID_FREEZE_MODES:
            raise ValueError(
                "freeze_llm must be one of {}".format(
                    sorted(VALID_FREEZE_MODES)
                )
            )

        if (
            self.resume_checkpoint_uri is None
        ) != (
            self.resume_checkpoint_hash is None
        ):
            raise ValueError(
                "resume_checkpoint_uri and "
                "resume_checkpoint_hash must either both "
                "exist or both be None"
            )

        if (
            self.trainable_parameter_count is not None
            and self.trainable_parameter_count < 0
        ):
            raise ValueError(
                "trainable_parameter_count must be non-negative"
            )

        if (
            self.total_parameter_count is not None
            and self.total_parameter_count < 0
        ):
            raise ValueError(
                "total_parameter_count must be non-negative"
            )

        if (
            self.trainable_parameter_count is not None
            and self.total_parameter_count is not None
            and self.trainable_parameter_count
            > self.total_parameter_count
        ):
            raise ValueError(
                "trainable_parameter_count cannot exceed "
                "total_parameter_count"
            )


@dataclass(frozen=True)
class TrainingBudget:
    """
    训练预算。

    正式比较时，不能只记录 epochs。
    至少需要记录：
    - optimizer steps
    - effective batch size
    - token/image exposure
    """

    learning_rate: float
    effective_batch_size: int
    gradient_accumulation_steps: int
    world_size: int
    max_seq_len: int

    max_steps: Optional[int] = None
    epochs: Optional[float] = None

    target_token_budget: Optional[int] = None
    target_image_exposure: Optional[int] = None

    checkpoint_interval_steps: Optional[int] = None
    evaluation_interval_steps: Optional[int] = None

    def validate(self) -> None:
        """
        检查训练预算是否合法。
        """

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if self.effective_batch_size <= 0:
            raise ValueError(
                "effective_batch_size must be greater than zero"
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be greater than zero"
            )

        if self.world_size <= 0:
            raise ValueError(
                "world_size must be greater than zero"
            )

        if self.max_seq_len <= 0:
            raise ValueError(
                "max_seq_len must be greater than zero"
            )

        if self.max_steps is None and self.epochs is None:
            raise ValueError(
                "at least one of max_steps or epochs must be set"
            )

        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(
                "max_steps must be greater than zero"
            )

        if self.epochs is not None and self.epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero"
            )

        optional_positive_fields = {
            "target_token_budget": self.target_token_budget,
            "target_image_exposure": self.target_image_exposure,
            "checkpoint_interval_steps": (
                self.checkpoint_interval_steps
            ),
            "evaluation_interval_steps": (
                self.evaluation_interval_steps
            ),
        }

        for field_name, value in optional_positive_fields.items():
            if value is not None and value <= 0:
                raise ValueError(
                    "{} must be greater than zero".format(
                        field_name
                    )
                )


@dataclass(frozen=True)
class EvaluationSpec:
    """
    本次运行使用的验证协议。

    test_allowed_in_feedback 必须保持 False。
    test 只允许用于最终 release evaluation。
    """

    suite_names: List[str]
    evaluator_version: str
    normalizer_version: str
    selection_metric: str

    min_prediction_coverage: float = 1.0
    test_allowed_in_feedback: bool = False

    def validate(self) -> None:
        """
        检查评估协议。
        """

        if not self.suite_names:
            raise ValueError(
                "suite_names must not be empty"
            )

        for suite_name in self.suite_names:
            if not suite_name:
                raise ValueError(
                    "suite_names cannot contain empty values"
                )

        if not self.evaluator_version:
            raise ValueError(
                "evaluator_version must not be empty"
            )

        if not self.normalizer_version:
            raise ValueError(
                "normalizer_version must not be empty"
            )

        if not self.selection_metric:
            raise ValueError(
                "selection_metric must not be empty"
            )

        if not 0 < self.min_prediction_coverage <= 1:
            raise ValueError(
                "min_prediction_coverage must be in (0, 1]"
            )

        if self.test_allowed_in_feedback:
            raise ValueError(
                "test data must not be allowed in feedback"
            )


@dataclass(frozen=True)
class RunManifest:
    """
    一次实验的固定运行合同。

    这个对象主要保存静态身份。

    动态 job 状态由 StateStore 保存，不在这里反复修改。
    """

    run_id: str
    experiment_name: str
    track: str
    round_id: str

    config_hash: str
    code_commit: str
    environment_hash: str

    seed_bundle: SeedBundle
    data: DataSnapshot
    model: ModelSnapshot
    training: TrainingBudget
    evaluation: EvaluationSpec

    manifest_version: str = MANIFEST_VERSION
    parent_run_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        检查完整 run manifest。
        """

        required = {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "track": self.track,
            "round_id": self.round_id,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "environment_hash": self.environment_hash,
            "manifest_version": self.manifest_version,
            "created_at": self.created_at,
        }

        for field_name, value in required.items():
            if not value:
                raise ValueError(
                    "run manifest field is empty: {}".format(
                        field_name
                    )
                )

        if self.track not in VALID_TRACKS:
            raise ValueError(
                "track must be one of {}".format(
                    sorted(VALID_TRACKS)
                )
            )

        self.seed_bundle.validate()
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.evaluation.validate()

    def identity_dict(self) -> Dict[str, Any]:
        """
        返回参与 manifest hash 的稳定字段。

        created_at 不参与 hash，因为创建时间不是实验语义。
        notes 也不参与 hash，因为修改说明文字不应改变实验身份。
        """

        return {
            "manifest_version": self.manifest_version,
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "track": self.track,
            "round_id": self.round_id,
            "parent_run_id": self.parent_run_id,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "environment_hash": self.environment_hash,
            "seed_bundle": asdict(self.seed_bundle),
            "data": asdict(self.data),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "evaluation": asdict(self.evaluation),
            "extra": self.extra,
        }

    @property
    def manifest_hash(self) -> str:
        """
        计算本次运行合同的稳定 hash。
        """

        return sha256_json(
            self.identity_dict()
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为最终写入 JSON 的结构。
        """

        payload = asdict(self)
        payload["manifest_hash"] = self.manifest_hash

        return payload

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
    ) -> "RunManifest":
        """
        从 JSON 字典恢复 RunManifest。
        """

        manifest = cls(
            run_id=payload["run_id"],
            experiment_name=payload["experiment_name"],
            track=payload["track"],
            round_id=payload["round_id"],
            config_hash=payload["config_hash"],
            code_commit=payload["code_commit"],
            environment_hash=payload["environment_hash"],
            seed_bundle=SeedBundle(
                **payload["seed_bundle"]
            ),
            data=DataSnapshot(
                **payload["data"]
            ),
            model=ModelSnapshot(
                **payload["model"]
            ),
            training=TrainingBudget(
                **payload["training"]
            ),
            evaluation=EvaluationSpec(
                **payload["evaluation"]
            ),
            manifest_version=payload.get(
                "manifest_version",
                MANIFEST_VERSION,
            ),
            parent_run_id=payload.get(
                "parent_run_id"
            ),
            created_at=payload["created_at"],
            notes=payload.get("notes"),
            extra=dict(
                payload.get("extra", {})
            ),
        )

        manifest.validate()

        stored_hash = payload.get(
            "manifest_hash"
        )

        if (
            stored_hash is not None
            and stored_hash != manifest.manifest_hash
        ):
            raise ValueError(
                "run manifest hash mismatch: "
                "stored={}, calculated={}".format(
                    stored_hash,
                    manifest.manifest_hash,
                )
            )

        return manifest


class RunManifestStore:
    """
    负责写入和读取 run_manifest.json。

    写入规则：

    1. 文件不存在：创建；
    2. 文件已经存在且 manifest_hash 一致：幂等复用；
    3. 文件已经存在但 manifest_hash 不同：拒绝覆盖。
    """

    def __init__(
        self,
        manifest_path: Path,
    ):
        self.manifest_path = manifest_path

    def write(
        self,
        manifest: RunManifest,
    ) -> RunManifest:
        """
        写入 run manifest。
        """

        manifest.validate()

        if self.manifest_path.exists():
            existing = self.load()

            if (
                existing.manifest_hash
                == manifest.manifest_hash
            ):
                return existing

            raise ValueError(
                "run manifest collision at {}: "
                "existing_hash={}, new_hash={}".format(
                    self.manifest_path,
                    existing.manifest_hash,
                    manifest.manifest_hash,
                )
            )

        write_json_atomic(
            self.manifest_path,
            manifest.to_dict(),
        )

        return manifest

    def load(self) -> RunManifest:
        """
        从磁盘加载并校验 run manifest。
        """

        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "run manifest does not exist: {}".format(
                    self.manifest_path
                )
            )

        payload = json.loads(
            self.manifest_path.read_text(
                encoding="utf-8",
            )
        )

        return RunManifest.from_dict(
            payload
        )


def build_demo_manifest() -> RunManifest:
    """
    构造一个完整的 A 线 demo run manifest。
    """

    return RunManifest(
        run_id="A-r1-seed42",
        experiment_name="clevr_curriculum",
        track="A",
        round_id="r1",
        config_hash="config-demo-hash",
        code_commit="git-demo-commit",
        environment_hash="environment-demo-hash",
        seed_bundle=SeedBundle(
            python=42,
            numpy=43,
            torch=44,
            torch_cuda=45,
            dataloader_worker_base=46,
            distributed_sampler=47,
            recipe_sampling=48,
            evaluation=49,
        ),
        data=DataSnapshot(
            dataset_version="clevr_spatial_v1",
            schema_version="canonical_v1",
            tag_version="static_v1",
            recipe_version="line_a_r1_v1",
            train_manifest_uri=(
                "data/manifests/clevr/"
                "line_a_r1_train.parquet"
            ),
            train_manifest_hash="train-demo-hash",
            val_manifest_uri=(
                "data/manifests/clevr/"
                "val_v1.parquet"
            ),
            val_manifest_hash="val-demo-hash",
            test_manifest_uri=(
                "data/manifests/clevr/"
                "test_v1.parquet"
            ),
            test_manifest_hash="test-demo-hash",
        ),
        model=ModelSnapshot(
            model_name="MiniMind-V",
            base_checkpoint_uri=(
                "checkpoints/minimind-v-base"
            ),
            base_checkpoint_hash="base-demo-hash",
            freeze_llm=2,
            trainable_parameter_count=1048576,
            total_parameter_count=26000000,
        ),
        training=TrainingBudget(
            learning_rate=1e-4,
            effective_batch_size=16,
            gradient_accumulation_steps=1,
            world_size=1,
            max_seq_len=512,
            max_steps=500,
            epochs=1.0,
            target_token_budget=1000000,
            target_image_exposure=8000,
            checkpoint_interval_steps=250,
            evaluation_interval_steps=250,
        ),
        evaluation=EvaluationSpec(
            suite_names=[
                "vqa",
                "bucket",
                "visual_dependency",
            ],
            evaluator_version="clevr_eval_v1",
            normalizer_version="answer_norm_v1",
            selection_metric="vqa.overall_accuracy",
            min_prediction_coverage=1.0,
            test_allowed_in_feedback=False,
        ),
        notes=(
            "Demo run manifest for projector warmup."
        ),
        extra={
            "stage_id": "stage_0",
        },
    )


def run_demo() -> None:
    """
    验证：

    1. manifest 可以写入；
    2. manifest 可以重新加载；
    3. 加载后的 hash 不变；
    4. 相同内容重复写入是幂等的；
    5. 相同路径写入不同实验配置会被拒绝。
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        run_root = (
            Path(temp_dir)
            / "runs"
            / "A"
            / "clevr_curriculum"
            / "A-r1-seed42"
        )

        manifest_path = (
            run_root
            / "run_manifest.json"
        )

        store = RunManifestStore(
            manifest_path
        )

        manifest = build_demo_manifest()

        written = store.write(
            manifest
        )

        loaded = store.load()

        assert (
            written.manifest_hash
            == loaded.manifest_hash
        )

        assert loaded.run_id == "A-r1-seed42"
        assert loaded.track == "A"
        assert loaded.model.freeze_llm == 2

        repeated = store.write(
            manifest
        )

        assert (
            repeated.manifest_hash
            == manifest.manifest_hash
        )

        changed_budget = replace(
            manifest.training,
            max_steps=1000,
        )

        changed_manifest = replace(
            manifest,
            training=changed_budget,
        )

        try:
            store.write(
                changed_manifest
            )
        except ValueError as exc:
            print(
                "run manifest collision rejected:",
                exc,
            )
        else:
            raise AssertionError(
                "changed run manifest was not rejected"
            )

        print("run manifest demo: PASS")
        print("run_id:", loaded.run_id)
        print("track:", loaded.track)
        print(
            "dataset_version:",
            loaded.data.dataset_version,
        )
        print(
            "recipe_version:",
            loaded.data.recipe_version,
        )
        print(
            "freeze_llm:",
            loaded.model.freeze_llm,
        )
        print(
            "max_steps:",
            loaded.training.max_steps,
        )
        print(
            "manifest_hash:",
            loaded.manifest_hash,
        )
        print(
            "manifest_path:",
            manifest_path,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run manifest utility and smoke test."
        )
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in smoke test",
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "reserved for future run manifest "
            "inspection commands"
        ),
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