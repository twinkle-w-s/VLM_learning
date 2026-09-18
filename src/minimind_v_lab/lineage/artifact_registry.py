# src/minimind_v_lab/lineage/artifact_registry.py
#
# Shared P0 基础组件：
# 1. 计算文件或目录的稳定 hash
# 2. 保存 artifact manifest
# 3. 查询 artifact
# 4. 判断 artifact 是否完整
#
# 第一版只使用 Python 标准库，不依赖数据库和第三方包。
# 后续如果并发量增大，可以在不改变接口的前提下替换成 SQLite、
# PostgreSQL 或对象存储上的 catalog。

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def canonical_json(value: Any) -> str:
    """
    把 Python 对象转换成稳定的 JSON 字符串。

    为什么需要稳定 JSON：
    同一个字典如果插入顺序不同，普通 json.dumps 可能产生不同字符串。
    如果直接对字符串计算 hash，就会导致相同配置得到不同 hash。

    sort_keys=True 可以让字段顺序固定。
    separators 可以去掉不必要的空格。
    """

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_bytes(data: bytes) -> str:
    """
    对字节内容计算 SHA-256。
    """

    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    流式计算单个文件的 SHA-256。

    不一次性把整个文件读入内存，适合图片、Parquet 和 checkpoint。
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def iter_directory_files(path: Path) -> Iterable[Path]:
    """
    按稳定顺序遍历目录中的所有普通文件。

    只返回文件，不返回目录。
    """

    files = [
        item
        for item in path.rglob("*")
        if item.is_file()
    ]

    return sorted(files, key=lambda item: item.relative_to(path).as_posix())


def sha256_directory(path: Path) -> str:
    """
    对目录内容计算稳定 hash。

    hash 同时包含：
    - 相对路径
    - 文件内容

    因此：
    1. 文件内容变化会改变 hash；
    2. 文件名变化会改变 hash；
    3. 文件顺序变化不会改变 hash。

    这里默认跳过由系统生成的元数据文件：
    - _SUCCESS
    - manifest.json
    - stats.json

    这样 artifact 的内容 hash 不会被提交时间和统计文件影响。
    """

    digest = hashlib.sha256()

    ignored_names = {
        "_SUCCESS",
        "manifest.json",
        "stats.json",
    }

    for file_path in iter_directory_files(path):
        if file_path.name in ignored_names:
            continue

        relative_path = file_path.relative_to(path).as_posix()

        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")

        with file_path.open("rb") as file:
            while True:
                chunk = file.read(1024 * 1024)

                if not chunk:
                    break

                digest.update(chunk)

        digest.update(b"\0")

    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    """
    根据路径类型选择文件 hash 或目录 hash。
    """

    if not path.exists():
        raise FileNotFoundError("path does not exist: {}".format(path))

    if path.is_file():
        return sha256_file(path)

    if path.is_dir():
        return sha256_directory(path)

    raise ValueError("unsupported path type: {}".format(path))


@dataclass(frozen=True)
class ArtifactManifest:
    """
    一个数据产物的元数据。

    artifact 可以是：
    - raw 数据
    - bronze/silver/gold 数据
    - train manifest
    - token cache
    - prediction 文件
    - checkpoint 目录
    """

    artifact_id: str
    logical_name: str
    artifact_type: str
    uri: str

    dataset_version: str
    schema_version: str
    schema_hash: str
    content_hash: str

    parent_artifact_ids: List[str]
    producer_job_id: str
    config_hash: str
    code_commit: str
    environment_hash: str

    row_count: int
    token_count: int
    byte_count: int

    created_at: str
    status: str = "SUCCEEDED"
    complete: bool = True

    partition_spec: Optional[Dict[str, Any]] = None
    stats_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        转换成可以写入 JSONL 的字典。
        """

        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "ArtifactManifest":
        """
        从 JSON 字典恢复 ArtifactManifest。
        """

        return cls(
            artifact_id=value["artifact_id"],
            logical_name=value["logical_name"],
            artifact_type=value["artifact_type"],
            uri=value["uri"],
            dataset_version=value["dataset_version"],
            schema_version=value["schema_version"],
            schema_hash=value["schema_hash"],
            content_hash=value["content_hash"],
            parent_artifact_ids=list(value.get("parent_artifact_ids", [])),
            producer_job_id=value["producer_job_id"],
            config_hash=value["config_hash"],
            code_commit=value["code_commit"],
            environment_hash=value["environment_hash"],
            row_count=int(value["row_count"]),
            token_count=int(value.get("token_count", 0)),
            byte_count=int(value.get("byte_count", 0)),
            created_at=value["created_at"],
            status=value.get("status", "SUCCEEDED"),
            complete=bool(value.get("complete", False)),
            partition_spec=value.get("partition_spec"),
            stats_hash=value.get("stats_hash"),
        )


class ArtifactRegistry:
    """
    基于 append-only JSONL 的 artifact registry。

    第一版假设：
    - 同一时间只有一个进程写 registry；
    - artifact 一旦登记，不允许覆盖；
    - 新版本必须产生新的 artifact_id 或 dataset_version。

    这套接口后续可以替换底层存储，但上层调用方式不变。
    """

    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.registry_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _read_all(self) -> List[ArtifactManifest]:
        """
        读取全部 artifact 记录。

        registry 是 append-only，因此同一个 artifact_id 可能理论上出现多条记录。
        这里保留最后一条，方便未来支持状态更新。
        """

        if not self.registry_path.exists():
            return []

        latest: Dict[str, ArtifactManifest] = {}

        with self.registry_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    payload = json.loads(line)
                    manifest = ArtifactManifest.from_dict(payload)
                except Exception as exc:
                    raise ValueError(
                        "invalid registry line {}: {}".format(
                            line_number,
                            exc,
                        )
                    )

                latest[manifest.artifact_id] = manifest

        return list(latest.values())

    def _validate_manifest(self, manifest: ArtifactManifest) -> None:
        """
        检查 manifest 的最基本合法性。

        复杂的数据质量检查不放在 registry 中。
        registry 只负责 artifact 身份和完整性契约。
        """

        required_strings = {
            "artifact_id": manifest.artifact_id,
            "logical_name": manifest.logical_name,
            "artifact_type": manifest.artifact_type,
            "uri": manifest.uri,
            "dataset_version": manifest.dataset_version,
            "schema_version": manifest.schema_version,
            "schema_hash": manifest.schema_hash,
            "content_hash": manifest.content_hash,
            "producer_job_id": manifest.producer_job_id,
            "config_hash": manifest.config_hash,
            "code_commit": manifest.code_commit,
            "environment_hash": manifest.environment_hash,
            "created_at": manifest.created_at,
        }

        for field_name, field_value in required_strings.items():
            if not field_value:
                raise ValueError(
                    "manifest field is empty: {}".format(field_name)
                )

        if manifest.row_count < 0:
            raise ValueError("row_count must be non-negative")

        if manifest.token_count < 0:
            raise ValueError("token_count must be non-negative")

        if manifest.byte_count < 0:
            raise ValueError("byte_count must be non-negative")

        if manifest.status not in {
            "PENDING",
            "RUNNING",
            "SUCCEEDED",
            "FAILED",
            "REVIEW",
        }:
            raise ValueError(
                "unsupported artifact status: {}".format(
                    manifest.status
                )
            )

    def register(self, manifest: ArtifactManifest) -> ArtifactManifest:
        """
        登记一个 artifact。

        规则：
        1. artifact_id 已存在且内容完全相同：视为幂等重复，直接复用；
        2. artifact_id 已存在但内容不同：拒绝覆盖；
        3. artifact_id 不存在：追加一行；
        """

        self._validate_manifest(manifest)

        existing = {
            item.artifact_id: item
            for item in self._read_all()
        }

        if manifest.artifact_id in existing:
            previous = existing[manifest.artifact_id]

            if previous.to_dict() == manifest.to_dict():
                return previous

            raise ValueError(
                "artifact collision: {} already exists with different metadata".format(
                    manifest.artifact_id
                )
            )

        with self.registry_path.open("a", encoding="utf-8") as file:
            file.write(
                canonical_json(manifest.to_dict())
                + "\n"
            )

        return manifest

    def resolve(
        self,
        logical_name: str,
        version: Optional[str] = None,
        content_hash: Optional[str] = None,
    ) -> ArtifactManifest:
        """
        根据逻辑名称查询 artifact。

        如果提供 version，则同时匹配 dataset_version。
        如果提供 content_hash，则同时匹配 content_hash。
        """

        candidates = [
            item
            for item in self._read_all()
            if item.logical_name == logical_name
            and (
                version is None
                or item.dataset_version == version
            )
            and (
                content_hash is None
                or item.content_hash == content_hash
            )
            and item.complete
            and item.status == "SUCCEEDED"
        ]

        if not candidates:
            raise KeyError(
                "artifact not found: logical_name={}, version={}, content_hash={}".format(
                    logical_name,
                    version,
                    content_hash,
                )
            )

        candidates.sort(key=lambda item: item.created_at)

        return candidates[-1]

    def get_by_id(self, artifact_id: str) -> ArtifactManifest:
        """
        通过 artifact_id 精确查询。
        """

        for item in self._read_all():
            if item.artifact_id == artifact_id:
                return item

        raise KeyError(
            "artifact not found: {}".format(artifact_id)
        )

    def is_complete(self, artifact_id: str) -> bool:
        """
        判断 artifact 是否满足最基本的完整条件：

        - registry 中存在；
        - status == SUCCEEDED；
        - complete == True；
        - uri 对应路径存在；
        - 目录型 artifact 存在 _SUCCESS。
        """

        manifest = self.get_by_id(artifact_id)

        if manifest.status != "SUCCEEDED":
            return False

        if not manifest.complete:
            return False

        path = Path(manifest.uri)

        if not path.exists():
            return False

        if path.is_dir():
            success_marker = path / "_SUCCESS"

            if not success_marker.exists():
                return False

        return True

    def list_artifacts(
        self,
        logical_name: Optional[str] = None,
    ) -> List[ArtifactManifest]:
        """
        列出 artifact。

        可选 logical_name 过滤。
        """

        items = self._read_all()

        if logical_name is not None:
            items = [
                item
                for item in items
                if item.logical_name == logical_name
            ]

        return sorted(
            items,
            key=lambda item: (
                item.logical_name,
                item.created_at,
                item.artifact_id,
            ),
        )


def build_demo_manifest(
    artifact_dir: Path,
) -> ArtifactManifest:
    """
    创建一个最小 demo artifact。

    这个函数只用于验证 registry，不参与正式数据链路。
    """

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = artifact_dir / "data.txt"
    data_path.write_text(
        "demo artifact\n",
        encoding="utf-8",
    )

    success_path = artifact_dir / "_SUCCESS"
    success_path.write_text(
        "complete=true\n",
        encoding="utf-8",
    )

    return ArtifactManifest(
        artifact_id="demo-artifact-v1",
        logical_name="demo",
        artifact_type="fixture",
        uri=str(artifact_dir.resolve()),
        dataset_version="demo_v1",
        schema_version="demo_schema_v1",
        schema_hash="schema-demo",
        content_hash=sha256_path(artifact_dir),
        parent_artifact_ids=[],
        producer_job_id="artifact-registry-demo",
        config_hash="config-demo",
        code_commit="code-demo",
        environment_hash="environment-demo",
        row_count=1,
        token_count=2,
        byte_count=data_path.stat().st_size,
        created_at="2026-09-18T00:00:00Z",
        partition_spec={"split": "demo"},
    )


def run_demo() -> None:
    """
    运行最小自检：

    1. 创建临时 artifact；
    2. 写入 registry；
    3. 查询 artifact；
    4. 检查完整性；
    5. 重复 register，验证幂等；
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        artifact_dir = root / "artifact"
        registry_path = root / "registry" / "artifacts.jsonl"

        manifest = build_demo_manifest(artifact_dir)
        registry = ArtifactRegistry(registry_path)

        registered = registry.register(manifest)

        assert registered.artifact_id == "demo-artifact-v1"
        assert registry.is_complete("demo-artifact-v1")

        resolved = registry.resolve(
            logical_name="demo",
            version="demo_v1",
        )

        assert resolved.artifact_id == "demo-artifact-v1"

        repeated = registry.register(manifest)

        assert repeated.to_dict() == manifest.to_dict()

        print("artifact registry demo: PASS")
        print("artifact_id:", resolved.artifact_id)
        print("content_hash:", resolved.content_hash)
        print("registry_path:", registry_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Artifact registry utility and smoke test."
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in smoke test",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="reserved for future registry inspection commands",
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