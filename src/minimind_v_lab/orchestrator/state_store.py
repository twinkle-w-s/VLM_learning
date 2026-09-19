# src/minimind_v_lab/orchestrator/state_store.py
#
# Shared P0 状态管理组件。
#
# 负责：
# 1. 记录 job 的状态；
# 2. 保存 append-only events.jsonl；
# 3. 保存当前 pipeline_state.json 快照；
# 4. 校验合法状态转移；
# 5. 支持失败 job 的恢复；
# 6. 提供最小 smoke test。
#
# 当前版本使用本地 JSONL 和 JSON 文件。
# 后续可以在不改变上层接口的情况下替换成 SQLite 或数据库。

from __future__ import annotations

import argparse
import json
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


PENDING = "PENDING"
RUNNING = "RUNNING"
SUCCEEDED = "SUCCEEDED"
FAILED = "FAILED"
REVIEW = "REVIEW"
SKIPPED = "SKIPPED"
STALE = "STALE"

VALID_STATUSES = {
    PENDING,
    RUNNING,
    SUCCEEDED,
    FAILED,
    REVIEW,
    SKIPPED,
    STALE,
}


ALLOWED_TRANSITIONS = {
    None: {
        PENDING,
    },
    PENDING: {
        RUNNING,
        SKIPPED,
    },
    RUNNING: {
        SUCCEEDED,
        FAILED,
        REVIEW,
        STALE,
    },
    FAILED: {
        RUNNING,
    },
    REVIEW: {
        RUNNING,
    },
    STALE: {
        RUNNING,
    },
    SUCCEEDED: set(),
    SKIPPED: set(),
}


def utc_now_iso() -> str:
    """
    返回当前 UTC 时间。

    状态日志统一使用 UTC，避免服务器时区不同导致排序混乱。
    """

    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(
    output_path: Path,
    payload: Dict[str, Any],
) -> None:
    """
    原子写 JSON 文件。

    先写 .tmp 文件，再替换正式文件。
    这样即使进程中途退出，也不会留下半个 JSON。
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


class StateStore:
    """
    基于本地文件的 pipeline 状态存储。

    目录结构：

    state_root/
    ├── events.jsonl
    ├── pipeline_state.json
    └── state.lock

    当前实现假设：
    - 同一个 run 同时只有一个 writer；
    - 状态变化必须经过 transition；
    - 不允许直接手动把 FAILED 改成 SUCCEEDED。
    """

    def __init__(self, state_root: Path):
        self.state_root = state_root
        self.events_path = state_root / "events.jsonl"
        self.snapshot_path = state_root / "pipeline_state.json"
        self.lock_path = state_root / "state.lock"

        self.state_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        if not self.snapshot_path.exists():
            write_json_atomic(
                self.snapshot_path,
                {
                    "run_id": state_root.parent.name,
                    "created_at": utc_now_iso(),
                    "updated_at": utc_now_iso(),
                    "jobs": {},
                },
            )

    def _read_snapshot(self) -> Dict[str, Any]:
        """
        读取当前状态快照。

        如果文件不存在，创建一个空快照。
        """

        if not self.snapshot_path.exists():
            return {
                "run_id": self.state_root.parent.name,
                "created_at": utc_now_iso(),
                "updated_at": utc_now_iso(),
                "jobs": {},
            }

        payload = json.loads(
            self.snapshot_path.read_text(
                encoding="utf-8",
            )
        )

        if "jobs" not in payload:
            raise ValueError(
                "pipeline state is missing jobs field"
            )

        return payload

    @contextmanager
    def _lock(self) -> Iterator[None]:
        """
        获取一个非常轻量的本地文件锁。

        使用独占创建模式：
        - 文件不存在：创建成功，获得锁；
        - 文件已经存在：说明另一个进程正在写。

        当前阶段每个 run 只允许一个 writer，因此这个实现足够。
        """

        self.state_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            lock_file = self.lock_path.open(
                "x",
                encoding="utf-8",
            )
        except FileExistsError:
            raise RuntimeError(
                "state store is locked: {}".format(
                    self.lock_path
                )
            )

        try:
            lock_file.write(
                "pid-lock\n"
            )
            lock_file.flush()
            yield
        finally:
            lock_file.close()

            if self.lock_path.exists():
                self.lock_path.unlink()

    def _append_event(
        self,
        event: Dict[str, Any],
    ) -> None:
        """
        向 events.jsonl 追加一条事件。

        events.jsonl 是 append-only，不覆盖历史事件。
        """

        with self.events_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    event,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    def get(
        self,
        job_id: str,
    ) -> Dict[str, Any]:
        """
        获取某个 job 的当前状态。
        """

        snapshot = self._read_snapshot()

        if job_id not in snapshot["jobs"]:
            raise KeyError(
                "job not found: {}".format(job_id)
            )

        return dict(
            snapshot["jobs"][job_id]
        )

    def list_jobs(
        self,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        列出当前 run 中的 job。

        可以按 status 过滤。
        """

        snapshot = self._read_snapshot()

        jobs = list(
            snapshot["jobs"].values()
        )

        if status is not None:
            if status not in VALID_STATUSES:
                raise ValueError(
                    "invalid status: {}".format(status)
                )

            jobs = [
                job
                for job in jobs
                if job["status"] == status
            ]

        return sorted(
            jobs,
            key=lambda item: item["job_id"],
        )

    def transition(
        self,
        job_id: str,
        new_status: str,
        expected_old: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        执行一次状态转移。

        参数：

        job_id：
            job 的稳定名称。

        new_status：
            要进入的新状态。

        expected_old：
            调用方认为的旧状态。
            如果实际旧状态不同，则拒绝执行。
            这可以避免两个进程互相覆盖状态。

        metadata：
            本次事件附加信息，例如：
            - input_hash
            - output_hash
            - rows_in
            - rows_out
            - error_class
            - retryable
            - resume_from_shard
        """

        if new_status not in VALID_STATUSES:
            raise ValueError(
                "invalid new status: {}".format(
                    new_status
                )
            )

        with self._lock():
            snapshot = self._read_snapshot()
            jobs = snapshot["jobs"]

            previous_job = jobs.get(job_id)

            if previous_job is None:
                old_status = None
            else:
                old_status = previous_job["status"]

            if (
                expected_old is not None
                and old_status != expected_old
            ):
                raise RuntimeError(
                    "unexpected old status for {}: "
                    "expected={}, actual={}".format(
                        job_id,
                        expected_old,
                        old_status,
                    )
                )

            allowed = ALLOWED_TRANSITIONS.get(
                old_status,
                set(),
            )

            if new_status not in allowed:
                raise RuntimeError(
                    "illegal state transition: {} -> {}".format(
                        old_status,
                        new_status,
                    )
                )

            now = utc_now_iso()
            event_metadata = dict(
                metadata or {}
            )

            event = {
                "event_type": "JOB_STATUS_CHANGED",
                "run_id": snapshot["run_id"],
                "job_id": job_id,
                "old_status": old_status,
                "new_status": new_status,
                "timestamp": now,
                "metadata": event_metadata,
            }

            job_state = {
                "job_id": job_id,
                "status": new_status,
                "created_at": (
                    previous_job["created_at"]
                    if previous_job is not None
                    else now
                ),
                "updated_at": now,
                "attempt": (
                    int(previous_job.get("attempt", 0))
                    if previous_job is not None
                    else 0
                ),
                "last_event_type": event["event_type"],
                "last_metadata": event_metadata,
            }

            if old_status in {
                FAILED,
                REVIEW,
                STALE,
            } and new_status == RUNNING:
                job_state["attempt"] += 1

            jobs[job_id] = job_state
            snapshot["jobs"] = jobs
            snapshot["updated_at"] = now

            self._append_event(event)
            write_json_atomic(
                self.snapshot_path,
                snapshot,
            )

            return dict(job_state)

    def initialize_job(
        self,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        初始化 job，使其进入 PENDING。
        """

        return self.transition(
            job_id=job_id,
            new_status=PENDING,
            expected_old=None,
            metadata=metadata,
        )

    def mark_succeeded(
        self,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        将 RUNNING job 标记为 SUCCEEDED。
        """

        return self.transition(
            job_id=job_id,
            new_status=SUCCEEDED,
            expected_old=RUNNING,
            metadata=metadata,
        )

    def mark_failed(
        self,
        job_id: str,
        error_class: str,
        error_message: str,
        retryable: bool,
        resume_from_shard: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        将 RUNNING job 标记为 FAILED。
        """

        metadata = {
            "error_class": error_class,
            "error_message": error_message,
            "retryable": retryable,
            "resume_from_shard": resume_from_shard,
        }

        return self.transition(
            job_id=job_id,
            new_status=FAILED,
            expected_old=RUNNING,
            metadata=metadata,
        )

    def resumable_jobs(self) -> List[Dict[str, Any]]:
        """
        找出可以被恢复的 job。

        当前允许恢复：

        - FAILED
        - STALE
        - REVIEW

        REVIEW 是否恢复，最终仍需由上层规则决定。
        """

        jobs = self.list_jobs()

        return [
            job
            for job in jobs
            if job["status"] in {
                FAILED,
                STALE,
                REVIEW,
            }
        ]

    def resume_job(
        self,
        job_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        将 FAILED、STALE 或 REVIEW job 恢复到 RUNNING。
        """

        current = self.get(job_id)

        if current["status"] not in {
            FAILED,
            STALE,
            REVIEW,
        }:
            raise RuntimeError(
                "job is not resumable: {} status={}".format(
                    job_id,
                    current["status"],
                )
            )

        return self.transition(
            job_id=job_id,
            new_status=RUNNING,
            expected_old=current["status"],
            metadata=metadata or {
                "resume_reason": "manual_resume",
            },
        )

    def resume_run(self) -> List[Dict[str, Any]]:
        """
        返回当前 run 中所有可恢复 job。
        """

        return self.resumable_jobs()


def run_demo() -> None:
    """
    运行最小 smoke test。

    测试内容：

    1. 初始化 job；
    2. 正常执行到 SUCCEEDED；
    3. 另一个 job 执行到 FAILED；
    4. 恢复 FAILED job；
    5. 检查非法状态转移会被拒绝；
    6. 检查 events.jsonl 和 pipeline_state.json 存在。
    """

    with tempfile.TemporaryDirectory() as temp_dir:
        run_root = Path(temp_dir) / "runs" / "demo-run"
        state_root = run_root / "state"

        store = StateStore(state_root)

        first = store.initialize_job(
            "ingest_demo",
            metadata={
                "input": "demo.jsonl",
            },
        )

        assert first["status"] == PENDING

        running = store.transition(
            job_id="ingest_demo",
            new_status=RUNNING,
            expected_old=PENDING,
            metadata={
                "attempt": 1,
            },
        )

        assert running["status"] == RUNNING

        succeeded = store.mark_succeeded(
            job_id="ingest_demo",
            metadata={
                "rows_in": 10,
                "rows_out": 10,
            },
        )

        assert succeeded["status"] == SUCCEEDED

        second = store.initialize_job(
            "clean_demo",
        )

        assert second["status"] == PENDING

        store.transition(
            job_id="clean_demo",
            new_status=RUNNING,
            expected_old=PENDING,
        )

        failed = store.mark_failed(
            job_id="clean_demo",
            error_class="transient_io",
            error_message="temporary demo failure",
            retryable=True,
            resume_from_shard="part-00001",
        )

        assert failed["status"] == FAILED
        assert failed["attempt"] == 0

        resumed = store.resume_job(
            "clean_demo",
            metadata={
                "resume_reason": "retry_failed_shard",
                "resume_from_shard": "part-00001",
            },
        )

        assert resumed["status"] == RUNNING
        assert resumed["attempt"] == 1

        store.mark_succeeded(
            "clean_demo",
            metadata={
                "rows_in": 10,
                "rows_out": 9,
            },
        )

        try:
            store.transition(
                job_id="ingest_demo",
                new_status=RUNNING,
                expected_old=SUCCEEDED,
            )
        except RuntimeError as exc:
            print("illegal transition rejected:", exc)
        else:
            raise AssertionError(
                "illegal transition was not rejected"
            )

        assert store.get(
            "ingest_demo"
        )["status"] == SUCCEEDED

        assert store.get(
            "clean_demo"
        )["status"] == SUCCEEDED

        assert store.events_path.exists()
        assert store.snapshot_path.exists()

        event_lines = [
            line
            for line in store.events_path.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        assert len(event_lines) == 8

        print("state store demo: PASS")
        print("state_path:", store.snapshot_path)
        print("event_count:", len(event_lines))
        print("jobs:")

        for job in store.list_jobs():
            print(
                " - {}: {}".format(
                    job["job_id"],
                    job["status"],
                )
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pipeline state store utility and smoke test."
    )

    parser.add_argument(
        "--demo",
        action="store_true",
        help="run the built-in smoke test",
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="reserved for future state inspection commands",
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