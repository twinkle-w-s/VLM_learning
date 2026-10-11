"""目录迁移回归检查：包导入、命令入口、提示与全量选择。

仅运行 CPU/标准数据操作，不加载模型、不训练、不访问服务器数据。
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from qwen.src.common.prompt_policy import (
    NEW_SUFFIX,
    OLD_SUFFIX,
    rewrite_record,
)
from qwen.src.data.prepare_clevr import select_clevr


ROOT = Path(__file__).resolve().parents[1]


class LayoutTests(unittest.TestCase):
    def test_internal_import_paths_exist(self):
        self.assertTrue((ROOT / "src/common/data_common.py").is_file())
        for path in (ROOT / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom):
                    continue
                module = node.module or ""
                if not module.startswith("qwen.src."):
                    continue
                target = ROOT.parent.joinpath(*module.split("."))
                with self.subTest(file=path.name, module=module):
                    self.assertTrue(
                        target.with_suffix(".py").is_file()
                        or (target / "__init__.py").is_file(),
                        str(target),
                    )
                    self.assertEqual(node.level, 0)

    def test_module_entrypoints_help(self):
        modules = (
            "data.materialize_recipe",
            "training.train_sft",
            "training.train_round1_typed",
            "training.verify_lora",
            "evaluation.evaluate_clevr",
            "evaluation.inspect_nll",
        )
        for module in modules:
            with self.subTest(module=module):
                result = subprocess.run(
                    [sys.executable, "-m", "qwen.src." + module, "--help"],
                    cwd=str(ROOT.parent),
                    capture_output=True,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(b"usage:", result.stdout)

    def test_prompt_is_idempotent_and_preserves_input(self):
        row = {
            "sample_id": "demo", "source": "clevr",
            "task_type": "comparison", "answer": "yes",
            "images": ["original.png"],
            "messages": [
                {"role": "user", "content": "Question\n" + OLD_SUFFIX},
                {"role": "assistant", "content": "yes"},
            ],
        }
        original = copy.deepcopy(row)
        changed, flag = rewrite_record(row)
        self.assertTrue(flag)
        self.assertTrue(changed["messages"][0]["content"].endswith(NEW_SUFFIX))
        self.assertEqual(row, original)
        self.assertEqual(changed["messages"][1], original["messages"][1])
        self.assertEqual(changed["images"], original["images"])
        repeated, flag = rewrite_record(changed)
        self.assertFalse(flag)
        self.assertEqual(repeated, changed)
        row["task_type"] = "count"
        self.assertEqual(rewrite_record(row), (row, False))
        # 纯文字记录没有 task_type，不应访问 CLEVR 专用字段。
        self.assertEqual(
            rewrite_record({"source": "replay_text"}),
            ({"source": "replay_text"}, False),
        )

    def test_null_limits_keep_all_and_use_shared_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            paths = {"image_root": str(base / "images")}
            for split, indices in (
                ("train", (1, 2)), ("val", (3,)), ("test", (4,)),
            ):
                rows = [
                    {
                        "question_index": index,
                        "image_filename": f"image_{index}.png",
                        "question": "Is this comparison true?",
                        "answer": "yes", "task_type": "comparison",
                        "difficulty": "easy",
                    }
                    for index in indices
                ]
                path = base / f"{split}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8",
                )
                paths[split] = str(path)
            cfg = {
                "seed": 42, "prompt_policy": "comparison_yes_no_v1",
                "data": {"clevr": paths, "train_pool_size": {"clevr": None}},
                "evaluation": {"clevr_val_size": None},
            }
            pools, stats = select_clevr(cfg)
            self.assertEqual(
                {key: len(rows) for key, rows in pools.items()},
                {"train": 2, "val": 1, "test": 1},
            )
            self.assertTrue(all(value == 0 for value in stats["image_overlap"].values()))
            for rows in pools.values():
                self.assertTrue(all(
                    row["messages"][0]["content"].endswith(NEW_SUFFIX)
                    for row in rows
                ))


if __name__ == "__main__":
    unittest.main()
