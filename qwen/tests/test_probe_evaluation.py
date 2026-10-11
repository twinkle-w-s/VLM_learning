"""固定 probe 生成评估测试；模拟推理接口，不加载真实模型。"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from qwen.src.evaluation import evaluate_clevr


def prepare_fixture(root):
    manifests = root / "manifests" / "clevr"
    manifests.mkdir(parents=True)
    image = root / "image.png"
    # 此测试只检查路径；图像解码由真实模型运行验收。
    image.touch()
    model = root / "model"
    model.mkdir()
    row = {
        "sample_id": "clevr:probe:1", "source": "clevr",
        "split": "train", "scorable": True, "bucket": "task:comparison",
        "task_type": "comparison", "difficulty": "hard", "answer": "yes",
        "images": [str(image)],
        "messages": [
            {"role": "user", "content": "<image>\nAre there more cubes than spheres?\nAnswer with a single word or number."},
            {"role": "assistant", "content": "yes"},
        ],
    }
    for subset, split in (("probe", "train"), ("val", "val")):
        (manifests / f"{subset}.jsonl").write_text(
            json.dumps(dict(row, split=split)) + "\n", encoding="utf-8"
        )
    config = root / "runtime.yaml"
    config.write_text(yaml.safe_dump({
        "output_root": str(root), "model": {"name_or_path": str(model)},
        "prompt_policy": "comparison_yes_no_v1", "seed": 42,
        "train": {"max_length": 2048, "max_pixels": 262144},
        "evaluation": {"clevr_val_size": 1, "probe_size": {"clevr": 1}, "qa_max_new_tokens": 32},
        "flywheel": {"min_bucket_val": 1},
    }), encoding="utf-8")
    return config, row


def run_main(arguments):
    output = io.StringIO()
    with patch.object(sys, "argv", ["evaluate_clevr"] + arguments), contextlib.redirect_stdout(output):
        evaluate_clevr.main()
    return output.getvalue()


class ProbeEvaluationTests(unittest.TestCase):
    def test_preview_probe_and_default_val(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _ = prepare_fixture(root)
            for subset in ("val", "probe"):
                arguments = ["--config", str(config)]
                if subset == "probe":
                    arguments += ["--subset", "probe"]
                printed = run_main(arguments)
                protocol = json.loads(printed[:printed.index("PREVIEW ONLY")])
                self.assertEqual(protocol["subset"], subset)
                self.assertTrue(protocol["complete_subset"])
                self.assertEqual(protocol["full_val"], subset == "val")
                expected = root / "baseline" / ("clevr" if subset == "val" else "clevr_probe")
                self.assertEqual(Path(protocol["output"]), expected)
                self.assertFalse(expected.exists())

    def test_probe_requires_train_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, row = prepare_fixture(root)
            probe = root / "manifests" / "clevr" / "probe.jsonl"
            self.assertEqual(len(evaluate_clevr.load_val(probe, 1, split="train")), 1)
            with self.assertRaises(ValueError):
                evaluate_clevr.load_val(probe, 1)
            row["split"] = "val"
            probe.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate_clevr.load_val(probe, 1, split="train")

    def run_mock_generation(self, root, limit=None):
        config, _ = prepare_fixture(root)
        captured = []

        class Engine:
            def __init__(self, *args, **kwargs):
                self.template = SimpleNamespace()
                self.model = SimpleNamespace(eval=lambda: None, requires_grad_=lambda flag: None)

            def infer(self, requests, request_config):
                captured.extend(requests)
                return [SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content="yes"), finish_reason="stop"
                )])]

        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True),
            bfloat16="bfloat16", manual_seed=lambda seed: None,
            inference_mode=contextlib.nullcontext,
        )
        fake_swift = SimpleNamespace(TransformersEngine=Engine, InferRequest=SimpleNamespace, RequestConfig=SimpleNamespace)
        arguments = ["--config", str(config), "--subset", "probe", "--execute"]
        if limit is not None:
            arguments += ["--limit", str(limit)]
        with patch.dict(sys.modules, {"torch": fake_torch, "swift": fake_swift}), patch.dict(
            "os.environ", {"CUDA_VISIBLE_DEVICES": "0"}
        ), patch.object(evaluate_clevr, "version", return_value="4.5.3"):
            run_main(arguments)
        output = root / "baseline" / "clevr_probe" if limit is None else root / "smoke" / "eval_base_clevr_probe"
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        return metrics, captured

    def test_generation_has_no_answer_input_and_no_val_feedback(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, requests = self.run_mock_generation(Path(directory))
            self.assertEqual(metrics["status"], "COMPLETE")
            self.assertTrue(metrics["probe_dynamics_available"])
            self.assertFalse(metrics["feedback_eligible"])
            self.assertFalse(metrics["by_bucket"]["task:comparison"]["feedback_eligible"])
            self.assertEqual([message["role"] for message in requests[0].messages], ["user"])
            self.assertEqual(metrics["accuracy"], 1.0)

    def test_explicit_limit_is_smoke_even_when_all_rows_are_used(self):
        with tempfile.TemporaryDirectory() as directory:
            metrics, _ = self.run_mock_generation(Path(directory), limit=1)
            self.assertEqual(metrics["status"], "SMOKE_COMPLETE")
            self.assertFalse(metrics["probe_dynamics_available"])


if __name__ == "__main__":
    unittest.main()
