"""MMStar 的合成数据测试，不下载官方数据、不运行真实模型。"""

from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
from PIL import Image

from qwen.src.data import prepare_mmstar
from qwen.src.evaluation import compare_mmstar, evaluate_mmstar
from qwen.src.evaluation.mmstar_metrics import (
    DATASET, EXPECTED_ROWS, METRIC, PARSER_POLICY, PROMPT_POLICY, REVISION,
    compare_results, make_prompt, parse_letter, summarize,
)


def predictions():
    return [{
        "sample_id": f"mmstar:{index}", "category": f"category_{index % 6}", "l2_category": "subcategory",
        "question": f"Question {index}? Options: A: Alpha, B: Beta, C: Gamma, D: Delta",
        "image": "fixed.png", "reference": "ABCD"[index % 4], "prediction": "ABCD"[index % 4],
        "parsed_prediction": "ABCD"[index % 4], "correct": True, "finish_reason": "stop",
    } for index in range(EXPECTED_ROWS)]


def metrics(rows, checkpoint=None):
    return {
        **summarize(rows), "status": "COMPLETE", "feedback_eligible": False,
        "protocol": {
            "metric": METRIC, "dataset": DATASET, "dataset_revision": REVISION,
            "dataset_config": "val", "dataset_split": "val", "dataset_rows": EXPECTED_ROWS,
            "data_path": "fixed_manifest.jsonl", "model_path": "fixed_model", "checkpoint": checkpoint,
            "prompt_policy": PROMPT_POLICY, "parser_policy": PARSER_POLICY,
            "max_length": 2048, "max_pixels": 262144, "max_new_tokens": 8,
            "batch_size": 4, "temperature": 0, "torch_dtype": "bfloat16", "attn_impl": "sdpa", "seed": 42,
            "complete_subset": True, "evaluated_rows": len(rows),
        },
    }


def change_prediction(row, text):
    row["prediction"] = text
    row["parsed_prediction"] = parse_letter(text)
    row["correct"] = row["parsed_prediction"] == row["reference"]


def invoke(module, arguments):
    printed = io.StringIO()
    with patch.object(sys, "argv", [module.__name__] + arguments), contextlib.redirect_stdout(printed):
        module.main()
    return printed.getvalue()


class MMStarTests(unittest.TestCase):
    def test_parser_accepts_single_choice_and_rejects_ambiguous_answers(self):
        for text, expected in (("A", "A"), (" b. ", "B"), ("(C)", "C"), ("[D]。", "D")):
            with self.subTest(text=text):
                self.assertEqual(parse_letter(text), expected)
        for text in ("", "A or B", "AB", "E", "The answer is A.", "A\nBecause it is obvious."):
            with self.subTest(text=text):
                self.assertIsNone(parse_letter(text))

    def test_prompt_preserves_options_and_has_only_one_image_marker(self):
        prompt = make_prompt("See <image 1>. Options: A: first, B: second, C: third, D: fourth")
        self.assertEqual(prompt.count("<image>"), 1)
        self.assertNotIn("<image 1>", prompt)
        self.assertIn("Options: A: first, B: second, C: third, D: fourth", prompt)

    def test_invalid_predictions_remain_in_denominator(self):
        rows = predictions()[:2]
        change_prediction(rows[1], "A or B")
        result = summarize(rows)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["invalid_predictions"], 1)

    def test_paired_change_and_percentage_points(self):
        base, after = predictions(), predictions()
        change_prediction(base[2], "A")
        change_prediction(after[0], "B")
        change_prediction(after[1], "A or B")
        pairs, result = compare_results(metrics(base), base, metrics(after, "r1/checkpoint-300"), after)
        self.assertEqual(len(pairs), EXPECTED_ROWS)
        self.assertEqual(result["correct_to_wrong"], 2)
        self.assertEqual(result["wrong_to_correct"], 1)
        self.assertAlmostEqual(result["delta_pp"], -100 / EXPECTED_ROWS)
        self.assertEqual(result["after_invalid"], 1)
        self.assertEqual(result["direction"], "observed_decrease")
        self.assertFalse(result["feedback_eligible"])

    def test_smoke_protocol_changed_sample_or_inconsistent_score_rejected(self):
        for mutation in ("smoke", "batch", "revision", "question", "duplicate", "score", "base_adapter"):
            base, after = predictions(), predictions()
            before_metrics, after_metrics = metrics(base), metrics(after, "r1/checkpoint-300")
            if mutation == "smoke":
                after_metrics["status"] = "SMOKE_COMPLETE"
            elif mutation == "batch":
                after_metrics["protocol"]["batch_size"] = 1
            elif mutation == "revision":
                after_metrics["protocol"]["dataset_revision"] = "different"
            elif mutation == "question":
                after[0]["question"] = "changed"
            elif mutation == "duplicate":
                after[0]["sample_id"] = after[1]["sample_id"]
            elif mutation == "score":
                after[0]["correct"] = False
            else:
                before_metrics["protocol"]["checkpoint"] = "another_adapter"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                compare_results(before_metrics, base, after_metrics, after)

    def raw_rows(self):
        buffer = io.BytesIO()
        Image.new("RGB", (2, 2), color="red").save(buffer, format="PNG")
        blob = buffer.getvalue()
        return [{
            "index": index, "question": "Options: A: first, B: second", "answer": "A",
            "image": {"bytes": blob, "path": None}, "category": "category", "l2_category": "subcategory",
            "meta_info": {"source": "synthetic", "split": "val", "image_path": f"{index}.png"},
        } for index in range(3)]

    def test_export_preserves_original_image_bytes_and_external_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = self.raw_rows()
            report = prepare_mmstar.export_rows(rows, root, expected_rows=3)
            self.assertEqual(report["rows"], 3)
            self.assertEqual((root / "images" / "0000.png").read_bytes(), rows[0]["image"]["bytes"])
            manifest = [json.loads(line) for line in (root / "manifest.jsonl").read_text().splitlines()]
            self.assertEqual(manifest[0]["split"], "external_eval")
            self.assertNotIn("messages", manifest[0])
            with self.assertRaises(FileExistsError):
                prepare_mmstar.export_rows(rows, root, expected_rows=3)

    def test_bad_image_or_duplicate_index_never_creates_ready_report(self):
        for mutation in ("image", "duplicate", "count"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                rows = self.raw_rows()
                if mutation == "image":
                    rows[0]["image"]["bytes"] = b"broken image"
                elif mutation == "duplicate":
                    rows[1]["index"] = rows[0]["index"]
                else:
                    rows.pop()
                with self.subTest(mutation=mutation), self.assertRaises((ValueError, OSError)):
                    prepare_mmstar.export_rows(rows, root, expected_rows=3)
                self.assertFalse((root / "report.json").exists())

    def test_prepare_preview_does_not_download_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new_mmstar"
            printed = invoke(prepare_mmstar, ["--data-root", str(output)])
            self.assertIn("PREVIEW ONLY", printed)
            self.assertFalse(output.exists())

    def prepared_fixture(self, root):
        data = root / "mmstar"
        data.mkdir()
        image = data / "fixed.png"
        Image.new("RGB", (2, 2)).save(image)
        rows = [{
            "sample_id": row["sample_id"], "source": "mmstar", "split": "external_eval",
            "image": str(image), "question": row["question"], "answer": row["reference"],
            "category": row["category"], "l2_category": row["l2_category"],
        } for row in predictions()]
        (data / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        (data / "report.json").write_text(json.dumps({
            "status": "MMSTAR_READY", "dataset": DATASET, "dataset_revision": REVISION,
            "rows": EXPECTED_ROWS, "categories": {f"category_{index}": 250 for index in range(6)},
        }), encoding="utf-8")
        model = root / "model"
        model.mkdir()
        cfg = {
            "model": {"name_or_path": str(model)}, "seed": 42,
            "train": {"max_length": 2048, "max_pixels": 262144},
            "evaluation": {"mmstar": {
                "dataset": DATASET, "config": "val", "split": "val", "max_new_tokens": 8, "feedback": False,
            }},
        }
        config = root / "runtime.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return data, config

    def test_evaluation_preview_does_not_load_gpu_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.prepared_fixture(root)
            output = root / "eval"
            printed = invoke(evaluate_mmstar, ["--config", str(config), "--data-root", str(data), "--output", str(output)])
            self.assertIn("PREVIEW ONLY", printed)
            self.assertFalse(output.exists())
            with self.assertRaises(ValueError):
                invoke(evaluate_mmstar, ["--config", str(config), "--output", str(output), "--batch-size", "0"])

    def test_changed_category_counts_are_rejected_before_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            data, _ = self.prepared_fixture(Path(directory))
            report_path = data / "report.json"
            report = json.loads(report_path.read_text())
            report["categories"]["category_0"] = 249
            report_path.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                evaluate_mmstar.load_mmstar(data)

    def test_mock_batched_evaluation_keeps_answers_out_of_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, config = self.prepared_fixture(root)
            requests_seen, batch_sizes = [], []

            class Engine:
                def __init__(self, *args, **kwargs):
                    self.template = SimpleNamespace()
                    self.model = SimpleNamespace(eval=lambda: None, requires_grad_=lambda flag: None)

                def infer(self, requests, request_config, use_tqdm=False):
                    requests_seen.extend(requests)
                    batch_sizes.append(len(requests))
                    return [SimpleNamespace(choices=[SimpleNamespace(
                        message=SimpleNamespace(content="ABCD"[int(re.search(r"Question (\d+)\?", request.messages[0]["content"]).group(1)) % 4]),
                        finish_reason="stop",
                    )]) for request in requests]

            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, is_bf16_supported=lambda: True),
                manual_seed=lambda seed: None, bfloat16="bfloat16", inference_mode=contextlib.nullcontext,
            )
            fake_swift = SimpleNamespace(TransformersEngine=Engine, InferRequest=SimpleNamespace, RequestConfig=SimpleNamespace)
            output = root / "eval"
            with patch.dict(sys.modules, {"torch": fake_torch, "swift": fake_swift}), patch.dict(
                "os.environ", {"CUDA_VISIBLE_DEVICES": "3"}
            ), patch.object(evaluate_mmstar, "version", return_value="4.5.3"):
                invoke(evaluate_mmstar, [
                    "--config", str(config), "--data-root", str(data), "--output", str(output),
                    "--batch-size", "2", "--limit", "5", "--execute",
                ])
            result = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "SMOKE_COMPLETE")
            self.assertEqual(result["n"], 5)
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(batch_sizes, [2, 2, 1])
            self.assertTrue(all([message["role"] for message in request.messages] == ["user"] for request in requests_seen))
            self.assertTrue(all(len(request.images) == 1 for request in requests_seen))
            self.assertFalse(result["feedback_eligible"])

    def test_comparison_cli_preview_execute_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = predictions()
            for name, checkpoint in (("base", None), ("after", "r1/checkpoint-300")):
                source = root / name
                source.mkdir()
                (source / "metrics.json").write_text(json.dumps(metrics(rows, checkpoint)), encoding="utf-8")
                (source / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "comparison"
            arguments = ["--base-dir", str(root / "base"), "--after-dir", str(root / "after"), "--output", str(output)]
            invoke(compare_mmstar, arguments)
            self.assertFalse(output.exists())
            invoke(compare_mmstar, arguments + ["--execute"])
            result = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
            self.assertEqual(result["delta_pp"], 0.0)
            self.assertEqual(len((output / "paired_changes.jsonl").read_text().splitlines()), EXPECTED_ROWS)
            with self.assertRaises(FileExistsError):
                invoke(compare_mmstar, arguments + ["--execute"])


if __name__ == "__main__":
    unittest.main()
