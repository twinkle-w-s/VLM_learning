"""学习动态的合成数据回归测试，不代表真实模型结果。"""

from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from qwen.src.learning.dynamics import build_dynamics
from qwen.src.learning.artifacts import load_history
from qwen.src.evaluation.nll_metrics import aggregate_nll


def snapshots():
    result = []
    for index, (sums, correct) in enumerate((
        ((2.0, 12.0, 10.0), (True, False)),
        ((1.0, 9.0, 12.0), (False, True)),
        ((0.5, 6.0, 8.0), (True, True)),
    )):
        rows = {}
        for sid, source, bucket, count, value in zip(
            ("a", "b", "t"), ("clevr", "clevr", "replay_text"),
            ("comparison|easy", "count|hard", "replay_text"),
            (1, 3, 2), sums,
        ):
            rows[sid] = {
                "sample_id": sid, "source": source, "bucket": bucket,
                "difficulty": "easy" if sid == "a" else None,
                "answer_token_count": count,
                "sum_nll": value, "mean_nll": value / count,
            }
        result.append({
            "tag": ("base", "r1", "r2")[index], "round": index,
            "cumulative_optimizer_steps": (0, 10, 30)[index],
            "nll_records": rows,
            "correctness": dict(zip(("a", "b"), correct)),
        })
    return result


def write_artifacts(directory: Path, stages=None):
    stages = stages or snapshots()
    entries = []
    for stage in stages:
        base = directory / stage["tag"]
        nll_dir, em_dir = base / "nll", base / "em"
        nll_dir.mkdir(parents=True)
        em_dir.mkdir(parents=True)
        checkpoint = None
        if stage["round"]:
            checkpoint = str(base / "checkpoint")
            Path(checkpoint).mkdir()
            previous = stages[stage["round"] - 1]["cumulative_optimizer_steps"]
            (Path(checkpoint) / "trainer_state.json").write_text(
                json.dumps({"global_step": stage["cumulative_optimizer_steps"] - previous}),
                encoding="utf-8",
            )
        records = list(stage["nll_records"].values())
        nll_protocol = {
            "subset": "probe", "complete_subset": True,
            "metric": "assistant_template_nll_v1",
            "token_scope": "assistant and supervised end markers",
            "aggregation": "sum_nll / supervised_token_count",
            "prompt_policy": "comparison_yes_no_v1",
            "model_path": "model", "checkpoint": checkpoint,
            "max_length": 2048, "max_pixels": 262144,
            "batch_size": 1, "torch_dtype": "bfloat16", "attn_impl": "sdpa",
            "evaluated_rows": len(records),
        }
        nll_metrics = {
            "status": "COMPLETE", "protocol": nll_protocol,
            "overall": aggregate_nll(records),
        }
        (nll_dir / "metrics.json").write_text(json.dumps(nll_metrics), encoding="utf-8")
        (nll_dir / "samples.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in records), encoding="utf-8"
        )
        predictions = [
            {
                "sample_id": sid, "bucket": stage["nll_records"][sid]["bucket"],
                "correct": flag, "reference": "yes", "normalized_reference": "yes",
                "prediction": "yes" if flag else "no",
                "normalized_prediction": "yes" if flag else "no",
                "question": f"Question {sid}", "image": f"{sid}.png",
            }
            for sid, flag in stage["correctness"].items()
        ]
        em_protocol = {
            key: value for key, value in nll_protocol.items()
            if key not in ("token_scope", "aggregation")
        }
        em_protocol.update({
            "metric": "clevr_normalized_em_v1", "temperature": 0,
            "max_new_tokens": 32, "evaluated_rows": len(predictions),
        })
        (em_dir / "metrics.json").write_text(json.dumps({
            "status": "COMPLETE", "protocol": em_protocol,
            "n": len(predictions), "correct": sum(stage["correctness"].values()),
            "accuracy": sum(stage["correctness"].values()) / len(predictions),
        }), encoding="utf-8")
        (em_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
        )
        entries.append({
            "tag": stage["tag"], "round": stage["round"],
            "nll_dir": str(nll_dir), "clevr_dir": str(em_dir),
        })
    return {"step_accounting": "round_local_reset", "snapshots": entries}


class DynamicsTests(unittest.TestCase):
    def test_learning_forgetting_and_recovery(self):
        rows, summary = build_dynamics(snapshots())
        a, b, text = rows
        self.assertEqual(a["forgetting_count"], 1)
        self.assertEqual(a["learning_count"], 1)
        self.assertEqual(a["latest_transition"], "learned")
        self.assertEqual(b["forgetting_count"], 0)
        self.assertEqual(b["learning_count"], 1)
        self.assertIsNone(text["forgetting_count"])
        self.assertEqual(text["correctness_history"], [None, None, None])
        self.assertAlmostEqual(a["latest"]["nll_delta_per_optimizer_step"], -0.025)
        self.assertEqual(summary["overall"]["forgetting_events"], 1)

    def test_group_nll_is_token_weighted(self):
        _, summary = build_dynamics(snapshots())
        group = summary["by_source"]["clevr"]
        self.assertEqual(group["history"][0]["mean_nll"], 3.5)
        self.assertEqual(group["history"][1]["mean_nll"], 2.5)
        self.assertEqual(group["history"][0]["accuracy"], 0.5)

    def test_baseline_only_has_no_invented_delta(self):
        rows, summary = build_dynamics(snapshots()[:1])
        self.assertEqual(summary["status"], "BASELINE_ONLY")
        self.assertIsNone(rows[0]["latest"]["nll_delta"])
        self.assertEqual(rows[0]["forgetting_count"], 0)

    def test_input_maps_are_not_modified(self):
        stages = snapshots()
        before = copy.deepcopy(stages)
        build_dynamics(stages)
        self.assertEqual(stages, before)

    def test_reject_missing_identity_changed_labels_or_steps(self):
        for mutation in ("identity", "bucket", "tokens", "steps", "correctness", "nan"):
            stages = snapshots()
            if mutation == "identity":
                stages[1]["nll_records"].pop("a")
            elif mutation == "bucket":
                stages[1]["nll_records"]["a"]["bucket"] = "different"
            elif mutation == "tokens":
                stages[1]["nll_records"]["a"]["answer_token_count"] = 2
            elif mutation == "steps":
                stages[1]["cumulative_optimizer_steps"] = 0
            elif mutation == "correctness":
                stages[1]["correctness"]["t"] = True
            else:
                stages[1]["nll_records"]["a"]["sum_nll"] = math.nan
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                build_dynamics(stages)

    def test_zero_baseline_nll_has_no_relative_gain(self):
        stages = snapshots()
        stages[0]["nll_records"]["a"].update(sum_nll=0.0, mean_nll=0.0)
        rows, _ = build_dynamics(stages[:2])
        self.assertIsNone(rows[0]["latest"]["relative_nll_gain"])

    def test_artifact_loading_uses_saved_checkpoint_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            stages = load_history(write_artifacts(Path(directory)))
            self.assertEqual([s["cumulative_optimizer_steps"] for s in stages], [0, 10, 30])
            self.assertEqual(build_dynamics(stages)[1]["status"], "DYNAMICS_READY")

    def test_reject_val_smoke_and_protocol_mismatch(self):
        for mutation in ("val", "smoke", "prompt", "checkpoint", "reference", "question", "image"):
            with tempfile.TemporaryDirectory() as directory:
                config = write_artifacts(Path(directory))
                stage = config["snapshots"][1]
                path = Path(stage["nll_dir"]) / "metrics.json"
                metrics = json.loads(path.read_text())
                if mutation == "val":
                    metrics["protocol"]["subset"] = "val"
                elif mutation == "smoke":
                    metrics["status"] = "SMOKE_COMPLETE"
                elif mutation == "prompt":
                    metrics["protocol"]["prompt_policy"] = "original_v1"
                elif mutation == "checkpoint":
                    metrics["protocol"]["checkpoint"] = None
                else:
                    predictions = Path(stage["clevr_dir"]) / "predictions.jsonl"
                    records = [json.loads(line) for line in predictions.read_text().splitlines()]
                    if mutation in ("question", "image"):
                        records[0][mutation] = "changed"
                    else:
                        records[0]["reference"] = "no"
                        records[0]["normalized_reference"] = "no"
                        records[0]["prediction"] = "yes"
                        records[0]["normalized_prediction"] = "yes"
                        records[0]["correct"] = False
                    predictions.write_text("".join(json.dumps(row) + "\n" for row in records))
                path.write_text(json.dumps(metrics))
                with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                    load_history(config)

    def test_cli_preview_execute_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "history.yaml"
            config_path.write_text(yaml.safe_dump(write_artifacts(root)), encoding="utf-8")
            output = root / "result"
            command = [
                sys.executable, "-m", "qwen.src.learning.collect_dynamics",
                "--history", str(config_path), "--output", str(output),
            ]
            preview = subprocess.run(command, capture_output=True)
            self.assertEqual(preview.returncode, 0, preview.stderr)
            self.assertFalse(output.exists())
            completed = subprocess.run(command + ["--execute"], capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue((output / "sample_dynamics.jsonl").is_file())
            self.assertTrue((output / "summary.json").is_file())
            repeated = subprocess.run(command + ["--execute"], capture_output=True)
            self.assertNotEqual(repeated.returncode, 0)


if __name__ == "__main__":
    unittest.main()
