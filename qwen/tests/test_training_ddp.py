"""训练启动器的 CPU 回归测试，不运行真实 Swift 或 GPU 训练。"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

from qwen.src.training import train_sft


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    with (ROOT / "configs" / "mvp.yaml").open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    cfg["output_root"] = "/tmp/qwen-test"
    cfg["train"]["micro_batch_size"] = 2
    return cfg


class TrainingDDPTests(unittest.TestCase):
    def test_same_global_batch_on_one_two_four_eight_gpus(self):
        cfg = configuration()
        before = copy.deepcopy(cfg)
        for cards, accumulation in ((1, 16), (2, 8), (4, 4), (8, 2)):
            with self.subTest(cards=cards):
                command, summary = train_sft.build_command(cfg, "round1", num_gpus=cards)
                self.assertEqual(summary["world_size"], cards)
                self.assertEqual(summary["gradient_accumulation_steps"], accumulation)
                self.assertEqual(summary["global_batch_size"], 32)
                self.assertEqual(summary["planned_sample_exposures"], 9600)
                self.assertEqual(summary["max_steps"], 300)
                options = dict(zip(
                    (key[2:] for key in command[command.index("--model")::2]),
                    command[command.index("--model") + 1::2],
                ))
                self.assertEqual(options["per_device_train_batch_size"], "2")
                self.assertEqual(options["gradient_accumulation_steps"], str(accumulation))
                if cards == 1:
                    self.assertEqual(command[:4], [sys.executable, "-m", "swift.cli.main", "sft"])
                else:
                    self.assertEqual(command[:3], [sys.executable, "-m", "torch.distributed.run"])
                    self.assertIn("--standalone", command)
                    self.assertEqual(command[command.index("--nproc_per_node") + 1], str(cards))
                    self.assertEqual(command[command.index("--module") + 1], "swift.cli.sft")
                    self.assertEqual(options["ddp_find_unused_parameters"], "false")
        self.assertEqual(cfg, before)

    def test_invalid_world_size_and_indivisible_batch(self):
        cfg = configuration()
        for cards in (0, -1, 3, 32):
            with self.subTest(cards=cards), self.assertRaises(ValueError):
                train_sft.build_command(cfg, "round1", num_gpus=cards)

    def test_output_override_does_not_change_data_or_budget(self):
        cfg = configuration()
        _, original = train_sft.build_command(cfg, "round1")
        _, trial = train_sft.build_command(cfg, "round1", num_gpus=8, output_dir="/tmp/qwen-test/sft_ddp8")
        self.assertEqual(original["train_path"], trial["train_path"])
        self.assertEqual(original["max_steps"], trial["max_steps"])
        self.assertEqual(Path(trial["output_dir"]), Path("/tmp/qwen-test/sft_ddp8"))

    def test_eight_gpu_micro_one_fallback(self):
        cfg = configuration()
        cfg["train"]["micro_batch_size"] = 1
        _, summary = train_sft.build_command(cfg, "smoke", num_gpus=8)
        self.assertEqual(summary["gradient_accumulation_steps"], 4)

    def test_default_single_gpu_behavior_is_preserved(self):
        cfg = configuration()
        cfg["train"]["micro_batch_size"] = 1
        command, summary = train_sft.build_command(cfg, "smoke")
        self.assertEqual(summary["world_size"], 1)
        self.assertEqual(summary["gradient_accumulation_steps"], 32)
        self.assertEqual(Path(summary["output_dir"]), Path(cfg["output_root"]) / "smoke" / "sft")
        self.assertNotIn("--ddp_find_unused_parameters", command)

    def prepare_cli(self, root):
        cfg = configuration()
        cfg["output_root"] = str(root)
        cfg["train"]["steps_per_round"] = 1
        data = root / "rounds" / "round01" / "data"
        data.mkdir(parents=True)
        (data / "train.jsonl").write_text("{}\n" * 32, encoding="utf-8")
        (data / "report.json").write_text(json.dumps({
            "status": "RECIPE_READY", "round": 1, "seed": cfg["seed"], "total_rows": 32,
        }), encoding="utf-8")
        model = root / "model"
        model.mkdir()
        (model / "weights.safetensors").write_bytes(b"fake")
        (model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"weight": "weights.safetensors"}}), encoding="utf-8"
        )
        cfg["model"]["name_or_path"] = str(model)
        config = root / "runtime.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        return config

    def run_main(self, arguments):
        with patch.object(sys, "argv", ["train_sft"] + arguments), contextlib.redirect_stdout(io.StringIO()):
            train_sft.main()

    def test_preview_accepts_cli_overrides_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare_cli(root)
            output = root / "smoke" / "sft_ddp8"
            self.run_main([
                "--config", str(config), "--mode", "smoke", "--num-gpus", "8",
                "--micro-batch-size", "2", "--smoke-steps", "60", "--output-dir", str(output),
            ])
            self.assertFalse(output.exists())

    def test_bad_gpu_selection_and_existing_output_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare_cli(root)
            output = root / "sft_ddp8"
            arguments = ["--config", str(config), "--num-gpus", "8", "--output-dir", str(output), "--execute"]
            for visible in ("", "3", "0,1,2,3,4,5,6,6", "0,1,2,3,4,5,6,"):
                with self.subTest(visible=visible), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": visible}), self.assertRaises(ValueError):
                    self.run_main(arguments)
                self.assertFalse(output.exists())
            output.mkdir()
            with self.assertRaises(FileExistsError):
                self.run_main(arguments)

    def test_actual_gpu_count_must_match_requested_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare_cli(root)
            output = root / "sft_ddp8"
            fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 7,
            ))
            with patch.object(train_sft, "version", return_value="4.5.3"), patch.dict(
                sys.modules, {"torch": fake_torch}
            ), patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7"}), self.assertRaises(RuntimeError):
                self.run_main([
                    "--config", str(config), "--num-gpus", "8",
                    "--output-dir", str(output), "--execute",
                ])
            self.assertFalse(output.exists())

    def test_smoke_step_override_cannot_change_formal_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.prepare_cli(Path(directory))
            with self.assertRaises(ValueError):
                self.run_main(["--config", str(config), "--mode", "round1", "--smoke-steps", "60"])

    def test_child_failure_is_logged_and_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.prepare_cli(root)
            output = root / "sft_ddp8"
            original_builder = train_sft.build_command
            child = [sys.executable, "-u", "-c", (
                "import os, sys; assert 'WORLD_SIZE' not in os.environ; "
                "assert 'NPROC_PER_NODE' not in os.environ; "
                "assert os.environ['SWIFT_SINGLE_DEVICE_MODE'] == '0'; "
                "print('stdout diagnostic'); print('stderr diagnostic', file=sys.stderr); sys.exit(7)"
            )]

            def fake_builder(*args, **kwargs):
                _, summary = original_builder(*args, **kwargs)
                return child, summary

            fake_torch = SimpleNamespace(cuda=SimpleNamespace(
                is_available=lambda: True, device_count=lambda: 8, is_bf16_supported=lambda: True,
            ))
            with patch.object(train_sft, "build_command", side_effect=fake_builder), patch.object(
                train_sft, "version", return_value="4.5.3"
            ), patch.dict(sys.modules, {"torch": fake_torch}), patch.dict(os.environ, {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "WORLD_SIZE": "99", "NPROC_PER_NODE": "99",
            }), self.assertRaises(subprocess.CalledProcessError):
                self.run_main([
                    "--config", str(config), "--mode", "smoke", "--num-gpus", "8",
                    "--output-dir", str(output), "--execute",
                ])
            log = (output / "console.log").read_text(encoding="utf-8")
            self.assertIn("stdout diagnostic", log)
            self.assertIn("stderr diagnostic", log)
            self.assertEqual(json.loads((output / "process_result.json").read_text())["returncode"], 7)
            launch = json.loads((output / "launch.json").read_text(encoding="utf-8"))
            self.assertEqual(launch["summary"]["world_size"], 8)


if __name__ == "__main__":
    unittest.main()
