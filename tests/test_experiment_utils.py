import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch import nn

from vgks.eval import evaluate_policy
from vgks.export import export_augmented_dataset
from vgks.experiment_logging import ExperimentLogger
from vgks.envs import resolve_env_name


class FakeSpace:
    def __init__(self, shape):
        self.shape = shape


class FakeEnv:
    def __init__(self):
        self.observation_space = FakeSpace((3,))
        self.action_space = FakeSpace((2,))
        self._steps = 0

    def reset(self):
        self._steps = 0
        return np.zeros(3, dtype=np.float32)

    def step(self, action):
        self._steps += 1
        reward = 1.0
        done = self._steps >= 3
        return np.ones(3, dtype=np.float32) * self._steps, reward, done, {}

    def get_normalized_score(self, raw_return):
        return raw_return / 3.0


class ConstantPolicy(nn.Module):
    def forward(self, observations):
        batch = observations.shape[0]
        return torch.zeros(batch, 2)


class ExperimentUtilsTests(unittest.TestCase):
    def test_logger_passes_wandb_mode_to_init(self):
        fake_wandb = mock.Mock()
        fake_run = mock.Mock()
        fake_wandb.init.return_value = fake_run

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(sys.modules, {"wandb": fake_wandb}):
            run_dir = Path(tmpdir)
            logger = ExperimentLogger(
                save_dir=run_dir,
                use_wandb=True,
                wandb_mode="offline",
                project="vgks-tests",
                group="td3bc",
                name="offline-run",
                config={"seed": 0},
            )
            logger.finish()

        fake_wandb.init.assert_called_once()
        self.assertEqual(fake_wandb.init.call_args[1]["mode"], "offline")

    def test_python_from_vgks_directory_still_imports_stdlib_logging(self):
        repo_root = Path("H:/codex_test/nips2026")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import logging; print(hasattr(logging, 'getLogger'))",
            ],
            cwd=repo_root / "vgks",
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "True")

    def test_evaluate_policy_returns_normalized_score(self):
        env = FakeEnv()
        policy = ConstantPolicy()

        metrics = evaluate_policy(env, policy, device="cpu", n_episodes=2)

        self.assertEqual(metrics["raw_return"], 3.0)
        self.assertEqual(metrics["normalized_score"], 100.0)
        self.assertEqual(metrics["episodes"], 2)

    def test_logger_writes_config_and_eval_locally(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            logger = ExperimentLogger(
                save_dir=run_dir,
                use_wandb=False,
                project="vgks-tests",
                group="bc",
                name="toy-run",
                config={"seed": 0},
            )
            logger.log_metrics({"loss": 1.2}, step=3)
            logger.write_eval({"normalized_score": 88.0})
            logger.finish()

            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "eval.json").exists())
            self.assertTrue((run_dir / "metrics.jsonl").exists())

            saved_eval = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_eval["normalized_score"], 88.0)

    def test_export_augmented_dataset_writes_npz(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = Path(tmpdir) / "augmented_dataset.npz"
            augmented = {
                "observations": torch.randn(4, 3),
                "actions": torch.randn(4, 2),
                "next_observations": torch.randn(4, 3),
                "q_values": torch.randn(4),
            }

            export_augmented_dataset(export_path, augmented)

            self.assertTrue(export_path.exists())
            with np.load(export_path) as data:
                self.assertEqual(tuple(data["observations"].shape), (4, 3))
                self.assertIn("q_values", data.files)

    def test_resolve_env_name_from_task_and_dataset(self):
        env_name = resolve_env_name(task="halfcheetah", dataset_name="medium-expert")
        self.assertEqual(env_name, "halfcheetah-medium-expert-v2")


if __name__ == "__main__":
    unittest.main()
