import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from vgks.train_corl_bc import build_corl_bc_training_data, run_corl_bc_training


class CORLBCTrainingTests(unittest.TestCase):
    def test_corl_bc_can_build_mixed_dataset_by_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.zeros((10, 3), dtype=np.float32),
                actions=np.zeros((10, 2), dtype=np.float32),
                next_observations=np.zeros((10, 3), dtype=np.float32),
                rewards=np.zeros(10, dtype=np.float32),
                terminals=np.zeros(10, dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.ones((8, 3), dtype=np.float32),
                actions=np.ones((8, 2), dtype=np.float32),
                next_observations=np.ones((8, 3), dtype=np.float32),
                rewards=np.ones(8, dtype=np.float32),
                terminals=np.zeros(8, dtype=np.float32),
            )

            data = build_corl_bc_training_data(
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.2,
                seed=0,
            )

            self.assertEqual(data["observations"].shape[0], 12)
            self.assertEqual(int((data["observations"] == 1.0).all(axis=1).sum()), 2)

    def test_corl_bc_training_accepts_mixed_paths_and_logs_progress(self):
        observations = np.random.randn(16, 3).astype(np.float32)
        actions = np.random.uniform(-1.0, 1.0, size=(16, 2)).astype(np.float32)
        next_observations = np.random.randn(16, 3).astype(np.float32)
        rewards = np.random.randn(16).astype(np.float32)
        terminals = np.zeros(16, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
                terminals=terminals,
            )
            np.savez(
                aug_path,
                observations=observations + 1.0,
                actions=np.clip(actions + 0.1, -1.0, 1.0),
                next_observations=next_observations + 1.0,
                rewards=rewards,
                terminals=terminals,
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                metrics = run_corl_bc_training(
                    dataset_path=None,
                    raw_dataset_path=raw_path,
                    aug_dataset_path=aug_path,
                    mix_aug_ratio=0.25,
                    env_name=None,
                    state_dim=3,
                    action_dim=2,
                    batch_size=8,
                    max_timesteps=4,
                    eval_freq=2,
                    log_every=1,
                    seed=0,
                    device="cpu",
                    save_dir=tmpdir / "runs" / "corl_bc",
                    use_wandb=False,
                    wandb_project="vgks-tests",
                    wandb_group="corl-bc",
                    wandb_name="toy-corl-bc",
                    eval_episodes=2,
                    frac=1.0,
                )

            self.assertTrue((tmpdir / "runs" / "corl_bc" / "checkpoint.pt").exists())
            self.assertIn("normalized_score", metrics["eval"])
            output = stdout.getvalue()
            self.assertIn("[Train][CORL-BC]", output)
            self.assertIn("[Eval][CORL-BC]", output)


if __name__ == "__main__":
    unittest.main()
