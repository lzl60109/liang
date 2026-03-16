import tempfile
import unittest
from pathlib import Path

import numpy as np

from vgks.train_kats import run_kats_training
from vgks.train_tgcvg import run_tgcvg_training


class MethodTrainingTests(unittest.TestCase):
    def test_kats_training_exports_augmented_dataset(self):
        observations = np.random.randn(16, 3).astype(np.float32)
        actions = np.random.randn(16, 2).astype(np.float32)
        next_observations = np.random.randn(16, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )

            run_dir = tmpdir / "runs" / "kats" / "toy-env" / "seed_0"
            metrics = run_kats_training(
                dataset_path=dataset_path,
                env_name=None,
                state_dim=3,
                action_dim=2,
                latent_dim=4,
                hidden_dim=16,
                batch_size=4,
                epochs=1,
                seed=0,
                device="cpu",
                save_dir=run_dir,
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="kats",
                wandb_name="toy-kats",
                eval_episodes=2,
            )

            self.assertTrue((run_dir / "augmented_dataset.npz").exists())
            self.assertTrue((run_dir / "eval.json").exists())
            self.assertIn("normalized_score", metrics["eval"])

    def test_tgcvg_training_exports_augmented_dataset(self):
        observations = np.random.randn(16, 3).astype(np.float32)
        actions = np.random.randn(16, 2).astype(np.float32)
        next_observations = np.random.randn(16, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )

            run_dir = tmpdir / "runs" / "tgcvg" / "toy-env" / "seed_0"
            metrics = run_tgcvg_training(
                dataset_path=dataset_path,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=4,
                epochs=1,
                seed=0,
                device="cpu",
                save_dir=run_dir,
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="tgcvg",
                wandb_name="toy-tgcvg",
                eval_episodes=2,
            )

            self.assertTrue((run_dir / "augmented_dataset.npz").exists())
            self.assertTrue((run_dir / "eval.json").exists())
            self.assertIn("normalized_score", metrics["eval"])


if __name__ == "__main__":
    unittest.main()
