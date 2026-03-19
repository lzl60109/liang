import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from vgks.train_bc import build_bc_training_data, run_bc_training


class BCTrainingTests(unittest.TestCase):
    def test_bc_can_build_mixed_dataset_by_ratio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.zeros((10, 3), dtype=np.float32),
                actions=np.zeros((10, 2), dtype=np.float32),
                next_observations=np.zeros((10, 3), dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.ones((8, 3), dtype=np.float32),
                actions=np.ones((8, 2), dtype=np.float32),
                next_observations=np.ones((8, 3), dtype=np.float32),
            )

            data = build_bc_training_data(
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.2,
                seed=0,
            )

            self.assertEqual(data["observations"].shape[0], 12)
            self.assertEqual(int((data["observations"] == 1.0).all(axis=1).sum()), 2)

    def test_bc_training_writes_eval_and_checkpoint(self):
        observations = np.random.randn(12, 3).astype(np.float32)
        actions = np.random.randn(12, 2).astype(np.float32)
        next_observations = np.random.randn(12, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )

            run_dir = tmpdir / "runs" / "bc" / "toy-env" / "seed_0"
            metrics = run_bc_training(
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
                wandb_group="bc",
                wandb_name="toy-bc",
                eval_episodes=2,
            )

            self.assertTrue((run_dir / "config.json").exists())
            self.assertTrue((run_dir / "eval.json").exists())
            self.assertTrue((run_dir / "checkpoint.pt").exists())
            self.assertIn("normalized_score", metrics["eval"])

    def test_bc_training_accepts_raw_and_aug_dataset_paths(self):
        observations = np.random.randn(12, 3).astype(np.float32)
        actions = np.random.randn(12, 2).astype(np.float32)
        next_observations = np.random.randn(12, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )
            np.savez(
                aug_path,
                observations=observations + 1.0,
                actions=actions + 1.0,
                next_observations=next_observations + 1.0,
            )

            run_dir = tmpdir / "runs" / "bc_mix" / "toy-env" / "seed_0"
            metrics = run_bc_training(
                dataset_path=None,
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.25,
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
                wandb_group="bc",
                wandb_name="toy-bc-mix",
                eval_episodes=2,
            )

            self.assertTrue((run_dir / "eval.json").exists())
            self.assertIn("normalized_score", metrics["eval"])

    def test_bc_training_prints_progress(self):
        observations = np.random.randn(12, 3).astype(np.float32)
        actions = np.random.randn(12, 2).astype(np.float32)
        next_observations = np.random.randn(12, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                run_bc_training(
                    dataset_path=dataset_path,
                    env_name=None,
                    state_dim=3,
                    action_dim=2,
                    hidden_dim=16,
                    batch_size=4,
                    epochs=2,
                    seed=0,
                    device="cpu",
                    save_dir=tmpdir / "runs" / "bc_print",
                    use_wandb=False,
                    wandb_project="vgks-tests",
                    wandb_group="bc",
                    wandb_name="toy-bc-print",
                    eval_episodes=2,
                )

            output = stdout.getvalue()
            self.assertIn("[Train][BC]", output)
            self.assertIn("[Eval][BC]", output)


if __name__ == "__main__":
    unittest.main()
