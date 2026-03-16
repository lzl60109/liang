import tempfile
import unittest
from pathlib import Path

import numpy as np

from vgks.train_bc import run_bc_training


class BCTrainingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
