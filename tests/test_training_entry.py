import tempfile
import unittest
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from vgks.data import OfflineReplayDataset, build_dataloader, load_offline_dataset
from vgks.generate_vgks import generate_augmented_dataset, save_generated_dataset
from vgks.train_vgks import build_trainer_from_args, infer_dims_from_dataset_source, run_training


class TrainingEntryTests(unittest.TestCase):
    def test_generate_vgks_stage_writes_augmented_dataset_without_eval(self):
        observations = np.random.randn(8, 3).astype(np.float32)
        actions = np.random.randn(8, 2).astype(np.float32)
        next_observations = np.random.randn(8, 3).astype(np.float32)
        rewards = np.random.randn(8).astype(np.float32)
        terminals = np.zeros(8, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
                rewards=rewards,
                terminals=terminals,
            )

            trainer = build_trainer_from_args(
                state_dim=3,
                action_dim=2,
                latent_dim=4,
                hidden_dim=8,
                lambda_q=0.1,
                lambda_state_anchor=1.0,
                lambda_latent_anchor=0.1,
                q_clip_min=-20.0,
                q_clip_max=20.0,
                sigma_warmup_steps=0,
                sigma_lr=1e-2,
                kats_checkpoint=None,
                critic_checkpoint=None,
                device="cpu",
            )

            augmented = generate_augmented_dataset(
                trainer=trainer,
                dataset_path=dataset_path,
                env_name=None,
                batch_size=4,
                epochs=1,
                num_workers=0,
            )
            save_prefix = save_generated_dataset(tmpdir / "aug", "toy-env", augmented)

            self.assertTrue((save_prefix.with_suffix(".pkl")).exists())
            self.assertTrue((save_prefix.with_suffix(".npy")).exists())
            self.assertTrue((tmpdir / "aug" / "toy-env.npz").exists())

    def test_infer_dims_from_dataset_source(self):
        observations = np.random.randn(8, 3).astype(np.float32)
        actions = np.random.randn(8, 2).astype(np.float32)
        next_observations = np.random.randn(8, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )

            dims = infer_dims_from_dataset_source(dataset_path)

        self.assertEqual(dims["state_dim"], 3)
        self.assertEqual(dims["action_dim"], 2)

    def test_inner_train_vgks_script_runs_from_package_directory(self):
        repo_root = Path("H:/codex_test/nips2026")
        result = subprocess.run(
            [sys.executable, "train_vgks.py", "--help"],
            cwd=repo_root / "vgks",
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Value-Guided Koopman Symmetry", result.stdout)

    def test_load_offline_dataset_from_npz(self):
        data = {
            "observations": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "actions": np.array([[0.1], [0.2]], dtype=np.float32),
            "next_observations": np.array([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "dataset.npz"
            np.savez(path, **data)
            loaded = load_offline_dataset(path)

        self.assertTrue(np.allclose(loaded["observations"], data["observations"]))
        self.assertTrue(np.allclose(loaded["actions"], data["actions"]))

    def test_build_dataloader_returns_expected_keys(self):
        dataset = OfflineReplayDataset(
            {
                "observations": np.random.randn(5, 3).astype(np.float32),
                "actions": np.random.randn(5, 2).astype(np.float32),
                "next_observations": np.random.randn(5, 3).astype(np.float32),
            }
        )
        loader = build_dataloader(dataset, batch_size=2, shuffle=False)
        batch = next(iter(loader))

        self.assertIn("observations", batch)
        self.assertIn("actions", batch)
        self.assertIn("next_observations", batch)
        self.assertIsInstance(batch["observations"], torch.Tensor)

    def test_run_training_executes_one_epoch_with_checkpoint_loading(self):
        observations = np.random.randn(8, 3).astype(np.float32)
        actions = np.random.randn(8, 2).astype(np.float32)
        next_observations = np.random.randn(8, 3).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=observations,
                actions=actions,
                next_observations=next_observations,
            )

            trainer = build_trainer_from_args(
                state_dim=3,
                action_dim=2,
                latent_dim=4,
                hidden_dim=8,
                lambda_q=0.1,
                lambda_state_anchor=1.0,
                lambda_latent_anchor=0.1,
                q_clip_min=-20.0,
                q_clip_max=20.0,
                sigma_warmup_steps=0,
                sigma_lr=1e-2,
                kats_checkpoint=None,
                critic_checkpoint=None,
                device="cpu",
            )

            save_dir = tmpdir / "outputs"
            metrics = run_training(
                trainer=trainer,
                dataset_path=dataset_path,
                env_name=None,
                batch_size=4,
                epochs=1,
                shuffle=False,
                num_workers=0,
                save_dir=save_dir,
                state_dim=3,
                action_dim=2,
                hidden_dim=8,
                seed=0,
                device="cpu",
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="vgks",
                wandb_name="toy-vgks",
                eval_episodes=2,
            )

            self.assertTrue((save_dir / "metrics.json").exists())
            self.assertTrue((save_dir / "checkpoint.pt").exists())
            self.assertTrue((save_dir / "augmented_dataset.npz").exists())
            self.assertTrue((save_dir / "eval.json").exists())

        self.assertIn("last", metrics)
        self.assertIn("sigma", metrics["last"])
        self.assertIn("eval", metrics["last"])


if __name__ == "__main__":
    unittest.main()
