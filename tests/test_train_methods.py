import tempfile
import unittest
from pathlib import Path

import numpy as np

from vgks.generate_vgks import generate_augmented_dataset, save_generated_dataset
from vgks.train_cql import run_cql_training
from vgks.train_iql import run_iql_training
from vgks.train_kats import run_kats_training
from vgks.train_td3bc import run_td3bc_training
from vgks.train_tgcvg import run_tgcvg_training
from vgks.train_vgks import build_trainer_from_args


class MethodTrainingTests(unittest.TestCase):
    def test_offline_rl_entrypoints_train_on_generated_vgks_dataset(self):
        observations = np.random.randn(16, 3).astype(np.float32)
        actions = np.random.randn(16, 2).astype(np.float32)
        next_observations = np.random.randn(16, 3).astype(np.float32)
        rewards = np.random.randn(16).astype(np.float32)
        terminals = np.zeros(16, dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_dataset_path = tmpdir / "dataset.npz"
            np.savez(
                raw_dataset_path,
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
                hidden_dim=16,
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
                dataset_path=raw_dataset_path,
                env_name=None,
                batch_size=4,
                epochs=1,
                num_workers=0,
            )
            generated_prefix = save_generated_dataset(tmpdir / "generated", "toy-env", augmented)
            generated_dataset_path = generated_prefix.with_suffix(".pkl")

            td3bc_run = tmpdir / "runs" / "td3bc"
            iql_run = tmpdir / "runs" / "iql"
            cql_run = tmpdir / "runs" / "cql"

            td3bc_metrics = run_td3bc_training(
                dataset_path=generated_dataset_path,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=4,
                epochs=1,
                seed=0,
                device="cpu",
                save_dir=td3bc_run,
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="td3bc",
                wandb_name="toy-td3bc",
                eval_episodes=2,
            )
            iql_metrics = run_iql_training(
                dataset_path=generated_dataset_path,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=4,
                epochs=1,
                seed=0,
                device="cpu",
                save_dir=iql_run,
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="iql",
                wandb_name="toy-iql",
                eval_episodes=2,
            )
            cql_metrics = run_cql_training(
                dataset_path=generated_dataset_path,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=4,
                epochs=1,
                seed=0,
                device="cpu",
                save_dir=cql_run,
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="cql",
                wandb_name="toy-cql",
                eval_episodes=2,
            )

            self.assertTrue((td3bc_run / "eval.json").exists())
            self.assertTrue((iql_run / "eval.json").exists())
            self.assertTrue((cql_run / "eval.json").exists())
            self.assertIn("normalized_score", td3bc_metrics["eval"])
            self.assertIn("normalized_score", iql_metrics["eval"])
            self.assertIn("normalized_score", cql_metrics["eval"])

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
