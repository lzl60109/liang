import tempfile
import unittest
from pathlib import Path

import numpy as np

from vgks.train_vgks import build_trainer_from_args, run_training


class VGKSFullTrainingTests(unittest.TestCase):
    def test_vgks_run_writes_eval_history_and_best_checkpoint(self):
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

            save_dir = tmpdir / "runs" / "vgks" / "toy-env" / "seed_0"
            history = run_training(
                trainer=trainer,
                dataset_path=dataset_path,
                env_name=None,
                batch_size=4,
                epochs=3,
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
                eval_interval=1,
                save_best=True,
                run_name="toy-run",
            )

            self.assertTrue((save_dir / "eval_history.json").exists())
            self.assertTrue((save_dir / "best_checkpoint.pt").exists())
            self.assertTrue((save_dir / "last_checkpoint.pt").exists())
            self.assertGreaterEqual(len(history["eval_history"]), 3)
            self.assertIn("best_normalized_score", history)

    def test_vgks_run_supports_multiple_seeds(self):
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

            save_root = tmpdir / "runs" / "vgks" / "toy-env"
            for seed in [0, 1]:
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
                run_training(
                    trainer=trainer,
                    dataset_path=dataset_path,
                    env_name=None,
                    batch_size=4,
                    epochs=1,
                    shuffle=False,
                    num_workers=0,
                    save_dir=save_root / f"seed_{seed}",
                    state_dim=3,
                    action_dim=2,
                    hidden_dim=8,
                    seed=seed,
                    device="cpu",
                    use_wandb=False,
                    wandb_project="vgks-tests",
                    wandb_group="vgks",
                    wandb_name=f"toy-vgks-{seed}",
                    eval_episodes=2,
                    eval_interval=1,
                    save_best=True,
                    run_name=f"toy-run-{seed}",
                )

            self.assertTrue((save_root / "seed_0" / "eval.json").exists())
            self.assertTrue((save_root / "seed_1" / "eval.json").exists())


if __name__ == "__main__":
    unittest.main()
