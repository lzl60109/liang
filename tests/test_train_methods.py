import tempfile
import unittest
from pathlib import Path
import subprocess
import sys
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from vgks.generate_vgks import generate_augmented_dataset, save_generated_dataset
import yaml

from vgks.offline_rl import (
    StableTD3BCTrainer,
    build_td3bc_training_data,
    build_td3bc_training_sources,
    format_eval_progress,
    format_train_progress,
)
from vgks.train_cql import run_cql_training
from vgks.train_iql import run_iql_training
from vgks.train_kats import run_kats_training
from vgks.train_td3bc import run_td3bc_training
from vgks.train_tgcvg import run_tgcvg_training
from vgks.train_vgks import build_trainer_from_args

REPO_ROOT = Path(__file__).resolve().parents[1]


class MethodTrainingTests(unittest.TestCase):
    def test_cql_config_includes_aug_mixture_fields(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "offline_rl" / "cql.yaml").read_text(encoding="utf-8"))

        for key in ("raw_dataset_path", "aug_dataset_path", "mix_aug_ratio"):
            self.assertIn(key, config)

    def test_build_td3bc_dataset_uses_raw_only_when_mix_ratio_zero(self):
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

            data = build_td3bc_training_data(raw_dataset_path=raw_path, aug_dataset_path=aug_path, mix_aug_ratio=0.0, seed=0)

            self.assertEqual(data["observations"].shape[0], 10)
            self.assertTrue(np.allclose(data["observations"], 0.0))

    def test_build_td3bc_dataset_mixes_augmented_subset(self):
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

            data = build_td3bc_training_data(raw_dataset_path=raw_path, aug_dataset_path=aug_path, mix_aug_ratio=0.2, seed=0)

            self.assertEqual(data["observations"].shape[0], 12)
            self.assertEqual(int((data["observations"] == 1.0).all(axis=1).sum()), 2)

    def test_build_td3bc_dataset_drops_aug_only_extra_fields(self):
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
                q_values=np.ones(8, dtype=np.float32),
            )

            data = build_td3bc_training_data(raw_dataset_path=raw_path, aug_dataset_path=aug_path, mix_aug_ratio=0.2, seed=0)

            self.assertNotIn("q_values", data)

    def test_build_td3bc_training_sources_keeps_raw_critic_data_and_mixed_actor_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.zeros((10, 3), dtype=np.float32),
                actions=np.zeros((10, 2), dtype=np.float32),
                next_observations=np.zeros((10, 3), dtype=np.float32),
                rewards=np.arange(10, dtype=np.float32),
                terminals=np.zeros(10, dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.ones((8, 3), dtype=np.float32),
                actions=np.ones((8, 2), dtype=np.float32),
                next_observations=np.ones((8, 3), dtype=np.float32),
                q_values=np.full(8, 5.0, dtype=np.float32),
            )

            sources = build_td3bc_training_sources(
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.2,
                seed=0,
            )

            self.assertEqual(sources["critic_data"]["observations"].shape[0], 10)
            self.assertEqual(sources["actor_data"]["observations"].shape[0], 12)
            self.assertTrue(np.allclose(sources["critic_data"]["observations"], 0.0))
            self.assertEqual(int((sources["actor_data"]["observations"] == 1.0).all(axis=1).sum()), 2)

    def test_td3bc_config_includes_aug_mixture_fields(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "offline_rl" / "td3bc.yaml").read_text(encoding="utf-8"))

        for key in ("raw_dataset_path", "aug_dataset_path", "mix_aug_ratio"):
            self.assertIn(key, config)

    def test_iql_config_includes_aug_mixture_fields(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "offline_rl" / "iql.yaml").read_text(encoding="utf-8"))

        for key in ("raw_dataset_path", "aug_dataset_path", "mix_aug_ratio"):
            self.assertIn(key, config)

    def test_run_td3bc_training_accepts_raw_and_aug_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.random.randn(24, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(24, 2)).astype(np.float32),
                next_observations=np.random.randn(24, 3).astype(np.float32),
                rewards=np.random.randn(24).astype(np.float32),
                terminals=np.zeros(24, dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.random.randn(12, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(12, 2)).astype(np.float32),
                next_observations=np.random.randn(12, 3).astype(np.float32),
                rewards=np.random.randn(12).astype(np.float32),
                terminals=np.zeros(12, dtype=np.float32),
            )

            metrics = run_td3bc_training(
                dataset_path=None,
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.25,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=8,
                max_timesteps=4,
                eval_freq=2,
                log_every=1,
                seed=0,
                device="cpu",
                save_dir=tmpdir / "runs" / "td3bc_mix",
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="td3bc",
                wandb_name="mix-td3bc",
                eval_episodes=2,
                num_workers=0,
            )

            self.assertIn("normalized_score", metrics["eval"])

    def test_run_iql_training_accepts_raw_and_aug_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.random.randn(24, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(24, 2)).astype(np.float32),
                next_observations=np.random.randn(24, 3).astype(np.float32),
                rewards=np.random.randn(24).astype(np.float32),
                terminals=np.zeros(24, dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.random.randn(12, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(12, 2)).astype(np.float32),
                next_observations=np.random.randn(12, 3).astype(np.float32),
                q_values=np.full(12, 5.0, dtype=np.float32),
            )

            metrics = run_iql_training(
                dataset_path=None,
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.25,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=8,
                max_timesteps=4,
                eval_freq=2,
                log_every=1,
                seed=0,
                device="cpu",
                save_dir=tmpdir / "runs" / "iql_mix",
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="iql",
                wandb_name="mix-iql",
                eval_episodes=2,
                num_workers=0,
            )

            self.assertIn("normalized_score", metrics["eval"])

    def test_run_cql_training_accepts_raw_and_aug_dataset_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.random.randn(24, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(24, 2)).astype(np.float32),
                next_observations=np.random.randn(24, 3).astype(np.float32),
                rewards=np.random.randn(24).astype(np.float32),
                terminals=np.zeros(24, dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.random.randn(12, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(12, 2)).astype(np.float32),
                next_observations=np.random.randn(12, 3).astype(np.float32),
                q_values=np.full(12, 5.0, dtype=np.float32),
            )

            metrics = run_cql_training(
                dataset_path=None,
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.25,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=8,
                max_timesteps=4,
                eval_freq=2,
                log_every=1,
                seed=0,
                device="cpu",
                save_dir=tmpdir / "runs" / "cql_mix",
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="cql",
                wandb_name="mix-cql",
                eval_episodes=2,
                num_workers=0,
            )

            self.assertIn("normalized_score", metrics["eval"])
            self.assertEqual(metrics["data"]["source"], "mixed")

    def test_generate_augmented_dataset_respects_q_threshold(self):
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
                dataset_path=dataset_path,
                env_name=None,
                batch_size=4,
                epochs=1,
                num_workers=0,
                q_threshold=100.0,
            )

            self.assertEqual(int(augmented["observations"].shape[0]), 0)
            self.assertEqual(int(augmented["num_kept"]), 0)

    def test_dual_loader_td3bc_uses_raw_state_stats_for_normalization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            raw_path = tmpdir / "raw.npz"
            aug_path = tmpdir / "aug.npz"
            np.savez(
                raw_path,
                observations=np.zeros((24, 3), dtype=np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(24, 2)).astype(np.float32),
                next_observations=np.zeros((24, 3), dtype=np.float32),
                rewards=np.random.randn(24).astype(np.float32),
                terminals=np.zeros(24, dtype=np.float32),
            )
            np.savez(
                aug_path,
                observations=np.full((12, 3), 10.0, dtype=np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(12, 2)).astype(np.float32),
                next_observations=np.full((12, 3), 10.0, dtype=np.float32),
                q_values=np.full(12, 5.0, dtype=np.float32),
            )

            run_dir = tmpdir / "runs" / "td3bc_mix"
            run_td3bc_training(
                dataset_path=None,
                raw_dataset_path=raw_path,
                aug_dataset_path=aug_path,
                mix_aug_ratio=0.25,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=8,
                max_timesteps=4,
                eval_freq=4,
                log_every=4,
                seed=0,
                device="cpu",
                save_dir=run_dir,
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="td3bc",
                wandb_name="mix-td3bc-raw-stats",
                eval_episodes=2,
                num_workers=0,
            )
            saved = __import__("torch").load(run_dir / "checkpoint.pt", map_location="cpu")

            self.assertTrue(np.allclose(saved["state_mean"], np.zeros(3, dtype=np.float32)))

    def test_stable_td3bc_train_step_uses_policy_delay(self):
        trainer = StableTD3BCTrainer(
            state_dim=3,
            action_dim=2,
            hidden_dim=16,
            max_action=1.0,
            device="cpu",
            policy_freq=2,
        )
        batch = {
            "observations": __import__("torch").randn(4, 3),
            "actions": __import__("torch").randn(4, 2).clamp(-1.0, 1.0),
            "next_observations": __import__("torch").randn(4, 3),
            "rewards": __import__("torch").randn(4),
            "terminals": __import__("torch").zeros(4),
        }

        step1 = trainer.train_step(batch)
        step2 = trainer.train_step(batch)

        self.assertIn("critic_loss", step1)
        self.assertIn("q_mean", step1)
        self.assertIsNone(step1["actor_loss"])
        self.assertIsNotNone(step2["actor_loss"])

    def test_stable_td3bc_actor_step_can_use_different_batch_from_critic_step(self):
        import torch

        trainer = StableTD3BCTrainer(
            state_dim=3,
            action_dim=2,
            hidden_dim=16,
            max_action=1.0,
            device="cpu",
            policy_freq=1,
        )
        raw_batch = {
            "observations": torch.zeros(4, 3),
            "actions": torch.zeros(4, 2),
            "next_observations": torch.zeros(4, 3),
            "rewards": torch.ones(4),
            "terminals": torch.zeros(4),
        }
        actor_batch = {
            "observations": torch.ones(4, 3),
            "actions": torch.ones(4, 2).clamp(-1.0, 1.0),
            "next_observations": torch.ones(4, 3),
            "rewards": torch.zeros(4),
            "terminals": torch.zeros(4),
        }

        trainer.train_critic_step(raw_batch)
        metrics = trainer.train_actor_step(actor_batch)

        self.assertIn("actor_loss", metrics)
        self.assertIn("bc_loss", metrics)
        self.assertIn("q_mean", metrics)
        self.assertIsNotNone(metrics["actor_loss"])

    def test_td3bc_config_includes_stability_hyperparameters(self):
        config = yaml.safe_load((REPO_ROOT / "configs" / "offline_rl" / "td3bc.yaml").read_text(encoding="utf-8"))

        for key in ("discount", "tau", "policy_noise", "noise_clip", "policy_freq", "alpha"):
            self.assertIn(key, config)

    def test_run_td3bc_training_reports_stable_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=np.random.randn(32, 3).astype(np.float32),
                actions=np.random.uniform(-1.0, 1.0, size=(32, 2)).astype(np.float32),
                next_observations=np.random.randn(32, 3).astype(np.float32),
                rewards=np.random.randn(32).astype(np.float32),
                terminals=np.zeros(32, dtype=np.float32),
            )

            metrics = run_td3bc_training(
                dataset_path=dataset_path,
                env_name=None,
                state_dim=3,
                action_dim=2,
                hidden_dim=16,
                batch_size=8,
                max_timesteps=4,
                eval_freq=2,
                log_every=1,
                seed=0,
                device="cpu",
                save_dir=tmpdir / "runs" / "td3bc_stable",
                use_wandb=False,
                wandb_project="vgks-tests",
                wandb_group="td3bc",
                wandb_name="stable-td3bc",
                eval_episodes=2,
                num_workers=0,
            )

            self.assertIn("q_mean", metrics["train"])
            self.assertIn("normalized_score", metrics["eval"])

    def test_format_train_progress_includes_step_and_losses(self):
        line = format_train_progress("td3bc", step=1000, total_steps=5000, metrics={"actor_loss": 1.25, "critic_loss": 0.5})

        self.assertIn("[Train][TD3BC]", line)
        self.assertIn("step=1000/5000", line)
        self.assertIn("actor_loss=1.2500", line)
        self.assertIn("critic_loss=0.5000", line)

    def test_format_eval_progress_includes_step_and_normalized_score(self):
        line = format_eval_progress(
            "td3bc",
            step=5000,
            total_steps=10000,
            metrics={"return": 123.0, "normalized_score": 45.6},
        )

        self.assertIn("[Eval][TD3BC]", line)
        self.assertIn("step=5000/10000", line)
        self.assertIn("return=123.0000", line)
        self.assertIn("normalized_score=45.6000", line)

    def test_offline_rl_cli_supports_max_timesteps_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=np.random.randn(8, 3).astype(np.float32),
                actions=np.random.randn(8, 2).astype(np.float32),
                next_observations=np.random.randn(8, 3).astype(np.float32),
                rewards=np.random.randn(8).astype(np.float32),
                terminals=np.zeros(8, dtype=np.float32),
            )
            config_path = tmpdir / "td3bc_steps.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"dataset_path: {dataset_path.as_posix()}",
                        "state_dim: 3",
                        "action_dim: 2",
                        "hidden_dim: 16",
                        "batch_size: 4",
                        "max_timesteps: 4",
                        "eval_freq: 2",
                        "log_every: 1",
                        "seed: 0",
                        "device: cpu",
                        f"save_dir: {(tmpdir / 'runs_steps').as_posix()}",
                        "use_wandb: false",
                        "wandb_project: vgks-tests",
                        "wandb_group: td3bc",
                        "wandb_name: cli-td3bc-steps",
                        "eval_episodes: 2",
                        "num_workers: 0",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "train_td3bc.py", "--config", str(config_path)],
                cwd=Path("H:/codex_test/nips2026"),
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

    def test_run_td3bc_training_prints_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=np.random.randn(8, 3).astype(np.float32),
                actions=np.random.randn(8, 2).astype(np.float32),
                next_observations=np.random.randn(8, 3).astype(np.float32),
                rewards=np.random.randn(8).astype(np.float32),
                terminals=np.zeros(8, dtype=np.float32),
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                run_td3bc_training(
                    dataset_path=dataset_path,
                    env_name=None,
                    state_dim=3,
                    action_dim=2,
                    hidden_dim=16,
                    batch_size=4,
                    max_timesteps=2,
                    eval_freq=1,
                    log_every=1,
                    seed=0,
                    device="cpu",
                    save_dir=tmpdir / "runs" / "td3bc",
                    use_wandb=False,
                    wandb_project="vgks-tests",
                    wandb_group="td3bc",
                    wandb_name="print-td3bc",
                    eval_episodes=2,
                    num_workers=0,
                )

            output = stdout.getvalue()
            self.assertIn("[Train][TD3BC]", output)
            self.assertIn("[Eval][TD3BC]", output)

    def test_run_iql_training_prints_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=np.random.randn(8, 3).astype(np.float32),
                actions=np.random.randn(8, 2).astype(np.float32),
                next_observations=np.random.randn(8, 3).astype(np.float32),
                rewards=np.random.randn(8).astype(np.float32),
                terminals=np.zeros(8, dtype=np.float32),
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                run_iql_training(
                    dataset_path=dataset_path,
                    env_name=None,
                    state_dim=3,
                    action_dim=2,
                    hidden_dim=16,
                    batch_size=4,
                    max_timesteps=2,
                    eval_freq=1,
                    log_every=1,
                    seed=0,
                    device="cpu",
                    save_dir=tmpdir / "runs" / "iql",
                    use_wandb=False,
                    wandb_project="vgks-tests",
                    wandb_group="iql",
                    wandb_name="print-iql",
                    eval_episodes=2,
                    num_workers=0,
                )

            output = stdout.getvalue()
            self.assertIn("[Train][IQL]", output)
            self.assertIn("[Eval][IQL]", output)

    def test_run_cql_training_prints_progress(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=np.random.randn(8, 3).astype(np.float32),
                actions=np.random.randn(8, 2).astype(np.float32),
                next_observations=np.random.randn(8, 3).astype(np.float32),
                rewards=np.random.randn(8).astype(np.float32),
                terminals=np.zeros(8, dtype=np.float32),
            )

            stdout = StringIO()
            with redirect_stdout(stdout):
                run_cql_training(
                    dataset_path=dataset_path,
                    env_name=None,
                    state_dim=3,
                    action_dim=2,
                    hidden_dim=16,
                    batch_size=4,
                    max_timesteps=2,
                    eval_freq=1,
                    log_every=1,
                    seed=0,
                    device="cpu",
                    save_dir=tmpdir / "runs" / "cql",
                    use_wandb=False,
                    wandb_project="vgks-tests",
                    wandb_group="cql",
                    wandb_name="print-cql",
                    eval_episodes=2,
                    num_workers=0,
                )

            output = stdout.getvalue()
            self.assertIn("[Train][CQL]", output)
            self.assertIn("[Eval][CQL]", output)

    def test_offline_rl_cli_accepts_config_without_save_dir_flag(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            dataset_path = tmpdir / "dataset.npz"
            np.savez(
                dataset_path,
                observations=np.random.randn(8, 3).astype(np.float32),
                actions=np.random.randn(8, 2).astype(np.float32),
                next_observations=np.random.randn(8, 3).astype(np.float32),
                rewards=np.random.randn(8).astype(np.float32),
                terminals=np.zeros(8, dtype=np.float32),
            )
            config_path = tmpdir / "td3bc.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        f"dataset_path: {dataset_path.as_posix()}",
                        "state_dim: 3",
                        "action_dim: 2",
                        "hidden_dim: 16",
                        "batch_size: 4",
                        "epochs: 1",
                        "seed: 0",
                        "device: cpu",
                        f"save_dir: {(tmpdir / 'runs').as_posix()}",
                        "use_wandb: false",
                        "wandb_project: vgks-tests",
                        "wandb_group: td3bc",
                        "wandb_name: cli-td3bc",
                        "eval_episodes: 2",
                        "num_workers: 0",
                    ]
                ),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, "train_td3bc.py", "--config", str(config_path)],
                cwd=Path("H:/codex_test/nips2026"),
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)

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
