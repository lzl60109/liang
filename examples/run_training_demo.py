import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgks.train_vgks import build_trainer_from_args, run_training


def main() -> None:
    rng = np.random.default_rng(0)
    observations = rng.normal(size=(32, 3)).astype(np.float32)
    actions = rng.normal(size=(32, 2)).astype(np.float32)
    next_observations = observations + 0.1 * rng.normal(size=(32, 3)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_path = Path(tmpdir) / "toy_dataset.npz"
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
            sigma_lr=1e-3,
            kats_checkpoint=None,
            critic_checkpoint=None,
        )
        metrics = run_training(
            trainer=trainer,
            dataset_path=dataset_path,
            env_name=None,
            batch_size=8,
            epochs=2,
            shuffle=True,
            num_workers=0,
            save_dir=Path(tmpdir) / "outputs",
            state_dim=3,
            action_dim=2,
            hidden_dim=16,
            seed=0,
            device="cpu",
            use_wandb=False,
            wandb_project="vgks-demo",
            wandb_group="vgks",
            wandb_name="toy-training-demo",
            eval_episodes=2,
        )

    print("Best normalized score:", metrics["best_normalized_score"])
    print("Last eval:", metrics["last"]["eval"])


if __name__ == "__main__":
    main()
