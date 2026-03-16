import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vgks import ConservativeCritic, KoopmanDynamicsModel, SigmaModel, ValueGuidedKoopmanTrainer


def main() -> None:
    dynamics = KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=16)
    sigma = SigmaModel(latent_dim=4)
    critic = ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=16)
    trainer = ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma,
        critic=critic,
        action_dim=2,
        lambda_q=0.1,
        lambda_state_anchor=1.0,
        lambda_latent_anchor=0.1,
    )

    batch = {
        "observations": np.array(
            [[0.2, -0.1, 0.3], [0.5, 0.4, -0.2], [0.1, 0.3, 0.6]],
            dtype=np.float32,
        ),
        "next_observations": np.array(
            [[0.25, -0.05, 0.35], [0.45, 0.35, -0.1], [0.15, 0.25, 0.55]],
            dtype=np.float32,
        ),
    }

    metrics = trainer.compute_sigma_loss(batch)
    augmented = trainer.augment_batch(batch, q_threshold=None)

    print("Sigma metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6f}")

    print("\nAugmented batch shapes:")
    for key, value in augmented.items():
        print(f"  {key}: {value.shape}")


if __name__ == "__main__":
    main()
