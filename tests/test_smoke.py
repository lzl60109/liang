import numpy as np

from vgks import (
    ConservativeCritic,
    KoopmanDynamicsModel,
    SigmaModel,
    ValueGuidedKoopmanTrainer,
)


def test_public_api_builds_trainer():
    dynamics = KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=8)
    sigma = SigmaModel(latent_dim=4)
    critic = ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=8)
    trainer = ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma,
        critic=critic,
        action_dim=2,
    )

    batch = {
        "observations": np.zeros((2, 3)),
        "next_observations": np.zeros((2, 3)),
    }
    metrics = trainer.compute_sigma_loss(batch)

    assert "total_loss" in metrics
