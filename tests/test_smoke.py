import unittest

import torch

from vgks import (
    ConservativeCritic,
    KoopmanDynamicsModel,
    SigmaModel,
    ValueGuidedKoopmanTrainer,
)


class SmokeTests(unittest.TestCase):
    def test_public_api_builds_trainer(self):
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
            "observations": torch.zeros(2, 3),
            "next_observations": torch.zeros(2, 3),
        }
        metrics = trainer.compute_sigma_loss(batch)

        self.assertIn("total_loss", metrics)


if __name__ == "__main__":
    unittest.main()
