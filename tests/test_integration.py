import tempfile
import unittest
from pathlib import Path

import torch

from vgks import ConservativeCritic, KoopmanDynamicsModel, SigmaModel, ValueGuidedKoopmanTrainer
from vgks.integration import load_kats_checkpoint, load_tgcvg_critic_checkpoint


class IntegrationTests(unittest.TestCase):
    def test_load_kats_checkpoint_maps_weights(self):
        model = KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=6)
        checkpoint = {
            "layer1.weight": torch.full_like(model.layer1.weight, 0.11),
            "layer1.bias": torch.full_like(model.layer1.bias, 0.12),
            "layer2.weight": torch.full_like(model.layer2.weight, 0.21),
            "layer2.bias": torch.full_like(model.layer2.bias, 0.22),
            "layer3.weight": torch.full_like(model.layer3.weight, 0.31),
            "layer3.bias": torch.full_like(model.layer3.bias, 0.32),
            "layerK.weight": torch.full_like(model.layerK.weight, 0.41),
            "layer3inv.weight": torch.full_like(model.layer3inv.weight, 0.51),
            "layer3inv.bias": torch.full_like(model.layer3inv.bias, 0.52),
            "layer2inv.weight": torch.full_like(model.layer2inv.weight, 0.61),
            "layer2inv.bias": torch.full_like(model.layer2inv.bias, 0.62),
            "layer1inv.weight": torch.full_like(model.layer1inv.weight, 0.71),
            "layer1inv.bias": torch.full_like(model.layer1inv.bias, 0.72),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "kats_sysmodel.pt"
            torch.save(checkpoint, path)
            load_kats_checkpoint(model, path)

        self.assertTrue(torch.allclose(model.layer1.weight, checkpoint["layer1.weight"]))
        self.assertTrue(torch.allclose(model.layerK.weight, checkpoint["layerK.weight"]))
        self.assertTrue(torch.allclose(model.layer1inv.bias, checkpoint["layer1inv.bias"]))

    def test_load_tgcvg_checkpoint_maps_critics(self):
        critic = ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=8)
        q1_state = critic.q1_net.state_dict()
        q2_state = critic.q2_net.state_dict()
        q1_state = {key: torch.full_like(value, 0.15) for key, value in q1_state.items()}
        q2_state = {key: torch.full_like(value, -0.25) for key, value in q2_state.items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "critic.pt"
            torch.save({"critic1": q1_state, "critic2": q2_state}, path)
            load_tgcvg_critic_checkpoint(critic, path)

        self.assertTrue(torch.allclose(critic.q1_net.network[0].weight, q1_state["network.0.weight"]))
        self.assertTrue(torch.allclose(critic.q2_net.network[0].weight, q2_state["network.0.weight"]))

    def test_train_sigma_epoch_returns_averaged_metrics(self):
        torch.manual_seed(0)
        trainer = ValueGuidedKoopmanTrainer(
            dynamics=KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=8),
            sigma_model=SigmaModel(latent_dim=4),
            critic=ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=8),
            action_dim=2,
            sigma_lr=1e-2,
        )
        batches = [
            {
                "observations": torch.randn(4, 3),
                "next_observations": torch.randn(4, 3),
            }
            for _ in range(3)
        ]

        metrics = trainer.train_sigma_epoch(batches)

        self.assertIn("total_loss", metrics)
        self.assertIn("mean_conservative_q", metrics)
        self.assertGreater(metrics["step_count"], 0)


if __name__ == "__main__":
    unittest.main()
