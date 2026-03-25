import unittest

import torch

from vgks.models import ConservativeCritic, KoopmanDynamicsModel, SigmaModel
from vgks.trainer import ValueGuidedKoopmanTrainer


def make_batch():
    return {
        "observations": torch.tensor(
            [[0.2, -0.1, 0.3], [0.5, 0.4, -0.2]], dtype=torch.float32
        ),
        "next_observations": torch.tensor(
            [[0.25, -0.05, 0.35], [0.45, 0.35, -0.1]], dtype=torch.float32
        ),
    }


def build_trainer(lambda_q=0.1):
    torch.manual_seed(0)
    dynamics = KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=12)
    sigma = SigmaModel(latent_dim=4)
    critic = ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=12)
    return ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma,
        critic=critic,
        action_dim=2,
        sigma_tau=0.01,
        lambda_q=lambda_q,
        lambda_state_anchor=1.0,
        lambda_action_anchor=0.5,
        lambda_latent_anchor=0.1,
        q_clip_min=-20.0,
        q_clip_max=20.0,
        sigma_lr=1e-2,
    )


class VGKSTrainerTests(unittest.TestCase):
    def test_sigma_metrics_include_value_guidance_terms(self):
        trainer = build_trainer()

        metrics = trainer.compute_sigma_loss(make_batch())

        self.assertTrue(
            {
                "total_loss",
                "commutation_loss",
                "value_loss",
                "state_anchor_loss",
                "latent_anchor_loss",
                "mean_conservative_q",
            }.issubset(metrics)
        )

    def test_conservative_q_uses_lower_critic_value(self):
        trainer = build_trainer()
        batch = make_batch()

        observations = batch["observations"]
        actions = torch.full((observations.shape[0], 2), 0.5)
        q1 = trainer.critic.q1(observations, actions)
        q2 = trainer.critic.q2(observations, actions)
        q = trainer.critic.conservative_value(observations, actions)

        self.assertTrue(torch.allclose(q, torch.minimum(q1, q2)))

    def test_lambda_q_changes_total_loss(self):
        batch = make_batch()
        trainer_without_value = build_trainer(lambda_q=0.0)
        trainer_with_value = build_trainer(lambda_q=0.5)

        without_value = trainer_without_value.compute_sigma_loss(batch)["total_loss"]
        with_value = trainer_with_value.compute_sigma_loss(batch)["total_loss"]

        self.assertNotEqual(without_value, with_value)

    def test_augment_batch_can_filter_by_q_threshold(self):
        trainer = build_trainer()
        batch = make_batch()

        unfiltered = trainer.augment_batch(batch)
        filtered = trainer.augment_batch(batch, q_threshold=100.0)

        self.assertTrue(
            {"observations", "actions", "next_observations", "q_values"}.issubset(unfiltered)
        )
        self.assertEqual(unfiltered["observations"].shape[0], batch["observations"].shape[0])
        self.assertEqual(filtered["observations"].shape[0], 0)
        self.assertIsInstance(unfiltered["observations"], torch.Tensor)

    def test_augment_batch_reports_kept_sample_count(self):
        trainer = build_trainer()
        batch = make_batch()

        augmented = trainer.augment_batch(batch, q_threshold=-100.0)

        self.assertIn("num_kept", augmented)
        self.assertEqual(int(augmented["num_kept"]), batch["observations"].shape[0])

    def test_sigma_loss_tensors_require_grad_and_step_updates_sigma(self):
        trainer = build_trainer()
        batch = make_batch()

        before = trainer.sigma_model.sigma_layer.weight.detach().clone()
        tensor_metrics = trainer.compute_sigma_loss_tensors(batch)

        self.assertTrue(tensor_metrics["total_loss"].requires_grad)

        trainer.train_sigma_step(batch)
        after = trainer.sigma_model.sigma_layer.weight.detach()

        self.assertFalse(torch.allclose(before, after))

    def test_sigma_loss_clamps_weight_exponent_to_avoid_overflow(self):
        trainer = build_trainer()
        batch = {
            "observations": torch.full((2, 3), 1e6, dtype=torch.float32),
            "next_observations": torch.full((2, 3), -1e6, dtype=torch.float32),
        }

        tensor_metrics = trainer.compute_sigma_loss_tensors(batch)

        self.assertTrue(torch.isfinite(tensor_metrics["commutation_loss"]))
        self.assertTrue(torch.isfinite(tensor_metrics["total_loss"]))

    def test_sigma_guidance_preserves_unclipped_q_signal(self):
        trainer = build_trainer()
        batch = make_batch()

        tensor_metrics = trainer.compute_sigma_loss_tensors(batch)

        self.assertIn("mean_conservative_q_unclipped", tensor_metrics)
        self.assertTrue(torch.isfinite(tensor_metrics["mean_conservative_q_unclipped"]))

    def test_sigma_metrics_include_advantage_and_state_shift(self):
        trainer = build_trainer()
        metrics = trainer.compute_sigma_loss_tensors(make_batch())

        self.assertIn("mean_advantage", metrics)
        self.assertIn("mean_state_shift", metrics)
        self.assertTrue(torch.isfinite(metrics["mean_advantage"]))
        self.assertTrue(torch.isfinite(metrics["mean_state_shift"]))

    def test_sigma_metrics_include_action_consistency_terms(self):
        trainer = build_trainer()
        batch = make_batch()
        batch["actions"] = torch.tensor([[0.1, -0.2], [0.0, 0.3]], dtype=torch.float32)

        metrics = trainer.compute_sigma_loss_tensors(batch)

        self.assertIn("action_anchor_loss", metrics)
        self.assertIn("mean_action_deviation", metrics)
        self.assertTrue(torch.isfinite(metrics["action_anchor_loss"]))
        self.assertTrue(torch.isfinite(metrics["mean_action_deviation"]))

    def test_augment_batch_can_filter_by_advantage_and_state_shift(self):
        trainer = build_trainer()
        batch = {
            "observations": torch.zeros(2, 3, dtype=torch.float32),
            "actions": torch.zeros(2, 2, dtype=torch.float32),
            "next_observations": torch.zeros(2, 3, dtype=torch.float32),
        }

        filtered = trainer.augment_batch(batch, q_threshold=None, q_delta=100.0, max_state_shift=0.01)

        self.assertEqual(int(filtered["observations"].shape[0]), 0)
        self.assertEqual(int(filtered["num_kept"]), 0)

    def test_augment_batch_can_filter_by_action_deviation(self):
        trainer = build_trainer()
        batch = {
            "observations": torch.zeros(2, 3, dtype=torch.float32),
            "actions": torch.zeros(2, 2, dtype=torch.float32),
            "next_observations": torch.zeros(2, 3, dtype=torch.float32),
        }

        filtered = trainer.augment_batch(batch, max_action_deviation=0.0)

        self.assertEqual(int(filtered["observations"].shape[0]), 0)
        self.assertEqual(int(filtered["num_kept"]), 0)


if __name__ == "__main__":
    unittest.main()
