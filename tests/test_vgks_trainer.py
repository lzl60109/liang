import numpy as np

from vgks.models import ConservativeCritic, KoopmanDynamicsModel, SigmaModel
from vgks.trainer import ValueGuidedKoopmanTrainer


def make_batch():
    return {
        "observations": np.array(
            [[0.2, -0.1, 0.3], [0.5, 0.4, -0.2]], dtype=np.float32
        ),
        "next_observations": np.array(
            [[0.25, -0.05, 0.35], [0.45, 0.35, -0.1]], dtype=np.float32
        ),
    }


def build_trainer(lambda_q=0.1):
    dynamics = KoopmanDynamicsModel(state_dim=3, action_dim=2, latent_dim=4, hidden_dim=12)
    sigma = SigmaModel(latent_dim=4)
    critic = ConservativeCritic(state_dim=3, action_dim=2, hidden_dim=12)
    return ValueGuidedKoopmanTrainer(
        dynamics=dynamics,
        sigma_model=sigma,
        critic=critic,
        action_dim=2,
        lambda_q=lambda_q,
        lambda_state_anchor=1.0,
        lambda_latent_anchor=0.1,
        q_clip_min=-20.0,
        q_clip_max=20.0,
    )


def test_sigma_metrics_include_value_guidance_terms():
    trainer = build_trainer()

    metrics = trainer.compute_sigma_loss(make_batch())

    assert set(
        [
            "total_loss",
            "commutation_loss",
            "value_loss",
            "state_anchor_loss",
            "latent_anchor_loss",
            "mean_conservative_q",
        ]
    ).issubset(metrics)


def test_conservative_q_uses_lower_critic_value():
    trainer = build_trainer()
    batch = make_batch()

    observations = batch["observations"]
    actions = np.full((observations.shape[0], 2), 0.5, dtype=np.float32)
    q1 = trainer.critic.q1(observations, actions)
    q2 = trainer.critic.q2(observations, actions)
    q = trainer.critic.conservative_value(observations, actions)

    assert np.allclose(q, np.minimum(q1, q2))


def test_lambda_q_changes_total_loss():
    batch = make_batch()
    trainer_without_value = build_trainer(lambda_q=0.0)
    trainer_with_value = build_trainer(lambda_q=0.5)

    without_value = trainer_without_value.compute_sigma_loss(batch)["total_loss"]
    with_value = trainer_with_value.compute_sigma_loss(batch)["total_loss"]

    assert without_value != with_value


def test_augment_batch_can_filter_by_q_threshold():
    trainer = build_trainer()
    batch = make_batch()

    unfiltered = trainer.augment_batch(batch)
    filtered = trainer.augment_batch(batch, q_threshold=100.0)

    assert set(["observations", "actions", "next_observations", "q_values"]).issubset(
        unfiltered
    )
    assert unfiltered["observations"].shape[0] == batch["observations"].shape[0]
    assert filtered["observations"].shape[0] == 0
