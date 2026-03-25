import unittest
from pathlib import Path

import yaml

from vgks.train_vgks import load_config_file

REPO_ROOT = Path(__file__).resolve().parents[1]


class VGKSPresetTests(unittest.TestCase):
    def test_shared_base_and_task_presets_exist(self):
        config_dir = REPO_ROOT / "configs"
        expected = [
            "vgks.base.yaml",
            "vgks.mujoco.base.yaml",
            "vgks.maze2d.base.yaml",
            "vgks.antmaze.base.yaml",
            "vgks.adroit.base.yaml",
            "vgks.kitchen.base.yaml",
            "vgks.halfcheetah-medium.yaml",
            "vgks.halfcheetah-medium-replay.yaml",
            "vgks.halfcheetah-medium-expert.yaml",
            "vgks.hopper-medium.yaml",
            "vgks.hopper-medium-replay.yaml",
            "vgks.hopper-medium-expert.yaml",
            "vgks.walker2d-medium.yaml",
            "vgks.walker2d-medium-replay.yaml",
            "vgks.walker2d-medium-expert.yaml",
            "vgks.maze2d-umaze.yaml",
            "vgks.maze2d-medium.yaml",
            "vgks.maze2d-large.yaml",
            "vgks.antmaze-umaze-diverse.yaml",
            "vgks.antmaze-umaze-play.yaml",
            "vgks.antmaze-medium-diverse.yaml",
            "vgks.antmaze-medium-play.yaml",
            "vgks.antmaze-large-diverse.yaml",
            "vgks.antmaze-large-play.yaml",
            "vgks.pen-human.yaml",
            "vgks.hammer-human.yaml",
            "vgks.door-human.yaml",
            "vgks.relocate-human.yaml",
            "vgks.kitchen-complete.yaml",
            "vgks.kitchen-partial.yaml",
            "vgks.kitchen-mixed.yaml",
            "vgks.kitchen-undirected.yaml",
        ]

        for name in expected:
            self.assertTrue((config_dir / name).exists(), msg=name)

    def test_task_preset_contains_dataset_and_save_dir(self):
        config = load_config_file(REPO_ROOT / "configs" / "vgks.walker2d-medium.yaml")

        self.assertEqual(config["dataset_path"], "data/d4rl/walker2d-medium-v2")
        self.assertEqual(config["save_dir"], "runs/vgks/walker2d-medium-v2/seed_0")

    def test_family_presets_inherit_domain_specific_defaults(self):
        maze_config = load_config_file(REPO_ROOT / "configs" / "vgks.maze2d-large.yaml")
        antmaze_config = load_config_file(REPO_ROOT / "configs" / "vgks.antmaze-large-play.yaml")
        adroit_config = load_config_file(REPO_ROOT / "configs" / "vgks.pen-human.yaml")
        kitchen_config = load_config_file(REPO_ROOT / "configs" / "vgks.kitchen-mixed.yaml")

        self.assertEqual(maze_config["dataset_path"], "data/d4rl/maze2d-large-v1")
        self.assertEqual(maze_config["epochs"], 30)
        self.assertEqual(maze_config["eval_episodes"], 20)

        self.assertEqual(antmaze_config["dataset_path"], "data/d4rl/antmaze-large-play-v2")
        self.assertEqual(antmaze_config["lambda_q"], 0.05)
        self.assertEqual(antmaze_config["sigma_warmup_steps"], 1000)

        self.assertEqual(adroit_config["dataset_path"], "data/d4rl/pen-human-v1")
        self.assertEqual(adroit_config["hidden_dim"], 512)
        self.assertEqual(adroit_config["sigma_lr"], 0.0005)
        self.assertIn("koopman_pretrain_epochs", adroit_config)
        self.assertIn("critic_pretrain_epochs", adroit_config)

        self.assertEqual(kitchen_config["dataset_path"], "data/d4rl/kitchen-mixed-v0")
        self.assertEqual(kitchen_config["lambda_state_anchor"], 2.0)
        self.assertEqual(kitchen_config["sigma_warmup_steps"], 500)

    def test_halfcheetah_medium_expert_preset_is_flattened(self):
        path = REPO_ROOT / "configs" / "vgks.halfcheetah-medium-expert.yaml"
        raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))

        self.assertNotIn("base_config", raw_config)
        self.assertEqual(raw_config["env_name"], "halfcheetah-medium-expert-v2")
        self.assertEqual(raw_config["q_delta"], 10.0)
        self.assertEqual(raw_config["max_state_shift"], 3.5)
        self.assertEqual(raw_config["max_action_deviation"], 0.8)
        self.assertEqual(raw_config["q_clip_max"], 50.0)
        self.assertEqual(raw_config["value_temperature"], 5.0)
        self.assertEqual(raw_config["commute_horizon"], 2)
        self.assertEqual(raw_config["sigma_warmup_steps"], 2000)


if __name__ == "__main__":
    unittest.main()
