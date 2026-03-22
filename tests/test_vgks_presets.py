import unittest
from pathlib import Path

from vgks.train_vgks import load_config_file


class VGKSPresetTests(unittest.TestCase):
    def test_shared_base_and_task_presets_exist(self):
        config_dir = Path("H:/codex_test/nips2026/configs")
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
        config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks.walker2d-medium.yaml"))

        self.assertEqual(config["dataset_path"], "data/d4rl/walker2d-medium-v2")
        self.assertEqual(config["save_dir"], "runs/vgks/walker2d-medium-v2/seed_0")

    def test_family_presets_inherit_domain_specific_defaults(self):
        maze_config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks.maze2d-large.yaml"))
        antmaze_config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks.antmaze-large-play.yaml"))
        adroit_config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks.pen-human.yaml"))
        kitchen_config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks.kitchen-mixed.yaml"))

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


if __name__ == "__main__":
    unittest.main()
