import unittest
from pathlib import Path

from vgks.train_vgks import load_config_file


class VGKSPresetTests(unittest.TestCase):
    def test_shared_base_and_task_presets_exist(self):
        config_dir = Path("H:/codex_test/nips2026/configs")
        expected = [
            "vgks.base.yaml",
            "vgks.halfcheetah-medium.yaml",
            "vgks.halfcheetah-medium-replay.yaml",
            "vgks.halfcheetah-medium-expert.yaml",
            "vgks.hopper-medium.yaml",
            "vgks.hopper-medium-replay.yaml",
            "vgks.hopper-medium-expert.yaml",
            "vgks.walker2d-medium.yaml",
            "vgks.walker2d-medium-replay.yaml",
            "vgks.walker2d-medium-expert.yaml",
        ]

        for name in expected:
            self.assertTrue((config_dir / name).exists(), msg=name)

    def test_task_preset_contains_dataset_and_save_dir(self):
        config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks.walker2d-medium.yaml"))

        self.assertEqual(config["dataset_path"], "data/d4rl/walker2d-medium-v2")
        self.assertEqual(config["save_dir"], "runs/vgks/walker2d-medium-v2/seed_0")


if __name__ == "__main__":
    unittest.main()
