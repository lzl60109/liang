import tempfile
import unittest
from pathlib import Path

import yaml

from vgks.train_vgks import load_config_file, merge_config_with_args


class VGKSConfigTests(unittest.TestCase):
    def test_default_generation_config_disables_wandb(self):
        config = load_config_file(Path("H:/codex_test/nips2026/configs/vgks/config.yaml"))

        self.assertFalse(config["use_wandb"])

    def test_load_config_file_reads_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "vgks.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "dataset_path": "data/d4rl/walker2d-medium-v2",
                        "latent_dim": 32,
                        "hidden_dim": 256,
                        "epochs": 20,
                    }
                ),
                encoding="utf-8",
            )

            config = load_config_file(path)

        self.assertEqual(config["dataset_path"], "data/d4rl/walker2d-medium-v2")
        self.assertEqual(config["latent_dim"], 32)

    def test_merge_config_with_args_prefers_explicit_cli_values(self):
        config = {
            "dataset_path": "data/d4rl/walker2d-medium-v2",
            "latent_dim": 32,
            "hidden_dim": 256,
            "epochs": 20,
            "device": "cuda:0",
        }
        merged = merge_config_with_args(
            config,
            {
                "hidden_dim": 512,
                "device": None,
                "epochs": None,
            },
        )

        self.assertEqual(merged["hidden_dim"], 512)
        self.assertEqual(merged["device"], "cuda:0")
        self.assertEqual(merged["epochs"], 20)


if __name__ == "__main__":
    unittest.main()
