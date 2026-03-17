import tempfile
import unittest
from pathlib import Path
import pickle

import numpy as np

from vgks.data import load_offline_dataset, save_trajectory_cache


class DownloadDatasetTests(unittest.TestCase):
    def test_save_trajectory_cache_writes_pkl_and_npy_files(self):
        data = {
            "observations": np.random.randn(6, 3).astype(np.float32),
            "actions": np.random.randn(6, 2).astype(np.float32),
            "next_observations": np.random.randn(6, 3).astype(np.float32),
            "rewards": np.random.randn(6).astype(np.float32),
            "terminals": np.zeros(6, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "halfcheetah-medium-v2"
            save_trajectory_cache(cache_dir, data)

            self.assertTrue((cache_dir / "dataset.pkl").exists())
            self.assertTrue((cache_dir / "observations.npy").exists())
            self.assertTrue((cache_dir / "actions.npy").exists())
            self.assertTrue((cache_dir / "next_observations.npy").exists())

    def test_load_offline_dataset_reads_cache_directory(self):
        data = {
            "observations": np.random.randn(6, 3).astype(np.float32),
            "actions": np.random.randn(6, 2).astype(np.float32),
            "next_observations": np.random.randn(6, 3).astype(np.float32),
            "rewards": np.random.randn(6).astype(np.float32),
            "terminals": np.zeros(6, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "hopper-medium-v2"
            save_trajectory_cache(cache_dir, data)
            loaded = load_offline_dataset(cache_dir)

            self.assertTrue(np.allclose(loaded["observations"], data["observations"]))
            self.assertTrue(np.allclose(loaded["actions"], data["actions"]))
            self.assertTrue(np.allclose(loaded["next_observations"], data["next_observations"]))

    def test_load_offline_dataset_reads_standalone_pkl_file(self):
        data = {
            "observations": np.random.randn(6, 3).astype(np.float32),
            "actions": np.random.randn(6, 2).astype(np.float32),
            "next_observations": np.random.randn(6, 3).astype(np.float32),
            "rewards": np.random.randn(6).astype(np.float32),
            "terminals": np.zeros(6, dtype=np.float32),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pkl_path = Path(tmpdir) / "halfcheetah-medium-v2.pkl"
            with pkl_path.open("wb") as handle:
                pickle.dump(data, handle)

            loaded = load_offline_dataset(pkl_path)

            self.assertTrue(np.allclose(loaded["observations"], data["observations"]))
            self.assertTrue(np.allclose(loaded["actions"], data["actions"]))
            self.assertTrue(np.allclose(loaded["next_observations"], data["next_observations"]))


if __name__ == "__main__":
    unittest.main()
