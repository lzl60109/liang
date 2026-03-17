import subprocess
import sys
import unittest
from pathlib import Path


class EnvCheckTests(unittest.TestCase):
    def test_check_env_script_runs_help(self):
        result = subprocess.run(
            [sys.executable, "check_env.py", "--help"],
            cwd=Path("H:/codex_test/nips2026"),
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Check VGKS runtime dependencies", result.stdout)


if __name__ == "__main__":
    unittest.main()
