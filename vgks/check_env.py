from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _version_of(module_name: str):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic path
        return f"missing ({exc})"


def run_check(env_name: str = "halfcheetah-medium-v2"):
    report = {
        "python": sys.version.split()[0],
        "gym": _version_of("gym"),
        "d4rl": _version_of("d4rl"),
        "mujoco": _version_of("mujoco"),
        "torch": _version_of("torch"),
        "env_name": env_name,
        "env_make": "not checked",
    }

    try:
        import gym  # type: ignore
        import d4rl  # type: ignore  # noqa: F401

        env = gym.make(env_name)
        report["env_make"] = "ok"
        report["state_dim"] = int(env.observation_space.shape[0])
        report["action_dim"] = int(env.action_space.shape[0])
    except Exception as exc:  # pragma: no cover - diagnostic path
        report["env_make"] = f"failed ({exc})"

    return report


def main() -> None:
    parser = __import__("argparse").ArgumentParser(description="Check VGKS runtime dependencies")
    parser.add_argument("--env-name", dest="env_name", type=str, default="halfcheetah-medium-v2")
    args = parser.parse_args()
    print(json.dumps(run_check(args.env_name), indent=2))


if __name__ == "__main__":
    main()
