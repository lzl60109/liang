import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Value-Guided Koopman Symmetry")
    parser.add_argument("--sigma-tau", dest="sigma_tau", type=float, default=0.01)
    parser.add_argument("--lambda-q", dest="lambda_q", type=float, default=0.1)
    parser.add_argument(
        "--lambda-state-anchor",
        dest="lambda_state_anchor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lambda-latent-anchor",
        dest="lambda_latent_anchor",
        type=float,
        default=0.1,
    )
    parser.add_argument("--q-clip-min", dest="q_clip_min", type=float, default=-20.0)
    parser.add_argument("--q-clip-max", dest="q_clip_max", type=float, default=20.0)
    parser.add_argument("--q-threshold", dest="q_threshold", type=float, default=None)
    parser.add_argument("--q-delta", dest="q_delta", type=float, default=0.0)
    parser.add_argument("--max-state-shift", dest="max_state_shift", type=float, default=None)
    parser.add_argument("--commute-horizon", dest="commute_horizon", type=int, default=1)
    parser.add_argument("--value-temperature", dest="value_temperature", type=float, default=1.0)
    parser.add_argument("--sigma-warmup-steps", dest="sigma_warmup_steps", type=int, default=0)
    parser.add_argument("--kats-checkpoint", dest="kats_checkpoint", type=str, default=None)
    parser.add_argument("--critic-checkpoint", dest="critic_checkpoint", type=str, default=None)
    parser.add_argument("--device", dest="device", type=str, default="cpu")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=0)
    return parser
