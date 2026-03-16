from vgks.cli import build_parser


def test_cli_exposes_value_guidance_flags():
    parser = build_parser()

    args = parser.parse_args([])

    assert args.lambda_q == 0.1
    assert args.lambda_state_anchor == 1.0
    assert args.lambda_latent_anchor == 0.1
    assert args.q_clip_min == -20.0
    assert args.q_clip_max == 20.0
    assert args.q_threshold is None
    assert args.sigma_warmup_steps == 0
