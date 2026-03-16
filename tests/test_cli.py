import unittest

from vgks.cli import build_parser


class CLITests(unittest.TestCase):
    def test_cli_exposes_value_guidance_flags(self):
        parser = build_parser()

        args = parser.parse_args([])

        self.assertEqual(args.lambda_q, 0.1)
        self.assertEqual(args.lambda_state_anchor, 1.0)
        self.assertEqual(args.lambda_latent_anchor, 0.1)
        self.assertEqual(args.q_clip_min, -20.0)
        self.assertEqual(args.q_clip_max, 20.0)
        self.assertIsNone(args.q_threshold)
        self.assertEqual(args.sigma_warmup_steps, 0)
        self.assertTrue(hasattr(args, "kats_checkpoint"))
        self.assertTrue(hasattr(args, "critic_checkpoint"))
        self.assertTrue(hasattr(args, "device"))
        self.assertTrue(hasattr(args, "num_workers"))
        self.assertIsNone(args.kats_checkpoint)
        self.assertIsNone(args.critic_checkpoint)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.num_workers, 0)


if __name__ == "__main__":
    unittest.main()
