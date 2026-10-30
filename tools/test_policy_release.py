"""CPU tests for release loading. No simulator instance or GPU is started."""

import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from supermimic.checkpoint import load_checkpoint
from supermimic.networks import Policy


class PolicyReleaseTest(unittest.TestCase):
    def test_legacy_and_release_actions_match(self):
        torch.manual_seed(7)
        pi = Policy(99, 730, hist=10).eval()
        pi.obs_norm.mean.normal_()
        pi.obs_norm.var.uniform_(0.1, 2)
        pi.goal_norm.mean.normal_()
        pi.goal_norm.var.uniform_(0.1, 2)
        pi.obs_norm.count.fill_(12345)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            source = {"pi": pi.state_dict(), "args": {"hist": 10, "goal_frames": 10, "amp": "none"}}
            torch.save(source, p / "legacy.pt")
            save_file(source["pi"], str(p / "policy.safetensors"))
            config = {"format": "supermimic-policy-v1", "policy": {"obs_dim": 99, "goal_dim": 730,
                      "goal_frames": 10, "hist": 10, "amp": "none"}, "checkpoint": {"iteration": 40000}}
            (p / "config.json").write_text(json.dumps(config))
            old, new = load_checkpoint(p / "legacy.pt"), load_checkpoint(p)
            self.assertEqual(set(old["pi"]), set(new["pi"]))
            loaded = Policy(99, 730, hist=10).eval()
            loaded.load_state_dict(new["pi"])
            for batch in [1, 4]:
                obs, goal = torch.randn(batch, 10, 99), torch.randn(batch, 730)
                torch.testing.assert_close(pi(obs, goal), loaded(obs, goal), rtol=0, atol=0)
            self.assertEqual(new["iteration"], 40000)
            config["format"] = "unknown"
            (p / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "Unsupported"):
                load_checkpoint(p)


if __name__ == "__main__":
    unittest.main()
