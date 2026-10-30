"""Verify exact tensors and policy actions against a local source checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from supermimic.checkpoint import load_checkpoint
import supermimic.networks as networks
from supermimic.networks import Policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", type=Path)
    p.add_argument("release", type=Path)
    p.add_argument("--motion", type=Path, help="Optional existing CSV for real reference-state inputs")
    a = p.parse_args()
    old, new = load_checkpoint(a.source), load_checkpoint(a.release)
    assert set(old["pi"]) == set(new["pi"]), "Missing policy tensor or normalization buffer"
    for k in old["pi"]:
        assert torch.equal(old["pi"][k], new["pi"][k]), f"Changed tensor: {k}"
    for name, expected in json.loads((a.release / "manifest.json").read_text()).items():
        assert hashlib.sha256((a.release / name).read_bytes()).hexdigest() == expected, name
    args = old["args"]
    hist, goal_dim = args["hist"], 73 * args["goal_frames"]
    torch.manual_seed(42)
    # Scale synthetic inputs with this checkpoint's actual normalization statistics.
    obs = torch.randn(8, hist, 99) * old["pi"]["obs_norm.var"].sqrt() + old["pi"]["obs_norm.mean"]
    goal = torch.randn(8, goal_dim) * old["pi"]["goal_norm.var"].sqrt() + old["pi"]["goal_norm.mean"]
    inputs = [("zero", torch.zeros(1, hist, 99), torch.zeros(1, goal_dim)), ("normalization_scaled", obs, goal)]
    if a.motion:
        from supermimic.kinematics import G1Kinematics
        from supermimic.motion import MotionLib
        from supermimic.mdp import MDP
        from supermimic.robot import default_joint_pos_list

        kin = G1Kinematics("cpu")
        motion = MotionLib([str(a.motion)], kin, "cpu")
        mdp = MDP(kin, motion, torch.tensor(default_joint_pos_list()))
        frames = torch.linspace(0, int(motion.clip_end[0]), steps=8).long()
        state = tuple(v[frames] for v in [motion.root_p, motion.root_R, motion.root_v, motion.root_w, motion.q, motion.qd])
        o = mdp.obs(state, torch.zeros(8, 29), torch.zeros(8, 33), frames)
        inputs.append(("reference_states", o[:, None].repeat(1, hist, 1), motion.goal(frames, args["goal_frames"])))
    results = []
    devices = ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else [])
    for dev in devices:
        networks.AMP_DTYPE = torch.bfloat16 if dev.startswith("cuda") and args.get("amp") == "bf16" else None
        original, restored = Policy(99, goal_dim, hist).to(dev).eval(), Policy(99, goal_dim, hist).to(dev).eval()
        original.load_state_dict(old["pi"])
        restored.load_state_dict(new["pi"])
        with torch.inference_mode():
            for name, o, g in inputs:
                x, y = original(o.to(dev), g.to(dev)), restored(o.to(dev), g.to(dev))
                assert torch.isfinite(x).all() and torch.isfinite(y).all(), name
                torch.testing.assert_close(x, y, rtol=0, atol=0)
                results.append({"device": dev, "case": name, "batch": len(o), "max_abs_action_difference": float((x-y).abs().max())})
    print(json.dumps({"iteration": old["iteration"], "tensor_and_buffer_equality": True,
                      "checksums": "PASS", "action_checks": results}, indent=2))


if __name__ == "__main__":
    main()
