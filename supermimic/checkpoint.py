"""Load local training checkpoints or the compact, tensor-only policy release."""

from pathlib import Path
import json

import torch


def load_checkpoint(path: str | Path, device="cpu") -> dict:
    path = Path(path)
    if path.is_dir():
        path = path / "policy.safetensors"
    if path.suffix != ".safetensors":
        return torch.load(path, map_location=device, weights_only=True)

    from safetensors.torch import load_file

    config = json.loads(path.with_name("config.json").read_text())
    if config.get("format") != "supermimic-policy-v1":
        raise ValueError("Unsupported policy release format")
    policy = config["policy"]
    if policy["obs_dim"] != 99 or policy["goal_dim"] != 73 * policy["goal_frames"]:
        raise ValueError("Policy observation/goal layout does not match this implementation")
    return {
        "pi": load_file(str(path), device=str(device)),
        "args": {"hist": policy["hist"], "goal_frames": policy["goal_frames"], "amp": policy["amp"]},
        **config["checkpoint"],
    }
