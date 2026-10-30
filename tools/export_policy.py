"""Export policy tensors and required metadata. Never modify the source checkpoint."""

from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import subprocess
import sys

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _hash(f)


def _hash(f):
    h = hashlib.sha256()
    for block in iter(lambda: f.read(1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def export(checkpoint: Path, output: Path, asset_root: Path):
    # Restricted loading is sufficient for our tensor/dict checkpoint format.
    ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {k: v.detach().cpu().contiguous() for k, v in ck["pi"].items()}
    obs_dim, goal_dim = state["obs_norm.mean"].numel(), state["goal_norm.mean"].numel()
    args = ck["args"]
    if obs_dim != 99 or goal_dim != 73 * args["goal_frames"]:
        raise ValueError("Unsupported policy dimensions")
    output.mkdir(parents=True, exist_ok=False)
    save_file(state, str(output / "policy.safetensors"), metadata={"format": "pt"})
    # Read the ordered names without importing the simulator-dependent robot module.
    tree = ast.parse((ROOT / "supermimic/robot.py").read_text())
    joints = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "G1_JOINT_NAMES" for t in n.targets))
    versions = {}
    for name in ["torch", "safetensors", "numpy", "mujoco", "warp-lang", "newton", "isaaclab"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not recorded by installed package metadata"
    code_files = ["networks.py", "mdp.py", "motion.py", "robot.py", "rotations.py", "kinematics.py", "env.py"]
    assets = {}
    for name in ["g1_description/g1_29dof_rev_1_0.urdf", "g1_description/g1_29dof.xml"]:
        p = asset_root / name
        if p.is_file():
            assets[name] = sha256(p)
    motions = {p.name: sha256(p) for p in sorted((asset_root / "lafan1_g1").glob("*.csv"))}
    config = {
        "format": "supermimic-policy-v1", "inference_only": True,
        "policy": {"obs_dim": obs_dim, "goal_dim": goal_dim, "goal_frames": args["goal_frames"],
                   "hist": args["hist"], "z_dim": 32, "amp": args.get("amp", "none"),
                   "parameters": sum(v.numel() for k, v in state.items() if not k.startswith(("obs_norm.", "goal_norm.")))},
        "checkpoint": {k: ck[k] for k in ["iteration", "wall_time", "sim_steps", "saved_at"]},
        "source": {"repository": "https://github.com/Manouselis/supermimic_vibed_fork",
                   "compatible_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "training_commit": None, "training_commit_note": "Not embedded in the original checkpoint.",
                   "checkpoint_sha256": sha256(checkpoint),
                   "runtime_source_sha256": {n: sha256(ROOT / "supermimic" / n) for n in code_files}},
        "control": {"dt_seconds": 0.02, "joint_names": joints,
                    "observation_order": ["joint_position_minus_default_minus_offset:29", "joint_velocity:29",
                                          "root_angular_velocity_body:3", "projected_gravity:3",
                                          "previous_action:29", "reference_relative_root_orientation_6d:6"],
                    "action": "q_target = default_joint_position + joint_offset + action * per_joint_action_scale",
                    "action_scale": "0.25 * effort_limit / PD_stiffness, see supermimic/robot.py",
                    "root_quaternion_input_order": "xyzw in source CSV, wxyz internally"},
        "training": {k: v for k, v in args.items() if k in ["num_envs", "iterations", "buffer_steps", "utd", "horizon",
                     "wm_window", "wm_batches", "wm_batch", "pi_batch", "lr", "gamma", "lam", "action_noise",
                     "critic_epochs", "critic_minibatches", "no_critic", "warmup_steps", "wm_only_iters", "lr_decay_iters", "seed"]},
        "runtime_versions": versions,
        "assets_sha256": assets,
        "dataset": {"repo_id": "lvhaidong/LAFAN1_Retargeting_Dataset", "source_fps": 30, "control_fps": 50,
                    "motion_sha256": motions, "license": "Source LAFAN1: CC BY-NC-ND 4.0. See release licensing notes."},
    }
    config["training"]["planned_iterations"] = config["training"].pop("iterations", None)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    manifest = {n: sha256(output / n) for n in ["policy.safetensors", "config.json"]}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"iteration": ck["iteration"], "tensor_values": sum(v.numel() for v in state.values()),
                      "policy_bytes": (output / "policy.safetensors").stat().st_size, "sha256": manifest}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path, help="New directory, never overwritten")
    parser.add_argument("--asset-root", type=Path, default=ROOT / "data")
    a = parser.parse_args()
    export(a.checkpoint, a.output, a.asset_root)
