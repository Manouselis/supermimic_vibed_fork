"""Read-only host report. Runs with system Python, even before installing PyTorch or SDK."""

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv):
    if not shutil.which(argv[0]):
        return {"available": False}
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=15)
        return {"returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": str(exc)}


def inspect():
    report = {"hostname": platform.node(), "architecture": platform.machine(),
              "python": sys.version, "executable": sys.executable, "kernel": platform.release(),
              "robot_connected": "NOT_CHECKED", "motor_commands": "NOT_IMPLEMENTED"}
    for name in ["/etc/os-release", "/etc/nv_tegra_release", "/sys/firmware/devicetree/base/model"]:
        p = Path(name)
        report[name] = p.read_text(errors="replace").strip("\0\n") if p.is_file() else None
    report["packages"] = {}
    for name in ["torch", "numpy", "safetensors", "PyYAML", "unitree_sdk2py", "cyclonedds"]:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
    report["interfaces"] = sorted(p.name for p in Path('/sys/class/net').glob('*'))
    report["jetpack_packages"] = command(["dpkg-query", "-W", "nvidia-jetpack", "nvidia-l4t-core"])
    report["nvcc"] = command(["nvcc", "--version"])
    report["nvidia_smi"] = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"])
    # A successful package import is insufficient: run a tiny operation on each available device.
    try:
        import torch
        report["torch"] = {"version": torch.__version__, "built_cuda": torch.version.cuda,
                           "cuda_available": torch.cuda.is_available()}
        for device in ["cpu"] + (["cuda:0"] if torch.cuda.is_available() else []):
            x = torch.ones(4, device=device)
            report["torch"][device] = float((x @ x).item())
        if torch.cuda.is_available():
            report["torch"]["gpu"] = torch.cuda.get_device_name(0)
            report["torch"]["capability"] = torch.cuda.get_device_capability(0)
    except Exception as exc:
        report["torch_error"] = str(exc)
    report["notes"] = ["CUDA build version is not the same as the installed toolkit or driver version.",
                       "Physical G1 revision, IMU frame, firmware and controller ownership need separate review."]
    return report


if __name__ == '__main__':
    print(json.dumps(inspect(), indent=2))
