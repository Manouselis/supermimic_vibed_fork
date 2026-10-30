"""Download a pinned HF policy release and check its declared tensor/config hashes."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id")
    parser.add_argument("--revision", required=True, help="Release tag or exact Hub commit")
    parser.add_argument("--output", required=True, type=Path)
    a = parser.parse_args()
    if a.output.exists():
        parser.error("Output directory already exists")
    from huggingface_hub import snapshot_download

    snapshot_download(a.repo_id, revision=a.revision, local_dir=a.output,
                      allow_patterns=["policy.safetensors", "config.json", "manifest.json"])
    manifest = json.loads((a.output / "manifest.json").read_text())
    for name in ["policy.safetensors", "config.json"]:
        digest = hashlib.sha256((a.output / name).read_bytes()).hexdigest()
        if digest != manifest[name]:
            raise ValueError(f"SHA-256 mismatch for {name}")
    print(f"Verified policy: {a.output}")
    print("Use evaluate.py --ckpt with this directory. This artifact cannot resume training.")


if __name__ == "__main__":
    main()
