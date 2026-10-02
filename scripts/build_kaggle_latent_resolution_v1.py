from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
SOURCE_KERNEL = ROOT / "kaggle" / "latent_resolution_v1" / "kernel_main.py"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    user = args.user.strip()
    if not user or "/" in user or "\\" in user:
        raise SystemExit("--user must be a Kaggle account slug")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_KERNEL, output / "kernel_main.py")

    metadata = {
        "id": f"{user}/vera-latent-resolution-v1",
        "title": "Vera Latent Resolution V1",
        "code_file": "kernel_main.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (output / "kernel-metadata.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "staged Kaggle kernel at "
        + str(output)
        + "\npush with: kaggle kernels push -p "
        + json.dumps(str(output))
        + " --accelerator NvidiaTeslaT4"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
