#!/usr/bin/env python3
"""
Download models using huggingface-cli (as specified in PLAN/03).
Falls back to Python API if huggingface-cli is unavailable.

Usage:
    python scripts/download_models_cli.py [--token YOUR_HF_TOKEN]
"""

import argparse
import subprocess
import sys
from pathlib import Path


MODELS = {
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "motion-module": "guoyww/animatediff-motion-adapter-v1-5-2",
    "clip-vit-large-patch14": "openai/clip-vit-large-patch14",
}


def run_cmd(cmd: list[str]) -> bool:
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def download_via_cli(repo_id: str, local_dir: Path, token: str | None) -> bool:
    # Try modern `hf` CLI first, then legacy `huggingface-cli`
    for cli in ["hf", "huggingface-cli"]:
        cmd = [
            cli, "download", repo_id,
            "--local-dir", str(local_dir),
            "--local-dir-use-symlinks", "False",
        ]
        if token:
            cmd += ["--token", token]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            return True
    return False


def download_via_python(repo_id: str, local_dir: Path):
    """Fallback using Python API."""
    from diffusers import StableDiffusionPipeline, MotionAdapter
    from transformers import CLIPModel, CLIPProcessor

    if "stable-diffusion" in repo_id:
        pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype="auto")
        pipe.save_pretrained(local_dir)
    elif "animatediff" in repo_id:
        adapter = MotionAdapter.from_pretrained(repo_id, torch_dtype="auto")
        adapter.save_pretrained(local_dir)
    elif "clip" in repo_id:
        model = CLIPModel.from_pretrained(repo_id)
        processor = CLIPProcessor.from_pretrained(repo_id)
        model.save_pretrained(local_dir)
        processor.save_pretrained(local_dir)


def main():
    parser = argparse.ArgumentParser(description="Download models for Urban Metamorphosis")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace access token")
    args = parser.parse_args()

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Urban Metamorphosis — Model Download")
    print("=" * 60)

    for name, repo_id in MODELS.items():
        local_dir = models_dir / name
        if local_dir.exists() and any(local_dir.iterdir()):
            print(f"\n[SKIP] {name} already exists at {local_dir}")
            continue

        print(f"\n[DOWNLOAD] {name} from {repo_id}")
        local_dir.mkdir(parents=True, exist_ok=True)

        # Try huggingface-cli first
        success = download_via_cli(repo_id, local_dir, args.token)
        if not success:
            print("  huggingface-cli failed, falling back to Python API...")
            download_via_python(repo_id, local_dir)
        print(f"  Saved to: {local_dir}")

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
