#!/usr/bin/env python3
"""
Model download script for Urban Metamorphosis project.
Downloads Stable Diffusion 1.5, AnimateDiff motion module,
and CLIP model to local ./models/ directory.

Usage:
    python scripts/download_models.py
"""

import os
import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import CLIPModel, CLIPProcessor
from diffusers import MotionAdapter, StableDiffusionPipeline


def download_models():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Urban Metamorphosis — Model Download")
    print("=" * 60)

    # 1. Stable Diffusion 1.5
    sd_path = models_dir / "stable-diffusion-v1-5"
    if not sd_path.exists():
        print("\n[1/3] Downloading Stable Diffusion 1.5...")
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype="auto",
        )
        pipe.save_pretrained(sd_path)
        print(f"    Saved to: {sd_path}")
    else:
        print(f"\n[1/3] Stable Diffusion 1.5 already exists at: {sd_path}")

    # 2. AnimateDiff Motion Module
    mm_path = models_dir / "motion-module"
    if not mm_path.exists():
        print("\n[2/3] Downloading AnimateDiff Motion Module v1-5-2...")
        adapter = MotionAdapter.from_pretrained(
            "guoyww/animatediff-motion-adapter-v1-5-2",
            torch_dtype="auto",
        )
        adapter.save_pretrained(mm_path)
        print(f"    Saved to: {mm_path}")
    else:
        print(f"\n[2/3] Motion Module already exists at: {mm_path}")

    # 3. CLIP ViT-L/14 (for evaluation)
    clip_path = models_dir / "clip-vit-large-patch14"
    if not clip_path.exists():
        print("\n[3/3] Downloading CLIP ViT-L/14...")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model.save_pretrained(clip_path)
        processor.save_pretrained(clip_path)
        print(f"    Saved to: {clip_path}")
    else:
        print(f"\n[3/3] CLIP already exists at: {clip_path}")

    print("\n" + "=" * 60)
    print("All models downloaded successfully!")
    print("=" * 60)
    print(f"\nTotal disk usage: {get_dir_size(models_dir):.1f} MB")


def get_dir_size(path: Path) -> float:
    total = 0
    for entry in os.scandir(path):
        if entry.is_file():
            total += entry.stat().st_size
        elif entry.is_dir():
            total += get_dir_size(Path(entry))
    return total / (1024 * 1024)


if __name__ == "__main__":
    download_models()
