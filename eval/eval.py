#!/usr/bin/env python3
"""
Full Evaluation Pipeline

Computes all metrics (CLIP-SIM, LPIPS, SSIM, Flow Warp, FVD)
for generated videos and produces aggregated ablation tables.

Usage:
    python eval/eval.py \
        --ablation-dir outputs/ablation \
        --prompts data/eval_prompts.json \
        --output outputs/metrics
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from eval.weakness_analysis import (
    load_frames,
    compute_per_frame_clip_sim,
    compute_flow_warping_error,
    compute_lpips_temporal,
    compute_ssim_temporal,
)


def load_prompt_data(prompts_path: str) -> dict[int, dict]:
    with open(prompts_path) as f:
        prompts = json.load(f)
    return {p["id"]: p for p in prompts}


def evaluate_video(video_path: str, prompt_data: dict, clip_model, clip_processor, device: str = "cuda") -> dict:
    """Compute all metrics for a single video."""
    frames = load_frames(video_path)
    if not frames:
        return {}

    n = len(frames)

    # CLIP-SIM dual curves
    day_scores, night_scores = compute_per_frame_clip_sim(
        frames,
        prompt_data["prompt_day"],
        prompt_data["prompt_night"],
        clip_model,
        clip_processor,
        device,
    )

    frame_indices = np.arange(n)
    day_slope = float(np.polyfit(frame_indices, day_scores, 1)[0])
    night_slope = float(np.polyfit(frame_indices, night_scores, 1)[0])
    mid_frame = n // 2
    mid_sep = float(night_scores[mid_frame] - day_scores[mid_frame])

    # Flow warping error
    fwe = compute_flow_warping_error(frames)
    fwe_mean = float(np.mean(fwe))
    fwe_dusk = float(np.mean(fwe[5:9])) if len(fwe) > 8 else fwe_mean

    # LPIPS
    lpips_scores = compute_lpips_temporal(frames)
    lpips_dusk = float(np.mean(lpips_scores[5:9])) if len(lpips_scores) > 8 else None
    lpips_mean = float(np.mean(lpips_scores)) if len(lpips_scores) else None

    # SSIM
    ssim_scores = compute_ssim_temporal(frames)
    ssim_dusk = float(np.mean(ssim_scores[5:9])) if len(ssim_scores) > 8 else None
    ssim_mean = float(np.mean(ssim_scores)) if len(ssim_scores) else None

    return {
        "clip_day_slope": day_slope,
        "clip_night_slope": night_slope,
        "clip_mid_separation": mid_sep,
        "fwe_mean": fwe_mean,
        "fwe_dusk": fwe_dusk,
        "lpips_mean": lpips_mean,
        "lpips_dusk": lpips_dusk,
        "ssim_mean": ssim_mean,
        "ssim_dusk": ssim_dusk,
        "num_frames": n,
    }


def main():
    parser = argparse.ArgumentParser(description="Full evaluation pipeline")
    parser.add_argument("--ablation-dir", default="outputs/ablation")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--output", default="outputs/metrics")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    prompt_map = load_prompt_data(args.prompts)
    ablation_dir = Path(args.ablation_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for config_dir in sorted(ablation_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        print(f"\nEvaluating config: {config_name}")

        for video_path in sorted(config_dir.glob("*.gif")):
            stem = video_path.stem  # prompt_{id}_seed{seed}
            parts = stem.split("_")
            prompt_id = int(parts[1])
            seed = int(parts[2].replace("seed", ""))

            prompt_data = prompt_map.get(prompt_id, {})
            if not prompt_data:
                continue

            metrics = evaluate_video(
                str(video_path), prompt_data, clip_model, clip_processor, args.device
            )
            if not metrics:
                continue

            entry = {
                "config": config_name,
                "prompt_id": prompt_id,
                "seed": seed,
                **metrics,
            }
            results.append(entry)
            print(f"  {stem}: clip_day_slope={metrics['clip_day_slope']:.4f}, "
                  f"clip_night_slope={metrics['clip_night_slope']:.4f}, "
                  f"fwe_dusk={metrics['fwe_dusk']:.4f}")

    # Save raw results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / "all_results.csv", index=False)

    # Aggregate per config
    if not df.empty:
        summary = df.groupby("config").agg({
            "clip_day_slope": ["mean", "std"],
            "clip_night_slope": ["mean", "std"],
            "clip_mid_separation": ["mean", "std"],
            "fwe_mean": ["mean", "std"],
            "fwe_dusk": ["mean", "std"],
            "lpips_dusk": ["mean", "std"],
            "ssim_dusk": ["mean", "std"],
        }).round(4)
        summary.to_csv(output_dir / "ablation_summary.csv")
        print("\n" + "=" * 60)
        print("Ablation Summary")
        print("=" * 60)
        print(summary)
        print("=" * 60)

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
