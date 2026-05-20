#!/usr/bin/env python3
"""
Weakness Analysis Runner — Baseline AnimateDiff Evaluation

Runs the full methodology from PLAN/07_weakness_analysis.md:
  1. Generate 150 baseline videos (50 prompts × 3 seeds)
  2. Compute CLIP-SIM dual curves, flow warping error, LPIPS, SSIM
  3. Aggregate metrics and produce weakness report

Output structure (matching the plan):
  outputs/baseline/
    prompt_{id}/
      seed_{seed}.gif
    metrics/
      per_frame_clip_day_night.csv
      flow_warping_error.csv
      lpips_temporal.csv
      summary.json

Usage:
    python3 eval/run_weakness_analysis.py \
        --prompts data/eval_prompts_10.json \
        --output-dir outputs/baseline \
        --base-model models/stable-diffusion-v1-5 \
        --motion-module models/motion-module \
        --skip-generation  # skip if videos already exist
"""

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.weakness_analysis import (
    load_frames,
    compute_per_frame_clip_sim,
    compute_flow_warping_error,
    compute_lpips_temporal,
    compute_ssim_temporal,
)

SEEDS = [42, 123, 999]


def generate_baseline_videos(
    prompts: list[dict],
    output_dir: str,
    base_model: str,
    motion_module: str,
    num_frames: int,
    num_steps: int,
    guidance_scale: float,
    num_gpus: int,
):
    """Generate 150 baseline videos using batch inference."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    total = len(prompts) * len(SEEDS)
    print(f"Generating {total} baseline videos ({len(prompts)} prompts × {len(SEEDS)} seeds)...")

    prompts_file = out / "eval_prompts_subset.json"
    with open(prompts_file, "w") as f:
        json.dump(prompts, f)

    cmd = [
        sys.executable, "scripts/batch_infer.py",
        "--base-model", base_model,
        "--motion-module", motion_module,
        "--prompts", str(prompts_file),
        "--output-dir", output_dir,
        "--seeds", *map(str, SEEDS),
        "--num-frames", str(num_frames),
        "--num-inference-steps", str(num_steps),
        "--guidance-scale", str(guidance_scale),
    ]
    if num_gpus > 0:
        cmd += ["--num-gpus", str(num_gpus)]

    print(f"  Command: {' '.join(cmd)}")
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        print("ERROR: Baseline generation failed.")
        sys.exit(1)

    # Rename to match the plan's prompt_{id}/seed_{seed} structure
    for gpu_dir in out.glob("gpu*"):
        for f in gpu_dir.iterdir():
            if f.is_file() and f.suffix in {".gif", ".mp4"}:
                parts = f.stem.split("_")
                pid = parts[1]
                seed = parts[2].replace("seed", "")
                target = out / f"prompt_{pid}" / f"seed_{seed}.gif"
                target.parent.mkdir(parents=True, exist_ok=True)
                f.rename(target)
        # Remove gpu dirs
        for f in gpu_dir.iterdir():
            f.unlink()
        gpu_dir.rmdir()

    # Clean up
    prompts_file.unlink()
    print("Baseline generation complete.")


def compute_metrics(
    prompts: list[dict],
    output_dir: str,
    clip_model_name: str,
    device: str,
) -> dict:
    """Compute all metrics for all videos and produce structured output."""
    from transformers import CLIPModel, CLIPProcessor

    out = Path(output_dir)
    metrics_dir = out / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    print("Loading CLIP model...")
    clip = CLIPModel.from_pretrained(clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(clip_model_name)

    prompt_map = {p["id"]: p for p in prompts}

    all_day_scores = []
    all_night_scores = []
    all_fwe = []
    all_lpips = []
    all_ssim = []
    video_summaries = []

    print(f"Computing metrics for {len(prompts) * len(SEEDS)} videos...")

    for p in prompts:
        pid = p["id"]
        prompt_dir = out / f"prompt_{pid}"
        if not prompt_dir.exists():
            continue

        for seed in SEEDS:
            video_path = prompt_dir / f"seed_{seed}.gif"
            if not video_path.exists():
                continue

            frames = load_frames(str(video_path))
            if len(frames) < 2:
                continue

            # CLIP-SIM dual curves
            day_scores, night_scores = compute_per_frame_clip_sim(
                frames, p["prompt_day"], p["prompt_night"],
                clip, processor, device,
            )
            all_day_scores.append(day_scores)
            all_night_scores.append(night_scores)

            # Flow warping
            fwe = compute_flow_warping_error(frames)
            all_fwe.append(fwe)

            # LPIPS
            lpips_scores = compute_lpips_temporal(frames)
            all_lpips.append(lpips_scores)

            # SSIM
            ssim_scores = compute_ssim_temporal(frames)
            all_ssim.append(ssim_scores)

            video_summaries.append({
                "prompt_id": pid,
                "seed": seed,
                "category": p.get("category", ""),
                "difficulty": p.get("difficulty", "normal"),
                "day_slope": float(np.polyfit(np.arange(len(day_scores)), day_scores, 1)[0]),
                "night_slope": float(np.polyfit(np.arange(len(night_scores)), night_scores, 1)[0]),
                "mid_separation": float(night_scores[len(night_scores)//2] - day_scores[len(day_scores)//2]),
                "fwe_mean": float(np.mean(fwe)),
                "fwe_dusk": float(np.mean(fwe[5:9])) if len(fwe) > 8 else float(np.mean(fwe)),
                "lpips_dusk": float(np.mean(lpips_scores[5:9])) if len(lpips_scores) > 8 else None,
                "ssim_dusk": float(np.mean(ssim_scores[5:9])) if len(ssim_scores) > 8 else None,
            })

    # Save per-frame CLIP-SIM CSV
    clip_csv_path = metrics_dir / "per_frame_clip_day_night.csv"
    with open(clip_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_idx", "frame", "day_score", "night_score"])
        for vidx in range(len(all_day_scores)):
            for f_idx in range(len(all_day_scores[vidx])):
                writer.writerow([vidx, f_idx,
                                  f"{all_day_scores[vidx][f_idx]:.6f}",
                                  f"{all_night_scores[vidx][f_idx]:.6f}"])
    print(f"  Saved: {clip_csv_path}")

    # Save FWE CSV
    fwe_csv_path = metrics_dir / "flow_warping_error.csv"
    with open(fwe_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_idx", "transition", "error"])
        for vidx in range(len(all_fwe)):
            for t_idx in range(len(all_fwe[vidx])):
                writer.writerow([vidx, t_idx, f"{all_fwe[vidx][t_idx]:.6f}"])
    print(f"  Saved: {fwe_csv_path}")

    # Save LPIPS CSV
    lpips_csv_path = metrics_dir / "lpips_temporal.csv"
    if all_lpips and all_lpips[0].size > 0:
        with open(lpips_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_idx", "transition", "lpips"])
            for vidx in range(len(all_lpips)):
                for t_idx in range(len(all_lpips[vidx])):
                    writer.writerow([vidx, t_idx, f"{all_lpips[vidx][t_idx]:.6f}"])
        print(f"  Saved: {lpips_csv_path}")
    else:
        print("  LPIPS skipped (lpips not installed)")

    # Save SSIM CSV
    ssim_csv_path = metrics_dir / "ssim_temporal.csv"
    if all_ssim and all_ssim[0].size > 0:
        with open(ssim_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_idx", "transition", "ssim"])
            for vidx in range(len(all_ssim)):
                for t_idx in range(len(all_ssim[vidx])):
                    writer.writerow([vidx, t_idx, f"{all_ssim[vidx][t_idx]:.6f}"])
        print(f"  Saved: {ssim_csv_path}")
    else:
        print("  SSIM skipped (skimage not installed)")

    # Compute aggregate summary
    day_slopes = [v["day_slope"] for v in video_summaries]
    night_slopes = [v["night_slope"] for v in video_summaries]
    mid_seps = [v["mid_separation"] for v in video_summaries]
    dusk_fwe = [v["fwe_dusk"] for v in video_summaries]
    dusk_lpips = [v["lpips_dusk"] for v in video_summaries if v["lpips_dusk"] is not None]
    dusk_ssim = [v["ssim_dusk"] for v in video_summaries if v["ssim_dusk"] is not None]

    avg_day_slope = float(np.mean(day_slopes))
    avg_night_slope = float(np.mean(night_slopes))
    avg_mid_sep = float(np.mean(mid_seps))
    avg_fwe_dusk = float(np.mean(dusk_fwe))
    avg_lpips_dusk = float(np.mean(dusk_lpips)) if dusk_lpips else None
    avg_ssim_dusk = float(np.mean(dusk_ssim)) if dusk_ssim else None

    summary = {
        "num_videos": len(video_summaries),
        "num_prompts": len(prompts),
        "seeds": SEEDS,
        "weakness_1_static_conditioning": {
            "hypothesis": "Same text embedding for all frames → no semantic progression",
            "avg_day_clip_sim_slope": avg_day_slope,
            "avg_night_clip_sim_slope": avg_night_slope,
            "avg_mid_separation": avg_mid_sep,
            "expected_day_slope_target": "Negative (should decrease across frames)",
            "expected_night_slope_target": "Positive (should increase across frames)",
            "root_cause": "Cross-attention receives same text embedding for every frame",
        },
        "weakness_2_temporal_flickering": {
            "hypothesis": "Motion module not trained on photometric transitions",
            "avg_fwe_dusk_mean": avg_fwe_dusk,
            "avg_lpips_dusk_mean": avg_lpips_dusk,
            "avg_ssim_dusk_mean": avg_ssim_dusk,
            "acceptable_thresholds": {
                "fwe_dusk": "< 0.02",
                "lpips_dusk": "< 0.20",
                "ssim_dusk": "> 0.85",
            },
            "root_cause": "Uniform temporal attention weights across 16-frame window",
        },
        "per_video": video_summaries,
    }

    summary_path = metrics_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {summary_path}")

    return summary


def print_weakness_report(summary: dict):
    """Print the ASCII weakness report matching the plan."""
    w1 = summary["weakness_1_static_conditioning"]
    w2 = summary["weakness_2_temporal_flickering"]

    print()
    print("┌──────────────────────────────────────────────────────────────┐")
    print("│ Weakness Analysis Report — Baseline AnimateDiff             │")
    print("├──────────────────────────────────────────────────────────────┤")
    print("│                                                              │")
    print("│ Weakness 1: Static Conditioning → Semantic Drift             │")
    print("│ ┌──────────────────────────────────────────────────────────┐│")
    print(f"│ │ Evidence:                                                 ││")
    print(f"│ │  - CLIP-SIM day-curve slope:   {w1['avg_day_clip_sim_slope']:.4f} (target: negative)  ││")
    print(f"│ │  - CLIP-SIM night-curve slope: {w1['avg_night_clip_sim_slope']:.4f} (target: positive) ││")
    print(f"│ │  - Mid-frame separation:       {w1['avg_mid_separation']:.4f} (target: > 0.10)  ││")
    print(f"│ │ Root Cause: {w1['root_cause']}     ││")
    print("│ └──────────────────────────────────────────────────────────┘│")
    print("│                                                              │")
    print("│ Weakness 2: Temporal Flickering (Dusk Window)                │")
    print("│ ┌──────────────────────────────────────────────────────────┐│")
    print("│ │ Evidence:                                                 ││")
    print(f"│ │  - Flow Warping Error (dusk): {w2['avg_fwe_dusk_mean']:.4f} (target: < 0.02)    ││")
    if w2["avg_lpips_dusk_mean"] is not None:
        print(f"│ │  - LPIPS (dusk):              {w2['avg_lpips_dusk_mean']:.4f} (target: < 0.20)    ││")
    if w2["avg_ssim_dusk_mean"] is not None:
        print(f"│ │  - SSIM (dusk):               {w2['avg_ssim_dusk_mean']:.4f} (target: > 0.85)    ││")
    print(f"│ │ Root Cause: {w2['root_cause']}    ││")
    print("│ └──────────────────────────────────────────────────────────┘│")
    print("└──────────────────────────────────────────────────────────────┘")


def main():
    parser = argparse.ArgumentParser(description="Run weakness analysis on baseline AnimateDiff")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--num-gpus", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-generation", action="store_true",
                        help="Skip video generation if videos already exist")
    args = parser.parse_args()

    with open(args.prompts) as f:
        all_prompts = json.load(f)

    print("=" * 60)
    print("Weakness Analysis — Baseline AnimateDiff")
    print("=" * 60)
    print(f"Prompts: {len(all_prompts)}")
    print(f"Seeds:   {SEEDS}")
    print(f"Videos:  {len(all_prompts) * len(SEEDS)}")
    print("=" * 60)

    out_path = Path(args.output_dir)

    # Phase 1: Generate baseline videos
    if not args.skip_generation:
        generate_baseline_videos(
            all_prompts,
            str(out_path),
            args.base_model,
            args.motion_module,
            args.num_frames,
            args.num_inference_steps,
            args.guidance_scale,
            args.num_gpus,
        )
    else:
        print("Skipping video generation (--skip-generation)")

    # Phase 2: Compute metrics
    summary = compute_metrics(
        all_prompts,
        str(out_path),
        args.clip_model,
        args.device,
    )

    # Phase 3: Print weakness report
    print_weakness_report(summary)

    print("\n" + "=" * 60)
    print("Weakness analysis complete!")
    print(f"Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
