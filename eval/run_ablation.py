#!/usr/bin/env python3
"""
Ablation Study Runner

Generates all 750 videos (5 configs × 50 prompts × 3 seeds) using
the enhanced inference pipeline. Supports multi-GPU distribution.

Usage:
    python eval/run_ablation.py \
        --prompts data/eval_prompts.json \
        --output-dir outputs/ablation \
        --base-model models/stable-diffusion-v1-5 \
        --motion-module models/motion-module
"""

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import torch


CONFIGS = {
    "C0_baseline":  {"config": "C0_baseline",  "blend_beta": 0.0},
    "C1_A_only":    {"config": "C1_A_only",    "blend_beta": 0.0},
    "C2_B_only":    {"config": "C2_B_only",    "blend_beta": 0.3},
    "C3_A_plus_B":  {"config": "C3_A_plus_B",  "blend_beta": 0.2},
    "C4_A_B_C":     {"config": "C4_A_B_C",     "blend_beta": 0.2},
}

SEEDS = [42, 123, 999]


def run_config_on_gpu(
    gpu_id: int,
    config_name: str,
    prompts: list[dict],
    output_dir: str,
    base_model: str,
    motion_module: str,
    num_frames: int,
    num_steps: int,
    guidance_scale: float,
):
    """Generate all videos for one config on a single GPU."""
    cfg = CONFIGS[config_name]
    cfg_out = Path(output_dir) / config_name
    cfg_out.mkdir(parents=True, exist_ok=True)

    print(f"[GPU{gpu_id}] Starting {config_name} ({len(prompts)} prompts × {len(SEEDS)} seeds)")

    for p in prompts:
        pid = p["id"]
        for seed in SEEDS:
            out_path = cfg_out / f"prompt_{pid}_seed{seed}.gif"
            if out_path.exists():
                continue

            cmd = [
                sys.executable, "inference/inference_enhanced.py",
                "--base-model", base_model,
                "--motion-module", motion_module,
                "--config", cfg["config"],
                "--prompt", p["prompt"],
                "--prompt-day", p["prompt_day"],
                "--prompt-night", p["prompt_night"],
                "--seed", str(seed),
                "--num-frames", str(num_frames),
                "--num-inference-steps", str(num_steps),
                "--guidance-scale", str(guidance_scale),
                "--blend-beta", str(cfg["blend_beta"]),
                "--output", str(out_path),
                "--device", f"cuda:{gpu_id}",
            ]
            subprocess.run(cmd, capture_output=True)

    print(f"[GPU{gpu_id}] Finished {config_name}")


def main():
    parser = argparse.ArgumentParser(description="Run full ablation study")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()),
                        help="Subset of configs to run")
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1  # CPU fallback

    total_videos = len(args.configs) * len(prompts) * len(SEEDS)
    print("=" * 60)
    print("Ablation Study Runner")
    print("=" * 60)
    print(f"Configs:   {len(args.configs)}")
    print(f"Prompts:   {len(prompts)}")
    print(f"Seeds:     {len(SEEDS)}")
    print(f"GPUs:      {num_gpus}")
    print(f"Total:     {total_videos} videos")
    print("=" * 60)

    # Option 1: Run sequentially (safer for limited VRAM)
    # Option 2: Parallelize across GPUs
    if num_gpus > 1:
        procs = []
        for i, config_name in enumerate(args.configs):
            gpu_id = i % num_gpus
            p = mp.Process(
                target=run_config_on_gpu,
                args=(
                    gpu_id, config_name, prompts, args.output_dir,
                    args.base_model, args.motion_module,
                    args.num_frames, args.num_inference_steps, args.guidance_scale,
                ),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join()
    else:
        for config_name in args.configs:
            run_config_on_gpu(
                0, config_name, prompts, args.output_dir,
                args.base_model, args.motion_module,
                args.num_frames, args.num_inference_steps, args.guidance_scale,
            )

    print("\n" + "=" * 60)
    print("Ablation generation complete!")
    print("Next: run eval/compute_ablation_metrics.py")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
