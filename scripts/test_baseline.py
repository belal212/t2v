#!/usr/bin/env python3
"""
Baseline inference verification script.
Generates a single 16-frame GIF using vanilla AnimateDiff
(SD 1.5 + motion module) to confirm the environment is functional.

Usage:
    python scripts/test_baseline.py [--output outputs/test_baseline.gif]

Expected VRAM: ~12-16 GB (enable_cpu_offload recommended on 6GB cards).
"""

import argparse
import sys
from pathlib import Path

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif


def main():
    parser = argparse.ArgumentParser(description="Baseline AnimateDiff inference test")
    parser.add_argument(
        "--base-model", type=str, default="models/stable-diffusion-v1-5",
        help="Path or HuggingFace ID of SD 1.5 base model",
    )
    parser.add_argument(
        "--motion-module", type=str, default="models/motion-module",
        help="Path or HuggingFace ID of AnimateDiff motion adapter",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/test_baseline.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--prompt", type=str,
        default="cinematic time-lapse of a city skyline transitioning from day to night",
        help="Text prompt for generation",
    )
    parser.add_argument(
        "--negative-prompt", type=str,
        default="static, flickering, low quality",
        help="Negative prompt",
    )
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu-offload", action="store_true", help="Enable model CPU offload to save VRAM")
    parser.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing to save VRAM")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Baseline Inference Test — AnimateDiff v2 (SD 1.5)")
    print("=" * 60)
    print(f"Base model:      {args.base_model}")
    print(f"Motion module:   {args.motion_module}")
    print(f"Output:          {args.output}")
    print(f"Prompt:          {args.prompt}")
    print(f"Frames:          {args.num_frames}")
    print(f"Steps:           {args.num_inference_steps}")
    print(f"CFG scale:       {args.guidance_scale}")
    print(f"Seed:            {args.seed}")
    print(f"CPU offload:     {args.cpu_offload}")
    print("=" * 60)

    # Load motion adapter
    print("\n[1/3] Loading motion adapter...")
    adapter = MotionAdapter.from_pretrained(
        args.motion_module,
        torch_dtype=torch.float16,
    )

    # Load pipeline
    print("[2/3] Loading AnimateDiff pipeline...")
    pipe = AnimateDiffPipeline.from_pretrained(
        args.base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    # VRAM optimizations
    if args.cpu_offload:
        print("  -> Enabling model CPU offload")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to("cuda")

    if args.vae_slicing:
        print("  -> Enabling VAE slicing")
        pipe.enable_vae_slicing()

    # Generate
    print("[3/3] Generating video...")
    generator = torch.manual_seed(args.seed)
    output = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        generator=generator,
    )

    # Export
    frames = output.frames[0]
    export_to_gif(frames, args.output)
    print(f"\n✓ Baseline inference successful!")
    print(f"  Saved {len(frames)} frames to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
