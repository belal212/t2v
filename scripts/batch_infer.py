#!/usr/bin/env python3
"""
Multi-GPU batch inference script.
Distributes prompts across available GPUs using process-level isolation.

Usage:
    python scripts/batch_infer.py \
        --config C0_baseline \
        --prompts data/eval_prompts.json \
        --output-dir outputs/baseline \
        --seeds 42 123 999

Each GPU runs a separate process to avoid CUDA context conflicts.
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif


def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_on_gpu(
    gpu_id: int,
    prompt_list: list[dict],
    base_model: str,
    motion_module: str,
    output_dir: str,
    seeds: list[int],
    num_frames: int,
    guidance_scale: float,
    num_steps: int,
    negative_prompt: str,
):
    """Worker function executed in a separate process per GPU."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    # Load model on this GPU
    adapter = MotionAdapter.from_pretrained(motion_module, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()

    gpu_out = Path(output_dir) / f"gpu{gpu_id}"
    gpu_out.mkdir(parents=True, exist_ok=True)

    for p in prompt_list:
        pid = p["id"]
        prompt_text = p["prompt"]

        for seed in seeds:
            out_path = gpu_out / f"prompt_{pid}_seed{seed}.gif"
            if out_path.exists():
                print(f"  [GPU{gpu_id}] Skip existing: {out_path.name}")
                continue

            print(f"  [GPU{gpu_id}] prompt={pid} seed={seed}")
            generator = torch.Generator(device=device).manual_seed(seed)
            output = pipe(
                prompt=prompt_text,
                negative_prompt=negative_prompt,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
                generator=generator,
            )
            export_to_gif(output.frames[0], str(out_path))

    print(f"[GPU{gpu_id}] Done.")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU batch inference")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--output-dir", default="outputs/baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--negative-prompt", default="static, flickering, low quality")
    parser.add_argument("--num-gpus", type=int, default=None, help="Override auto-detect")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    num_gpus = args.num_gpus or torch.cuda.device_count()

    if num_gpus == 0:
        print("ERROR: No CUDA GPUs detected.")
        sys.exit(1)

    print("=" * 60)
    print("Multi-GPU Batch Inference")
    print("=" * 60)
    print(f"Prompts:     {len(prompts)}")
    print(f"Seeds:       {args.seeds}")
    print(f"GPUs:        {num_gpus}")
    print(f"Total runs:  {len(prompts) * len(args.seeds)}")
    print(f"Output:      {args.output_dir}")
    print("=" * 60)

    # Distribute prompts round-robin across GPUs
    splits = [prompts[i::num_gpus] for i in range(num_gpus)]

    procs = []
    for gpu_id in range(num_gpus):
        if not splits[gpu_id]:
            continue
        p = mp.Process(
            target=run_on_gpu,
            args=(
                gpu_id,
                splits[gpu_id],
                args.base_model,
                args.motion_module,
                args.output_dir,
                args.seeds,
                args.num_frames,
                args.guidance_scale,
                args.num_inference_steps,
                args.negative_prompt,
            ),
        )
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    print("\n" + "=" * 60)
    print("All GPUs finished.")
    print("=" * 60)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
