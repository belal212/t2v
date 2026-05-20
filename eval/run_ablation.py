#!/usr/bin/env python3
"""
Ablation Study Runner — Multi-GPU, no subprocess overhead.

Each GPU loads the model ONCE and generates all its assigned videos in a loop.
5 configs across 4 GPUs: GPU0=C0+C4, GPU1=C1, GPU2=C2, GPU3=C3.

Usage:
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 eval/run_ablation.py \
        --prompts data/eval_prompts_10.json \
        --configs C0_baseline C1_A_only C2_B_only C3_A_plus_B C4_A_B_C \
        --num-gpus 4
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.inference_enhanced import setup_pipeline, generate_enhanced
from diffusers.utils import export_to_gif, export_to_video


CONFIGS = {
    "C0_baseline":  {"interp": False, "blend": 0.0, "bias": False},
    "C1_A_only":    {"interp": True,  "blend": 0.0, "bias": False},
    "C2_B_only":    {"interp": False, "blend": 0.3, "bias": False},
    "C3_A_plus_B":  {"interp": True,  "blend": 0.2, "bias": False},
    "C4_A_B_C":     {"interp": True,  "blend": 0.2, "bias": True},
    "C5_B_plus_C":  {"interp": False, "blend": 0.3, "bias": True},
}

SEEDS = [42, 123, 999]


def assign_configs(config_names: list[str], num_gpus: int) -> list[list[str]]:
    """Distribute configs across GPUs — each GPU gets ~equal total load."""
    assignments = [[] for _ in range(num_gpus)]
    for i, name in enumerate(config_names):
        assignments[i % num_gpus].append(name)
    return assignments


def worker(gpu_id: int, configs: list[str], prompts: list[dict], args):
    """Load model once, generate all videos for assigned configs."""
    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    pipe = setup_pipeline(args.base_model, args.motion_module, device=device)

    for config_name in configs:
        cfg_flags = CONFIGS[config_name]
        out_dir = Path(args.output_dir) / config_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"[GPU{gpu_id}] {config_name} — {len(prompts)} prompts × {len(SEEDS)} seeds")

        for p in prompts:
            pid = p["id"]
            for seed in SEEDS:
                if args.ext == ".gif":
                    out_path = out_dir / f"prompt_{pid}_seed{seed}.gif"
                else:
                    out_path = out_dir / f"prompt_{pid}_seed{seed}.mp4"

                if out_path.exists():
                    continue

                frames = generate_enhanced(
                    pipe,
                    prompt=p["prompt"],
                    prompt_day=p["prompt_day"],
                    prompt_night=p["prompt_night"],
                    num_frames=args.num_frames,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    seed=seed,
                    config=config_name,
                    blend_beta=cfg_flags["blend"],
                    device=device,
                )

                if args.ext == ".mp4":
                    export_to_video(frames, str(out_path), fps=args.fps, quality=9)
                else:
                    export_to_gif(frames, str(out_path))

                print(f"  [GPU{gpu_id}] {out_path.name} ✓")

        print(f"[GPU{gpu_id}] {config_name} done")

    print(f"[GPU{gpu_id}] All done")


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU ablation runner (no subprocess)")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--ext", default=".gif", choices=[".gif", ".mp4"])
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()

    with open(args.prompts) as f:
        prompts = json.load(f)

    num_gpus = args.num_gpus or torch.cuda.device_count()
    if num_gpus == 0:
        num_gpus = 1

    assignments = assign_configs(args.configs, num_gpus)

    total = len(args.configs) * len(prompts) * len(SEEDS)
    print("=" * 60)
    print("Ablation — Direct GPU (no subprocess)")
    print("=" * 60)
    print(f"Configs: {len(args.configs)}  Prompts: {len(prompts)}  Seeds: {len(SEEDS)}")
    print(f"GPUs:    {num_gpus}  Total videos: {total}")
    for gid, configs in enumerate(assignments):
        videos = sum(len(prompts) * len(SEEDS) for _ in configs)
        print(f"  GPU{gid}: {configs} ({videos} videos)")
    print("=" * 60)

    procs = []
    for gpu_id, configs in enumerate(assignments):
        if not configs:
            continue
        p = mp.Process(target=worker, args=(gpu_id, configs, prompts, args))
        procs.append(p)
        p.start()

    for p in procs:
        p.join()

    print("\n✓ All done! Run: python3 eval/eval.py --ablation-dir outputs/ablation ...")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
