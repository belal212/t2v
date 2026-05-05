#!/usr/bin/env python3
"""
Batch Weakness Analysis

Computes CLIP-SIM curves, flow warping error, LPIPS, and SSIM
for a folder of baseline videos/GIFs.

Usage:
    python eval/run_weakness_batch.py \
        --inputs outputs/baseline \
        --prompts data/eval_prompts.json \
        --output outputs/baseline/metrics \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).parent
sys.path.insert(0, str(EVAL_DIR))

from weakness_analysis import (
    load_frames,
    compute_per_frame_clip_sim,
    compute_flow_warping_error,
    compute_lpips_temporal,
    compute_ssim_temporal,
)


def _parse_prompt_map(prompts_path: str) -> dict[int, dict]:
    data = json.loads(Path(prompts_path).read_text())
    return {p["id"]: p for p in data}


def _find_videos(inputs: Path) -> list[Path]:
    patterns = ["*.gif", "*.mp4", "*.mov"]
    results: list[Path] = []
    for pat in patterns:
        results.extend(inputs.rglob(pat))
    return sorted(results)


def _parse_id_seed(path: Path) -> tuple[int | None, int | None]:
    m = re.search(r"prompt_(\d+)_seed(\d+)", path.stem)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def _save_plot(output_path: Path, x: np.ndarray, series: dict[str, np.ndarray], title: str, ylabel: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4))
    for name, y in series.items():
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel("Frame" if len(x) > 1 else "Transition")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch weakness analysis")
    parser.add_argument("--inputs", default="outputs/baseline")
    parser.add_argument("--prompts", default="data/eval_prompts.json")
    parser.add_argument("--output", default="outputs/baseline/metrics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()

    from transformers import CLIPModel, CLIPProcessor

    inputs = Path(args.inputs)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_map = _parse_prompt_map(args.prompts)
    videos = _find_videos(inputs)

    if not videos:
        print(f"No videos found under {inputs}")
        return

    print(f"Found {len(videos)} videos")
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(args.clip_model).to(args.device)
    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)

    per_video_rows = []
    day_curves = []
    night_curves = []
    fwe_curves = []
    lpips_curves = []
    ssim_curves = []

    for vp in videos:
        pid, seed = _parse_id_seed(vp)
        if pid is None or seed is None:
            continue
        prompt = prompt_map.get(pid)
        if not prompt:
            continue

        frames = load_frames(str(vp))
        if not frames:
            continue

        day_scores, night_scores = compute_per_frame_clip_sim(
            frames,
            prompt["prompt_day"],
            prompt["prompt_night"],
            clip_model,
            clip_processor,
            args.device,
        )
        fwe = compute_flow_warping_error(frames)
        lpips_scores = compute_lpips_temporal(frames)
        ssim_scores = compute_ssim_temporal(frames)

        n = len(frames)
        frame_idx = np.arange(n)
        day_slope = float(np.polyfit(frame_idx, day_scores, 1)[0])
        night_slope = float(np.polyfit(frame_idx, night_scores, 1)[0])
        mid_frame = n // 2
        mid_sep = float(night_scores[mid_frame] - day_scores[mid_frame])

        per_video_rows.append(
            {
                "prompt_id": pid,
                "seed": seed,
                "video_path": str(vp),
                "num_frames": n,
                "clip_day_slope": day_slope,
                "clip_night_slope": night_slope,
                "clip_mid_separation": mid_sep,
                "fwe_mean": float(np.mean(fwe)),
                "fwe_dusk": float(np.mean(fwe[5:9])) if len(fwe) > 8 else float(np.mean(fwe)),
                "lpips_mean": float(np.mean(lpips_scores)) if len(lpips_scores) else None,
                "lpips_dusk": float(np.mean(lpips_scores[5:9])) if len(lpips_scores) > 8 else None,
                "ssim_mean": float(np.mean(ssim_scores)) if len(ssim_scores) else None,
                "ssim_dusk": float(np.mean(ssim_scores[5:9])) if len(ssim_scores) > 8 else None,
            }
        )

        day_curves.append(day_scores)
        night_curves.append(night_scores)
        fwe_curves.append(fwe)
        if len(lpips_scores):
            lpips_curves.append(lpips_scores)
        if len(ssim_scores):
            ssim_curves.append(ssim_scores)

    # Save per-video metrics
    import pandas as pd

    df = pd.DataFrame(per_video_rows)
    df.to_csv(output_dir / "per_video_metrics.csv", index=False)

    if df.empty:
        print("No metrics computed.")
        return

    # Aggregate per-frame curves (trim to minimum length)
    min_frames = min(len(c) for c in day_curves)
    min_trans = min(len(c) for c in fwe_curves)

    day_stack = np.stack([c[:min_frames] for c in day_curves])
    night_stack = np.stack([c[:min_frames] for c in night_curves])
    fwe_stack = np.stack([c[:min_trans] for c in fwe_curves])

    day_mean = day_stack.mean(axis=0)
    night_mean = night_stack.mean(axis=0)
    fwe_mean = fwe_stack.mean(axis=0)

    # Save curve CSVs
    frame_idx = np.arange(min_frames)
    trans_idx = np.arange(min_trans)

    np.savetxt(
        output_dir / "per_frame_clip_day_night.csv",
        np.column_stack([frame_idx, day_mean, night_mean]),
        delimiter=",",
        header="frame,clip_day_mean,clip_night_mean",
        comments="",
    )
    np.savetxt(
        output_dir / "flow_warping_error.csv",
        np.column_stack([trans_idx, fwe_mean]),
        delimiter=",",
        header="transition,fwe_mean",
        comments="",
    )

    # Optional LPIPS/SSIM curves
    if lpips_curves:
        min_lp = min(len(c) for c in lpips_curves)
        lp_stack = np.stack([c[:min_lp] for c in lpips_curves])
        lp_mean = lp_stack.mean(axis=0)
        np.savetxt(
            output_dir / "lpips_temporal.csv",
            np.column_stack([np.arange(min_lp), lp_mean]),
            delimiter=",",
            header="transition,lpips_mean",
            comments="",
        )

    if ssim_curves:
        min_ss = min(len(c) for c in ssim_curves)
        ss_stack = np.stack([c[:min_ss] for c in ssim_curves])
        ss_mean = ss_stack.mean(axis=0)
        np.savetxt(
            output_dir / "ssim_temporal.csv",
            np.column_stack([np.arange(min_ss), ss_mean]),
            delimiter=",",
            header="transition,ssim_mean",
            comments="",
        )

    # Summary
    summary = {
        "num_videos": int(df.shape[0]),
        "clip_day_slope_mean": float(df["clip_day_slope"].mean()),
        "clip_night_slope_mean": float(df["clip_night_slope"].mean()),
        "clip_mid_separation_mean": float(df["clip_mid_separation"].mean()),
        "fwe_dusk_mean": float(df["fwe_dusk"].mean()),
        "lpips_dusk_mean": float(df["lpips_dusk"].mean()) if df["lpips_dusk"].notna().any() else None,
        "ssim_dusk_mean": float(df["ssim_dusk"].mean()) if df["ssim_dusk"].notna().any() else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Plots
    _save_plot(
        output_dir / "clip_curves_mean.png",
        frame_idx,
        {"day": day_mean, "night": night_mean},
        "Per-Frame CLIP-SIM (Mean)",
        "CLIP-SIM",
    )
    _save_plot(
        output_dir / "flow_warp_mean.png",
        trans_idx,
        {"fwe": fwe_mean},
        "Flow Warping Error (Mean)",
        "FWE",
    )

    summary_md = [
        "# Weakness Analysis Summary",
        "",
        f"Videos analyzed: {summary['num_videos']}",
        f"CLIP day slope mean: {summary['clip_day_slope_mean']:.4f}",
        f"CLIP night slope mean: {summary['clip_night_slope_mean']:.4f}",
        f"CLIP mid separation mean: {summary['clip_mid_separation_mean']:.4f}",
        f"FWE dusk mean: {summary['fwe_dusk_mean']:.4f}",
        f"LPIPS dusk mean: {summary['lpips_dusk_mean']}",
        f"SSIM dusk mean: {summary['ssim_dusk_mean']}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_md) + "\n")

    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
