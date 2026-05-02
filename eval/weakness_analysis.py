#!/usr/bin/env python3
"""
Weakness Analysis — Baseline AnimateDiff Evaluation

Computes per-frame CLIP-SIM curves (day vs night) and flow warping error
to diagnose semantic drift and temporal flickering.

Usage:
    python eval/weakness_analysis.py \
        --video outputs/baseline/prompt_0_seed42.gif \
        --prompt-day "bright sunny day in a city" \
        --prompt-night "dark night city skyline" \
        --output outputs/metrics/prompt_0_seed42.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_frames(video_path: str) -> list[np.ndarray]:
    """Load video frames as RGB numpy arrays [H, W, 3]."""
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    # Fallback for GIF / image sequences
    if not frames:
        img = Image.open(video_path)
        try:
            while True:
                frames.append(np.array(img.convert("RGB")))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

    return frames


def compute_per_frame_clip_sim(
    frames: list[np.ndarray],
    prompt_day: str,
    prompt_night: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns two arrays of length N:
        clip_day_scores, clip_night_scores
    """
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        day_inputs = processor(text=[prompt_day], return_tensors="pt", padding=True)
        night_inputs = processor(text=[prompt_night], return_tensors="pt", padding=True)
        day_feat = model.get_text_features(
            input_ids=day_inputs["input_ids"].to(device),
            attention_mask=day_inputs["attention_mask"].to(device),
        )
        night_feat = model.get_text_features(
            input_ids=night_inputs["input_ids"].to(device),
            attention_mask=night_inputs["attention_mask"].to(device),
        )
        day_feat = F.normalize(day_feat, dim=-1)
        night_feat = F.normalize(night_feat, dim=-1)

    day_scores, night_scores = [], []

    with torch.no_grad():
        for frame in frames:
            pil_img = Image.fromarray(frame)
            img_inputs = processor(images=pil_img, return_tensors="pt")
            img_feat = model.get_image_features(
                pixel_values=img_inputs["pixel_values"].to(device),
            )
            img_feat = F.normalize(img_feat, dim=-1)

            day_sim = (img_feat * day_feat).sum(dim=-1).item()
            night_sim = (img_feat * night_feat).sum(dim=-1).item()
            day_scores.append(day_sim)
            night_scores.append(night_sim)

    return np.array(day_scores), np.array(night_scores)


def compute_flow_warping_error(frames: list[np.ndarray]) -> np.ndarray:
    """
    Returns array of length N-1: FWE per transition.
    Uses Farneback optical flow (no RAFT dependency).
    """
    errors = []
    for i in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(
            np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32)
        )
        map_x += flow[..., 0]
        map_y += flow[..., 1]
        warped = cv2.remap(
            frames[i].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR
        )
        error = np.mean((warped - frames[i + 1].astype(np.float32)) ** 2)
        errors.append(error)
    return np.array(errors)


def compute_lpips_temporal(frames: list[np.ndarray]) -> np.ndarray:
    """
    Compute LPIPS between consecutive frames.
    Returns array of length N-1.
    """
    try:
        import lpips
    except ImportError:
        return np.array([])

    loss_fn = lpips.LPIPS(net="alex").to("cuda")
    scores = []
    for i in range(len(frames) - 1):
        # Convert to [-1, 1] torch tensors [1, 3, H, W]
        t1 = torch.from_numpy(frames[i]).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        t2 = torch.from_numpy(frames[i + 1]).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        with torch.no_grad():
            dist = loss_fn(t1.to("cuda"), t2.to("cuda")).item()
        scores.append(dist)
    return np.array(scores)


def compute_ssim_temporal(frames: list[np.ndarray]) -> np.ndarray:
    """
    Compute SSIM between consecutive frames.
    Returns array of length N-1.
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        return np.array([])

    scores = []
    for i in range(len(frames) - 1):
        score = ssim(frames[i], frames[i + 1], channel_axis=-1, data_range=255.0)
        scores.append(score)
    return np.array(scores)


def analyze_video(
    video_path: str,
    prompt_day: str,
    prompt_night: str,
    clip_model_name: str = "openai/clip-vit-large-patch14",
    device: str = "cuda",
) -> dict:
    """Run full weakness analysis on a single video."""
    frames = load_frames(video_path)
    if not frames:
        raise ValueError(f"No frames loaded from {video_path}")

    print(f"Loaded {len(frames)} frames from {video_path}")

    # Load CLIP
    print("Loading CLIP model...")
    clip_model = CLIPModel.from_pretrained(clip_model_name).to(device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    # CLIP-SIM curves
    print("Computing CLIP-SIM curves...")
    day_scores, night_scores = compute_per_frame_clip_sim(
        frames, prompt_day, prompt_night, clip_model, clip_processor, device
    )

    # Linear regression slopes
    frame_indices = np.arange(len(day_scores))
    day_slope = np.polyfit(frame_indices, day_scores, 1)[0]
    night_slope = np.polyfit(frame_indices, night_scores, 1)[0]

    mid_frame = len(frames) // 2
    mid_separation = night_scores[mid_frame] - day_scores[mid_frame]

    # Flow warping error
    print("Computing flow warping error...")
    fwe = compute_flow_warping_error(frames)

    # Dusk window = frames 6-9 (transitions 5-8)
    dusk_fwe = np.mean(fwe[5:9]) if len(fwe) > 8 else np.mean(fwe)

    # LPIPS & SSIM
    print("Computing LPIPS & SSIM...")
    lpips_scores = compute_lpips_temporal(frames)
    ssim_scores = compute_ssim_temporal(frames)

    dusk_lpips = np.mean(lpips_scores[5:9]) if len(lpips_scores) > 8 else np.mean(lpips_scores)
    dusk_ssim = np.mean(ssim_scores[5:9]) if len(ssim_scores) > 8 else np.mean(ssim_scores)

    results = {
        "video_path": video_path,
        "num_frames": len(frames),
        "clip_sim": {
            "day_scores": day_scores.tolist(),
            "night_scores": night_scores.tolist(),
            "day_slope": float(day_slope),
            "night_slope": float(night_slope),
            "mid_frame_separation": float(mid_separation),
        },
        "flow_warping_error": {
            "per_transition": fwe.tolist(),
            "mean": float(np.mean(fwe)),
            "dusk_window_mean": float(dusk_fwe),
        },
        "lpips": {
            "per_transition": lpips_scores.tolist() if len(lpips_scores) else [],
            "dusk_window_mean": float(dusk_lpips) if len(lpips_scores) else None,
        },
        "ssim": {
            "per_transition": ssim_scores.tolist() if len(ssim_scores) else [],
            "dusk_window_mean": float(dusk_ssim) if len(ssim_scores) else None,
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Weakness analysis for a single video")
    parser.add_argument("--video", required=True, help="Path to video/GIF file")
    parser.add_argument("--prompt-day", required=True, help="Day prompt for CLIP-SIM")
    parser.add_argument("--prompt-night", required=True, help="Night prompt for CLIP-SIM")
    parser.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    results = analyze_video(
        args.video,
        args.prompt_day,
        args.prompt_night,
        args.clip_model,
        args.device,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Weakness Analysis Results")
    print("=" * 60)
    print(f"  Day CLIP-SIM slope:     {results['clip_sim']['day_slope']:.4f} (target: negative)")
    print(f"  Night CLIP-SIM slope:   {results['clip_sim']['night_slope']:.4f} (target: positive)")
    print(f"  Mid-frame separation:   {results['clip_sim']['mid_frame_separation']:.4f}")
    print(f"  Flow Warp Error (mean): {results['flow_warping_error']['mean']:.4f}")
    print(f"  Flow Warp Error (dusk): {results['flow_warping_error']['dusk_window_mean']:.4f}")
    if results['lpips']['dusk_window_mean'] is not None:
        print(f"  LPIPS (dusk):           {results['lpips']['dusk_window_mean']:.4f}")
    if results['ssim']['dusk_window_mean'] is not None:
        print(f"  SSIM (dusk):            {results['ssim']['dusk_window_mean']:.4f}")
    print("=" * 60)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
