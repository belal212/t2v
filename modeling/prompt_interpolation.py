#!/usr/bin/env python3
"""
Enhancement A — Time-Dependent Prompt Conditioning

Interpolates between day and night CLIP text embeddings per frame
using a sigmoid schedule. Replaces the single static text embedding
used for all frames in baseline AnimateDiff.

Usage:
    from modeling.prompt_interpolation import interpolate_prompt_embeddings
    embeddings = interpolate_prompt_embeddings(
        pipe, prompt_day, prompt_night, num_frames=16, k=6.0
    )
"""

import math
from typing import Optional

import torch


def sigmoid_schedule(t: float, k: float = 6.0, center: float = 0.5) -> float:
    """
    Sigmoid interpolation weight.

    Args:
        t: Normalized time in [0, 1].
        k: Steepness parameter (higher = sharper transition).
        center: Center point of the transition.

    Returns:
        Alpha weight in [0, 1] representing night influence.
    """
    return 1.0 / (1.0 + math.exp(-k * (t - center)))


def interpolate_prompt_embeddings(
    pipe,
    prompt_day: str,
    prompt_night: str,
    num_frames: int = 16,
    k: float = 6.0,
    center: float = 0.5,
) -> torch.Tensor:
    """
    Generate per-frame text embeddings by sigmoid-interpolating
    between day and night prompt embeddings.

    Args:
        pipe: AnimateDiffPipeline (or any pipeline with tokenizer + text_encoder).
        prompt_day: Daytime text prompt.
        prompt_night: Nighttime text prompt.
        num_frames: Number of video frames (N).
        k: Sigmoid steepness.
        center: Sigmoid center point.

    Returns:
        Tensor of shape [num_frames, 77, 768] containing per-frame embeddings.
    """
    device = pipe.device if hasattr(pipe, "device") else "cuda"

    with torch.no_grad():
        # Tokenize both prompts
        tok_day = pipe.tokenizer(
            prompt_day,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        tok_night = pipe.tokenizer(
            prompt_night,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )

        # Encode to CLIP embeddings
        emb_day = pipe.text_encoder(
            tok_day.input_ids.to(device)
        )[0]  # [1, 77, 768]
        emb_night = pipe.text_encoder(
            tok_night.input_ids.to(device)
        )[0]  # [1, 77, 768]

    # Compute sigmoid alphas for each frame
    alphas = [
        sigmoid_schedule(i / (num_frames - 1), k=k, center=center)
        for i in range(num_frames)
    ]

    # Interpolate embeddings per frame
    embeddings = [
        (1.0 - alpha) * emb_day + alpha * emb_night
        for alpha in alphas
    ]

    return torch.stack(embeddings).squeeze(1)  # [num_frames, 77, 768]


def get_negative_embedding(pipe, negative_prompt: str) -> torch.Tensor:
    """
    Encode a negative prompt into a single embedding tensor.

    Args:
        pipe: Diffusion pipeline with tokenizer + text_encoder.
        negative_prompt: Negative prompt text.

    Returns:
        Tensor of shape [1, 77, 768].
    """
    device = pipe.device if hasattr(pipe, "device") else "cuda"
    with torch.no_grad():
        tok = pipe.tokenizer(
            negative_prompt,
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        emb = pipe.text_encoder(tok.input_ids.to(device))[0]
    return emb
