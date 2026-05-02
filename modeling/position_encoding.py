#!/usr/bin/env python3
"""
Position Encodings — Sinusoidal for Timestep and Frame Index

Implements Section 4 of PLAN/05_architecture_deep_dive.md.

Key equations:
  PE(pos, 2i)   = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

Used for:
  - Diffusion timestep t ∈ {0, ..., T-1}
  - Frame index i ∈ {0, ..., N-1} (added to temporal attention K, V)

Usage:
    from modeling.position_encoding import SinusoidalEncoding
    time_pe = SinusoidalEncoding(max_len=1000, dim=320)
    emb = time_pe(torch.tensor([100, 500]))  # [2, 320]
"""

import math

import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """
    Sinusoidal position encoding.

    Precomputes PE matrix of shape [max_len, dim].
    Supports both timestep and frame index encoding.
    """

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.max_len = max_len
        self.dim = dim

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [max_len, 1]

        # div_term = 1 / 10000^(2i/d)
        div_term = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / dim)
        )  # [dim/2]

        # sin on even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # cos on odd indices
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)  # [max_len, dim]

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            indices: Integer indices [batch] or [batch, ...].

        Returns:
            Embeddings of shape [batch, dim] or [*indices.shape, dim].
        """
        indices = indices.long().clamp(0, self.max_len - 1)
        return self.pe[indices]

    def __repr__(self):
        return f"SinusoidalEncoding(max_len={self.max_len}, dim={self.dim})"


class TimestepEncoding(SinusoidalEncoding):
    """Sinusoidal encoding for diffusion timesteps t ∈ [0, T-1]."""

    def __init__(self, num_timesteps: int = 1000, dim: int = 320):
        super().__init__(max_len=num_timesteps, dim=dim)


class FramePositionEncoding(SinusoidalEncoding):
    """
    Sinusoidal encoding for frame indices i ∈ [0, N-1].

    Added to temporal attention K, V in the motion module.
    """

    def __init__(self, num_frames: int = 16, dim: int = 320):
        super().__init__(max_len=num_frames, dim=dim)


def get_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Functional API: compute sinusoidal timestep embeddings.

    Args:
        timesteps: Timestep indices [batch].
        dim: Embedding dimension.

    Returns:
        Embeddings [batch, dim].
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def get_frame_position_embedding(frame_indices: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Functional API: compute sinusoidal frame position embeddings.

    Args:
        frame_indices: Frame indices [N].
        dim: Embedding dimension.

    Returns:
        Embeddings [N, dim].
    """
    return get_timestep_embedding(frame_indices, dim)
