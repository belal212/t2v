#!/usr/bin/env python3
"""
VAE Utilities — Encode/Decode Frames to/from Latent Space

Implements Section 6 of PLAN/05_architecture_deep_dive.md.

Key facts:
  - Encoder ℰ: Frame [3, 512, 512] → Latent [4, 64, 64]  (8× spatial compression)
  - Decoder 𝒟: Latent [4, 64, 64] → Frame [3, 512, 512]
  - Each frame processed independently
  - Latent diffusion runs in 64×64 space (16× cheaper than pixel)

Usage:
    from modeling.vae_utils import VAEProcessor
    proc = VAEProcessor(pipe.vae)
    latents = proc.encode_frames(frames)      # [N, 4, 64, 64]
    frames = proc.decode_latents(latents)     # [N, 3, 512, 512]
"""

import torch
import torch.nn as nn


class VAEProcessor:
    """
    Wrapper around diffusers VAE for explicit encode/decode.
    """

    def __init__(self, vae, scaling_factor: float = None):
        self.vae = vae
        self.scaling_factor = scaling_factor or getattr(vae.config, "scaling_factor", 0.18215)

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Encode pixel frames to latent space.

        Args:
            frames: Pixel frames [N, 3, H, W] in range [-1, 1] or [0, 1].

        Returns:
            Latents [N, 4, H/8, W/8].
        """
        # Ensure correct range for VAE
        if frames.max() <= 1.0 and frames.min() >= 0.0:
            frames = frames * 2.0 - 1.0  # [0,1] → [-1,1]

        # Encode in batches if needed
        latents = self.vae.encode(frames).latent_dist.sample()
        latents = latents * self.scaling_factor
        return latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to pixel frames.

        Args:
            latents: Latents [N, 4, H/8, W/8].

        Returns:
            Pixel frames [N, 3, H, W] in range [0, 1].
        """
        latents = latents / self.scaling_factor
        frames = self.vae.decode(latents).sample
        frames = (frames / 2.0 + 0.5).clamp(0, 1)  # [-1,1] → [0,1]
        return frames

    def encode_single(self, frame: torch.Tensor) -> torch.Tensor:
        """Encode a single frame [3, H, W] → [4, H/8, W/8]."""
        return self.encode_frames(frame.unsqueeze(0)).squeeze(0)

    def decode_single(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode a single latent [4, H/8, W/8] → [3, H, W]."""
        return self.decode_latents(latent.unsqueeze(0)).squeeze(0)

    @property
    def spatial_compression(self) -> int:
        """VAE spatial compression factor (8× for SD 1.5)."""
        return 8

    @property
    def latent_channels(self) -> int:
        """Number of latent channels (4 for SD 1.5)."""
        return 4

    def compute_cost_ratio(self) -> float:
        """
        Compute the computational cost ratio of latent vs pixel space.
        For SD 1.5: (512×512) / (64×64) = 64, but VAE adds overhead.
        Actual diffusion cost: ~16× cheaper in latent space.
        """
        return (512 * 512) / (64 * 64)  # = 64


def create_vae_processor(pipe) -> VAEProcessor:
    """Factory from a diffusers pipeline."""
    return VAEProcessor(pipe.vae)
