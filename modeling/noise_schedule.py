#!/usr/bin/env python3
"""
Noise Schedules — Diffusion Model Beta/Bar-Alpha Computations

Implements the linear and cosine noise schedules from Section 4 of
PLAN/04_mathematical_background.md.

Key equations:
  Linear:   β_t = β_1 + (t-1)/(T-1) * (β_T - β_1)
  Cosine:   ᾱ_t = f(t)/f(0),  f(t) = cos²( (t/T + s)/(1+s) * π/2 )

Usage:
    from modeling.noise_schedule import LinearSchedule, CosineSchedule
    sched = LinearSchedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02)
    alpha_bar = sched.alpha_bar(torch.tensor([250, 500, 750]))
"""

import math
from typing import Union

import torch
import torch.nn as nn


class NoiseSchedule(nn.Module):
    """Base class for diffusion noise schedules."""

    def __init__(self, num_timesteps: int = 1000):
        super().__init__()
        self.num_timesteps = num_timesteps

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β_t at timestep t (1-indexed in math, 0-indexed in code)."""
        raise NotImplementedError

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """α_t = 1 - β_t."""
        return 1.0 - self.beta(t)

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ᾱ_t = ∏_{s=1}^t α_s — cumulative product of alphas."""
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(T={self.num_timesteps})"


class LinearSchedule(NoiseSchedule):
    """
    Linear beta schedule (SD 1.5 default).

    β_t = β_start + (t / T) * (β_end - β_start)
    """

    def __init__(self, num_timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        super().__init__(num_timesteps)
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Precompute all schedules for efficiency
        self.register_buffer("betas", self._compute_betas())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))

    def _compute_betas(self) -> torch.Tensor:
        """Compute β_t for all t = 0 ... T-1."""
        t = torch.arange(self.num_timesteps, dtype=torch.float32)
        beta = self.beta_start + (t / (self.num_timesteps - 1)) * (self.beta_end - self.beta_start)
        return beta

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """β_t — clamped to valid range."""
        t = t.long().clamp(0, self.num_timesteps - 1)
        return self.betas[t]

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long().clamp(0, self.num_timesteps - 1)
        return self.alpha_bars[t]

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward noising: x_t = √(ᾱ_t) * x_0 + √(1-ᾱ_t) * ε

        Args:
            x0: Clean data [..., C, H, W].
            t: Timestep indices [...].
            noise: Optional pre-sampled noise (same shape as x0).

        Returns:
            (x_t, noise) — noisy data and the noise that was added.
        """
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def __repr__(self):
        return f"LinearSchedule(T={self.num_timesteps}, β=[{self.beta_start}, {self.beta_end}])"


class CosineSchedule(NoiseSchedule):
    """
    Cosine alpha-bar schedule (improved over linear).

    ᾱ_t = f(t) / f(0),  f(t) = cos²( (t/T + s) / (1+s) * π/2 )
    """

    def __init__(self, num_timesteps: int = 1000, s: float = 0.008):
        super().__init__(num_timesteps)
        self.s = s

        self.register_buffer("alpha_bars", self._compute_alpha_bars())
        self.register_buffer("betas", self._compute_betas_from_alpha_bars())
        self.register_buffer("alphas", 1.0 - self.betas)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(self.alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - self.alpha_bars))

    def _f(self, t: torch.Tensor) -> torch.Tensor:
        """f(t) = cos²( (t/T + s) / (1+s) * π/2 )."""
        return torch.cos(
            ((t / self.num_timesteps) + self.s) / (1.0 + self.s) * (math.pi / 2.0)
        ) ** 2

    def _compute_alpha_bars(self) -> torch.Tensor:
        t = torch.arange(self.num_timesteps + 1, dtype=torch.float32)
        alpha_bars = self._f(t) / self._f(torch.tensor(0.0))
        # Clip to avoid numerical issues at t=0
        alpha_bars = torch.clip(alpha_bars, min=1e-6, max=1.0)
        return alpha_bars[1:]  # t = 1 ... T

    def _compute_betas_from_alpha_bars(self) -> torch.Tensor:
        """Recover β_t from ᾱ_t: β_t = 1 - ᾱ_t / ᾱ_{t-1}."""
        alpha_bars_prev = torch.cat([
            torch.tensor([1.0]),
            self.alpha_bars[:-1]
        ])
        betas = 1.0 - self.alpha_bars / alpha_bars_prev
        return torch.clip(betas, min=0.0, max=0.999)

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long().clamp(0, self.num_timesteps - 1)
        return self.betas[t]

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        t = t.long().clamp(0, self.num_timesteps - 1)
        return self.alpha_bars[t]

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward noising using cosine schedule."""
        if noise is None:
            noise = torch.randn_like(x0)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)

        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
        return xt, noise

    def __repr__(self):
        return f"CosineSchedule(T={self.num_timesteps}, s={self.s})"


def get_schedule(name: str, num_timesteps: int = 1000, **kwargs) -> NoiseSchedule:
    """Factory for noise schedules."""
    if name.lower() == "linear":
        return LinearSchedule(num_timesteps, **kwargs)
    elif name.lower() == "cosine":
        return CosineSchedule(num_timesteps, **kwargs)
    else:
        raise ValueError(f"Unknown schedule: {name}. Choose 'linear' or 'cosine'.")
