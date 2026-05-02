#!/usr/bin/env python3
"""
Diffusion Utilities — Sampling Steps, CFG, Score Function

Implements the mathematical primitives from Sections 1, 2, 7, and 8 of
PLAN/04_mathematical_background.md.

Key equations implemented:
  - Forward noising:   x_t = √(ᾱ_t) x_0 + √(1-ᾱ_t) ε
  - DDIM step:         x_{t-1} = √(ᾱ_{t-1}) x̂_0 + √(1-ᾱ_{t-1}) ε_θ
  - DDPM step:         x_{t-1} = μ_θ + σ_t z
  - CFG:               ε̂ = ε_∅ + s · (ε_c - ε_∅)
  - Score function:    s(x) ≈ -ε / σ_t

Usage:
    from modeling.diffusion_utils import ddim_step, ddim_step_with_cfg, score_from_noise
    from modeling.noise_schedule import LinearSchedule

    sched = LinearSchedule(num_timesteps=1000)
    x_t, noise = sched.forward_process(x0, t)
    x_t_minus_1 = ddim_step(sched, x_t, noise_pred, t)
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F


def score_from_noise(noise: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
    r"""
    Score function approximation:  s(x_t) ≈ -ε / σ_t

    Args:
        noise: Predicted noise ε_θ(x_t, t).
        sigma_t: Noise level σ_t = √(1 - ᾱ_t).

    Returns:
        Score vector (direction toward cleaner data).
    """
    return -noise / (sigma_t + 1e-8)


def apply_cfg(
    noise_uncond: torch.Tensor,
    noise_cond: torch.Tensor,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    r"""
    Classifier-Free Guidance (CFG).

    ε̂_θ = ε_θ(x_t, ∅) + s · (ε_θ(x_t, c) - ε_θ(x_t, ∅))

    Args:
        noise_uncond: Unconditional noise prediction ε_∅.
        noise_cond: Conditional noise prediction ε_c.
        guidance_scale: CFG scale s (1 = no guidance, 7.5 = default).

    Returns:
        Guided noise prediction.
    """
    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


def ddim_step(
    schedule,
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    t: torch.Tensor,
    eta: float = 0.0,
) -> torch.Tensor:
    r"""
    DDIM deterministic sampling step.

    x_{t-1} = √(ᾱ_{t-1}) · x̂_0 + √(1 - ᾱ_{t-1} - σ_t²) · ε_θ + σ_t · z

    where x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √(ᾱ_t)

    With eta=0, this is fully deterministic (no random noise added).
    With eta=1, this becomes DDPM-like stochastic sampling.

    Args:
        schedule: NoiseSchedule instance (LinearSchedule or CosineSchedule).
        x_t: Noisy latent at timestep t.
        noise_pred: Model prediction ε_θ(x_t, t).
        t: Current timestep (0-indexed, 0 = clean, T-1 = noisy).
        eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM-like).

    Returns:
        x_{t-1}: Denoised latent at previous timestep.
    """
    alpha_bar_t = schedule.alpha_bar(t)
    alpha_bar_prev = schedule.alpha_bar(t - 1) if (t > 0).all() else torch.ones_like(alpha_bar_t)

    # Predict x_0 from noise prediction
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    pred_x0 = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / (sqrt_alpha_bar_t + 1e-8)

    # Direction pointing to x_t
    sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
    sqrt_one_minus_alpha_bar_prev = torch.sqrt(1.0 - alpha_bar_prev)

    # Stochastic sigma
    sigma_t = eta * torch.sqrt(
        (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * (1.0 - alpha_bar_t / alpha_bar_prev)
    )

    # Noise component
    noise_direction = torch.sqrt(1.0 - alpha_bar_prev - sigma_t ** 2) * noise_pred

    # Combine
    x_t_minus_1 = sqrt_alpha_bar_prev * pred_x0 + noise_direction

    if eta > 0:
        noise = torch.randn_like(x_t)
        x_t_minus_1 = x_t_minus_1 + sigma_t * noise

    return x_t_minus_1


def ddim_step_with_cfg(
    schedule,
    x_t: torch.Tensor,
    noise_uncond: torch.Tensor,
    noise_cond: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
) -> torch.Tensor:
    """
    DDIM step with integrated Classifier-Free Guidance.

    Convenience wrapper: applies CFG then DDIM step.
    """
    noise_pred = apply_cfg(noise_uncond, noise_cond, guidance_scale)
    return ddim_step(schedule, x_t, noise_pred, t, eta)


def ddpm_step(
    schedule,
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    r"""
    DDPM stochastic sampling step.

    μ_θ = (1/√α_t) · (x_t - (β_t / √(1-ᾱ_t)) · ε_θ)
    x_{t-1} = μ_θ + √β_t · z

    Args:
        schedule: NoiseSchedule with betas and alphas.
        x_t: Noisy latent at timestep t.
        noise_pred: Model prediction ε_θ(x_t, t).
        t: Current timestep.

    Returns:
        x_{t-1}: Sampled latent at previous timestep.
    """
    beta_t = schedule.beta(t)
    alpha_t = schedule.alpha(t)
    alpha_bar_t = schedule.alpha_bar(t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    # Mean prediction
    pred_mean = (x_t - (beta_t / (sqrt_one_minus_alpha_bar_t + 1e-8)) * noise_pred) / torch.sqrt(alpha_t)

    # Variance
    if (t == 0).all():
        # At t=0, no noise added
        return pred_mean
    else:
        noise = torch.randn_like(x_t)
        variance = torch.sqrt(beta_t)
        return pred_mean + variance * noise


def predict_x0_from_noise(
    x_t: torch.Tensor,
    noise_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    r"""
    Predict clean data x_0 from noisy x_t and noise prediction.

    x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ) / √(ᾱ_t)

    Args:
        x_t: Noisy data.
        noise_pred: Predicted noise ε_θ.
        alpha_bar_t: Cumulative product ᾱ_t.

    Returns:
        Predicted clean data x̂_0.
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    return (x_t - sqrt_one_minus_alpha_bar * noise_pred) / (sqrt_alpha_bar + 1e-8)


def predict_noise_from_x0(
    x_t: torch.Tensor,
    x0_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    r"""
    Predict noise from clean data prediction (for x0-prediction models).

    ε̂ = (x_t - √(ᾱ_t) · x̂_0) / √(1-ᾱ_t)

    Args:
        x_t: Noisy data.
        x0_pred: Predicted clean data x̂_0.
        alpha_bar_t: Cumulative product ᾱ_t.

    Returns:
        Implied noise prediction.
    """
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
    return (x_t - sqrt_alpha_bar * x0_pred) / (sqrt_one_minus_alpha_bar + 1e-8)


def velocity_target(
    x0: torch.Tensor,
    noise: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    r"""
    Compute v-prediction target.

    v = α_t · ε - σ_t · x_0
      = √(ᾱ_t) · ε - √(1-ᾱ_t) · x_0

    Args:
        x0: Clean data.
        noise: Sampled noise ε.
        alpha_bar_t: Cumulative product ᾱ_t.

    Returns:
        Velocity target v.
    """
    alpha_t = torch.sqrt(alpha_bar_t)
    sigma_t = torch.sqrt(1.0 - alpha_bar_t)
    return alpha_t * noise - sigma_t * x0


def velocity_to_noise(
    v_pred: torch.Tensor,
    alpha_bar_t: torch.Tensor,
) -> torch.Tensor:
    r"""
    Convert velocity prediction to noise prediction.

    ε = α_t · v + σ_t · x_0   ... but we need x_0 first.

    Alternative:  v = α·ε - σ·x_0  and  x_t = α·x_0 + σ·ε
    Solving:  ε = α·v + σ·x_t
    """
    alpha_t = torch.sqrt(alpha_bar_t)
    sigma_t = torch.sqrt(1.0 - alpha_bar_t)
    return alpha_t * v_pred + sigma_t  # This needs x_t for full conversion; placeholder
