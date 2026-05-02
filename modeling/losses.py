#!/usr/bin/env python3
"""
Loss Functions — Diffusion Loss Variants + Temporal Losses

Implements all loss formulations from PLAN/06_loss_functions.md:
  1. ε-prediction loss (AnimateDiff default)
  2. x₀-prediction loss
  3. v-prediction loss
  4. Temporal smoothness loss (Enhancement B)
  5. Perceptual LPIPS temporal loss (Enhancement B)

Key equations:
  L_ε     = ‖ε - ε_θ(x_t, t)‖²
  L_x0    = ‖x_0 - x_θ(x_t, t)‖²
  L_v     = ‖v - v_θ(x_t, t)‖²
  L_smooth = (1/(N-1)) Σ ‖z^{i+1} - z^i‖²
  L_LPIPS  = (1/(N-1)) Σ LPIPS(D(z^{i+1}), D(z^i))

Usage:
    from modeling.losses import EpsilonLoss, TemporalSmoothnessLoss
    loss_fn = EpsilonLoss()
    loss = loss_fn(noise_pred, noise_target)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EpsilonLoss(nn.Module):
    """
    Standard noise-prediction loss (AnimateDiff default).

    L_ε = 𝔼_{t, x_0, ε} [ ‖ε - ε_θ(x_t, t)‖² ]
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, noise_pred: torch.Tensor, noise_target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noise_pred: Model output ε_θ(x_t, t).
            noise_target: Ground-truth noise ε.

        Returns:
            Scalar MSE loss.
        """
        return F.mse_loss(noise_pred, noise_target, reduction=self.reduction)


class X0Loss(nn.Module):
    """
    Image-prediction loss (direct x_0 reconstruction).

    L_x0 = 𝔼_{t, x_0, ε} [ ‖x_0 - x_θ(x_t, t)‖² ]

    Note: Less stable at high noise levels since x_0 prediction is unbounded.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, x0_pred: torch.Tensor, x0_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(x0_pred, x0_target, reduction=self.reduction)


class VPredictionLoss(nn.Module):
    r"""
    Velocity-prediction loss.

    v = α_t · ε - σ_t · x_0,  where α_t = √(ᾱ_t), σ_t = √(1-ᾱ_t)

    L_v = 𝔼_{t, x_0, ε} [ ‖v - v_θ(x_t, t)‖² ]

    More stable for fast samplers and high-resolution video.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, v_pred: torch.Tensor, v_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(v_pred, v_target, reduction=self.reduction)

    @staticmethod
    def compute_target(
        x0: torch.Tensor,
        noise: torch.Tensor,
        alpha_bar_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute velocity target from clean data and noise.

        Args:
            x0: Clean data [B, C, H, W].
            noise: Sampled noise ε.
            alpha_bar_t: Cumulative alpha [B].

        Returns:
            Velocity target v.
        """
        alpha_t = torch.sqrt(alpha_bar_t).view(-1, 1, 1, 1)
        sigma_t = torch.sqrt(1.0 - alpha_bar_t).view(-1, 1, 1, 1)
        return alpha_t * noise - sigma_t * x0


class TemporalSmoothnessLoss(nn.Module):
    r"""
    Temporal smoothness loss for Enhancement B.

    Encourages consecutive frame latents to be close in L2 space.

    L_smooth = (1/(N-1)) Σ_{i=1}^{N-1} ‖z^{i+1} - z^i‖²

    where z^i is the latent of frame i.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: Frame latents [N, C, H, W] or [B, N, C, H, W].

        Returns:
            Scalar smoothness loss.
        """
        if latents.dim() == 5:
            # Batch mode: compute per-batch then average
            losses = [self._smoothness_loss_single(latents[b]) for b in range(latents.shape[0])]
            loss = torch.stack(losses).mean()
        else:
            loss = self._smoothness_loss_single(latents)
        return loss

    def _smoothness_loss_single(self, latents: torch.Tensor) -> torch.Tensor:
        """Compute smoothness for a single video's latents [N, C, H, W]."""
        diffs = latents[1:] - latents[:-1]  # [N-1, C, H, W]
        return torch.mean(diffs ** 2)


class LPIPSTemporalLoss(nn.Module):
    r"""
    Perceptual temporal loss using LPIPS (Enhancement B).

    L_LPIPS = (1/(N-1)) Σ_{i=1}^{N-1} LPIPS(D(z^{i+1}), D(z^i))

    where D is the VAE decoder. Operates in pixel space for perceptual quality.
    """

    def __init__(self, net: str = "alex", device: str = "cuda"):
        super().__init__()
        try:
            import lpips
            self.lpips_fn = lpips.LPIPS(net=net).to(device)
            self.available = True
        except ImportError:
            self.lpips_fn = None
            self.available = False

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frames: Pixel frames [N, 3, H, W] in range [-1, 1] or [0, 1].

        Returns:
            Scalar LPIPS temporal loss.
        """
        if not self.available:
            raise RuntimeError("lpips is not installed. Run: pip install lpips")

        if frames.dim() == 4:
            losses = []
            for i in range(len(frames) - 1):
                dist = self.lpips_fn(
                    frames[i].unsqueeze(0),
                    frames[i + 1].unsqueeze(0),
                )
                losses.append(dist)
            return torch.stack(losses).mean()
        else:
            raise ValueError(f"Expected frames of shape [N, 3, H, W], got {frames.shape}")


class CombinedLoss(nn.Module):
    r"""
    Combined training objective for Enhancement B fine-tuning.

    L_total = L_ε + λ_s · L_smooth + λ_p · L_LPIPS

    with λ_s = 0.01, λ_p = 0.001 by default.
    """

    def __init__(
        self,
        lambda_smooth: float = 0.01,
        lambda_lpips: float = 0.001,
        device: str = "cuda",
    ):
        super().__init__()
        self.epsilon_loss = EpsilonLoss()
        self.smoothness_loss = TemporalSmoothnessLoss()
        self.lpips_loss = LPIPSTemporalLoss(device=device)
        self.lambda_smooth = lambda_smooth
        self.lambda_lpips = lambda_lpips

    def forward(
        self,
        noise_pred: torch.Tensor,
        noise_target: torch.Tensor,
        latents: torch.Tensor = None,
        decoded_frames: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            noise_pred: Model noise prediction.
            noise_target: Ground-truth noise.
            latents: Optional frame latents for smoothness loss [N, C, H, W].
            decoded_frames: Optional decoded frames for LPIPS loss [N, 3, H, W].

        Returns:
            Dict with individual and total losses.
        """
        loss_eps = self.epsilon_loss(noise_pred, noise_target)
        total = loss_eps

        result = {
            "loss": total,
            "loss_epsilon": loss_eps,
        }

        if latents is not None:
            loss_smooth = self.smoothness_loss(latents)
            total = total + self.lambda_smooth * loss_smooth
            result["loss_smooth"] = loss_smooth
            result["loss"] = total

        if decoded_frames is not None and self.lpips_loss.available:
            loss_lpips = self.lpips_loss(decoded_frames)
            total = total + self.lambda_lpips * loss_lpips
            result["loss_lpips"] = loss_lpips
            result["loss"] = total

        return result


def aninate_diff_video_loss(
    noise_pred: torch.Tensor,
    noise_target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    r"""
    AnimateDiff video extension of ε-prediction loss.

    L_AD = 𝔼_{t, {ε^i}, {x_0^i}, c} [ Σ_{i=1}^N ‖ε^i - ε_θ({x_t^i}, t, c, {pe_i})‖² ]

    This is simply MSE summed (or meaned) over all frames.

    Args:
        noise_pred: [B, N, C, H, W] or [B*N, C, H, W].
        noise_target: Same shape as noise_pred.

    Returns:
        Scalar loss.
    """
    return F.mse_loss(noise_pred, noise_target, reduction=reduction)
