#!/usr/bin/env python3
"""
Enhancement B — Temporal Smoothness Fine-Tuning

Fine-tunes the AnimateDiff motion module with an additional
latent L2 smoothness loss to reduce flickering in photometric transitions.

Usage:
    python training/train_smoothness.py \
        --video-dir data/day-to-night-clips \
        --output-dir outputs/trained_motion \
        --steps 1000 \
        --lr 1e-5
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class DayToNightVideoDataset(Dataset):
    """
    Simple dataset that loads video clips as frame tensors.
    Expects directory of .mp4 or .gif files.
    """

    def __init__(self, video_dir: str, num_frames: int = 16, resolution: int = 512):
        self.video_dir = Path(video_dir)
        self.video_paths = sorted(list(self.video_dir.glob("*.mp4")) +
                                   list(self.video_dir.glob("*.gif")))
        self.num_frames = num_frames
        self.resolution = resolution
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        import cv2
        path = str(self.video_paths[idx])
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            pil = Image.fromarray(frame)
            frames.append(self.transform(pil))
        cap.release()

        if len(frames) == 0:
            # Fallback for GIFs
            from PIL import Image
            img = Image.open(path)
            try:
                while True:
                    frames.append(self.transform(img.convert("RGB")))
                    img.seek(img.tell() + 1)
            except EOFError:
                pass

        # Sample or pad to num_frames
        if len(frames) >= self.num_frames:
            indices = torch.linspace(0, len(frames) - 1, self.num_frames).long()
            frames = [frames[i] for i in indices]
        else:
            while len(frames) < self.num_frames:
                frames.append(frames[-1])

        return torch.stack(frames)  # [N, 3, H, W]


def compute_smoothness_loss(latents: torch.Tensor) -> torch.Tensor:
    """
    L2 smoothness loss between consecutive frame latents.

    Args:
        latents: [N, C, H, W] latent tensor.

    Returns:
        Scalar loss.
    """
    diffs = latents[1:] - latents[:-1]
    return torch.mean(diffs ** 2)


def train(
    video_dir: str,
    base_model: str,
    motion_module: str,
    output_dir: str,
    num_frames: int = 16,
    resolution: int = 512,
    steps: int = 1000,
    lr: float = 1e-5,
    lambda_smooth: float = 0.01,
    batch_size: int = 1,
    device: str = "cuda",
):
    """
    Fine-tune motion module with temporal smoothness loss.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load pipeline
    adapter = MotionAdapter.from_pretrained(motion_module, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Freeze spatial UNet
    for param in pipe.unet.parameters():
        param.requires_grad = False

    # Only train motion module parameters
    motion_params = list(adapter.parameters())
    optimizer = AdamW(motion_params, lr=lr)

    # Dataset
    dataset = DayToNightVideoDataset(video_dir, num_frames, resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    global_step = 0
    epoch = 0
    losses = []

    print("=" * 60)
    print("Motion Module Fine-Tuning — Temporal Smoothness")
    print("=" * 60)
    print(f"Dataset: {len(dataset)} clips")
    print(f"Steps: {steps}, LR: {lr}, λ_smooth: {lambda_smooth}")
    print(f"Trainable params: {sum(p.numel() for p in motion_params):,}")
    print("=" * 60)

    while global_step < steps:
        epoch += 1
        for batch in dataloader:
            if global_step >= steps:
                break

            # batch: [B, N, 3, H, W]
            frames = batch.to(device).float()
            b, n, c, h, w = frames.shape

            # Encode frames to latents via VAE
            with torch.no_grad():
                # VAE expects [B, C, H, W]
                flat_frames = frames.view(b * n, c, h, w)
                latents = pipe.vae.encode(flat_frames).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                latents = latents.view(b, n, *latents.shape[1:])  # [B, N, 4, H/8, W/8]

            # Sample random timestep
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (b,), device=device
            ).long()

            # Add noise
            noise = torch.randn_like(latents)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = pipe.unet(
                noisy_latents.view(b * n, *noisy_latents.shape[2:]),
                timesteps.repeat(n),
            ).sample
            noise_pred = noise_pred.view(b, n, *noise_pred.shape[1:])

            # Standard diffusion MSE loss
            loss_diffusion = F.mse_loss(noise_pred, noise)

            # Temporal smoothness loss (on latents)
            loss_smooth = compute_smoothness_loss(latents[0])  # batch=1

            # Combined loss
            loss = loss_diffusion + lambda_smooth * loss_smooth

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            losses.append({
                "step": global_step,
                "loss": loss.item(),
                "loss_diffusion": loss_diffusion.item(),
                "loss_smooth": loss_smooth.item(),
            })

            if global_step % 100 == 0:
                print(f"Step {global_step}/{steps} | "
                      f"loss={loss.item():.4f} | "
                      f"diff={loss_diffusion.item():.4f} | "
                      f"smooth={loss_smooth.item():.6f}")

    # Save fine-tuned motion module
    adapter.save_pretrained(output_path / "motion-module-smooth")

    # Save training log
    with open(output_path / "training_log.json", "w") as f:
        json.dump(losses, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Training complete! Saved to: {output_path / 'motion-module-smooth'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune motion module for smoothness")
    parser.add_argument("--video-dir", required=True, help="Directory of day-to-night video clips")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--output-dir", default="outputs/trained_motion")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lambda-smooth", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train(
        video_dir=args.video_dir,
        base_model=args.base_model,
        motion_module=args.motion_module,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
        resolution=args.resolution,
        steps=args.steps,
        lr=args.lr,
        lambda_smooth=args.lambda_smooth,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
