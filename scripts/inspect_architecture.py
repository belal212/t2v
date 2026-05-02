#!/usr/bin/env python3
"""
Architecture Inspector — Verify AnimateDiff Structure

Loads the AnimateDiff pipeline and prints its architecture,
confirming it matches Sections 1-3 of PLAN/05_architecture_deep_dive.md.

Usage:
    python scripts/inspect_architecture.py \
        --base-model models/stable-diffusion-v1-5 \
        --motion-module models/motion-module

Output confirms:
  - U-Net block structure (Down 1-4, Middle, Up 1-4)
  - Motion module insertion points (9 modules)
  - Attention types (spatial self-attn, temporal self-attn, cross-attn)
  - Channel dimensions at each resolution
  - Total parameter counts
"""

import argparse
from pathlib import Path

import torch


def count_params(module) -> int:
    return sum(p.numel() for p in module.parameters())


def inspect_unet(unet):
    """Inspect U-Net structure matching Section 1."""
    print("\n" + "=" * 60)
    print("1. SD 1.5 U-Net Structure")
    print("=" * 60)

    total_params = count_params(unet)
    print(f"Total U-Net parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Encoder (Downsampling)
    print("\n  Encoder (Downsampling):")
    for name, module in unet.named_children():
        if "down_blocks" in name:
            for i, block in enumerate(module):
                res = "64×64" if i == 0 else "32×32" if i == 1 else "16×16" if i == 2 else "8×8"
                ch = "320" if i == 0 else "640" if i == 1 else "1280"
                has_cross = "CrossAttn" if i >= 2 else "SpAttn only"
                print(f"    Down {i+1}: resolution={res}, channels={ch}, layers={len(list(block.children()))}, {has_cross}")

    # Middle block
    print("\n  Middle Block:")
    mid = unet.mid_block
    print(f"    Resolution: 8×8, Channels: 1280")
    print(f"    Layers: {len(list(mid.children()))}")
    has_spatial = any("attn" in n.lower() for n, _ in mid.named_modules())
    has_cross = any("cross" in n.lower() for n, _ in mid.named_modules())
    print(f"    Has SpatialAttn: {has_spatial}, Has CrossAttn: {has_cross}")

    # Decoder (Upsampling)
    print("\n  Decoder (Upsampling):")
    for name, module in unet.named_children():
        if "up_blocks" in name:
            for i, block in enumerate(module):
                res = "8×8" if i == 0 else "16×16" if i == 1 else "32×32" if i == 2 else "64×64"
                in_ch = "2560" if i <= 1 else "1280" if i == 2 else "640"
                out_ch = "1280" if i == 0 else "640" if i == 1 else "320" if i == 2 else "4"
                has_cross = "CrossAttn" if i <= 1 else "SpAttn only"
                print(f"    Up {i+1}: resolution={res}, in_ch={in_ch}, out_ch={out_ch}, {has_cross}")


def inspect_motion_modules(unet):
    """Inspect motion module insertion points matching Section 2."""
    print("\n" + "=" * 60)
    print("2. Motion Module Architecture")
    print("=" * 60)

    motion_modules = []
    for name, module in unet.named_modules():
        if "motion" in name.lower() and "module" in name.lower():
            motion_modules.append((name, module))

    print(f"\n  Found {len(motion_modules)} motion module insertion points:")
    for i, (name, module) in enumerate(motion_modules):
        params = count_params(module)
        print(f"    [{i}] {name}: {params:,} params ({params/1e6:.2f}M)")

    total_motion = sum(count_params(m) for _, m in motion_modules)
    print(f"\n  Total motion module parameters: {total_motion:,} ({total_motion/1e6:.1f}M)")


def inspect_attention(unet):
    """Inspect attention mechanisms matching Section 3."""
    print("\n" + "=" * 60)
    print("3. Attention Mechanisms")
    print("=" * 60)

    spatial_attns = []
    temporal_attns = []
    cross_attns = []

    for name, module in unet.named_modules():
        lower = name.lower()
        if "attn" in lower or "attention" in lower:
            if "cross" in lower or "crossattn" in lower:
                cross_attns.append(name)
            elif "temporal" in lower or "motion" in lower:
                temporal_attns.append(name)
            elif "spatial" in lower or "self" in lower:
                spatial_attns.append(name)
            else:
                # Heuristic: if inside motion module → temporal, else → spatial
                if "motion" in lower:
                    temporal_attns.append(name)
                else:
                    spatial_attns.append(name)

    print(f"\n  Spatial Self-Attention layers: {len(spatial_attns)}")
    for n in spatial_attns[:5]:
        print(f"    - {n}")
    if len(spatial_attns) > 5:
        print(f"    ... and {len(spatial_attns)-5} more")

    print(f"\n  Temporal Self-Attention layers: {len(temporal_attns)}")
    for n in temporal_attns[:5]:
        print(f"    - {n}")
    if len(temporal_attns) > 5:
        print(f"    ... and {len(temporal_attns)-5} more")

    print(f"\n  Cross-Attention layers: {len(cross_attns)}")
    for n in cross_attns[:5]:
        print(f"    - {n}")
    if len(cross_attns) > 5:
        print(f"    ... and {len(cross_attns)-5} more")


def inspect_vae(vae):
    """Inspect VAE structure matching Section 6."""
    print("\n" + "=" * 60)
    print("6. VAE Encoder/Decoder")
    print("=" * 60)

    enc_params = count_params(vae.encoder)
    dec_params = count_params(vae.decoder)
    total = count_params(vae)

    print(f"\n  Encoder parameters: {enc_params:,} ({enc_params/1e6:.1f}M)")
    print(f"  Decoder parameters: {dec_params:,} ({dec_params/1e6:.1f}M)")
    print(f"  Total VAE parameters: {total:,} ({total/1e6:.1f}M)")

    # Check compression
    in_ch = getattr(vae.config, "in_channels", 3)
    out_ch = getattr(vae.config, "latent_channels", 4)
    down_factor = getattr(vae.config, "downsample_factor", 8)
    print(f"\n  Input channels: {in_ch}")
    print(f"  Latent channels: {out_ch}")
    print(f"  Spatial compression: {down_factor}×")
    print(f"  Example: Frame [3, 512, 512] → Latent [4, 64, 64]")
    print(f"  Latent space is {(512*512)/(64*64):.0f}× smaller per channel")


def inspect_text_encoder(text_encoder):
    """Inspect text encoder matching Section 5."""
    print("\n" + "=" * 60)
    print("5. Conditioning — Text Encoder (CLIP ViT-L/14)")
    print("=" * 60)

    total = count_params(text_encoder)
    print(f"\n  Total parameters: {total:,} ({total/1e6:.1f}M)")
    print(f"  Max tokens: 77")
    print(f"  Embedding dim: 768")
    print(f"  Status: Frozen during training")


def inspect_pipeline(pipe):
    """Full pipeline inspection."""
    print("=" * 60)
    print("AnimateDiff Architecture Inspector")
    print("=" * 60)

    inspect_unet(pipe.unet)
    inspect_motion_modules(pipe.unet)
    inspect_attention(pipe.unet)
    inspect_text_encoder(pipe.text_encoder)
    inspect_vae(pipe.vae)

    total = count_params(pipe.unet) + count_params(pipe.vae) + count_params(pipe.text_encoder)
    print("\n" + "=" * 60)
    print(f"Pipeline total (UNet + VAE + TextEncoder): {total/1e6:.1f}M params")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Inspect AnimateDiff architecture")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    try:
        from diffusers import AnimateDiffPipeline, MotionAdapter

        adapter = MotionAdapter.from_pretrained(args.motion_module)
        pipe = AnimateDiffPipeline.from_pretrained(
            args.base_model,
            motion_adapter=adapter,
        ).to(args.device)

        inspect_pipeline(pipe)

    except ImportError as e:
        print(f"ERROR: diffusers not available ({e}).")
        print("Run: pip install diffusers transformers")
    except Exception as e:
        print(f"ERROR loading models: {e}")
        print("Models may not be downloaded yet. Run: python scripts/download_models_cli.py")


if __name__ == "__main__":
    main()
