# Model Selection — AnimateDiff

## Text-to-Video Backbone

### AnimateDiff v2 (SD 1.5)
| Property | Value |
|----------|-------|
| **Base Model** | Stable Diffusion 1.5 |
| **Motion Module** | `guoyww/animatediff-motion-adapter-v1-5-2` |
| **Resolution** | 512×512 |
| **Max Frames** | 16 (can extend to 24+) |
| **VRAM Required** | ~12-16 GB |
| **Text Encoder** | CLIP ViT-L/14 (77 tokens, 768-dim) |
| **Sampling** | DDIM (25-50 steps) |
| **Guidance** | CFG (scale 7.5) |

**Why AnimateDiff over alternatives:**
- Strict 3D U-Net compliance with rubric requirements
- Highly modular — motion module layers are clearly separated
- Runs cleanly on a single A6000 (48GB)
- Extensive community support, easy to modify inference loop

### Architecture Summary
```
Text Prompt
    │
    ▼
CLIP Text Encoder (Frozen) ──────┐
                                  ▼
                            ┌─────────────┐
                            │  SD 1.5     │
                            │  U-Net      │
                            │  (Spatial)  │
                            │             │
                            │ ┌─────────┐ │
                            │ │ Motion  │ │  ← Temporal self-attn ×9
                            │ │ Module  │ │    (inserted after each
                            │ │ (Train) │ │     residual block)
                            │ └─────────┘ │
                            └─────────────┘
                                  │
                                  ▼
                        ┌─────────────────┐
                        │  VAE Decoder    │
                        │  (per frame)    │
                        └─────────────────┘
                                  │
                                  ▼
                        Final Video Frames
```

## Qwen-TTS (Bonus)

| Property | Value |
|----------|-------|
| **Model** | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` |
| **Type** | Neural TTS (not basic library) |
| **Key Features** | Multi-voice support, emotion conditioning |
| **Parameters** | ~600M |
| **Hardware** | CPU or GPU (lightweight) |
| **Integration** | Post-processing after video generation |

## Why No QLoRA / Fine-Tuning of UNet
- The spatial UNet layers are frozen (standard AnimateDiff practice)
- Our enhancements operate at:
  - **Inference level** (Enh A: prompt embedding interpolation)
  - **Loss level** (Enh B: additional smoothness term)
  - **Parameter level** (Enh C: small learned bias matrix)
- Only Enhancement B (full version) requires motion module fine-tuning (~450M params, ~500 steps)

## Hardware Requirements
| Component | Requirement |
|-----------|-------------|
| GPU | 1× RTX A6000 (48GB) minimum, 2× for parallel inference |
| CUDA | 12.1+ |
| RAM | 32 GB |
| Storage | 50 GB (models + datasets + outputs) |
