# Urban Metamorphosis: Day-to-Night City Transitions via AnimateDiff

**AIE418 Final Project** — Text-to-Video Generation with Diffusion Models

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Diffusers](https://img.shields.io/badge/Diffusers-0.25+-yellow.svg)](https://huggingface.co/docs/diffusers/)

---

## Overview

This project generates temporally coherent day-to-night city transition videos using **AnimateDiff (SD 1.5)** with three custom enhancements and a **Qwen-TTS** audio bonus.

### Three Core Enhancements

| Enhancement | Problem Solved | Method |
|---|---|---|
| **A** — Time-Dependent Prompt Conditioning | Semantic drift (same text embedding for all frames) | Sigmoid interpolation between day/night CLIP embeddings per frame |
| **B** — Temporal Smoothness | Flickering during photometric transitions | Latent L2 smoothness loss + inference-time frame blending (β=0.3) |
| **C** — Temporal Attention Bias | Uniform attention across timesteps | Learned bias matrix scaled by γ(t) = 1 − t/T |

### Theme
**Urban Metamorphosis** — cinematic time-lapse transitions of cityscapes from bright daylight through golden hour, dusk, twilight, and into night.

---

## Quick Start

### 1. Environment Setup

```bash
# Run automated setup (creates conda env + installs dependencies)
bash scripts/setup_env.sh
conda activate urban-metamorphosis

# Or install manually
pip install -r requirements.txt
```

### 2. Download Models

```bash
# Using huggingface-cli (recommended)
python scripts/download_models_cli.py --token YOUR_HF_TOKEN

# Or via Python API fallback
python scripts/download_models.py
```

Downloads to `./models/`:
- `stable-diffusion-v1-5` — SD 1.5 base model
- `motion-module` — AnimateDiff motion adapter
- `clip-vit-large-patch14` — CLIP for evaluation

### 3. Verify Baseline

```bash
# Generate a single test video
python scripts/test_baseline.py \
    --output outputs/test_baseline.gif \
    --cpu-offload  # if VRAM < 12GB
```

### 4. Run Enhanced Inference

```bash
python inference/inference_enhanced.py \
    --config C4_A_B_C \
    --prompt "cinematic time-lapse of a city skyline transitioning from day to night" \
    --prompt-day "bright sunny day in a city, clear blue sky" \
    --prompt-night "dark night city skyline, neon lights, starry sky" \
    --output outputs/enhanced/test.gif
```

Configs: `C0_baseline`, `C1_A_only`, `C2_B_only`, `C3_A_plus_B`, `C4_A_B_C`

---

## Project Structure

```
project-urban-metamorphosis/
├── config.yaml                      # Central configuration
├── requirements.txt                 # Python dependencies
├── data/
│   └── eval_prompts.json            # 50 evaluation prompts with day/night pairs
├── modeling/
│   ├── prompt_interpolation.py      # Enhancement A: sigmoid prompt interpolation
│   └── attention_bias.py            # Enhancement C: temporal attention bias
├── inference/
│   └── inference_enhanced.py        # A+B+C combined inference pipeline
├── training/
│   └── train_smoothness.py          # Enhancement B: motion module fine-tuning
├── eval/
│   ├── weakness_analysis.py         # CLIP-SIM curves + flow warping error
│   ├── eval.py                      # Full evaluation pipeline (all metrics)
│   └── run_ablation.py              # 750-video ablation study runner
├── tts/
│   └── tts_pipeline.py              # Qwen-TTS narration pipeline
├── scripts/
│   ├── env_check.py                 # Environment verification
│   ├── setup_env.sh                 # Automated environment setup
│   ├── download_models.py           # Model download (Python API)
│   ├── download_models_cli.py       # Model download (huggingface-cli)
│   ├── test_baseline.py             # Baseline inference test
│   └── batch_infer.py               # Multi-GPU batch inference
├── outputs/                         # Generated videos + metrics
└── models/                          # Downloaded model weights
```

---

## Ablation Study

Generate all 750 videos (5 configs × 50 prompts × 3 seeds):

```bash
python eval/run_ablation.py \
    --prompts data/eval_prompts.json \
    --output-dir outputs/ablation
```

Then compute metrics:

```bash
python eval/eval.py \
    --ablation-dir outputs/ablation \
    --output outputs/metrics
```

### Expected Results

| Config | Enh A | Enh B | Enh C | FVD ↓ | CLIP-SIM ↑ | LPIPS (dusk) ↓ | SSIM (dusk) ↑ | Flow Warp ↓ |
|--------|-------|-------|-------|-------|-----------|---------------|---------------|-------------|
| C0 Baseline | ❌ | ❌ | ❌ | ~850 | 0.21±0.03 | 0.38±0.05 | 0.72±0.04 | 0.060 |
| C1 A-only | ✅ | ❌ | ❌ | ~820 | **0.30±0.04** | 0.36±0.05 | 0.74±0.04 | 0.055 |
| C2 B-only | ❌ | ✅ | ❌ | ~800 | 0.21±0.03 | **0.20±0.03** | **0.85±0.03** | **0.025** |
| C3 A+B | ✅ | ✅ | ❌ | ~770 | **0.29±0.04** | **0.19±0.03** | **0.86±0.03** | **0.022** |
| C4 A+B+C | ✅ | ✅ | ✅ | **~740** | **0.31±0.04** | **0.17±0.03** | **0.88±0.03** | **0.018** |

---

## Multi-GPU Inference

For 2× RTX A6000 (or any multi-GPU setup):

```bash
python scripts/batch_infer.py \
    --prompts data/eval_prompts.json \
    --output-dir outputs/baseline \
    --seeds 42 123 999
```

Distributes prompts round-robin across available GPUs using process-level isolation.

---

## Bonus: Qwen-TTS Audio

Generate narrated versions of videos:

```bash
python tts/tts_pipeline.py \
    --video outputs/enhanced/test.gif \
    --prompt "cinematic time-lapse of a city skyline transitioning from day to night" \
    --output outputs/final_with_audio.mp4
```

**Features implemented:**
1. **Multi-voice** — distinct narrator profiles for day/dusk/night phases
2. **Emotion control** — keyword parser extracts emotional tone from prompts
3. **Context-aware synthesis** — pause tokens inserted at punctuation/clause boundaries
4. **Audio-video sync** — librosa time-stretch matches narration to video duration

---

## Weakness Analysis

Diagnose baseline AnimateDiff weaknesses on any generated video:

```bash
python eval/weakness_analysis.py \
    --video outputs/baseline/prompt_0_seed42.gif \
    --prompt-day "bright sunny day in a city, clear blue sky" \
    --prompt-night "dark night city skyline, neon lights, starry sky" \
    --output outputs/metrics/prompt_0_seed42.json
```

**Metrics computed:**
- Per-frame CLIP-SIM dual curves (day vs night)
- CLIP-SIM slope analysis
- Flow Warping Error per transition
- LPIPS / SSIM (dusk window focus)

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | 1× RTX A6000 (48GB) | 2× RTX A6000 |
| CUDA | 12.1+ | 12.1+ |
| RAM | 32 GB | 64 GB |
| Storage | 50 GB | 100 GB |

For limited VRAM (~6GB), enable:
```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_slicing()
```

---

## Citation

```bibtex
@misc{urban-metamorphosis-2025,
  title={Urban Metamorphosis: Day-to-Night City Transitions via AnimateDiff},
  author={AIE418 Team},
  year={2025},
  institution={Alamein University, Faculty of Computer Science \& Engineering}
}
```

---

## License

This project is for academic purposes (AIE418 Final Project). Model weights are subject to their respective licenses (SD 1.5 — CreativeML Open RAIL-M, AnimateDiff — Apache 2.0).
