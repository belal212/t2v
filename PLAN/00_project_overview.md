# AIE418 Final Project — Project Blueprint

## Course Info
- **Course:** AIE418 — Selected Topics in AI 2
- **University:** Alamein University, Faculty of Computer Science & Engineering
- **Topic:** Text-to-Video Generation with Diffusion Models

## Project Title
**Urban Metamorphosis: Day-to-Night City Transitions via AnimateDiff + Time-Dependent Conditioning + Temporal Smoothness + Qwen-TTS**

## Theme
Urban Metamorphosis — generating temporally coherent day-to-night video clips of city scenes.

## Core Technical Challenge
Standard AnimateDiff applies the same static text embedding to all frames, causing:
1. **Semantic Drift** — frames at different temporal positions (noon vs dusk) receive identical text conditioning
2. **Temporal Flickering** — motion module not trained on smooth photometric transitions

## Proposed Solution (3 Enhancements)
| Enhancement | Problem Solved | Method |
|-------------|---------------|--------|
| **A:** Time-Dependent Prompt Conditioning | Semantic Drift | Sigmoid interpolation between day/night CLIP embeddings per frame |
| **B:** Temporal Smoothness | Flickering | Latent L2 smoothness loss + inference-time frame blending |
| **C:** Temporal Attention Bias | Global-local balance | Learned bias matrix scaled by denoising timestep |

## Bonus: Qwen-TTS Audio
Multi-voice, emotion-controlled, context-aware TTS narration synced to video timeline.

## Hardware
- **Lambda Labs** — 2× RTX A6000 (48GB)
- CUDA 12.1, PyTorch, diffusers, bitsandbytes, peft

## Two-Phase Timeline
| Phase | Week | Focus | Deliverables |
|-------|------|-------|-------------|
| **Phase 1: Foundation & Analysis** | Week 13 | Baselines, weakness analysis, math docs | Paper Sections 1-5, 50 baseline eval videos |
| **Phase 2: Enhancements + TTS + Paper** | Weeks 14-15 | All 3 enhancements + Qwen-TTS + ablation | Final video, full paper, code repo |

## Ablation Configurations (5 variants)
| Config | Enh A | Enh B | Enh C |
|--------|-------|-------|-------|
| Baseline | ❌ | ❌ | ❌ |
| A-only | ✅ | ❌ | ❌ |
| B-only | ❌ | ✅ | ❌ |
| A+B | ✅ | ✅ | ❌ |
| A+B+C | ✅ | ✅ | ✅ |
