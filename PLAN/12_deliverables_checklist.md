# Deliverables Checklist

## Phase 1 — Week 13: Foundation & Analysis

### Environment & Baseline
- [ ] PyTorch + diffusers + CUDA verified
- [ ] AnimateDiff cloned and working
- [ ] SD 1.5 + motion module downloaded
- [ ] Baseline inference test: 1 video generated successfully
- [ ] Multi-GPU inference script ready

### Weakness Analysis (50 prompts × 3 seeds = 150 videos)
- [ ] Per-frame CLIP-SIM dual curves plotted (day vs night prompt)
- [ ] CLIP-SIM slopes computed (should be ~0 for baseline)
- [ ] Flow Warping Error per-transition computed
- [ ] LPIPS dusk-window mean computed
- [ ] SSIM dusk-window mean computed
- [ ] Weakness 1 documented (static conditioning → flat CLIP curves)
- [ ] Weakness 2 documented (temporal flickering → FWE dusk spike)

### Math & Architecture Docs
- [ ] DDPM forward/reverse process
- [ ] Score function intuition
- [ ] ELBO → MSE simplification
- [ ] Noise schedules (linear vs cosine)
- [ ] Video modeling + temporal attention explanation
- [ ] Loss variants ($\epsilon$, $x_0$, $v$) with formulas
- [ ] CFG with formula
- [ ] Sampling (DDPM vs DDIM)
- [ ] Architecture diagram (AnimateDiff UNet + motion module)
- [ ] Attention types documented (spatial, temporal, cross)
- [ ] Position encodings + conditioning mechanisms

### Paper Draft
- [ ] Abstract (draft)
- [ ] Section 1: Introduction (complete)
- [ ] Section 2: Background & Math (complete)
- [ ] Section 3: Model Architecture (complete)
- [ ] Section 4: Loss Functions (complete)
- [ ] Section 5: Weakness Analysis (with real numbers)

---

## Phase 2a — Week 14, Part 1: Enhancements A & B

### Enhancement A (Prompt Interpolation)
- [ ] `modeling/prompt_interpolation.py` written
- [ ] Day/night prompt pairs defined for all 50 eval prompts
- [ ] Inference loop modified to use per-frame embeddings
- [ ] C1 (A-only) 150 videos generated
- [ ] Per-frame CLIP-SIM slopes verified (day ↓, night ↑)

### Enhancement B (Temporal Smoothness)
- [ ] Inference blending function written (`apply_frame_blend`)
- [ ] Optimal $\beta$ tuned on 5 validation prompts
- [ ] C2 (B-only) 150 videos generated
- [ ] Flow Warping Error verified (dusk spike flattened)
- [ ] (Optional) Fine-tuning dataset collected (10-15 clips)
- [ ] (Optional) Motion module fine-tuning launched

### Combined
- [ ] C3 (A+B) 150 videos generated

---

## Phase 2b — Week 14, Part 2: Enhancement C

### Enhancement C (Temporal Attention Bias)
- [ ] `modeling/attention_bias.py` written
- [ ] Bias matrix injected into motion module forward pass
- [ ] $\gamma(t)$ timestep scheduling verified
- [ ] C4 (A+B+C) 150 videos generated
- [ ] Metrics computed for all 5 configs

### Paper Draft (Updated)
- [ ] Section 6: Enhancements (with equations for all 3)
- [ ] Preliminary results table

---

## Phase 3a — Week 15, Part 1: TTS Bonus

### Qwen-TTS Setup
- [ ] Qwen3-TTS model loaded
- [ ] Basic inference → `.wav` output

### Advanced Features (≥2 required)
- [ ] **Feature 1: Multi-voice** — narrator profiles for day/dusk/night
- [ ] **Feature 2: Emotion control** — keyword parser → conditioning params
- [ ] **Feature 3: Context-aware synthesis** — pause tokens at punctuation
- [ ] **Feature 4: Sync** — librosa time-stretch to match video length

### End-to-End
- [ ] Prompt → video → narration → merged MP4

---

## Phase 3b — Week 15, Part 2: Evaluation & Paper

### Full Eval (750 videos, 5 configs)
- [ ] FVD computed (requires real video reference)
- [ ] CLIP-SIM dual curves for all 5 configs
- [ ] LPIPS/SSIM per window (day, dusk, night)
- [ ] Flow Warping Error per config
- [ ] Ablation table complete

### User Study
- [ ] Google Form deployed (≥10 raters)
- [ ] Results collected and analyzed

### Final Video
- [ ] Best config selected (C4: A+B+C)
- [ ] Final video ≥4 seconds (48+ frames at 12fps)
- [ ] High-quality MP4 export
- [ ] TTS-narrated version

### Final Paper
- [ ] Section 7: Experiments & Results (tables + figures)
- [ ] Section 8: Discussion (interpretation + limitations)
- [ ] Section 9: Conclusion
- [ ] All references (IEEE format)
- [ ] IEEE double-column formatting
- [ ] PDF exported

### Code Repository
- [ ] `modeling/prompt_interpolation.py`
- [ ] `modeling/attention_bias.py`
- [ ] `inference/inference_enhanced.py`
- [ ] `tts/tts_pipeline.py`
- [ ] `eval/eval.py`
- [ ] `eval/run_ablation.py`
- [ ] `data/eval_prompts.json`
- [ ] `requirements.txt`
- [ ] `README.md` — setup + usage instructions

---

## Final Submission Package
- [ ] Scientific paper (PDF, IEEE double-column)
- [ ] Code repository
- [ ] Final video (≥4s, day-to-night)
- [ ] Ablation results (table)
- [ ] Weakness analysis report
- [ ] User study results
- [ ] (Bonus) TTS-narrated video
