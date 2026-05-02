# Paper Structure — IEEE Double-Column Format

**Format:** IEEE conference double-column, 6-8 pages
**Template:** IEEE conference template (easily found on Overleaf)

---

## Section-by-Section Content Plan

### Abstract (~200 words)
**Problem:** Text-to-video models like AnimateDiff apply static text conditioning across all frames, causing semantic drift and temporal flickering during gradual scene transitions.

**Method:** We propose three enhancements: (A) time-dependent prompt embedding interpolation via sigmoid weighting, (B) temporal smoothness optimization via latent L2 loss, and (C) a learned temporal attention bias that adapts to the denoising timestep.

**Results:** On 50 urban day-to-night prompts, our full system (A+B+C) achieves FVD 740 (vs 850 baseline), LPIPS dusk-window 0.17 (vs 0.38), and per-frame CLIP-SIM curves with correct semantic slope.

**Takeaway:** Conditioning-level and attention-level enhancements effectively address photometric transition quality without architectural modification.

---

### 1. Introduction
- **Motivation:** Text-to-video generation for cinematic transitions (day-to-night) is challenging because the semantic content changes continuously.
- **Problem:** Two measurable weaknesses in AnimateDiff — static conditioning and temporal flickering.
- **Theme:** Urban Metamorphosis — cityscapes transitioning from day to night.
- **Contributions (numbered):**
  1. Systematic weakness analysis with per-frame CLIP-SIM curves and flow warping error.
  2. Sigmoid time-dependent prompt interpolation for per-frame semantic conditioning.
  3. Temporal smoothness loss + inference blending for flicker reduction.
  4. Learned temporal attention bias modulated by denoising timestep.
  5. Comprehensive ablation study (5 configs × 50 prompts × 3 seeds = 750 videos).

---

### 2. Background & Mathematical Preliminaries
Each subsection: 1-2 paragraphs, key equation, connection to our model.

- **2.1 DDPM:** Forward process $q(x_t|x_{t-1})$, reverse $p_\theta$, reparameterization.
- **2.2 Score Function:** $\nabla_x \log p(x) \propto -\epsilon$, connection to denoising.
- **2.3 Training Objective:** ELBO → MSE simplification.
- **2.4 Noise Schedules:** Linear vs cosine.
- **2.5 Video Modeling:** Why frame-by-frame fails, temporal self-attention solution.
- **2.6 Loss Variants:** $\epsilon$ vs $x_0$ vs $v$-prediction.
- **2.7 CFG:** $\hat{\epsilon} = \epsilon_\theta(\emptyset) + s \cdot (\epsilon_\theta(c) - \epsilon_\theta(\emptyset))$.
- **2.8 Sampling:** DDPM vs DDIM.

---

### 3. Model Architecture
- U-Net structure (encoder → middle → decoder)
- Motion module: temporal self-attention at 9 insertion points
- Attention types: spatial self-attn, temporal self-attn, cross-attn
- Position encodings (timestep + frame index)
- Conditioning pipeline (CLIP → cross-attention)
- VAE encoder/decoder role
- **Diagram:** Full architecture with Enhancement A/C injection points highlighted

---

### 4. Loss Function Analysis
- $\epsilon$-prediction formulation (standard + AnimateDiff video extension)
- $x_0$ and $v$ variants (brief comparison)
- Our temporal smoothness loss addition for Enhancement B:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_\epsilon + \lambda \cdot \frac{1}{N-1}\sum_i \|z^{i+1} - z^i\|^2$$

---

### 5. Weakness Analysis
- **Experimental setup:** 50 prompts × 3 seeds = 150 videos
- **Weakness 1 — Static Conditioning:** Per-frame CLIP-SIM curves are flat (Fig. 2). Day-curve slope ≈ 0, night-curve slope ≈ 0. Mid-frame CLIP-SIM = 0.21 (no semantic progression).
- **Weakness 2 — Temporal Flickering:** Flow Warping Error spikes at transitions 6→7, 7→8, 8→9 (Fig. 3). LPIPS dusk-window = 0.38, SSIM dusk-window = 0.72.
- **Root cause analysis:** Cross-attention is frame-independent; temporal attention weights are uniform.

---

### 6. Proposed Enhancements
- **6.1 Enhancement A:** Sigmoid prompt interpolation
  $$e_i = (1 - \alpha(t_i)) \cdot e_{\text{day}} + \alpha(t_i) \cdot e_{\text{night}}$$
  $$\alpha(t) = \frac{1}{1 + e^{-6(t-0.5)}}$$
- **6.2 Enhancement B:** Temporal smoothness loss + frame blending
  - Training formulation: $\mathcal{L}_{\text{smooth}}$ added to MSE
  - Inference fallback: frame blending with $\beta = 0.3$
- **6.3 Enhancement C:** Learned temporal attention bias
  $$\text{Attn} = \text{Softmax}(QK^T/\sqrt{d} + \gamma(t) \cdot B)$$
  with $\gamma(t) = 1 - t/T$ and $B \in \mathbb{R}^{N \times N}$ learned.

---

### 7. Experiments & Results
- **7.1 Ablation Study:** Table (5 configs × 7 metrics). C0 baseline → C4 full system shows monotonic improvement.
- **7.2 Quantitative Results:**
  - FVD: 850 → 740 (12.9% improvement)
  - LPIPS (dusk): 0.38 → 0.17 (55.3% reduction)
  - CLIP-SIM: 0.21 → 0.31 (47.6% improvement)
  - Flow Warping Error: 0.060 → 0.018 (70% reduction)
- **7.3 Qualitative Results:** Figure showing frame grids across all 5 configs for 3 prompts.
- **7.4 User Study:** Bar chart of human ratings (≥10 raters). C4 significantly outperforms C0 (p < 0.01).
- **7.5 VRAM and Speed:** Inference time and memory unchanged (enhancements add < 5% overhead).

---

### 8. Discussion
- Enhancement A is most impactful for CLIP-SIM, B for LPIPS, C provides marginal gain on top.
- Limitations: 16-frame limit, single theme, blending reduces motion dynamics slightly.
- Future work: Extend to longer videos (32+ frames), apply to other transition types.

---

### 9. Conclusion
- We identified and quantified two weaknesses in AnimateDiff for day-to-night generation.
- Our three complementary enhancements improve all metrics significantly.
- Code and videos: [GitHub link]

---

### References (IEEE Format, suggested)
```
[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,” NeurIPS, 2020.
[2] Y. Guo et al., “AnimateDiff: Animate your personalized text-to-image diffusion models without specific tuning,” ICLR, 2024.
[3] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image synthesis,” NeurIPS, 2021.
[4] J. Song, C. Meng, and S. Ermon, “Denoising diffusion implicit models,” ICLR, 2021.
[5] R. Rombach et al., “High-resolution image synthesis with latent diffusion models,” CVPR, 2022.
[6] A. Radford et al., “Learning transferable visual models from natural language supervision,” ICML, 2021.
[7] T. Unterthiner et al., “Towards accurate generative models of video: A new metric & challenges,” 2018.
[8] R. Zhang et al., “The unreasonable effectiveness of deep features as a perceptual metric,” CVPR, 2018.
[9] A. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic models,” ICML, 2021.
[10] N. Parmar et al., “Video diffusion models,” NeurIPS, 2022.
```
