# Proposed Enhancements — 3 Enhancements

---

## Enhancement A: Time-Dependent Prompt Conditioning

### Problem
The same text embedding $e$ is used for all 16 frames. Frame $i=12$ (near-night) receives identical semantic conditioning as frame $i=0$ (full day). This causes semantic drift in mid-transition frames.

### Mathematical Formulation

Let $e_{\text{day}} = E_{\text{CLIP}}(\text{prompt}_{\text{day}})$ and $e_{\text{night}} = E_{\text{CLIP}}(\text{prompt}_{\text{night}})$.

For frame $i$ at normalized position $t_i = i/(N-1) \in [0, 1]$:

$$e_i = (1 - \alpha(t_i)) \cdot e_{\text{day}} + \alpha(t_i) \cdot e_{\text{night}}$$

### Interpolation Function: Sigmoid (k=6)

$$\alpha(t) = \frac{1}{1 + e^{-6(t - 0.5)}}$$

**Why sigmoid over linear?** A linear schedule treats the transition as uniform. The sigmoid matches perceptual reality: the scene stays predominantly day, transitions sharply around dusk, then stabilizes at night.

| $t$ | Frame | $\alpha(t)$ | Day Weight | Night Weight |
|-----|-------|-------------|------------|--------------|
| 0.0 | 0 (Day) | 0.05 | 0.95 | 0.05 |
| 0.33 | 5 (Afternoon) | 0.12 | 0.88 | 0.12 |
| 0.5 | 8 (Dusk) | 0.50 | 0.50 | 0.50 |
| 0.67 | 10 (Twilight) | 0.88 | 0.12 | 0.88 |
| 1.0 | 15 (Night) | 0.95 | 0.05 | 0.95 |

### Implementation
```python
# modeling/prompt_interpolation.py
import torch, math

def interpolate_prompt_embeddings(pipe, prompt_day, prompt_night, num_frames, k=6.0):
    with torch.no_grad():
        tok_day = pipe.tokenizer(prompt_day, return_tensors="pt",
            padding="max_length", max_length=77, truncation=True)
        tok_night = pipe.tokenizer(prompt_night, return_tensors="pt",
            padding="max_length", max_length=77, truncation=True)
        emb_day = pipe.text_encoder(tok_day.input_ids.to(pipe.device))[0]
        emb_night = pipe.text_encoder(tok_night.input_ids.to(pipe.device))[0]
    
    alphas = [1.0 / (1.0 + math.exp(-k * (i / (num_frames - 1) - 0.5)))
              for i in range(num_frames)]
    
    embeddings = [(1 - a) * emb_day + a * emb_night for a in alphas]
    return torch.stack(embeddings)  # [num_frames, 77, 768]
```

### Integration into Inference
Replace the single encoder hidden state with per-frame embeddings:

```python
# In the denoising loop:
for i in range(num_frames):
    noise_pred = pipe.unet(
        latents[:, i], t,
        encoder_hidden_states=embeddings[i].unsqueeze(0),  # frame-specific
    )
```

### Expected Effect
- Per-frame CLIP-SIM curves slope correctly (day-score decreases, night-score increases across frames)
- Mid-transition frames show clearer semantic identity
- No additional VRAM cost (embeddings are small)

---

## Enhancement B: Temporal Smoothness

### Problem
The motion module produces flickering during photometric transitions (especially frames 6-9) because it was not trained on smooth lighting changes.

### Approach A: Fine-tuning with Smoothness Loss

**Additional loss terms:**

$$\mathcal{L}_{\text{smooth}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \| z_t^{i+1} - z_t^i \|^2$$

$$\mathcal{L}_{\text{total}} = \mathcal{L}_\epsilon + \lambda_s \cdot \mathcal{L}_{\text{smooth}}, \quad \lambda_s = 0.01$$

**Training setup:**
- Parameters: Motion module only (~450M)
- Spatial UNet: Frozen
- Dataset: 10-15 day-to-night time-lapse clips (Pexels/Pixabay)
- Optimizer: AdamW, lr=1e-5
- Steps: 500-1000
- GPUs: 1× A6000 (or 2× with DataParallel)

### Approach B: Inference-Time Frame Blending (Fallback)

If fine-tuning is not completed, apply at the final latent step:

$$z_0^{i,\text{blend}} = (1-\beta) \cdot z_0^i + \beta \cdot z_0^{i+1}, \quad \beta = 0.3$$

```python
def apply_frame_blend(latents, beta=0.3):
    blended = latents.clone()
    for i in range(len(latents) - 1):
        blended[i]   = (1 - beta) * latents[i] + beta * latents[i+1]
        blended[i+1] = beta * latents[i] + (1 - beta) * latents[i+1]
    return blended
```

### Expected Effect
| Metric | Baseline | After B |
|--------|----------|---------|
| LPIPS (dusk window) | ~0.38 | ~0.20 |
| SSIM (dusk window) | ~0.72 | ~0.85 |
| Flow Warping Error | ~0.06 | ~0.02 |

---

## Enhancement C: Temporal Attention Bias

### Problem
The motion module attends equally across all frame pairs at all denoising timesteps. Early in denoising (high $t$), long-range attention is useful for establishing the global transition arc. Late in denoising (low $t$), attending to distant frames introduces noise into fine detail.

### Mathematical Formulation

Add a learned bias matrix $B \in \mathbb{R}^{N \times N}$ scaled by timestep-dependent $\gamma(t)$:

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \gamma(t) \cdot B\right)V$$

where:

$$\gamma(t) = 1 - \frac{t}{T} \in [0, 1]$$

### Intuition
- At $t = T$ (pure noise): $\gamma = 0$, no bias, full global attention — model learns the overall transition arc
- At $t = 0$ (clean image): $\gamma = 1$, full bias, local attention dominates — model refines frame-specific details

### Implementation
```python
# modeling/attention_bias.py
import torch.nn as nn

class TemporalBias(nn.Module):
    def __init__(self, num_frames=16):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_frames, num_frames))
    
    def forward(self, attn_weights, t, T):
        # attn_weights: [batch, heads, N, N]
        # t: current timestep (0 = clean, T = noisy)
        gamma = 1.0 - t / T  # ∈ [0, 1]
        return attn_weights + gamma * self.bias
```

### Integration
Inject this into the motion module's temporal self-attention by adding the bias to the attention logits before softmax. Total added code: ~25 lines.

### Expected Effect
- Sharper fine-grained textures in final frames
- Reduced global bleeding between semantically distant frames
- Complementary to Enhancement A (which fixes semantics) and B (which fixes flicker)

---

## Combined Pipeline (A+B+C)

```python
# inference/inference_enhanced.py
def generate_enhanced(pipe, prompt_day, prompt_night, num_frames=16, blend_beta=0.3):
    # Enhancement A: per-frame embeddings
    embeddings = interpolate_prompt_embeddings(pipe, prompt_day, prompt_night, num_frames)
    
    # Denoising loop with Enhancement C
    latents = torch.randn(1, num_frames, 4, 64, 64, device=pipe.device)
    for t in reversed(range(num_steps)):
        for i in range(num_frames):
            noise_pred = pipe.unet(
                latents[:, i], t,
                encoder_hidden_states=embeddings[i].unsqueeze(0),
                # Enhancement C bias applied inside motion module forward
            )
            latents[:, i] = ddim_step(latents[:, i], noise_pred, t)
    
    # Enhancement B: frame blending
    latents = apply_frame_blend(latents[0], beta=blend_beta)
    
    # Decode
    frames = pipe.vae.decode(latents)
    return frames
```

## Ablation Configurations
| Config | Enh A | Enh B | Enh C | Description |
|--------|-------|-------|-------|-------------|
| C0: Baseline | ❌ | ❌ | ❌ | Standard AnimateDiff |
| C1: A-only | ✅ | ❌ | ❌ | Prompt interpolation |
| C2: B-only | ❌ | ✅ | ❌ | Temporal smoothness |
| C3: A+B | ✅ | ✅ | ❌ | Combined semantics + smoothness |
| C4: A+B+C | ✅ | ✅ | ✅ | Full enhanced system |
