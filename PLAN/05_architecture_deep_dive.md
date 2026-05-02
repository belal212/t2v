# Architecture Deep Dive — AnimateDiff

## Overall Pipeline

```
Text Prompt
    │
    ▼
CLIP Text Encoder (Frozen) ──────────────────┐
(ViT-L/14, 77 tokens × 768-dim)              │
                                              ▼
                                       ┌──────────────┐
                                       │  U-Net       │
┌──────────────────┐                  │  (Frozen     │
│ Latent Noise     │───────►──────────►   Spatial)   │
│ [N, 4, 64, 64]   │                  │              │
└──────────────────┘                  │  Down Block 1│
       ▲                              │  ┌─────────┐ │
       │                              │  │ResNet   │ │
       │                              │  │Motion   │─┼───► Temporal self-attn
       │                              │  └─────────┘ │
       │                              │       ▼       │
       │                              │  Down Block 2│
       │                              │  ┌─────────┐ │
       │                              │  │ResNet   │ │
       │                              │  │Motion   │─┼───► Temporal self-attn
       │                              │  └─────────┘ │
       │                              │      ...      │
       │                              │  Middle Block │
       │                              │  ┌─────────┐ │
       │                              │  │ResNet   │ │
       │                              │  │Spatial  │─┼───► Spatial self-attn
       │                              │  │Cross    │─┼───► Text conditioning
       │                              │  │Motion   │─┼───► Temporal self-attn
       │                              │  └─────────┘ │
       │                              │  Up Block 4  │
       │                              │     ...       │
       │                              └──────────────┘
       │                                      │
       │                                      ▼
       │                              ┌──────────────┐
       │        DDIM Denoising Loop   │ VAE Decoder  │
       └──────────────────────────────┤ (per frame)  │
                                      │ [N, 3, 512,  │
                                      │  512]         │
                                      └──────────────┘
                                              │
                                              ▼
                                     Final Video (N frames)
```

## 1. SD 1.5 U-Net Structure

### Encoder (Downsampling)
| Block | Resolution | Channels | Input | Layers |
|-------|-----------|----------|-------|--------|
| Down 1 | 64×64 | 320 | Latents  | ResNet ×2, SpAttn |
| Down 2 | 32×32 | 640 |          | ResNet ×2, SpAttn |
| Down 3 | 16×16 | 1280 |          | ResNet ×2, SpAttn + CrossAttn |
| Down 4 | 8×8 | 1280 |           | ResNet ×2, SpAttn + CrossAttn |

### Middle Block
ResNet → SpAttn + CrossAttn → ResNet (8×8, 1280ch)

### Decoder (Upsampling, with skip connections)
| Block | Resolution | Input Channels | Output Channels |
|-------|-----------|----------------|-----------------|
| Up 1 | 8×8 | 2560 (concat skip) | 1280 |
| Up 2 | 16×16 | 2560 | 640 |
| Up 3 | 32×32 | 1280 | 320 |
| Up 4 | 64×64 | 640 | 4 |

## 2. Motion Module Architecture

### Insertion Points
9 motion modules — one after each ResNet block in every down/up block. NOT inserted after attention blocks.

### Internal Structure
```
Input: [N, C, H, W]    (N frames at spatial position)
    │
    ▼
┌─────────────────────────────────────┐
│ LayerNorm (across features)         │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Temporal Self-Attention              │
│                                      │
│ Reshape: [N, C, H, W]               │
│       → [H*W, N, C]                 │
│                                      │
│ Q, K, V projections (each C → C)    │
│ + frame position encoding on K, V   │
│                                      │
│ Attn = Softmax(QK^T / √d) V         │
│                                      │
│ Reshape back: [N, C, H, W]          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Residual Connection (+ input)       │
└─────────────┬───────────────────────┘
              │
              ▼
        Output: [N, C, H, W]
```

## 3. Attention Mechanisms

| Type | Scope | Purpose | Dimensions |
|------|-------|---------|------------|
| **Spatial Self-Attn** | Within one frame | Spatial relationships | [H*W, C] × [H*W, C] |
| **Temporal Self-Attn** (Motion Module) | Across frames at same pixel | Motion consistency | [N, C] × [N, C] per (h,w) |
| **Cross-Attn** | Frame → Text tokens | Text conditioning | [H*W, C] × [77, C] |

All use standard scaled dot-product: $\text{Attention}(Q,K,V) = \text{Softmax}(QK^T/\sqrt{d_k})V$.

## 4. Position Encodings

### Timestep Encoding
Sinusoidal for diffusion step $t$:
$$\text{PE}(t, 2i) = \sin(t / 10000^{2i/d})$$
$$\text{PE}(t, 2i+1) = \cos(t / 10000^{2i/d})$$

### Frame Position Encoding
Same sinusoidal encoding applied to frame indices $i \in \{0, 1, \ldots, N-1\}$, added to temporal attention K, V.

## 5. Conditioning Mechanisms

### Text Encoder (CLIP ViT-L/14)
- Input: Raw text (max 77 tokens)
- Output: 77 × 768-dim embeddings
- Frozen during all training

### Cross-Attention Injection
In down blocks 3-4, middle, and up blocks 1-2:
- Query: spatial features from U-Net
- Key/Value: text embeddings (same for all frames — **this is the limitation Enhancement A fixes**)

## 6. VAE Encoder/Decoder

| Component | Input | Output | Compression |
|-----------|-------|--------|-------------|
| **Encoder** $\mathcal{E}$ | Frame [3, 512, 512] | Latent [4, 64, 64] | 8× spatial |
| **Decoder** $\mathcal{D}$ | Latent [4, 64, 64] | Frame [3, 512, 512] | 8× spatial |

- Each frame encoded/decoded independently
- Latent diffusion: denoising runs in 64×64 space (16× cheaper than pixel)

## 7. Enhancement C Injection Point

**Location:** Inside each temporal self-attention layer within the motion module.

**Modification:**
$$\text{Attention}(Q,K,V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + \gamma(t) \cdot B\right)V$$

Where $B \in \mathbb{R}^{N \times N}$ is a learned bias matrix and $\gamma(t) = 1 - t/T \in [0,1]$.

## 8. Data Flow (Denoising Loop)
```
For t = T down to 1:
    z_t = current noisy latents [N, 4, 64, 64]
    
    For each block in UNet (down → mid → up):
        1. ResNet(z_t)           → spatial features
        2. MotionModule(z_t)     → temporal self-attn (across frames)
        3. SpatialSelfAttn(z_t)  → per-frame spatial
        4. CrossAttn(z_t, text)  → text conditioning
    
    z_{t-1} = DDIM_step(z_t, epsilon_theta, t)

z_0 → VAE Decoder → pixel frames [N, 3, 512, 512]
```
