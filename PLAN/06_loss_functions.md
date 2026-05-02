# Loss Functions — Detailed Analysis

## 1. Noise Prediction ($\epsilon$-prediction)

### Standard Formulation
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

$$\mathcal{L}_\epsilon = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

### In AnimateDiff (Video Extension)
$$\mathcal{L}_\epsilon^{\text{AD}} = \mathbb{E}_{t, \{\epsilon^i\}, \{x_0^i\}, c} \left[ \sum_{i=1}^N \| \epsilon^i - \epsilon_\theta(\{x_t^i\}, t, c, \{pe_i\}) \|^2 \right]$$

where $N$ = frames, $pe_i$ = frame position encoding, $c$ = text embedding.

## 2. Image Prediction ($x_0$-prediction)

$$\mathcal{L}_{x_0} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| x_0 - x_\theta(x_t, t) \|^2 \right]$$

- Direct optimization target
- Less stable at high noise levels (prediction is unbounded)
- Rarely used in practice

## 3. Velocity Prediction ($v$-prediction)

$$v = \alpha_t \epsilon - \sigma_t x_0, \quad \alpha_t = \sqrt{\bar{\alpha}_t}, \quad \sigma_t = \sqrt{1-\bar{\alpha}_t}$$

$$\mathcal{L}_v = \mathbb{E}_{t, x_0, \epsilon} \left[ \| v - v_\theta(x_t, t) \|^2 \right]$$

- Target is continuous across timesteps
- At $t=0$: $v \propto -x_0$ (toward data)
- At $t=T$: $v \propto \epsilon$ (toward noise)
- More stable training for fast samplers

## 4. Comparison Table

| Variant | Output | Stability | When to Use |
|---------|--------|-----------|-------------|
| $\epsilon$ | Unit Gaussian | High | General purpose; **AnimateDiff default** |
| $x_0$ | Pixel values | Low at high noise | Clean image reconstruction |
| $v$ | Velocity | High | Fast samplers, high-res video |

## 5. Our Choice: $\epsilon$-prediction

**AnimateDiff uses $\epsilon$-prediction** inherited from SD 1.5. The frozen spatial layers expect this formulation. The motion module is trained to predict temporally-consistent noise residuals.

We keep $\epsilon$-prediction for all training and inference. The temporal bias (Enhancement C) does not change the prediction target — it only modifies attention weights.

## 6. Additional Loss for Enhancement B (Fine-tuning)

### Temporal Smoothness Loss
$$\mathcal{L}_{\text{smooth}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \| z_t^{i+1} - z_t^i \|^2$$

where $z_t^i$ is the latent of frame $i$ at timestep $t$.

### Perceptual Temporal Loss (Final Step)
$$\mathcal{L}_{\text{LPIPS}} = \frac{1}{N-1} \sum_{i=1}^{N-1} \text{LPIPS}(\mathcal{D}(z_0^{i+1}), \mathcal{D}(z_0^i))$$

### Combined Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_\epsilon + \lambda_s \cdot \mathcal{L}_{\text{smooth}} + \lambda_p \cdot \mathcal{L}_{\text{LPIPS}}$$

with $\lambda_s = 0.01$, $\lambda_p = 0.001$.

**Training details:** Only motion module parameters are updated (~450M). Spatial UNet frozen. 500-1000 steps on ~15 day-to-night time-lapse clips.
