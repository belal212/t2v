# Mathematical Background

## 1. Denoising Diffusion Probabilistic Models (DDPM)

### Forward Process
Gradually add Gaussian noise to data $x_0 \sim p_{\text{data}}(x)$ over $T$ timesteps:

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\,x_{t-1},\ \beta_t \mathbf{I})$$

By the reparameterization trick:

$$x_t = \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\,\epsilon,\quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

where $\alpha_t = 1 - \beta_t$ and $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$.

As $t \to T$, $\bar{\alpha}_t \to 0$, so $x_T \approx \mathcal{N}(0, \mathbf{I})$.

### Reverse Process
The model learns to reverse the forward process:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t))$$

Instead of predicting $\mu_\theta$ directly, predict the noise $\epsilon_\theta(x_t, t)$:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right)$$

### Visual Intuition
| $t$ | $\bar{\alpha}_t$ | Signal | Noise | Appearance |
|-----|------------------|--------|-------|------------|
| 0 | 1.0 | $x_0$ | 0 | Clean frame |
| 250 | 0.5 | $0.7 x_0$ | $0.7 \epsilon$ | Blurry, outlines visible |
| 500 | 0.2 | $0.45 x_0$ | $0.89 \epsilon$ | Mostly noise, structure faint |
| 1000 | ~0 | ~0 | ~$\epsilon$ | Pure Gaussian noise |

### Connection to Our Model
AnimateDiff extends this to video: each frame follows the same forward/reverse process independently, but the motion module enforces consistent noise predictions across frames through temporal attention.

---

## 2. Score Function

### Definition
The score function is the gradient of the log-probability density:

$$s(x) = \nabla_x \log p(x)$$

### Connection to Denoising
For $x_t = x_0 + \sigma_t \epsilon$:

$$\nabla_{x_t} \log p(x_t) \approx -\frac{\epsilon}{\sigma_t}$$

Learning $\epsilon_\theta$ is equivalent to learning the score $\nabla_x \log p(x)$. The model learns to point from noisy data toward cleaner data — the direction of steepest increase in the data distribution.

---

## 3. Training Objective (ELBO → MSE)

Diffusion models minimize the negative variational lower bound (ELBO):

$$\mathcal{L} = \mathbb{E}_q[-\log p_\theta(x_0)] \leq \mathbb{E}_q\left[\sum_{t=2}^T D_{\text{KL}}(q(x_{t-1}|x_t,x_0) \| p_\theta(x_{t-1}|x_t))\right]$$

The KL divergence between two Gaussians simplifies to MSE:

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$$

**Why this works:** Uniform sampling over $t$ ensures the model learns all noise levels. The simplification drops weighting terms but produces empirically better samples.

### In Our Model
AnimateDiff extends this to video:

$$\mathcal{L}_{\text{AD}} = \mathbb{E}_{t, \{\epsilon^i\}, \{x_0^i\}, c}\left[\sum_{i=1}^N \|\epsilon^i - \epsilon_\theta(\{x_t^i\}, t, c, \{pe_i\})\|^2\right]$$

---

## 4. Noise Schedules

### Linear Schedule
$$\beta_t = \beta_1 + \frac{t-1}{T-1}(\beta_T - \beta_1),\quad \beta_1=10^{-4},\ \beta_T=0.02$$

- Simple, widely used in SD 1.5
- Adds noise too quickly at early timesteps

### Cosine Schedule
$$\bar{\alpha}_t = \frac{f(t)}{f(0)},\quad f(t) = \cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right),\ s=0.008$$

- Slower noise addition in early/mid steps
- Better preservation of structural information
- Produces higher quality samples

### Our Choice
AnimateDiff (SD 1.5-based) uses the **linear schedule**. We keep this for compatibility.

---

## 5. Video Modeling — Why Frame-by-Frame Fails

### Naive Approach
Generate each frame independently with SD → no temporal consistency.

**Problems:** Lighting jumps between frames, objects change position erratically, no motion coherence → perceptible flickering.

### AnimateDiff's Solution
Temporal self-attention layers model inter-frame dependencies:

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right)V$$

For a given spatial position $(h, w)$:
- $Q \in \mathbb{R}^{N \times d}$ — queries from each frame
- $K, V \in \mathbb{R}^{N \times d}$ — keys and values from all frames

Each frame's representation is updated based on all other frames, enforcing consistency.

---

## 6. Loss Variants

| Variant | Target | Formula | When Useful |
|---------|--------|---------|-------------|
| **Noise pred** ($\epsilon$) | Added noise | $\|\epsilon - \epsilon_\theta\|^2$ | General purpose; **our choice** |
| **Image pred** ($x_0$) | Clean image | $\|x_0 - x_\theta\|^2$ | Direct optimization; unstable at high noise |
| **Velocity pred** ($v$) | $v = \alpha_t\epsilon - \sigma_t x_0$ | $\|v - v_\theta\|^2$ | Better for high-res; smoother training |

---

## 7. Classifier-Free Guidance (CFG)

### Formulation
During training, randomly drop conditioning (10% of the time) to learn unconditional $\epsilon_\theta(x_t, \varnothing)$.

At inference:

$$\hat{\epsilon}_\theta = \epsilon_\theta(x_t, \varnothing) + s \cdot (\epsilon_\theta(x_t, c) - \epsilon_\theta(x_t, \varnothing))$$

### Effect of Guidance Scale $s$
| $s$ | Alignment | Diversity | Visual Effect |
|-----|-----------|-----------|---------------|
| 1 | Low | High | Ignores prompt |
| 7.5 | Good | Moderate | **Default — our choice** |
| 15 | High | Low | Over-saturated, reduced motion |

---

## 8. Sampling Methods

### DDPM (Stochastic)
Full reverse process $t = T, T-1, \ldots, 1$ with noise added at each step. Higher quality but slow (1000 steps).

### DDIM (Deterministic)
$$x_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \underbrace{\frac{x_t - \sqrt{1-\bar{\alpha}_t}\,\epsilon_\theta}{\sqrt{\bar{\alpha}_t}}}_{\text{predicted }x_0} + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \epsilon_\theta$$

Can skip steps (25-50). Near-identical quality to DDPM. **Our choice** for efficiency.

### DPM-Solver (Optional)
ODE-based solver, 10-20 steps. Fast but may introduce temporal artifacts in video.

---

## Summary for Paper
Each subsection above corresponds to a paragraph in Paper Section 2 (Background & Mathematical Preliminaries). Keep equations minimal (1-2 per concept) and focus on intuition and connection to our model.
