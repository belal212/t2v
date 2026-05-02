# Evaluation Metrics

## Metric Summary

| Metric | What It Measures | Tool | Our Focus |
|--------|-----------------|------|-----------|
| **FVD** | Overall video quality + realism | `torch-fidelity` | Full 5-config comparison |
| **IS** | Frame sharpness + diversity | `torchmetrics` | Sanity check (should not degrade) |
| **CLIP-SIM (per-frame)** | Semantic alignment per frame | `openai/clip-vit` | **Primary for Enh A** — dual curves |
| **SSIM / PSNR** | Frame-level pixel consistency | `skimage.metrics` | Dusk window focus |
| **LPIPS (temporal)** | Perceptual flickering | `lpips` library | **Primary for Enh B** — dusk window |
| **Flow Warping Error** | Optical-flow temporal consistency | RAFT + custom | **Primary for Enh B** — per-transition |
| **User Study** | Subjective quality (≥10 raters) | Google Form | 1-5 scale, 4 criteria |

---

## 1. Fréchet Video Distance (FVD)

### Definition
Distributional distance between real and generated video features in a pretrained I3D space:

$$\text{FVD} = \|\mu_r - \mu_g\|^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$

### Implementation
```python
from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1="outputs/ablation/C4_A+B+C",  # generated
    input2="data/real_day_to_night",      # reference
    cuda=True, batch_size=8, fvd=True, verbose=False,
)
```

**Reference dataset:** 20-30 real day-to-night time-lapse city videos from Pexels/Pixabay.

---

## 2. Inception Score (IS)

Measures per-frame sharpness and diversity using an Inception-v3 classifier:

$$\text{IS} = \exp\left(\mathbb{E}_x[D_{\text{KL}}(p(y|x) \| p(y))]\right)$$

**Note:** Per-frame metric. Does not capture temporal quality. Used as a sanity check.

---

## 3. CLIP-SIM (Semantic Alignment) — Primary for Enhancement A

### Per-Frame Dual Curves
Compute two similarity scores per frame:
1. $\text{sim}(x_i, e_{\text{day}})$ — how "daytime" does this frame look?
2. $\text{sim}(x_i, e_{\text{night}})$ — how "nighttime" does this frame look?

### Expected Shape After Enhancement A
```
CLIP-SIM to Day:  [0.33, 0.32, 0.30, 0.28, 0.26, 0.23, 0.20, 0.17]  ↓ slope
CLIP-SIM to Night: [0.15, 0.17, 0.20, 0.23, 0.26, 0.29, 0.32, 0.34]  ↑ slope
```

### Key Metrics
| Metric | Formula | Baseline | Target |
|--------|---------|----------|--------|
| Day slope | $\beta_{\text{day}}$ in sim ~ frame | ~0.0 | Negative |
| Night slope | $\beta_{\text{night}}$ in sim ~ frame | ~0.0 | Positive |
| Mid-frame separation | $\text{sim}_{\text{night}} - \text{sim}_{\text{day}}$ at frame 8 | ~0.0 | > 0.10 |

---

## 4. SSIM / PSNR — Dusk Window Focus

Compute between consecutive frames, group by region:
- **Day window:** frames 0-5
- **Dusk window:** frames 6-9 **(our focus)**
- **Night window:** frames 10-15

```python
def compute_windowed_ssim(frames):
    ssims = [ssim(frames[i], frames[i+1], channel_axis=-1, data_range=1.0)
             for i in range(len(frames)-1)]
    return {
        "day_mean":  np.mean(ssims[0:5]),
        "dusk_mean": np.mean(ssims[5:9]),   # ← problem area
        "night_mean": np.mean(ssims[9:15]),
    }
```

---

## 5. LPIPS (Temporal) — Primary for Enhancement B

Learned perceptual similarity between consecutive frames:

```python
import lpips
lpips_fn = lpips.LPIPS(net='alex')

scores = []
for i in range(len(frames) - 1):
    img1 = tensor_from_frame(frames[i])  # [1, 3, H, W], [-1, 1]
    img2 = tensor_from_frame(frames[i+1])
    scores.append(lpips_fn(img1, img2).item())
```

**Interpretation:**
| LPIPS | Quality |
|-------|---------|
| < 0.15 | Very smooth |
| 0.15-0.30 | Noticeable but acceptable |
| > 0.30 | Flickering (baseline weakness) |

---

## 6. Flow Warping Error

### Algorithm
1. Compute dense optical flow between frame $i$ and $i+1$ (Farneback method in OpenCV)
2. Warp frame $i$ using the flow
3. Compute L2 error between warped and actual frame $i+1$

```python
def flow_warp_error(frames):
    errors = []
    for i in range(len(frames) - 1):
        g1, g2 = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in [frames[i], frames[i+1]]]
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x += flow[..., 0]; map_y += flow[..., 1]
        warped = cv2.remap(frames[i].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        errors.append(np.mean((warped - frames[i+1].astype(np.float32))**2))
    return np.array(errors)  # shape: [N-1]
```

---

## 7. User Study

### Criteria (1-5 scale)
1. **Visual quality:** Realism and clarity
2. **Temporal smoothness:** Absence of flickering
3. **Semantic alignment:** Does it clearly show day→night?
4. **Overall impression:** Engaging? Impressive?

### Design
- Compare C0 (Baseline) vs C4 (Full Enhanced)
- With and without TTS audio
- ≥10 raters
- Google Form with embedded MP4s

---

## Automated Eval Pipeline
```python
# eval/eval.py
def evaluate_video(video_path, prompt_data):
    frames = load_frames(video_path)
    return {
        "lpips": compute_windowed_lpips(frames),
        "ssim": compute_windowed_ssim(frames),
        "clip_day": compute_clip_sim(frames, prompt_data["prompt_day"]),
        "clip_night": compute_clip_sim(frames, prompt_data["prompt_night"]),
        "flow_warp_mean": np.mean(flow_warp_error(frames)),
        "flow_warp_dusk": np.mean(flow_warp_error(frames)[5:9]),
    }

def aggregate_results(config_dirs):
    rows = []
    for config_dir in config_dirs:
        for vp in Path(config_dir).glob("*.mp4"):
            row = {"config": config_dir.name}
            row.update(evaluate_video(vp, load_prompt(vp)))
            rows.append(row)
    return pd.DataFrame(rows)
```
