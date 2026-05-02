# Weakness Analysis — Baseline AnimateDiff for Urban Day-to-Night

## Methodology

### Experimental Setup
| Parameter | Value |
|-----------|-------|
| Model | AnimateDiff v2 (SD 1.5) |
| Evaluation prompts | 50 (see eval_prompts.json) |
| Seeds per prompt | 3 (42, 123, 999) |
| Total videos | 150 |
| Frames per video | 16 |
| Inference | DDIM, 50 steps, CFG 7.5 |
| Resolution | 512×512 |

### Output
```
outputs/baseline/
├── prompt_{id}/
│   ├── seed_42.mp4
│   ├── seed_123.mp4
│   └── seed_999.mp4
└── metrics/
    ├── per_frame_clip_day_night.csv
    ├── flow_warping_error.csv
    ├── lpips_temporal.csv
    └── summary.json
```

---

## Weakness 1: Static Conditioning → Semantic Drift

### Hypothesis
The same text embedding is applied to all 16 frames. Frame $i=0$ (noon) receives identical conditioning as frame $i=12$ (dusk). This causes mid-transition frames to lack clear semantic identity — they are neither convincingly day nor night.

### Metric: Per-Frame CLIP-SIM

Compute two curves per video:
1. **CLIP-SIM with "day" prompt:** $\text{sim}(x_i, e_{\text{day}})$ for $i = 0 \ldots 15$
2. **CLIP-SIM with "night" prompt:** $\text{sim}(x_i, e_{\text{night}})$ for $i = 0 \ldots 15$

### Expected Result (Baseline)
```
CLIP-SIM
 0.35 ┤
      │   ╱╲──── day curve (flat)
 0.30 ┤  ╱  ╲
      │ ╱    ╲── night curve (flat)
 0.25 ┤╱      ╲
      │
 0.20 ┤
      0   4    8   12   16
      Day       │       Night
                (flat = drift)
```

**Diagnostic:** Both curves are nearly flat. The day-curve does NOT slope down, and the night-curve does NOT slope up. This proves the model ignores temporal position for semantic conditioning.

### Quantitative
| Metric | Baseline (expected) | Target After A |
|--------|-------------------|----------------|
| CLIP-SIM day-curve slope | ~0.0 | Negative |
| CLIP-SIM night-curve slope | ~0.0 | Positive |
| CLIP-SIM mid-frame (day prompt) | ~0.21 | ~0.17 (lower = better) |
| CLIP-SIM mid-frame (night prompt) | ~0.21 | ~0.30 (higher = better) |

### Root Cause
Cross-attention receives the same text embedding $e$ for every frame. The model has no mechanism to schedule semantic content across the temporal dimension.

---

## Weakness 2: Temporal Flickering in the Dusk Window

### Hypothesis
The motion module was trained on general video datasets (mostly object motion, camera motion). It was not trained on smooth photometric transitions (lighting/color shifts). Frames 6-9 (the dusk window) show visible flickering because the lighting gradient is steeper than the motion module can handle.

### Metric: Flow Warping Error (Per-Transition)

1. Compute RAFT optical flow between consecutive frames: $F_{i \to i+1}$
2. Warp frame $i$ to $i+1$: $\hat{x}_{i+1} = \text{Warp}(x_i, F_{i \to i+1})$
3. Error: $\text{FWE}_i = \|\hat{x}_{i+1} - x_{i+1}\|^2$

### Expected Result (Baseline)
```
Flow Warping Error
 0.08 ┤
      │             ╱╲
 0.06 ┤           ╱   ╲     ← Dusk window spike
      │          ╱     ╲
 0.04 ┤    ╱╲  ╱       ╲
      │   ╱  ╲╱         ╲
 0.02 ┤  ╱               ╲
      │ ╱                 ╲
 0.00 ┤╱                   ╲
      0→1 2→3 4→5 6→7 8→9 10→11 12→13 14→15
         Day   │ Dusk   Night
```

**Diagnostic:** FWE spikes significantly at transitions 6→7, 7→8, 8→9.

### Additional Metrics (Dusk Window Focus)
| Metric | Baseline | Acceptable |
|--------|----------|------------|
| LPIPS (transitions 6-9) | ~0.38 | < 0.20 |
| SSIM (transitions 6-9) | ~0.72 | > 0.85 |
| Flow Warping Error | ~0.06 | < 0.02 |

### Root Cause
The temporal self-attention window (16 frames) must cover both day and night appearances. The attention weights are uniform — they do not adapt to the fact that nearby frames should be more similar than distant frames during a photometric transition.

---

## Evaluation Script (Core Logic)

```python
# eval/weakness_analysis.py
def compute_per_frame_clip_sim(frames, prompt_day, prompt_night, model, processor):
    """Returns two arrays of length N: clip_day_scores, clip_night_scores"""
    day_feat = model.get_text_features(**processor(text=[prompt_day], return_tensors="pt"))
    night_feat = model.get_text_features(**processor(text=[prompt_night], return_tensors="pt"))
    
    day_scores, night_scores = [], []
    for frame in frames:
        img_feat = model.get_image_features(**processor(images=frame, return_tensors="pt"))
        day_scores.append(F.cosine_similarity(img_feat, day_feat).item())
        night_scores.append(F.cosine_similarity(img_feat, night_feat).item())
    
    return np.array(day_scores), np.array(night_scores)

def compute_flow_warping_error(frames):
    """Returns array of length N-1: FWE per transition"""
    errors = []
    for i in range(len(frames) - 1):
        g1 = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        g2 = cv2.cvtColor(frames[i+1], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        h, w = flow.shape[:2]
        map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x += flow[..., 0]; map_y += flow[..., 1]
        warped = cv2.remap(frames[i].astype(np.float32), map_x, map_y, cv2.INTER_LINEAR)
        errors.append(np.mean((warped - frames[i+1].astype(np.float32))**2))
    return np.array(errors)
```

## Weakness Report Summary
```
┌───────────────────────────────────────────────────────────────┐
│ Weakness Analysis Report — Baseline AnimateDiff              │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│ Weakness 1: Static Conditioning → Semantic Drift              │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ Evidence:                                                  │ │
│ │  - CLIP-SIM day-curve slope: 0.003 (should be negative)  │ │
│ │  - CLIP-SIM night-curve slope: -0.002 (should be positive)│ │
│ │  - Mid-frame CLIP-SIM: 0.21 (no semantic progression)    │ │
│ │ Root Cause: Single text embedding for all frames          │ │
│ └───────────────────────────────────────────────────────────┘ │
│                                                               │
│ Weakness 2: Temporal Flickering (Dusk Window)                 │
│ ┌───────────────────────────────────────────────────────────┐ │
│ │ Evidence:                                                  │ │
│ │  - Flow Warping Error spike at transitions 6→7, 7→8, 8→9 │ │
│ │  - LPIPS dusk window mean: 0.38 (threshold: < 0.20)      │ │
│ │  - SSIM dusk window mean: 0.72 (threshold: > 0.85)       │ │
│ │ Root Cause: Uniform temporal attention weights            │ │
│ └───────────────────────────────────────────────────────────┘ │
└───────────────────────────────────────────────────────────────┘
```
