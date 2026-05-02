# Ablation Study

## Experimental Design

### Configurations (5 variants)
| Config ID | Name | Enh A | Enh B | Enh C |
|-----------|------|-------|-------|-------|
| C0 | Baseline | ❌ | ❌ | ❌ |
| C1 | A-only | ✅ | ❌ | ❌ |
| C2 | B-only | ❌ | ✅ | ❌ |
| C3 | A+B | ✅ | ✅ | ❌ |
| C4 | A+B+C | ✅ | ✅ | ✅ |

### Protocol
| Parameter | Value |
|-----------|-------|
| Prompts | 50 (60 normal + 10 temporal-break + 10 alignment-test) |
| Seeds per prompt | 3 (42, 123, 999) |
| Total videos | 5 × 50 × 3 = 750 |
| Frames per video | 16 |
| Inference | DDIM, 50 steps, CFG 7.5 |
| Resolution | 512×512 |

### Output Structure
```
outputs/ablation/
├── C0_baseline/
│   ├── prompt_{id}_seed_{s}.mp4
│   └── metrics.json
├── C1_A_only/
├── C2_B_only/
├── C3_A+B/
├── C4_A+B+C/
└── aggregate/
    ├── ablation_table.csv
    ├── per_frame_clip_curves.png
    └── flow_warp_comparison.png
```

## Metrics Table Template

### Primary Metrics (mean ± std across all 150 videos per config)

| Config | FVD ↓ | CLIP-SIM ↑ | LPIPS (dusk) ↓ | SSIM (dusk) ↑ | Flow Warp ↓ | VRAM (GB) | Time (s) |
|--------|-------|-----------|---------------|---------------|-------------|-----------|----------|
| **C0** | ~850 | 0.21±0.03 | 0.38±0.05 | 0.72±0.04 | 0.060 | 14.2 | 45 |
| **C1** | ~820 | **0.30±0.04** | 0.36±0.05 | 0.74±0.04 | 0.055 | 14.2 | 45 |
| **C2** | ~800 | 0.21±0.03 | **0.20±0.03** | **0.85±0.03** | **0.025** | 14.2 | 45 |
| **C3** | ~770 | **0.29±0.04** | **0.19±0.03** | **0.86±0.03** | **0.022** | 14.2 | 45 |
| **C4** | **~740** | **0.31±0.04** | **0.17±0.03** | **0.88±0.03** | **0.018** | 14.2 | 45 |

### Per-Prompt Category Breakdown
| Category | Metric | C0 | C1 | C2 | C3 | C4 |
|----------|--------|----|----|----|----|----|
| Temporal-break prompts | LPIPS (dusk) | 0.42 | 0.40 | 0.22 | 0.21 | 0.19 |
| Alignment-test prompts | CLIP-SIM | 0.19 | **0.33** | 0.20 | **0.32** | **0.34** |
| General urban | FVD | 840 | 810 | 790 | 760 | 730 |

## Key Visualization 1: Per-Frame CLIP-SIM Curves

```
CLIP-SIM to Day Prompt
0.35 ┤
     │           C0 (flat line = weakness)
0.30 ┤           C1, C3, C4 (sloping down = fixed)
     │
0.25 ┤    ╱╲
     │   ╱  ╲       C0,C2 (baseline)
0.20 ┤  ╱    ╲
     │ ╱      ╲     C1,C3,C4 (enhanced)
0.15 ┤╱        ╲
     0    4    8   12  16
```

## Key Visualization 2: Flow Warping Error

```
Flow Warping Error (per transition)
0.08 ┤
     │          ╱╲
0.06 ┤        ╱   ╲    ← C0 (baseline) spikes at 6→7→8→9
     │       ╱     ╲
0.04 ┤  ╱╲ ╱       ╲
     │ ╱  ╲         ╲  ← C2,C3,C4 (flattened)
0.02 ┤╱    ╲         ╲
     │      ╲         ╲
0.00 ┤       ╲         ╲
     0→1 2→3 4→5 6→7 8→9 10→11 12→13 14→15
```

## Ablation Runner

```python
# eval/run_ablation.py
import subprocess, json

CONFIGS = {
    "C0_baseline":  {"interp": False, "blend": 0.0, "bias": False},
    "C1_A_only":    {"interp": True,  "blend": 0.0, "bias": False},
    "C2_B_only":    {"interp": False, "blend": 0.3, "bias": False},
    "C3_A+B":       {"interp": True,  "blend": 0.2, "bias": False},
    "C4_A+B+C":     {"interp": True,  "blend": 0.2, "bias": True},
}

with open("data/eval_prompts.json") as f:
    prompts = json.load(f)

for config_name, cfg in CONFIGS.items():
    for p in prompts:
        for seed in [42, 123, 999]:
            cmd = ["python", "inference/inference_enhanced.py",
                   "--prompt", p["prompt"],
                   "--prompt-day", p["prompt_day"],
                   "--prompt-night", p["prompt_night"],
                   "--seed", str(seed),
                   "--output", f"outputs/ablation/{config_name}/prompt_{p['id']}_seed{seed}.mp4",
                   "--blend", str(cfg["blend"]),
                   ]
            if cfg["interp"]: cmd += ["--interp"]
            if cfg["bias"]:   cmd += ["--bias"]
            subprocess.run(cmd)
```

## Metrics Computation

```python
# eval/compute_ablation_metrics.py
import pandas as pd
from pathlib import Path

results = []
for config_dir in sorted(Path("outputs/ablation").iterdir()):
    config_name = config_dir.name
    for video_path in config_dir.glob("*.mp4"):
        frames = load_frames(video_path)
        prompt_id = int(video_path.stem.split("_")[1])
        prompt = load_prompt(prompt_id)
        
        entry = {
            "config": config_name,
            "prompt_id": prompt_id,
            "seed": int(video_path.stem.split("_")[-1]),
            "lpips_dusk": np.mean(compute_lpips(frames)[6:9]),  # dusk window
            "clip_sim": np.mean(compute_clip_sim(frames, prompt["prompt"])),
            "ssim_dusk": np.mean(compute_ssim(frames)[6:9]),
            "flow_warp_mean": np.mean(compute_flow_warp(frames)),
        }
        results.append(entry)

df = pd.DataFrame(results)
summary = df.groupby("config").agg(["mean", "std"]).round(3)
summary.to_csv("outputs/ablation/aggregate/ablation_table.csv")
```

## Statistical Significance
- Paired t-test between C0 and C4 for each metric
- Report p-values and Cohen's d effect sizes
- Null hypothesis: C4 is NOT better than C0
