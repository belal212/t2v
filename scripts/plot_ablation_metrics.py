#!/usr/bin/env python3
"""
Generate ablation summary plots from outputs/metrics/all_results.csv.

Usage:
    python scripts/plot_ablation_metrics.py \
        --results outputs/metrics/all_results.csv \
        --output reports/plots
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ablation metrics")
    parser.add_argument("--results", default="outputs/metrics/all_results.csv")
    parser.add_argument("--output", default="reports/plots")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        raise SystemExit(f"Missing results file: {results_path}")

    df = pd.read_csv(results_path)
    if df.empty:
        raise SystemExit("Results file is empty.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        "clip_mid_separation",
        "fwe_dusk",
        "lpips_dusk",
        "ssim_dusk",
    ]
    agg = df.groupby("config")[metrics].mean().reset_index()
    agg.to_csv(output_dir / "ablation_metrics_mean.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        ax.bar(agg["config"], agg[metric])
        ax.set_title(metric)
        ax.set_xticklabels(agg["config"], rotation=30, ha="right")

    fig.tight_layout()
    fig.savefig(output_dir / "ablation_metrics.png", dpi=150)
    plt.close(fig)

    # Clip slope plot
    slope_metrics = ["clip_day_slope", "clip_night_slope"]
    if all(m in df.columns for m in slope_metrics):
        slopes = df.groupby("config")[slope_metrics].mean().reset_index()
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        x = range(len(slopes["config"]))
        ax2.bar([i - 0.15 for i in x], slopes["clip_day_slope"], width=0.3, label="day_slope")
        ax2.bar([i + 0.15 for i in x], slopes["clip_night_slope"], width=0.3, label="night_slope")
        ax2.set_xticks(list(x))
        ax2.set_xticklabels(slopes["config"], rotation=30, ha="right")
        ax2.set_title("CLIP-SIM Slopes")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(output_dir / "clip_slopes.png", dpi=150)
        plt.close(fig2)

    print(f"Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
