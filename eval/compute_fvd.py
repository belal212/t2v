#!/usr/bin/env python3
"""
FVD & Inception Score — Distributional Video Quality Metrics

Computes Fréchet Video Distance (FVD) and Inception Score (IS)
using torch-fidelity and torchmetrics.

FVD measures the distributional distance between real and generated
videos in pretrained I3D feature space.

Usage:
    # FVD: compares generated vs real videos
    python eval/compute_fvd.py \
        --generated outputs/ablation/C4_A_B_C \
        --reference data/real_day_to_night \
        --output outputs/metrics/fvd_results.json

    # IS: per-frame sharpness/diversity
    python eval/compute_fvd.py --is-only \
        --generated outputs/ablation/C0_baseline \
        --output outputs/metrics/is_results.json
"""

import argparse
import json
from pathlib import Path


def compute_fvd(generated_dir: str, reference_dir: str, output_path: str = None):
    """Compute FVD between generated and real videos using torch-fidelity."""
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print("ERROR: torch-fidelity not installed. Run: pip install torch-fidelity")
        return None

    print(f"Computing FVD...")
    print(f"  Generated: {generated_dir}")
    print(f"  Reference: {reference_dir}")

    metrics = calculate_metrics(
        input1=generated_dir,
        input2=reference_dir,
        cuda=True,
        batch_size=8,
        fvd=True,
        isc=False,
        verbose=True,
    )

    result = {
        "fvd": float(metrics["frechet_video_distance"]),
        "fvd_num_videos": metrics.get("fvd_num_input", None),
    }
    print(f"  FVD: {result['fvd']:.2f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to: {output_path}")

    return result


def compute_is(generated_dir: str, output_path: str = None):
    """Compute Inception Score on generated frames."""
    try:
        from torch_fidelity import calculate_metrics
    except ImportError:
        print("ERROR: torch-fidelity not installed. Run: pip install torch-fidelity")
        return None

    print(f"Computing Inception Score...")
    print(f"  Videos: {generated_dir}")

    metrics = calculate_metrics(
        input1=generated_dir,
        cuda=True,
        batch_size=8,
        fvd=False,
        isc=True,
        verbose=True,
    )

    result = {
        "inception_score_mean": float(metrics["inception_score_mean"]),
        "inception_score_std": float(metrics["inception_score_std"]),
    }
    print(f"  IS: {result['inception_score_mean']:.4f} ± {result['inception_score_std']:.4f}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved to: {output_path}")

    return result


def compute_all_ablation(ablation_dir: str, reference_dir: str, output_dir: str):
    """Compute FVD for all 5 ablation configs + IS for baseline."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    configs = ["C0_baseline", "C1_A_only", "C2_B_only", "C3_A_plus_B", "C4_A_B_C"]
    fvd_results = {}

    for config in configs:
        gen_dir = Path(ablation_dir) / config
        if not gen_dir.exists():
            print(f"  SKIP {config}: directory not found")
            continue

        result = compute_fvd(str(gen_dir), reference_dir)
        if result:
            fvd_results[config] = result["fvd"]

    # Save all FVD results
    summary = {
        "fvd_per_config": fvd_results,
        "best_config": min(fvd_results, key=fvd_results.get) if fvd_results else None,
    }

    summary_path = output_path / "fvd_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nFVD summary saved to: {summary_path}")
    for config, fvd in fvd_results.items():
        print(f"  {config}: FVD = {fvd:.1f}")

    # IS for baseline
    baseline_dir = Path(ablation_dir) / "C0_baseline"
    if baseline_dir.exists():
        compute_is(str(baseline_dir), str(output_path / "is_baseline.json"))


def main():
    parser = argparse.ArgumentParser(description="FVD and IS evaluation")
    parser.add_argument("--generated", help="Generated video directory")
    parser.add_argument("--reference", default="data/real_day_to_night", help="Real video reference directory")
    parser.add_argument("--ablation-dir", help="Ablation directory (computes all 5 configs)")
    parser.add_argument("--output", default="outputs/metrics/fvd_results.json")
    parser.add_argument("--is-only", action="store_true", help="Compute only Inception Score")
    parser.add_argument("--fvd-only", action="store_true", help="Compute only FVD")
    args = parser.parse_args()

    if args.ablation_dir:
        compute_all_ablation(args.ablation_dir, args.reference, Path(args.output).parent)
    elif args.is_only and args.generated:
        compute_is(args.generated, args.output)
    elif args.fvd_only and args.generated and args.reference:
        compute_fvd(args.generated, args.reference, args.output)
    elif args.generated and args.reference:
        compute_fvd(args.generated, args.reference, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
