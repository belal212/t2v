#!/usr/bin/env python3
"""
Run plan-aligned tasks with logging and status updates.

This runner is designed for long jobs and resumability. It writes:
- reports/logs/<step>.log
- reports/run_status.json
- reports/run_status.md

Usage examples:
  python scripts/run_plan.py --steps env_check,baseline_smoke,enhanced_smoke
  python scripts/run_plan.py --steps baseline_batch,weakness_analysis \
      --prompt-start 0 --prompt-count 10
  python scripts/run_plan.py --steps ablation_generate,ablation_eval,ablation_plots \
      --configs C0_baseline C4_A_B_C
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports"
LOGS_DIR = REPORTS_DIR / "logs"
STATUS_JSON = REPORTS_DIR / "run_status.json"
STATUS_MD = REPORTS_DIR / "run_status.md"

DEFAULT_PROMPT = "cinematic time-lapse of a city skyline transitioning from day to night"
DEFAULT_PROMPT_DAY = "bright sunny day in a city, clear blue sky"
DEFAULT_PROMPT_NIGHT = "dark night city skyline, neon lights, starry sky"


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_status() -> dict:
    if STATUS_JSON.exists():
        return json.loads(STATUS_JSON.read_text())
    return {}


def _write_status(status: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_JSON.write_text(json.dumps(status, indent=2))

    lines = ["# Run Status", "", "| Step | Status | Last Run | Details |", "|---|---|---|---|"]
    for step, info in sorted(status.items()):
        lines.append(
            f"| {step} | {info.get('status')} | {info.get('last_run')} | {info.get('details')} |"
        )
    STATUS_MD.write_text("\n".join(lines) + "\n")


def _update_status(step: str, status: str, details: str) -> None:
    data = _load_status()
    data[step] = {
        "status": status,
        "last_run": _now(),
        "details": details,
    }
    _write_status(data)


def _log_header(log_path: Path, cmd: Iterable[str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[{_now()}] $ {' '.join(cmd)}\n")
        f.write("=" * 80 + "\n")


def _run_step(step: str, cmd: list[str], dry_run: bool) -> bool:
    if dry_run:
        print(f"[DRY RUN] {step}: {' '.join(cmd)}")
        return True

    log_path = LOGS_DIR / f"{step}.log"
    _log_header(log_path, cmd)

    with log_path.open("a", encoding="utf-8") as f:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if result.returncode == 0:
        _update_status(step, "ok", f"log={log_path}")
        return True

    _update_status(step, "fail", f"log={log_path} (exit={result.returncode})")
    return False


def _subset_prompts(prompts_path: str, start: int | None, count: int | None) -> Path:
    if start is None and count is None:
        return Path(prompts_path)

    prompts = json.loads(Path(prompts_path).read_text())
    s = start or 0
    if count is None:
        subset = prompts[s:]
    else:
        subset = prompts[s : s + count]

    out_path = REPORTS_DIR / f"prompts_subset_{s}_{len(subset)}.json"
    out_path.write_text(json.dumps(subset, indent=2))
    return out_path


def _parse_steps(steps_arg: str) -> list[str]:
    steps = [s.strip() for s in steps_arg.split(",") if s.strip()]
    if len(steps) == 1 and steps[0] == "all":
        return [
            "env_check",
            "baseline_smoke",
            "enhanced_smoke",
            "baseline_batch",
            "weakness_analysis",
            "ablation_generate",
            "ablation_eval",
            "ablation_plots",
            "tts_narration",
        ]
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run plan-aligned tasks with logging")
    parser.add_argument("--steps", default="env_check,baseline_smoke,enhanced_smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--prompt-start", type=int, default=None)
    parser.add_argument("--prompt-count", type=int, default=None)
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--vae-slicing", action="store_true")
    parser.add_argument("--weakness-inputs", default="outputs/baseline")
    parser.add_argument("--tts-video", default="outputs/enhanced_test.gif")
    parser.add_argument("--tts-prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    steps = _parse_steps(args.steps)
    prompts_path = _subset_prompts("data/eval_prompts.json", args.prompt_start, args.prompt_count)

    for step in steps:
        if step == "env_check":
            cmd = [sys.executable, "scripts/env_check.py"]
        elif step == "baseline_smoke":
            cmd = [
                sys.executable, "scripts/test_baseline.py",
                "--output", "outputs/test_baseline.gif",
                "--num-frames", str(args.num_frames),
                "--num-inference-steps", str(args.num_inference_steps),
                "--guidance-scale", str(args.guidance_scale),
            ]
            if args.cpu_offload:
                cmd.append("--cpu-offload")
            if args.vae_slicing:
                cmd.append("--vae-slicing")
        elif step == "enhanced_smoke":
            cmd = [
                sys.executable, "inference/inference_enhanced.py",
                "--config", "C4_A_B_C",
                "--prompt", DEFAULT_PROMPT,
                "--prompt-day", DEFAULT_PROMPT_DAY,
                "--prompt-night", DEFAULT_PROMPT_NIGHT,
                "--num-frames", str(args.num_frames),
                "--num-inference-steps", str(args.num_inference_steps),
                "--guidance-scale", str(args.guidance_scale),
                "--output", "outputs/enhanced_test.gif",
                "--device", args.device,
            ]
            if args.cpu_offload:
                cmd.append("--cpu-offload")
        elif step == "baseline_batch":
            cmd = [
                sys.executable, "scripts/batch_infer.py",
                "--prompts", str(prompts_path),
                "--output-dir", "outputs/baseline",
                "--num-frames", str(args.num_frames),
                "--num-inference-steps", str(args.num_inference_steps),
                "--guidance-scale", str(args.guidance_scale),
            ]
            if args.num_gpus is not None:
                cmd.extend(["--num-gpus", str(args.num_gpus)])
        elif step == "weakness_analysis":
            cmd = [
                sys.executable, "eval/run_weakness_batch.py",
                "--inputs", args.weakness_inputs,
                "--prompts", str(prompts_path),
                "--output", "outputs/baseline/metrics",
                "--device", args.device,
            ]
        elif step == "ablation_generate":
            cmd = [
                sys.executable, "eval/run_ablation.py",
                "--prompts", str(prompts_path),
                "--output-dir", "outputs/ablation",
                "--num-frames", str(args.num_frames),
                "--num-inference-steps", str(args.num_inference_steps),
                "--guidance-scale", str(args.guidance_scale),
            ]
            if args.configs:
                cmd.extend(["--configs", *args.configs])
            if args.num_gpus is not None:
                cmd.extend(["--num-gpus", str(args.num_gpus)])
        elif step == "ablation_eval":
            cmd = [
                sys.executable, "eval/eval.py",
                "--ablation-dir", "outputs/ablation",
                "--prompts", str(prompts_path),
                "--output", "outputs/metrics",
                "--device", args.device,
            ]
        elif step == "ablation_plots":
            cmd = [
                sys.executable, "scripts/plot_ablation_metrics.py",
                "--results", "outputs/metrics/all_results.csv",
                "--output", "reports/plots",
            ]
        elif step == "tts_narration":
            cmd = [
                sys.executable, "tts/tts_pipeline.py",
                "--video", args.tts_video,
                "--prompt", args.tts_prompt,
                "--output", "outputs/final_with_audio.mp4",
            ]
        else:
            print(f"Unknown step: {step}")
            continue

        ok = _run_step(step, cmd, args.dry_run)
        if not ok:
            print(f"Step failed: {step}")
            break

    print("Done. Status written to reports/run_status.md")


if __name__ == "__main__":
    main()
