#!/usr/bin/env python3
"""
Environment verification script.
Checks GPU, CUDA, Python, disk space, and key Python packages.

Usage:
    python scripts/env_check.py
"""

import importlib
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as e:
        return f"ERROR: {e}"


def check_gpu():
    print("=" * 60)
    print("GPU / CUDA Check")
    print("=" * 60)

    nvidia = run(["nvidia-smi"])
    if "ERROR" in nvidia:
        print("  nvidia-smi: NOT FOUND")
    else:
        print("  nvidia-smi:\n" + "\n".join(f"    {l}" for l in nvidia.splitlines()[:12]))

    nvcc = run(["nvcc", "--version"])
    if "ERROR" in nvcc:
        print("  nvcc: NOT FOUND")
    else:
        print(f"  nvcc:\n    {nvcc.splitlines()[3] if len(nvcc.splitlines()) > 3 else nvcc}")

    # PyTorch CUDA
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version (PyTorch): {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                total = props.total_memory / 1024**3
                print(f"      Total memory: {total:.1f} GB")
    except ImportError:
        print("  torch: NOT INSTALLED")


def check_python():
    print("\n" + "=" * 60)
    print("Python Check")
    print("=" * 60)
    print(f"  Python executable: {sys.executable}")
    print(f"  Python version: {sys.version.split()[0]}")


def check_disk():
    print("\n" + "=" * 60)
    print("Disk Space Check")
    print("=" * 60)
    total, used, free = shutil.disk_usage(".")
    print(f"  Total: {total / (1024**3):.1f} GB")
    print(f"  Used:  {used  / (1024**3):.1f} GB")
    print(f"  Free:  {free  / (1024**3):.1f} GB")


def check_packages():
    print("\n" + "=" * 60)
    print("Python Packages Check")
    print("=" * 60)

    packages = [
        "torch", "torchvision", "torchaudio",
        "diffusers", "transformers", "accelerate",
        "safetensors", "lpips", "skimage",
        "cv2", "imageio", "librosa",
        "soundfile", "pandas", "matplotlib",
        "yaml",
    ]

    for pkg in packages:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {pkg:<20} {version}")
        except Exception as e:
            print(f"  {pkg:<20} ERROR: {type(e).__name__}")


def check_models():
    print("\n" + "=" * 60)
    print("Model Cache Check")
    print("=" * 60)
    models_dir = Path("models")
    if not models_dir.exists():
        print("  models/ directory: NOT FOUND")
        return

    for sub in models_dir.iterdir():
        if sub.is_dir():
            size = sum(f.stat().st_size for f in sub.rglob("*") if f.is_file()) / (1024**3)
            print(f"  {sub.name:<35} {size:.2f} GB")


def main():
    check_python()
    check_gpu()
    check_disk()
    check_packages()
    check_models()
    print("\n" + "=" * 60)
    print("Environment check complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
