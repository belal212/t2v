#!/usr/bin/env bash
# Environment setup script for Urban Metamorphosis.
# Creates conda env, installs PyTorch (CUDA 12.1), and all dependencies.
#
# Usage:
#   bash scripts/setup_env.sh

set -e

ENV_NAME="urban-metamorphosis"
PYTHON_VERSION="3.10"

echo "========================================"
echo "Urban Metamorphosis — Environment Setup"
echo "========================================"

# Check conda
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found. Please install Miniconda/Anaconda first."
    exit 1
fi

# Create or update conda environment
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Conda env '${ENV_NAME}' already exists. Updating..."
else
    echo "Creating conda env '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo ""
echo "Installing PyTorch with CUDA 12.1..."
conda run -n "${ENV_NAME}" pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing core ML dependencies..."
conda run -n "${ENV_NAME}" pip install diffusers transformers accelerate safetensors

echo ""
echo "Installing quantization + LoRA tools..."
conda run -n "${ENV_NAME}" pip install bitsandbytes peft

echo ""
echo "Installing video processing libraries..."
conda run -n "${ENV_NAME}" pip install av ffmpeg-python opencv-python imageio imageio-ffmpeg

echo ""
echo "Installing metrics libraries..."
conda run -n "${ENV_NAME}" pip install lpips scikit-image torch-fidelity

echo ""
echo "Installing CLIP evaluation dependencies..."
conda run -n "${ENV_NAME}" pip install ftfy regex tqdm

echo ""
echo "Installing TTS / audio dependencies..."
conda run -n "${ENV_NAME}" pip install pydub librosa soundfile

echo ""
echo "Installing utility libraries..."
conda run -n "${ENV_NAME}" pip install matplotlib seaborn pandas tqdm pyyaml

echo ""
echo "========================================"
echo "Setup complete! Activate with:"
echo "  conda activate ${ENV_NAME}"
echo "Then run baseline test:"
echo "  python scripts/test_baseline.py"
echo "========================================"
