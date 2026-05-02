# Environment Setup — Lambda Labs (2× RTX A6000)

## System Check
```bash
nvidia-smi
# Expected: 2× RTX A6000, 48GB each, CUDA 12.1+
nvcc --version
df -h
```

## Conda Environment
```bash
conda create -n urban-metamorphosis python=3.10 -y
conda activate urban-metamorphosis

# PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Core Dependencies
```bash
# AnimateDiff / diffusers
pip install diffusers transformers accelerate safetensors

# Quantization + LoRA (if fine-tuning)
pip install bitsandbytes peft

# Video processing
pip install av ffmpeg-python opencv-python imageio imageio-ffmpeg

# Metrics
pip install lpips scikit-image torch-fidelity

# CLIP evaluation
pip install ftfy regex tqdm

# TTS
pip install git+https://github.com/huggingface/transformers.git
pip install pydub librosa soundfile

# Utility
pip install matplotlib seaborn pandas tqdm
```

## Clone & Setup AnimateDiff
```bash
git clone https://github.com/guoyww/AnimateDiff.git
cd AnimateDiff
pip install -r requirements.txt
```

## Download Models
```bash
mkdir -p models
huggingface-cli login  # (ensure access)

# SD 1.5 base
huggingface-cli download runwayml/stable-diffusion-v1-5 \
    --local-dir models/stable-diffusion-v1-5

# AnimateDiff motion module
huggingface-cli download guoyww/animatediff-motion-adapter-v1-5-2 \
    --local-dir models/motion-module

# (Optional for eval) RAFT optical flow
git clone https://github.com/princeton-vl/RAFT.git
```

## Verify Baseline Inference
```python
# test_baseline.py
import torch
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

adapter = MotionAdapter.from_pretrained("models/motion-module")
pipe = AnimateDiffPipeline.from_pretrained(
    "models/stable-diffusion-v1-5", motion_adapter=adapter
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

output = pipe(
    prompt="cinematic time-lapse of a city skyline transitioning from day to night",
    negative_prompt="static, flickering, low quality",
    num_frames=16,
    guidance_scale=7.5,
    num_inference_steps=25,
    generator=torch.manual_seed(42),
)
export_to_gif(output.frames[0], "outputs/test_baseline.gif")
print("✓ Baseline inference successful")
```

## Multi-GPU Inference
```python
# batch_infer.py
import torch, os
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

PROMPTS = [...]  # all 50 eval prompts
NUM_GPUS = torch.cuda.device_count()

def run_on_gpu(gpu_id, prompt_list):
    torch.cuda.set_device(gpu_id)
    adapter = MotionAdapter.from_pretrained("models/motion-module")
    pipe = AnimateDiffPipeline.from_pretrained(
        "models/stable-diffusion-v1-5", motion_adapter=adapter
    ).to(f"cuda:{gpu_id}")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    
    for i, prompt in enumerate(prompt_list):
        for seed in [42, 123, 999]:
            out = pipe(
                prompt=prompt, negative_prompt="static, flickering",
                num_frames=16, guidance_scale=7.5,
                num_inference_steps=25, generator=torch.manual_seed(seed),
            )
            os.makedirs(f"outputs/gpu{gpu_id}", exist_ok=True)
            export_to_gif(out.frames[0], f"outputs/gpu{gpu_id}/prompt{i}_seed{seed}.gif")

# Distribute prompts across GPUs
split = [PROMPTS[i::NUM_GPUS] for i in range(NUM_GPUS)]
import multiprocessing as mp
procs = [mp.Process(target=run_on_gpu, args=(i, split[i])) for i in range(NUM_GPUS)]
for p in procs: p.start()
for p in procs: p.join()
```

## Project Structure
```
project-urban-metamorphosis/
├── modeling/
│   ├── prompt_interpolation.py   # Enhancement A
│   └── attention_bias.py         # Enhancement C
├── training/
│   └── train_smoothness.py       # Enhancement B (fine-tuning)
├── inference/
│   └── inference_enhanced.py     # A + B + C pipeline
├── tts/
│   └── tts_pipeline.py           # Qwen-TTS pipeline
├── eval/
│   └── eval.py                   # All metrics
├── data/
│   └── eval_prompts.json         # 50 prompts
├── outputs/
├── models/
├── README.md
└── requirements.txt
```

## Troubleshooting
| Issue | Solution |
|-------|----------|
| CUDA OOM | Enable `pipe.enable_model_cpu_offload()`, reduce batch |
| Motion module not found | Check HuggingFace cache, use absolute path |
| Slow generation | Use `torch.compile()` or reduce to 20 steps |
| Multi-GPU race conditions | Use process-level isolation (not threads) |
