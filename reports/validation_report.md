# Urban Metamorphosis Validation Report (May 3, 2026)

## Pass/Fail Summary

| Step | Task | Status | Notes |
|---|---|---|---|
| 1 | Reconfigure Python + install deps | PASS | CUDA-enabled PyTorch 2.5.1+cu121; requirements installed; `env_check.py` clean |
| 2 | Download pretrained models | PASS | `stable-diffusion-v1-5`, `motion-module`, `clip-vit-large-patch14` present |
| 3 | Run full test suite | PASS | 65/65 tests passed |
| 4 | Baseline inference | PASS | 16 frames, 25 steps, output `outputs/test.gif` |
| 5 | Enhanced pipeline (C4) | PASS | 16 frames, 25 steps on `cuda:0` produced `outputs/enhanced_test.gif` |

## Environment Status

- Python: 3.12.13 (venv)
- PyTorch: 2.5.1+cu121
- CUDA available: True (driver 535.288.01; PyTorch CUDA 12.1)
- GPU count: 8x RTX A6000

## Model Cache (On Disk)

- models/stable-diffusion-v1-5: 45G
- models/motion-module: 3.4G
- models/clip-vit-large-patch14: 6.4G

## Performance Notes

- Baseline inference: 16 frames, 25 steps completed successfully.
- Enhanced inference: succeeded on GPU 0 with 16 frames / 25 steps (C4 A+B+C).

## Issues / Warnings

- GPU 0 was heavily utilized; enhanced inference OOM at default 16 frames on GPU 0.
- `xformers` not installed; script continues without memory-efficient attention.
- `enable_vae_slicing` deprecation warning from diffusers 0.38.0.

## Code Changes

- Enhanced inference fixes for diffusers API, device handling, scheduler setup, and VAE decoding.
