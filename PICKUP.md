# PICKUP

## Current Status
- Ablation generation completed (outputs/ablation).
- Ablation evaluation fails due to CUDA init errors for user "honeypot" (torch reports no CUDA devices).
- Ablation plots and TTS narration are pending because ablation eval fails.
- Baseline batch and weakness analysis completed.

## Blockers
- CUDA init fails inside Python for honeypot (NVML init error, torch.cuda.is_available() = False).
- Stuck honeypot GPU processes were observed (PIDs 28198, 28200) and could not be killed without admin help.
- Two GPUs on this host are known bad (NVML "Unknown Error" on bus IDs C3/C4).

## Resume Commands
- Preferred GPU path (after CUDA is working):
  CUDA_VISIBLE_DEVICES=2,3 /home/honeypot/.venv/bin/python scripts/run_plan.py --steps ablation_eval,ablation_plots,tts_narration --device cuda:0 --num-gpus 2
- CPU fallback (slow but should work):
  /home/honeypot/.venv/bin/python scripts/run_plan.py --steps ablation_eval,ablation_plots,tts_narration --device cpu

## Quick Diagnostics
- CUDA availability:
  /home/honeypot/.venv/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
- GPU usage by user:
  nvidia-smi --query-compute-apps=pid,used_memory,gpu_name --format=csv

## Notes
- Do not commit generated artifacts or weights (models/, outputs/, reports/logs/, reports/run_status.*).
