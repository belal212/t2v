#!/usr/bin/env python3
"""
Comprehensive Test Suite — Urban Metamorphosis

Run this on the server after setting up the environment to verify
all modules, scripts, and logic work correctly.

Usage:
    python scripts/run_tests.py
"""

import importlib
import subprocess
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_syntax(files: list[Path]) -> list[str]:
    """Syntax-check all Python files."""
    results = []
    for f in sorted(files):
        try:
            import py_compile
            py_compile.compile(str(f), doraise=True)
            results.append(f"PASS  syntax  {f.relative_to(Path.cwd())}")
        except py_compile.PyCompileError as e:
            results.append(f"FAIL  syntax  {f}: {e}")
    return results


def test_imports(modules: list[tuple[str, list[str]]]) -> list[str]:
    """Test importing all modules and checking exported names."""
    results = []
    for mod_name, names in modules:
        try:
            mod = importlib.import_module(mod_name)
            missing = [n for n in names if not hasattr(mod, n)]
            if missing:
                results.append(f"FAIL  import  {mod_name}: missing {missing}")
            else:
                results.append(f"PASS  import  {mod_name}")
        except Exception as e:
            results.append(f"FAIL  import  {mod_name}: {type(e).__name__}: {e}")
    return results


def test_cli(scripts: list[tuple[str, str]]) -> list[str]:
    """Test --help on all executable scripts."""
    results = []
    cwd = Path(__file__).parent.parent
    for script, keyword in scripts:
        try:
            proc = subprocess.run(
                [sys.executable, script, "--help"],
                capture_output=True, text=True, timeout=15, cwd=cwd
            )
            if proc.returncode == 0 and keyword.lower() in proc.stdout.lower():
                results.append(f"PASS  cli     {script}")
            else:
                err = proc.stderr[:200].replace("\n", " ")
                results.append(f"FAIL  cli     {script}: rc={proc.returncode} err={err}")
        except Exception as e:
            results.append(f"FAIL  cli     {script}: {e}")
    return results


def test_logic() -> list[str]:
    """Run unit tests on core math/modeling functions."""
    results = []

    # Noise schedules
    try:
        import torch
        from modeling.noise_schedule import LinearSchedule, CosineSchedule, get_schedule
        lin = LinearSchedule(1000)
        cos = CosineSchedule(1000)
        assert lin.beta(torch.tensor(0)).item() < lin.beta(torch.tensor(999)).item()
        assert lin.alpha_bar(torch.tensor(0)).item() > lin.alpha_bar(torch.tensor(500)).item()
        assert cos.alpha_bar(torch.tensor(0)).item() > cos.alpha_bar(torch.tensor(500)).item()
        x0 = torch.randn(2, 4, 64, 64)
        xt, noise = lin.forward_process(x0, torch.tensor([100, 500]))
        assert xt.shape == x0.shape
        assert torch.isfinite(xt).all()
        results.append("PASS  logic   noise_schedules")
    except Exception as e:
        results.append(f"FAIL  logic   noise_schedules: {e}")

    # Diffusion utils
    try:
        from modeling.diffusion_utils import apply_cfg, ddim_step, predict_x0_from_noise, score_from_noise
        sched = LinearSchedule(1000)
        x_t = torch.randn(1, 4, 64, 64)
        noise_pred = torch.randn(1, 4, 64, 64)
        x_t1 = ddim_step(sched, x_t, noise_pred, torch.tensor([500]))
        assert x_t1.shape == x_t.shape
        n_g = apply_cfg(torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64), 7.5)
        assert n_g.shape == (2, 4, 64, 64)
        results.append("PASS  logic   diffusion_utils")
    except Exception as e:
        results.append(f"FAIL  logic   diffusion_utils: {e}")

    # Losses
    try:
        from modeling.losses import EpsilonLoss, TemporalSmoothnessLoss, CombinedLoss, VPredictionLoss
        assert EpsilonLoss()(torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64)).item() >= 0
        assert TemporalSmoothnessLoss()(torch.randn(16, 4, 64, 64)).item() >= 0
        v_targ = VPredictionLoss.compute_target(
            torch.randn(1, 4, 64, 64), torch.randn(1, 4, 64, 64), torch.tensor([0.5])
        )
        assert VPredictionLoss()(torch.randn_like(v_targ), v_targ).item() >= 0
        r = CombinedLoss()(
            torch.randn(2, 4, 64, 64), torch.randn(2, 4, 64, 64),
            latents=torch.randn(16, 4, 64, 64)
        )
        assert "loss" in r and r["loss"].item() >= 0
        results.append("PASS  logic   losses")
    except Exception as e:
        results.append(f"FAIL  logic   losses: {e}")

    # Prompt interpolation
    try:
        from modeling.prompt_interpolation import sigmoid_schedule, interpolate_prompt_embeddings
        assert abs(sigmoid_schedule(0.0) - 0.0474) < 0.01
        assert abs(sigmoid_schedule(0.5) - 0.5) < 0.01
        assert abs(sigmoid_schedule(1.0) - 0.9526) < 0.01

        class MockPipe:
            device = "cpu"
            class Tok:
                def __call__(self, text, **kwargs):
                    class O: input_ids = torch.randint(0, 1000, (1, 77))
                    return O()
            tokenizer = Tok()
            class TE:
                def __call__(self, input_ids):
                    return (torch.randn(1, 77, 768),)
            text_encoder = TE()

        emb = interpolate_prompt_embeddings(MockPipe(), "day", "night", 16)
        assert emb.shape == torch.Size([16, 77, 768])
        results.append("PASS  logic   prompt_interpolation")
    except Exception as e:
        results.append(f"FAIL  logic   prompt_interpolation: {e}")

    # Attention bias
    try:
        from modeling.attention_bias import TemporalBias
        bias = TemporalBias(16)
        attn = torch.randn(2, 8, 16, 16)
        out = bias(attn, t=500, T=1000)
        assert out.shape == attn.shape
        expected = attn + 0.5 * bias.bias.unsqueeze(0).unsqueeze(0)
        assert torch.allclose(out, expected)
        results.append("PASS  logic   attention_bias")
    except Exception as e:
        results.append(f"FAIL  logic   attention_bias: {e}")

    # Position encodings
    try:
        from modeling.position_encoding import SinusoidalEncoding, TimestepEncoding, FramePositionEncoding, get_timestep_embedding
        pe = SinusoidalEncoding(1000, 320)
        emb = pe(torch.tensor([0, 100, 500, 999]))
        assert emb.shape == torch.Size([4, 320])
        assert not torch.allclose(emb[0], emb[3])
        te = TimestepEncoding(1000, 320)
        fe = FramePositionEncoding(16, 320)
        assert te(torch.tensor([500])).shape == torch.Size([1, 320])
        assert fe(torch.arange(16)).shape == torch.Size([16, 320])
        results.append("PASS  logic   position_encoding")
    except Exception as e:
        results.append(f"FAIL  logic   position_encoding: {e}")

    # VAE utils
    try:
        from modeling.vae_utils import VAEProcessor
        class MockDist:
            def sample(self): return torch.randn(2, 4, 64, 64)
        class MockOut:
            latent_dist = MockDist()
        class MockDecodeOut:
            sample = torch.randn(2, 3, 512, 512)
        class MockVAE:
            class Config:
                scaling_factor = 0.18215
            config = Config()
            def encode(self, x): return MockOut()
            def decode(self, z): return MockDecodeOut()
        proc = VAEProcessor(MockVAE())
        latents = proc.encode_frames(torch.rand(2, 3, 512, 512))
        assert latents.shape == torch.Size([2, 4, 64, 64])
        decoded = proc.decode_latents(latents)
        assert decoded.shape == torch.Size([2, 3, 512, 512])
        assert proc.spatial_compression == 8
        results.append("PASS  logic   vae_utils")
    except Exception as e:
        results.append(f"FAIL  logic   vae_utils: {e}")

    # Denoising loop tracer
    try:
        from modeling.denoising_loop import DenoisingLoopTracer
        tracer = DenoisingLoopTracer(num_frames=16, num_timesteps=25)
        trace = tracer.trace_step(24)
        assert trace["z_t_shape"] == [16, 4, 64, 64], f"z_t_shape={trace['z_t_shape']}"
        assert len(trace["blocks"]) > 0, f"blocks len={len(trace['blocks'])}"
        assert trace["blocks"][0]["name"] == "Down 1", f"first block={trace['blocks'][0]['name']}"
        full = tracer.trace_full()
        assert len(full) == 25, f"full len={len(full)}"
        results.append("PASS  logic   denoising_loop")
    except Exception as e:
        results.append(f"FAIL  logic   denoising_loop: {e}")

    # TTS pipeline
    try:
        from tts.tts_pipeline import select_voice, parse_emotion, insert_pause_tokens, generate_narration_script
        assert select_voice(0, 16)["style"] == "warm"
        assert select_voice(15, 16)["style"] == "mystery"
        assert parse_emotion("bustling city")["energy"] == "high"
        assert "<|pause|>" in insert_pause_tokens("Hello, world.")
        segs = generate_narration_script("city skyline neon", 16)
        assert len(segs) >= 4
        results.append("PASS  logic   tts_pipeline")
    except Exception as e:
        results.append(f"FAIL  logic   tts_pipeline: {e}")

    return results


def main():
    project_root = Path(__file__).parent.parent
    py_files = list(project_root.rglob("*.py"))
    py_files = [f for f in py_files if ".git" not in str(f) and "PLAN" not in str(f)]

    modules = [
        ("modeling.noise_schedule", ["LinearSchedule", "CosineSchedule", "get_schedule"]),
        ("modeling.diffusion_utils", ["ddim_step", "apply_cfg", "predict_x0_from_noise"]),
        ("modeling.losses", ["EpsilonLoss", "TemporalSmoothnessLoss", "CombinedLoss"]),
        ("modeling.prompt_interpolation", ["interpolate_prompt_embeddings", "sigmoid_schedule"]),
        ("modeling.attention_bias", ["TemporalBias", "BiasAttentionProcessor"]),
        ("modeling.position_encoding", ["SinusoidalEncoding", "TimestepEncoding", "FramePositionEncoding"]),
        ("modeling.vae_utils", ["VAEProcessor", "create_vae_processor"]),
        ("modeling.denoising_loop", ["DenoisingLoopTracer"]),
        ("eval.weakness_analysis", ["load_frames", "compute_flow_warping_error"]),
        ("eval.run_weakness_analysis", ["compute_metrics"]),
        ("tts.tts_pipeline", ["QwenTTSPipeline", "select_voice", "parse_emotion"]),
    ]

    scripts = [
        ("scripts/test_baseline.py", "Baseline"),
        ("scripts/batch_infer.py", "Multi-GPU"),
        ("scripts/download_models_cli.py", "HuggingFace"),
        ("scripts/env_check.py", "Environment"),
        ("scripts/inspect_architecture.py", "Inspect"),
        ("inference/inference_enhanced.py", "Enhanced"),
        ("training/train_smoothness.py", "Fine-tune"),
        ("eval/eval.py", "evaluation"),
        ("eval/run_ablation.py", "ablation"),
        ("eval/weakness_analysis.py", "Weakness"),
        ("eval/run_weakness_analysis.py", "Weakness"),
        ("eval/compute_fvd.py", "FVD"),
        ("eval/visualize.py", "Ablation"),
        ("tts/tts_pipeline.py", "Qwen"),
        ("modeling/denoising_loop.py", "Denoising"),
    ]

    print("=" * 60)
    print("Urban Metamorphosis — Comprehensive Test Suite")
    print("=" * 60)

    all_results = []
    all_results.extend(test_syntax(py_files))
    all_results.extend(test_imports(modules))
    all_results.extend(test_logic())
    all_results.extend(test_cli(scripts))

    print("\n".join(all_results))

    passed = sum(1 for r in all_results if r.startswith("PASS"))
    failed = sum(1 for r in all_results if r.startswith("FAIL"))

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(all_results)}")
    print("=" * 60)

    if failed > 0:
        print("\nFailed tests:")
        for r in all_results:
            if r.startswith("FAIL"):
                print(f"  {r}")
        sys.exit(1)
    else:
        print("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
