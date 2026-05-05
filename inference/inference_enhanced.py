#!/usr/bin/env python3
"""
Enhanced Inference Pipeline — AnimateDiff with A + B + C

This is the CORE implementation of the project's three enhancements.
Uses a custom denoising loop to enable all three simultaneously.

Design decisions:
  - Enhancement A: Separate UNet forward passes for uncond (single neg emb)
                   and cond (per-frame day/night embeddings [N, 77, 768]).
                   The UNet's cross-attention naturally uses the N-dim batch.
  - Enhancement B: Latent-space blending applied after denoising, before VAE decode.
  - Enhancement C: Patches the motion module's temporal self-attention (attn1 in
                   AnimateDiffTransformer3D transformer blocks) to add γ(t)·B
                   to attention scores via manual attention computation.

Usage:
    python inference/inference_enhanced.py \
        --config C4_A_B_C \
        --prompt "cinematic time-lapse of a city skyline..." \
        --prompt-day "bright sunny day in a city" \
        --prompt-night "dark night city skyline, neon lights" \
        --output outputs/enhanced/test.gif

Configs:
    C0_baseline  → no enhancements
    C1_A_only    → Enhancement A (sigmoid prompt interpolation)
    C2_B_only    → Enhancement B (latent frame blending)
    C3_A_plus_B  → A + B
    C4_A_B_C     → A + B + C (full)
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
from diffusers.utils import export_to_gif

sys.path.insert(0, str(Path(__file__).parent.parent))

from modeling.prompt_interpolation import interpolate_prompt_embeddings, get_negative_embedding
from modeling.attention_bias import TemporalBias


# ─── Enhancement B: Latent-Space Frame Blending ───────────────────────────

def apply_frame_blend(latents: torch.Tensor, beta: float = 0.3) -> torch.Tensor:
    """
    Blends adjacent frame latents to smooth the dusk transition.

    z_blend^i   = (1-β)·z^i + β·z^{i+1}
    z_blend^{i+1} = β·z^i + (1-β)·z^{i+1}

    Args:
        latents: [4, N, H/8, W/8] latent tensor.
        beta: Blending coefficient.

    Returns:
        Blended latents of same shape.
    """
    blended = latents.clone()
    num_frames = latents.shape[1]
    for i in range(num_frames - 1):
        blended[:, i] = (1.0 - beta) * latents[:, i] + beta * latents[:, i + 1]
        blended[:, i + 1] = beta * latents[:, i] + (1.0 - beta) * latents[:, i + 1]
    return blended


# ─── Enhancement C: Temporal Bias Injection ────────────────────────────────

class _BiasTemporalAttnProcessor:
    """
    Attention processor for temporal self-attention in AnimateDiff motion modules.

    Computes: Softmax(QK^T / sqrt(d_k) + γ(t)·B) · V
    where B is the learned bias from TemporalBias and γ(t) = 1 - t/T ∈ [0,1].

    Uses manual attention (not F.scaled_dot_product_attention) to allow
    bias injection into the scores before softmax.
    """

    def __init__(self, bias_module: TemporalBias, t_source):
        self.bias_module = bias_module
        self._t_source = t_source

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        from diffusers.utils import deprecate
        if len(args) > 0 or (kwargs and kwargs.get("scale")):
            deprecate("scale", "1.0.0", "")

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            B, C, H, W = hidden_states.shape
            hidden_states = hidden_states.view(B, C, H * W).transpose(1, 2)

        batch_size, seq_len, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        Q = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            K = attn.to_k(hidden_states)
            V = attn.to_v(hidden_states)
        else:
            if attn.norm_cross:
                encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
            K = attn.to_k(encoder_hidden_states)
            V = attn.to_v(encoder_hidden_states)

        inner_dim = K.shape[-1]
        head_dim = inner_dim // attn.heads

        Q = Q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, h, L, d]
        K = K.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, h, S, d]
        V = V.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)  # [B, h, S, d]

        if attn.norm_q is not None: Q = attn.norm_q(Q)
        if attn.norm_k is not None: K = attn.norm_k(K)

        # ── Enhancement C: Add temporal bias to attention scores ──
        # Temporal self-attention operates on sequence of N frames.
        # When the key sequence length matches num_frames, we inject bias.
        N = self.bias_module.num_frames
        if K.shape[-2] == N and encoder_hidden_states is None:
            t = self._t_source.current_t
            T = self._t_source.max_T
            gamma = 1.0 - (t / max(T, 1))
            # QK^T / sqrt(d_k) + γ(t)·B
            scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
            scores = scores + gamma * self.bias_module.bias.to(
                device=scores.device,
                dtype=scores.dtype,
            )
            attn_weights = F.softmax(scores, dim=-1)
            hidden_states = torch.matmul(attn_weights, V)
        else:
            hidden_states = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(Q.dtype)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(B, C, H, W)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


class TemporalBiasInjector:
    """
    Injects Enhancement C bias into AnimateDiff motion modules.

    Finds all transformer blocks inside AnimateDiffTransformer3D modules
    and patches their attn1 (temporal self-attention) processor.
    """

    def __init__(self, unet, bias_module: TemporalBias):
        self.unet = unet
        self.bias_module = bias_module
        self.current_t = 999
        self.max_T = 1000
        self._hooks = []  # (module, original_processor)
        self._patch()

    def set_timestep(self, t, T=1000):
        self.current_t = t
        self.max_T = T

    def _patch(self):
        custom_proc = _BiasTemporalAttnProcessor(self.bias_module, self)

        for name, module in self.unet.named_modules():
            # Target: BasicTransformerBlock inside AnimateDiffTransformer3D
            # These are motion module transformer blocks
            if "transformer_block" in name.lower() and hasattr(module, "attn1"):
                # attn1 is temporal self-attention in AnimateDiff
                old = module.attn1.processor
                module.attn1.processor = custom_proc
                self._hooks.append((module.attn1, old))
                # Note: attn_temp is the spatial self-attention (double_self_attention)
                # We do NOT bias that — only temporal attn

    def remove(self):
        for attn_module, old_proc in self._hooks:
            attn_module.processor = old_proc
        self._hooks.clear()


# ─── Pipeline Setup ────────────────────────────────────────────────────────

def setup_pipeline(
    base_model: str,
    motion_module: str,
    device: str = "cuda",
    cpu_offload: bool = False,
):
    adapter = MotionAdapter.from_pretrained(motion_module, torch_dtype=torch.float16)
    pipe = AnimateDiffPipeline.from_pretrained(
        base_model,
        motion_adapter=adapter,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_slicing()
    if cpu_offload:
        print("[INFO] Enabling model CPU offload")
        gpu_id = 0
        if device.startswith("cuda:"):
            try:
                gpu_id = int(device.split(":", 1)[1])
            except ValueError:
                gpu_id = 0
        pipe.to("cpu")
        pipe.enable_model_cpu_offload(gpu_id=gpu_id)
    else:
        pipe = pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except ModuleNotFoundError:
        print("[WARN] xformers not installed; continuing without memory-efficient attention.")
    return pipe


# ─── Core Denoising ────────────────────────────────────────────────────────

def compute_noise_pred(
    pipe,
    latents: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float,
    uncond_emb: torch.Tensor,
    cond_emb: torch.Tensor,
    per_frame_emb: torch.Tensor | None,
    enh_a: bool,
    enh_c_injector: TemporalBiasInjector | None,
):
    """
    Compute CFG-guided noise prediction.
    Enhancement A: conditional forward uses per-frame embeddings [N, 77, 768].
    Enhancement C: injector timestep is updated for bias gamma.
    """
    latent_input = pipe.scheduler.scale_model_input(latents, t)

    if enh_c_injector is not None:
        enh_c_injector.set_timestep(t.item(), pipe.scheduler.config.num_train_timesteps)

    # Unconditional: same negative embedding for all frames
    noise_uncond = pipe.unet(
        latent_input, t,
        encoder_hidden_states=uncond_emb,
    ).sample

    # Conditional: per-frame embeddings or single cond embedding
    if enh_a and per_frame_emb is not None:
        noise_cond = pipe.unet(
            latent_input, t,
            encoder_hidden_states=per_frame_emb,
        ).sample
    else:
        noise_cond = pipe.unet(
            latent_input, t,
            encoder_hidden_states=cond_emb,
        ).sample

    return noise_uncond + guidance_scale * (noise_cond - noise_uncond)


def generate_enhanced(
    pipe,
    prompt: str,
    prompt_day: str,
    prompt_night: str,
    negative_prompt: str = "static, flickering, low quality",
    num_frames: int = 16,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    config: str = "C4_A_B_C",
    blend_beta: float = 0.3,
    device: str = "cuda",
):
    cfg_map = {
        "C0_baseline":  {"interp": False, "blend": 0.0, "bias": False},
        "C1_A_only":    {"interp": True,  "blend": 0.0, "bias": False},
        "C2_B_only":    {"interp": False, "blend": blend_beta, "bias": False},
        "C3_A_plus_B":  {"interp": True,  "blend": blend_beta, "bias": False},
        "C4_A_B_C":     {"interp": True,  "blend": blend_beta, "bias": True},
    }
    if config not in cfg_map:
        raise ValueError(f"Unknown config: {config}. Choose from {list(cfg_map.keys())}")
    flags = cfg_map[config]

    print(f"Config: {config}")
    print(f"  Enhancement A: {flags['interp']}")
    print(f"  Enhancement B: beta={flags['blend']}")
    print(f"  Enhancement C: {flags['bias']}")

    # ── Text embeddings ──
    if flags["interp"]:
        print("Precomputing per-frame day/night embeddings (Enh A)...")
        per_frame_emb = interpolate_prompt_embeddings(
            pipe, prompt_day, prompt_night, num_frames, k=6.0
        )
    else:
        per_frame_emb = None

    with torch.no_grad():
        uncond_emb = get_negative_embedding(pipe, negative_prompt)     # [1, 77, 768]
        cond_emb, _ = pipe.encode_prompt(
            prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )  # [1, 77, 768]

    # AnimateDiff flattens frames into the batch dimension; repeat embeddings to match.
    uncond_emb = uncond_emb.repeat(num_frames, 1, 1)
    if per_frame_emb is None:
        cond_emb = cond_emb.repeat(num_frames, 1, 1)

    # ── Latents ──
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = torch.randn(
        1, 4, num_frames, 64, 64,
        generator=generator, device=device, dtype=torch.float16,
    )

    # ── Enhancement C: inject bias into motion module temporal attention ──
    enh_c_injector = None
    if flags["bias"]:
        print("Injecting temporal attention bias (Enh C)...")
        bias_mod = TemporalBias(num_frames=num_frames).to(device, dtype=torch.float16)
        enh_c_injector = TemporalBiasInjector(pipe.unet, bias_mod)
        pipe.unet = pipe.unet  # re-register to apply processor changes

    # ── Denoising loop ──
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    print(f"Denoising {len(timesteps)} steps...")

    with torch.no_grad():
        for step_idx, t in enumerate(timesteps):
            noise_pred = compute_noise_pred(
                pipe, latents, t.unsqueeze(0), guidance_scale,
                uncond_emb, cond_emb, per_frame_emb,
                flags["interp"], enh_c_injector,
            )
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample

            if step_idx % 5 == 0 or step_idx == num_inference_steps - 1:
                print(f"  Step {step_idx+1}/{num_inference_steps}  "
                      f"t={t.item():>4.0f}  noise_std={noise_pred.std().item():.4f}  "
                      f"latent_std={latents.std().item():.4f}")

    # Remove bias hooks
    if enh_c_injector is not None:
        enh_c_injector.remove()

    # ── Enhancement B: latent-space frame blending ──
    if flags["blend"] > 0.0:
        print(f"Latent frame blending (beta={flags['blend']})...")
        latents[0] = apply_frame_blend(latents[0], beta=flags["blend"])

    # ── Decode ──
    print("Decoding to pixel space...")
    with torch.no_grad():
        b, c, f, h, w = latents.shape
        latents_2d = latents.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)
        frames = pipe.vae.decode(
            (latents_2d / pipe.vae.config.scaling_factor).to(torch.float16)
        ).sample
        frames = frames.reshape(b, f, 3, frames.shape[-2], frames.shape[-1]).squeeze(0)

    return _tensor_to_pil(frames)


# ─── Helpers ───────────────────────────────────────────────────────────────

def _tensor_to_pil(tensor: torch.Tensor):
    from PIL import Image
    tensor = (tensor / 2.0 + 0.5).clamp(0, 1)
    return [
        Image.fromarray((tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
        for i in range(tensor.shape[0])
    ]


def main():
    parser = argparse.ArgumentParser(description="Enhanced AnimateDiff Inference")
    parser.add_argument("--base-model", default="models/stable-diffusion-v1-5")
    parser.add_argument("--motion-module", default="models/motion-module")
    parser.add_argument("--config", default="C4_A_B_C",
                        choices=["C0_baseline", "C1_A_only", "C2_B_only",
                                 "C3_A_plus_B", "C4_A_B_C"])
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--prompt-day", required=True)
    parser.add_argument("--prompt-night", required=True)
    parser.add_argument("--negative-prompt", default="static, flickering, low quality")
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--blend-beta", type=float, default=0.3)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-offload", action="store_true")
    args = parser.parse_args()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Enhanced Inference — Urban Metamorphosis")
    print("=" * 60)

    pipe = setup_pipeline(
        args.base_model,
        args.motion_module,
        device=args.device,
        cpu_offload=args.cpu_offload,
    )
    frames = generate_enhanced(
        pipe,
        prompt=args.prompt,
        prompt_day=args.prompt_day,
        prompt_night=args.prompt_night,
        negative_prompt=args.negative_prompt,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        config=args.config,
        blend_beta=args.blend_beta,
        device=args.device,
    )
    export_to_gif(frames, args.output)
    print(f"\n✓ Saved {len(frames)} frames to: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
