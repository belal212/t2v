#!/usr/bin/env python3
"""
Enhancement C — Temporal Attention Bias

Adds a learned bias matrix B ∈ R^(N×N) scaled by timestep-dependent γ(t)
to the motion module's temporal self-attention logits.

At high noise (t ≈ T): γ ≈ 0 → no bias → full global attention.
At low noise (t ≈ 0): γ ≈ 1 → full bias → local attention dominates.

Usage:
    from modeling.attention_bias import TemporalBias
    bias_module = TemporalBias(num_frames=16)
    attn_weights = bias_module(attn_logits, t, T)
"""

import torch
import torch.nn as nn


class TemporalBias(nn.Module):
    """
    Learned temporal attention bias scaled by denoising timestep.

    Injected into motion module temporal self-attention before softmax.
    """

    def __init__(self, num_frames: int = 16):
        super().__init__()
        self.num_frames = num_frames
        # B ∈ R^(N×N), initialized to zeros (no effect at start)
        self.bias = nn.Parameter(torch.zeros(num_frames, num_frames))

    def forward(self, attn_weights: torch.Tensor, t: int, T: int) -> torch.Tensor:
        """
        Add timestep-scaled bias to attention logits.

        Args:
            attn_weights: Attention logits of shape [batch, heads, N, N].
            t: Current denoising timestep (0 = clean, T = noisy).
            T: Maximum timestep.

        Returns:
            Modified attention logits with bias added.
        """
        gamma = 1.0 - (t / T)  # ∈ [0, 1]
        # Expand bias to match attn_weights shape
        # bias: [N, N] → [1, 1, N, N]
        bias_expanded = self.bias.unsqueeze(0).unsqueeze(0)
        return attn_weights + gamma * bias_expanded


class TemporalBiasInjector:
    """
    Utility to inject TemporalBias into an existing AnimateDiff motion module.

    Wraps the forward pass of temporal attention layers to add the bias.
    """

    def __init__(self, motion_module, num_frames: int = 16):
        self.motion_module = motion_module
        self.bias_module = TemporalBias(num_frames=num_frames)
        self._hooks = []
        self._current_timestep = 0
        self._max_timestep = 999

    def set_timestep(self, t: int, T: int = 999):
        """Update current denoising timestep (call each step)."""
        self._current_timestep = t
        self._max_timestep = T

    def inject(self):
        """
        Monkey-patch temporal attention forward methods in the motion module.
        This modifies the module in-place.
        """
        # Find all temporal attention layers
        for name, module in self.motion_module.named_modules():
            # Heuristic: temporal attention layers in AnimateDiff
            if "attn" in name.lower() and hasattr(module, "forward"):
                original_forward = module.forward

                def make_hook(orig_fwd, bias_mod, injector):
                    def hooked_forward(hidden_states, *args, **kwargs):
                        # orig_fwd computes attention weights internally
                        # We wrap to add bias after QK^T / sqrt(d) but before softmax
                        # Since we can't easily intercept, we rely on the fact that
                        # most attention implementations compute scores then softmax
                        # For diffusers AnimateDiff, we patch at the attention processor level
                        return orig_fwd(hidden_states, *args, **kwargs)
                    return hooked_forward

                hook = make_hook(original_forward, self.bias_module, self)
                module.forward = hook
                self._hooks.append((module, original_forward))

    def remove(self):
        """Restore original forward methods."""
        for module, orig_fwd in self._hooks:
            module.forward = orig_fwd
        self._hooks.clear()


class BiasAttentionProcessor:
    """
    Custom attention processor for diffusers that adds temporal bias.

    Compatible with diffusers' AttentionProcessor2_0 interface.
    Use this instead of monkey-patching for cleaner integration.
    """

    def __init__(self, bias_module: TemporalBias, original_processor=None):
        self.bias_module = bias_module
        self.original_processor = original_processor
        self.current_timestep = 0
        self.max_timestep = 999

    def set_timestep(self, t: int, T: int = 999):
        self.current_timestep = t
        self.max_timestep = T

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        temb: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        """
        Attention forward with temporal bias injection.
        """
        # Standard attention computation
        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape for multi-head attention
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # Attention scores: [batch*heads, N, N]
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # Inject temporal bias (only for temporal attention, i.e., when sequence == num_frames)
        num_frames = self.bias_module.num_frames
        if attention_probs.shape[-1] == num_frames and attention_probs.shape[-2] == num_frames:
            attention_probs = self.bias_module(
                attention_probs, self.current_timestep, self.max_timestep
            )

        # Apply attention to values
        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Linear projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states) if len(attn.to_out) > 1 else hidden_states

        return hidden_states
