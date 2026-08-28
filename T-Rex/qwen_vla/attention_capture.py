"""Attention-map capture for Qwen3VLAttentionMoT's joint per-expert attention.

Validation-only (see scripts/train.py's run_validation). Passing
output_attentions=True forces the eager (non-fused, matmul+softmax) attention
path in modeling_qwen3vl_mot.py for every decoder layer of that one forward
call -- slower and more memory-hungry than the default SDPA path, so this
must never run during the actual training step, only on a capped number of
validation batches every so often (see train.py's --capture_attn_every_n_val).

Scope (be aware of this before trusting a map that isn't here): this only
captures the joint self/cross-attention inside Qwen3VLModelMoT's decoder
layers, from the ONE `model.model(...)` call in run_validation that computes
the action-expert's full-flow loss. It does NOT (yet) reach inside
`forward_flow_action_partial` / `tactile_flow_continue` in modeling_vla.py,
which have their own separate `self.model(...)` calls and their own KV-cache
bookkeeping -- so tactile<->action cross-attention (the cascaded mechanism
itself) is not captured by this module as-is. Extending there is the natural
next step if you need that specific map; it wasn't done here because those
methods carry the real training-time cascaded loss and a half-verified patch
there is a much higher-stakes mistake than one in a validation-only utility.

What you get from what IS wired up, per captured layer:
    latent_to_latent   -- vision/language self-attention (does the model look
                           at the paper/fold region, or elsewhere?)
    action_to_latent   -- does the action expert actually attend to vision at
                           all, or mostly to its own tokens? (the same
                           vision-under-use question this project asked of
                           the vitacformer++ ACT policy)
    action_to_action   -- action expert self-attention
    latent_to_action, tactile_to_* etc. -- saved too when both sides are
                           non-empty, for completeness
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LAYER_FRACTIONS = (0.0, 0.5, 1.0)  # first, middle, last -- independent of depth (28 here)


def _select_layers(n_layers: int, fractions: Sequence[float] = _LAYER_FRACTIONS) -> list:
    idxs = []
    for f in fractions:
        i = min(n_layers - 1, int(round(f * (n_layers - 1))))
        if i not in idxs:
            idxs.append(i)
    return idxs


def save_attention_maps(
    attentions: Tuple[torch.Tensor, ...],
    expert_indexes: Dict[str, torch.Tensor],
    out_dir: str,
    n_samples: int = 2,
    layer_fractions: Sequence[float] = _LAYER_FRACTIONS,
    tag_prefix: str = "",
) -> int:
    """attentions: outputs.attentions from a model.model(..., output_attentions=True)
    call -- a tuple of [B, heads, total_seq, total_seq] tensors, one per decoder
    layer, un-averaged over heads (eager_attention_forward returns per-head weights).

    expert_indexes: {"latent": latent_indexes, "action": action_indexes,
    "tactile": tactile_indexes} -- the exact index tensors passed into that
    same model.model(...) call. Empty tensors are skipped (no error) so this
    works whether or not tactile tokens were present in that particular call.

    Saves, per selected layer and per (query-expert, key-expert) pair where
    both sides are non-empty: one head-averaged .npy [n_samples, len(q), len(k)]
    and one heatmap PNG per sample. Returns the number of .npy files written.
    """
    if not attentions:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    n_layers = len(attentions)
    layer_idxs = _select_layers(n_layers, layer_fractions)
    n = min(n_samples, attentions[0].shape[0])

    live_experts = {name: idx for name, idx in expert_indexes.items() if len(idx) > 0}

    saved = 0
    for li in layer_idxs:
        # Head-average for a plottable 2D map; eager_attention_forward's raw
        # output is per-head ([B, heads, S, S]), unlike vitacformer++'s
        # nn.MultiheadAttention which averages heads for you by default.
        attn = attentions[li].float().mean(dim=1)[:n]  # [n, S, S]
        for q_name, q_idx in live_experts.items():
            q_idx = q_idx.to(attn.device)
            for k_name, k_idx in live_experts.items():
                k_idx = k_idx.to(attn.device)
                sliced = attn.index_select(1, q_idx).index_select(2, k_idx)
                sliced = sliced.detach().cpu().numpy()  # [n, len(q_idx), len(k_idx)]

                tag = f"{tag_prefix}layer{li}_{q_name}_to_{k_name}"
                np.save(os.path.join(out_dir, f"{tag}.npy"), sliced)
                saved += 1

                for s in range(sliced.shape[0]):
                    fig, ax = plt.subplots(figsize=(4, 4))
                    im = ax.imshow(sliced[s], aspect="auto", cmap="viridis")
                    ax.set_title(f"{tag} sample{s}", fontsize=8)
                    ax.set_xlabel(f"{k_name} keys")
                    ax.set_ylabel(f"{q_name} queries")
                    fig.colorbar(im, ax=ax, fraction=0.046)
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, f"{tag}_sample{s}.png"), dpi=100)
                    plt.close(fig)
    return saved


def save_tactile_attention_maps(
    attentions: Tuple[torch.Tensor, ...],
    n_action_in_cache: int,
    n_tac_seq: int,
    out_dir: str,
    n_samples: int = 2,
    layer_fractions: Sequence[float] = _LAYER_FRACTIONS,
    tag_prefix: str = "",
) -> int:
    """The one map save_attention_maps can't produce: tactile queries against
    the cached [latent | action] KV from the slow tick, from
    tactile_flow_continue(output_attentions=True)'s first Euler step.

    Rectangular, not square, unlike save_attention_maps: the query axis is
    only this call's n_tac_seq tactile tokens (latent/action indexes are both
    empty in tactile_flow_continue -- it reads their KV from cache, never
    recomputes them), while the key axis is the full cache (latent + action
    from the slow tick) plus this step's own tactile keys. One index tensor
    can't serve both axes here the way it does in save_attention_maps, hence
    a separate function rather than a generalization of it.

    n_action_in_cache, n_tac_seq: the same values tactile_flow_continue was
    called with / returned -- used to split the key axis into named ranges by
    simple arithmetic (total_keys - n_action_in_cache - n_tac_seq = however
    many latent keys must be at the front), not by guessing.
    """
    if not attentions:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    n_layers = len(attentions)
    layer_idxs = _select_layers(n_layers, layer_fractions)
    n = min(n_samples, attentions[0].shape[0])

    saved = 0
    for li in layer_idxs:
        attn = attentions[li].float().mean(dim=1)[:n]  # [n, n_tac_seq, total_keys]
        total_keys = attn.shape[-1]
        n_latent_in_cache = total_keys - n_action_in_cache - n_tac_seq
        if n_latent_in_cache < 0:
            continue  # the assumed cache layout doesn't hold here -- skip rather than guess

        key_ranges = {
            "latent":  (0, n_latent_in_cache),
            "action":  (n_latent_in_cache, n_latent_in_cache + n_action_in_cache),
            "tactile": (n_latent_in_cache + n_action_in_cache, total_keys),
        }
        for k_name, (lo, hi) in key_ranges.items():
            if hi <= lo:
                continue
            k_idx = torch.arange(lo, hi, device=attn.device)
            sliced = attn.index_select(2, k_idx).detach().cpu().numpy()  # [n, n_tac_seq, hi-lo]

            tag = f"{tag_prefix}layer{li}_tactile_to_{k_name}"
            np.save(os.path.join(out_dir, f"{tag}.npy"), sliced)
            saved += 1

            for s in range(sliced.shape[0]):
                fig, ax = plt.subplots(figsize=(4, 4))
                im = ax.imshow(sliced[s], aspect="auto", cmap="viridis")
                ax.set_title(f"{tag} sample{s}", fontsize=8)
                ax.set_xlabel(f"{k_name} keys")
                ax.set_ylabel("tactile queries")
                fig.colorbar(im, ax=ax, fraction=0.046)
                fig.tight_layout()
                fig.savefig(os.path.join(out_dir, f"{tag}_sample{s}.png"), dpi=100)
                plt.close(fig)
    return saved
