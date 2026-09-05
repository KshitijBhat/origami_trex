"""Monkeypatches on T-Rex's attention/cache-read path. REDESIGN_PLAN.md §7.1.

Needed only because ``train_origami.py`` enables gradient checkpointing on one GPU (§7.3);
both patches are no-ops otherwise. ``apply()`` asserts the pinned upstream source hashes
(mirrors ``origami/tests/test_upstream_drift.py``'s hash-assertion pattern) before patching,
and is idempotent -- safe to call more than once (e.g. import-time + explicit call in a test).

Patch 1 -- ``Qwen3VLAttentionMoT.forward`` cached-prefix read (modeling_qwen3vl_mot.py).
Upstream always calls ``past_key_value.update(...)``, which appends. ``update()`` is not
idempotent: under ``torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`` (used by
``Qwen3VLModelMoT.forward`` when ``self.gradient_checkpointing and self.training``), the
non-reentrant checkpoint's real forward pass runs under ``torch.no_grad()`` (real append),
and the backward-time recompute runs under ``torch.enable_grad()`` -- calling the same layer
forward, and hence ``past_key_value.update(...)``, a SECOND time. The second append grows the
cache past the length the causal mask (built for ``past_len + seq_len``) expects, and the
cascaded tactile step (which chains multiple forward calls against the same cache object
across a training step) is exactly the place this compounds. Fix: when
``torch.is_grad_enabled()`` (true only for the enable-grad recompute call, never for the
real no-grad forward call that actually appends), read the already-appended prefix without
appending again and concat with the locally computed (recomputed) key/value -- reproducing
exactly what ``update()`` would have returned, without a second append. Nothing downstream
consumes the appended cache a third time, so the read-only concat is behaviorally equivalent.

Patch 2 -- assertion only, not a functional patch (modeling_vla.py). Upstream already omits
``attention_mask`` in the ``past_kv is not None`` branches of ``forward_flow_action_full`` /
``forward_flow_action_partial`` (only the ``past_kv is None`` branches pass it) -- a
``[B, L_slow]`` mask against an action-only ``inputs_embeds`` would be a length mismatch.
Verified at import time by source inspection; would need a real patch if a future upstream
version added it back, which is not the case at the pinned commit.
"""
from __future__ import annotations

import hashlib
import inspect
import re
from pathlib import Path

import torch

from origami.constants import REPO_ROOT

TREX_ROOT = REPO_ROOT / "T-Rex"

# Expected sha256 of the two patched files at the pinned f88e10c commit (see
# origami/upstream_manifest.json / test_upstream_drift.py -- same manifest, same hashes).
_EXPECTED_HASHES = {
    "qwen_vla/modeling_vla.py":
        "0b8b172de03852c78d7cee567b7b64b9fcb12dfc4c889e13e68a829e29f9b059",
    "qwen_vla/modeling_qwen3vl_mot.py":
        "6ff0314e1cd196b7e319e5aa39e15fb04ba858c1a8f0f5cad8c6a1453a8a7e08",
}

_PATCHED_MARKER = "_origami_trex_patch_applied"


def _assert_upstream_hashes() -> None:
    for rel_path, expected in _EXPECTED_HASHES.items():
        actual_bytes = (TREX_ROOT / rel_path).read_bytes()
        actual = hashlib.sha256(actual_bytes).hexdigest()
        assert actual == expected, (
            f"T-Rex/{rel_path} has drifted from the pinned f88e10c hash recorded for "
            f"trex_patch.py -- re-verify the patch still applies cleanly before updating "
            f"the expected hash."
        )


def read_cached_kv(cache, layer_idx: int):
    """Read the already-appended K/V prefix for one layer without appending.

    Returns ``(keys, values)`` or ``None`` if nothing has been cached yet for this layer.
    Handles both cache layouts seen across transformers versions: the newer
    ``DynamicCache.layers`` (list of per-layer objects with ``.keys``/``.values``) and the
    older top-level ``key_cache``/``value_cache`` lists. Exact code from REDESIGN_PLAN.md §7.1.
    """
    layers = getattr(cache, "layers", None)
    if isinstance(layers, list):
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            k = getattr(layer, "keys", None)
            if k is not None and k.numel() > 0:
                return k, layer.values
        return None
    k = getattr(cache, "key_cache", None)
    if k is not None and layer_idx < len(k) and k[layer_idx] is not None:
        return k[layer_idx], cache.value_cache[layer_idx]
    return None


def _patched_attention_forward(self, hidden_states, position_embeddings, attention_mask,
                                past_key_value=None, cache_position=None,
                                latent_indexes=None, action_indexes=None,
                                tactile_indexes=None, **kwargs):
    from qwen_vla.modeling_qwen3vl_mot import repeat_kv

    input_shape = hidden_states.shape[:-1]

    lat_h = hidden_states[:, latent_indexes] if len(latent_indexes) > 0 else hidden_states[:, :0]
    act_h = hidden_states[:, action_indexes] if len(action_indexes) > 0 else hidden_states[:, :0]
    tac_h = hidden_states[:, tactile_indexes] if len(tactile_indexes) > 0 else hidden_states[:, :0]

    lat_q, lat_k, lat_v = self._proj_qkv(lat_h, self.q_proj, self.k_proj, self.v_proj, self.q_norm, self.k_norm)
    act_q, act_k, act_v = self._proj_qkv(act_h, self.q_proj_action, self.k_proj_action, self.v_proj_action, self.q_norm_action, self.k_norm_action)
    tac_q, tac_k, tac_v = self._proj_qkv(tac_h, self.q_proj_tactile, self.k_proj_tactile, self.v_proj_tactile, self.q_norm_tactile, self.k_norm_tactile)

    query_states = torch.cat([lat_q, act_q, tac_q], dim=2)
    key_states = torch.cat([lat_k, act_k, tac_k], dim=2)
    value_states = torch.cat([lat_v, act_v, tac_v], dim=2)

    cos, sin = position_embeddings
    from qwen_vla.modeling_qwen3vl_mot import _apply_rope_fn, apply_rotary_pos_emb_1d
    if _apply_rope_fn is not None:
        query_states, key_states = _apply_rope_fn(query_states, key_states, cos, sin)
    else:
        query_states, key_states = apply_rotary_pos_emb_1d(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # ORIGAMI-PATCH (§7.1 patch 1): idempotent cache read under gradient
        # checkpointing's backward recompute -- see module docstring.
        if torch.is_grad_enabled():
            cached = read_cached_kv(past_key_value, self.layer_idx)
            if cached is not None:
                prefix_k, prefix_v = cached
                key_states = torch.cat([prefix_k, key_states], dim=2)
                value_states = torch.cat([prefix_v, value_states], dim=2)
        else:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    key_s = repeat_kv(key_states, self.num_key_value_groups)
    value_s = repeat_kv(value_states, self.num_key_value_groups)
    causal_mask = attention_mask[:, :, :, :key_s.shape[-2]] if attention_mask is not None else None
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states, key_s, value_s,
        attn_mask=causal_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
    )
    attn_weights = None

    attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

    lat_out = self.o_proj(attn_output[:, latent_indexes]) if len(latent_indexes) > 0 else attn_output[:, :0]
    act_out = self.o_proj_action(attn_output[:, action_indexes]) if len(action_indexes) > 0 else attn_output[:, :0]
    tac_out = self.o_proj_tactile(attn_output[:, tactile_indexes]) if len(tactile_indexes) > 0 else attn_output[:, :0]

    out = torch.cat([lat_out, act_out, tac_out], dim=1)
    return out, attn_weights


def _assert_attention_mask_absent_from_cached_prefix_calls() -> None:
    """Patch 2 (assertion only): verify `modeling_vla.py`'s `forward_flow_action_{full,
    partial}` still omit `attention_mask=` in their `past_kv is not None` branch. If a future
    upstream version starts passing it there, this raises so the assumption gets re-examined
    (and a real patch written) instead of silently producing a shape-mismatched mask.
    """
    import qwen_vla.modeling_vla as modeling_vla

    for fn_name in ("forward_flow_action_full", "forward_flow_action_partial"):
        source = inspect.getsource(getattr(modeling_vla.Qwen3VLVLAModel, fn_name))
        # Split at the `else:` that starts the `past_kv is not None` branch (both methods
        # have exactly one top-level `if past_kv is None: ... else: ...` inside their loop).
        parts = re.split(r"\n(\s*)else:\n", source, maxsplit=1)
        assert len(parts) == 3, (
            f"{fn_name}: could not locate the `if past_kv is None / else` split -- "
            f"upstream source shape has changed, re-examine patch 2's assumption."
        )
        indent = parts[1]
        else_body_and_after = parts[2]
        # The else-branch body is the indented block following `else:`; stop at the first
        # line that dedents back to `indent` level or less (end of the else block).
        else_lines = []
        for line in else_body_and_after.splitlines():
            if line.strip() and not line.startswith(indent + " "):
                break
            else_lines.append(line)
        else_body = "\n".join(else_lines)
        assert "attention_mask" not in else_body, (
            f"{fn_name}: upstream now passes `attention_mask` in the cached-prefix "
            f"(`past_kv is not None`) branch -- REDESIGN_PLAN.md §7.1 patch 2 needs to "
            f"become a real patch (strip it back out) instead of just this assertion."
        )


def apply() -> None:
    """Apply both patches. Idempotent; safe to call multiple times."""
    import qwen_vla.modeling_qwen3vl_mot as modeling_qwen3vl_mot

    if getattr(modeling_qwen3vl_mot.Qwen3VLAttentionMoT, _PATCHED_MARKER, False):
        return

    _assert_upstream_hashes()
    _assert_attention_mask_absent_from_cached_prefix_calls()

    modeling_qwen3vl_mot.Qwen3VLAttentionMoT.forward = _patched_attention_forward
    setattr(modeling_qwen3vl_mot.Qwen3VLAttentionMoT, _PATCHED_MARKER, True)
