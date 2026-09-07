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

Patch 3 -- ``Qwen3VLVLAModel.prepare_inputs_embeds``'s vision-output unwrapping
(modeling_vla.py). Found while building ``diagnose_shift.py``/``eval_offline.py`` (step 10b),
which are the first callers to ever actually run this method against a real Qwen3-VL vision
tower + real images in this project. Upstream (written against T-Rex's own transformers pin,
``4.57.0.dev0`` per the released checkpoint's ``config.json``) does
``image_features = out[0] if isinstance(out, (tuple, list)) else out`` on the vision tower's
return value, with a comment claiming ``out[0]`` is ``[total_merged_tokens, hidden_size]`` --
i.e. it assumes ``self.visual(...)`` returns a plain ``(merged_hidden_states,
deepstack_feature_lists)`` tuple. On our pinned transformers (5.16.x), confirmed by direct
source inspection and by calling it,
``Qwen3VLVisionModel.forward`` instead always returns a ``BaseModelOutputWithDeepstackFeatures``
-- a ``ModelOutput``, which subclasses ``OrderedDict`` (**not** tuple/list, so
``isinstance(..., (tuple, list))`` is ``False``), whose ``.last_hidden_state`` is the
**pre-merge** sequence (wrong token count -- doesn't match the ``<image_pad>`` mask) and whose
``.pooler_output`` is the actual post-``self.merger(...)`` merged sequence upstream meant to
grab. Unpatched, ``prepare_inputs_embeds`` hands the *whole ModelOutput object* to
``image_features`` (the ``else`` branch fires) and crashes on ``image_features.to(dtype)``
(``AttributeError`` -- dicts have no ``.to``) the moment any real image goes through it --
i.e. every real forward pass in ``train_origami.py``'s ``train()``/``run_validation()`` and
``scripts/test.py::CascadedServer`` on this transformers version. Fix: unwrap
``.pooler_output`` when present, else ``.last_hidden_state``, else fall back to upstream's own
tuple/plain-tensor handling -- covers both transformers eras without needing to know which one
is installed.

Patch 4 -- ``Qwen3VLRotaryEmbeddingWrapper.__init__``'s synthetic rope config
(modeling_qwen3vl_mot.py). Also found while getting a real forward pass to run for step 10b.
Upstream builds a throwaway ``_RopeCfg`` type carrying the separate, pre-consolidation rope
fields (``rope_theta``, ``rope_scaling``, ``head_dim``, ...) that transformers' Qwen3-VL rotary
embedding expected on T-Rex's own pin. On our pinned transformers (5.16.x), confirmed by
direct source inspection, `Qwen3VLTextRotaryEmbedding.__init__` instead reads
``config.rope_parameters["rope_type"]`` and, inside ``compute_default_rope_parameters``,
``config.rope_parameters["rope_theta"]`` -- a single consolidated dict transformers introduced
at some point between the two pins. ``_RopeCfg`` has no such attribute, so **every** model
construction that reaches this wrapper (i.e. every real model on this transformers version,
whether built via ``from_pretrained_qwen3vl`` or ``_build_qwen3vl_from_config``) raises
``AttributeError: '_RopeCfg' object has no attribute 'rope_parameters'`` before a single
forward pass can run. Fix: build the same ``_RopeCfg`` upstream builds, plus a
``rope_parameters`` dict assembled from the same ``rope_theta``/``rope_scaling`` values --
satisfies both the old-style direct-attribute reads (kept, harmless if unused) and the new
consolidated-dict read.

Patch 5 -- ``Qwen3VLVLAModel.get_rope_index``'s call into the base model's M-RoPE indexer
(modeling_vla.py). Also found while getting a real forward pass to run for step 10b, one call
further than patch 4. Upstream calls ``self._rope_index_fn(input_ids=..., image_grid_thw=...,
attention_mask=...)``. On T-Rex's own pin, ``Qwen3VLModel.get_rope_index`` took exactly those
three keyword arguments. On our pinned transformers (5.16.x), confirmed by direct source
inspection, its signature gained a new **required** positional parameter,
``mm_token_type_ids`` (an ``int`` tensor shaped like ``input_ids``, marking each token
text=0/image=1/video=2) -- upstream's call is missing it and raises
``TypeError: get_rope_index() missing 1 required positional argument``. `_rope_index_fn` is
bound one of two ways (``scripts/test.py::model_load``'s ``_build_qwen3vl_from_config`` path
binds a closure-local ``_RopeStub`` whose own ``get_rope_index`` wrapper *also* doesn't accept
``mm_token_type_ids`` and can't be monkeypatched by name since it's a new class object per
call; ``modeling_vla.py::from_pretrained_qwen3vl`` binds the real
``Qwen3VLModel.get_rope_index`` directly), so the fix builds ``mm_token_type_ids`` from
``input_ids == self.image_token_id`` (we never feed video, so text/image is the whole
picture) and, when the bound function's own signature doesn't accept it (the ``_RopeStub``
case), calls the real ``transformers`` method directly against the stub's bound `self` --
exactly what the stub itself does one layer up, since `Qwen3VLModel.get_rope_index` only reads
`self.config` internally, so a `_RopeStub` duck-types fine as `self` here too.

Patch 6 -- ``Qwen3VLVLAModel.from_pretrained_qwen3vl``'s visual-tower copy (modeling_vla.py).
Found running the FIRST real ``train_origami.py`` invocation against the real 101-season
``eef62_train`` root (§12 step 11's pilot, pulled forward into step 10b's verification pass --
patches 1-5 above only ever exercised the *inference* path via `CascadedServer`/`model_load`;
nothing had called `from_pretrained_qwen3vl`, the training-time model constructor, for real
before this). Upstream does ``vla.visual = base_model.visual``, with a comment claiming
``base_model.visual`` is a ``@property`` that forwards to ``base_model.model.visual`` on
T-Rex's own transformers pin. On our pinned transformers (5.16.x), confirmed by direct source
inspection of ``Qwen3VLForConditionalGeneration.__init__`` (constructs only ``self.model =
Qwen3VLModel(config)`` and ``self.lm_head``, no ``visual`` property at all) and
``Qwen3VLModel.__init__`` (constructs ``self.visual``/``self.language_model`` directly) --
the property upstream's comment describes no longer exists; ``base_model.visual`` raises
``AttributeError: 'Qwen3VLForConditionalGeneration' object has no attribute 'visual'`` at
**model-construction time**, before any data or forward pass. The rest of the method already
handles both eras correctly (its own comments show the author was aware ``base_model.model``
restructures between Qwen2-VL and Qwen3-VL for the state-dict-prefix logic just below) -- only
this one line assumes the now-gone delegating property. Fix: replace the whole classmethod
(needed since it is a classmethod body, not an isolable helper) with an identical copy except
``vla.visual = base_model.visual if hasattr(base_model, "visual") else base_model.model.visual``
-- covers both transformers eras without needing to know which one is installed, exactly
patch 3's strategy for the same kind of drift.
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


def extract_merged_vision_features(out):
    """Unwrap a vision-tower call's return value into the merged ``[total_merged_tokens,
    hidden_size]`` tensor `prepare_inputs_embeds` needs, across both the tuple-returning
    upstream transformers era T-Rex was written against and the ``ModelOutput``-returning one
    we're pinned to. See patch 3's module-docstring section for why this exists.
    """
    pooler = getattr(out, "pooler_output", None)
    if pooler is not None:
        return pooler
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def _patched_prepare_inputs_embeds(self, input_ids, pixel_values=None, image_grid_thw=None):
    """ORIGAMI-PATCH (§7.1 patch 3): identical to upstream except for the vision-output
    unwrap -- see `extract_merged_vision_features`."""
    inputs_embeds = self.model.get_input_embeddings()(input_ids)
    if pixel_values is not None and self.visual is not None:
        pixel_values = pixel_values.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
        out = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_features = extract_merged_vision_features(out)
        image_mask = (input_ids == self.image_token_id)
        if image_mask.any():
            inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)
    return inputs_embeds


def _patched_rope_wrapper_init(self, config, device=None):
    """ORIGAMI-PATCH (§7.1 patch 4): identical to upstream except `_rope_cfg` also carries a
    consolidated `rope_parameters` dict -- see patch 4's module-docstring section."""
    import torch.nn as nn
    from qwen_vla.modeling_qwen3vl_mot import _VLRotaryEmbedding

    nn.Module.__init__(self)
    if _VLRotaryEmbedding is not None:
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = {"rope_type": "default", "mrope_section": [16, 24, 24]}
        rope_theta = getattr(config, "rope_theta", 1000000.0)
        rope_parameters = {
            "rope_type": rope_scaling.get("rope_type", "default"),
            "rope_theta": rope_theta,
            **{k: v for k, v in rope_scaling.items() if k != "rope_type"},
        }
        _rope_cfg = type("_RopeCfg", (), {
            "hidden_size":             config.hidden_size,
            "num_attention_heads":     config.num_attention_heads,
            "head_dim":                getattr(config, "head_dim",
                                               config.hidden_size // config.num_attention_heads),
            "max_position_embeddings": getattr(config, "max_position_embeddings", 32768),
            "rope_theta":              rope_theta,
            "rope_scaling":            rope_scaling,
            "rope_parameters":         rope_parameters,
            "partial_rotary_factor":   getattr(config, "partial_rotary_factor", 1.0),
        })()
        self._rope = _VLRotaryEmbedding(config=_rope_cfg, device=device)
    else:
        rope_theta = getattr(config, "rope_theta", 10000.0)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        inv_freq = 1.0 / (
            rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        self._rope = None
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0


def _patched_get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
    """ORIGAMI-PATCH (§7.1 patch 5): identical to upstream except it builds and forwards
    `mm_token_type_ids` -- see patch 5's module-docstring section."""
    if self._rope_index_fn is None:
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        return position_ids, None

    fn = self._rope_index_fn
    mm_token_type_ids = (input_ids == self.image_token_id).to(torch.int64)
    try:
        accepts_mm = "mm_token_type_ids" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        accepts_mm = False

    if accepts_mm:
        return fn(input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
                  image_grid_thw=image_grid_thw, attention_mask=attention_mask)

    bound_self = getattr(fn, "__self__", None)
    if bound_self is not None:
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        if isinstance(bound_self, Qwen3VLModel):
            target_self = bound_self
        else:
            # scripts/test.py's `_RopeStub` duck-types only `.config`, but the real
            # `get_rope_index` also calls other bound methods on `self`
            # (`get_vision_position_ids`) that a bare config object doesn't have.
            # Build an uninitialized real `Qwen3VLModel` (bypassing __init__, so no
            # weights are ever allocated) and reuse the stub's `.config` on it --
            # every method these two calls touch is config/tensor-math only.
            target_self = Qwen3VLModel.__new__(Qwen3VLModel)
            target_self.config = bound_self.config
        return Qwen3VLModel.get_rope_index(
            target_self, input_ids=input_ids, mm_token_type_ids=mm_token_type_ids,
            image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    return fn(input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)


def _patched_from_pretrained_qwen3vl(
    cls,
    pretrained_path: str,
    action_dim: int = 29,
    action_chunk: int = 8,
    tacf6_dim: int = 6,
    use_tactile_deform: bool = False,
    use_robot_state: bool = False,
    torch_dtype=torch.bfloat16,
    tactile_intermediate_size: int = None,
    n_flare_tokens_per_frame: int = 0,
    n_flare_steps: int = 0,
    flare_layer_index: int = -1,
    use_tactile_code: bool = False,
    vqvae_codebook_size: int = 64,
    use_tactile_vqvae: bool = False,
    vqvae_config: dict = None,
):
    """ORIGAMI-PATCH (§7.1 patch 6): identical to upstream `from_pretrained_qwen3vl` except
    the visual-tower copy tolerates both transformers eras -- see patch 6's module-docstring
    section. Whole classmethod copied (not isolable into a small helper)."""
    import gc

    from qwen_vla.modeling_vla import _DEFAULT_IMAGE_TOKEN_ID

    try:
        from transformers import Qwen3VLForConditionalGeneration
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            pretrained_path, torch_dtype=torch_dtype, trust_remote_code=True
        )
    except Exception:
        from transformers import Qwen2VLForConditionalGeneration
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            pretrained_path, torch_dtype=torch_dtype, trust_remote_code=True
        )

    config = base_model.config
    text_config = getattr(config, "text_config", config)
    image_token_id = getattr(config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID)

    vla = cls(
        config=text_config,
        action_dim=action_dim,
        action_chunk=action_chunk,
        tacf6_dim=tacf6_dim,
        use_tactile_deform=use_tactile_deform,
        use_robot_state=use_robot_state,
        image_token_id=image_token_id,
        tactile_intermediate_size=tactile_intermediate_size,
        n_flare_tokens_per_frame=n_flare_tokens_per_frame,
        n_flare_steps=n_flare_steps,
        flare_layer_index=flare_layer_index,
        use_tactile_code=use_tactile_code,
        vqvae_codebook_size=vqvae_codebook_size,
        use_tactile_vqvae=use_tactile_vqvae,
        vqvae_config=vqvae_config,
    )
    _inner = base_model.model if hasattr(base_model, "model") else base_model
    if hasattr(_inner, "get_rope_index"):
        object.__setattr__(vla, "_rope_index_fn", _inner.get_rope_index)

    # ORIGAMI-PATCH (§7.1 patch 6): base_model.visual no longer exists as a delegating
    # property on our transformers pin -- reach through base_model.model.visual instead.
    vla.visual = base_model.visual if hasattr(base_model, "visual") else base_model.model.visual

    raw_sd = base_model.model.state_dict()
    lang_prefix = "language_model."
    has_lang_prefix = any(k.startswith(lang_prefix) for k in raw_sd)

    text_sd = {}
    if has_lang_prefix:
        for k, v in raw_sd.items():
            if k.startswith(lang_prefix):
                text_sd[k[len(lang_prefix):]] = v
    else:
        text_sd = dict(raw_sd)

    missing, unexpected = vla.model.load_state_dict(text_sd, strict=False)
    print(f"[Qwen3VLVLAModel] Text model loaded – missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        truly_missing = [k for k in missing if "_action" not in k and "_tactile" not in k]
        expected_missing = len(missing) - len(truly_missing)
        if truly_missing:
            print(f"  WARNING – unexpected missing base keys: {truly_missing[:10]} ...")
        print(f"  New MoT expert weights (expected missing): {expected_missing}")

    if hasattr(_inner, "language_model"):
        del _inner.language_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[Qwen3VLVLAModel] Freed base model language_model to save memory.")

    return vla


def apply() -> None:
    """Apply all six patches. Idempotent; safe to call multiple times."""
    import qwen_vla.modeling_qwen3vl_mot as modeling_qwen3vl_mot
    import qwen_vla.modeling_vla as modeling_vla

    if getattr(modeling_qwen3vl_mot.Qwen3VLAttentionMoT, _PATCHED_MARKER, False):
        return

    _assert_upstream_hashes()
    _assert_attention_mask_absent_from_cached_prefix_calls()

    modeling_qwen3vl_mot.Qwen3VLAttentionMoT.forward = _patched_attention_forward
    setattr(modeling_qwen3vl_mot.Qwen3VLAttentionMoT, _PATCHED_MARKER, True)

    modeling_vla.Qwen3VLVLAModel.prepare_inputs_embeds = _patched_prepare_inputs_embeds
    setattr(modeling_vla.Qwen3VLVLAModel, _PATCHED_MARKER, True)

    modeling_qwen3vl_mot.Qwen3VLRotaryEmbeddingWrapper.__init__ = _patched_rope_wrapper_init
    setattr(modeling_qwen3vl_mot.Qwen3VLRotaryEmbeddingWrapper, _PATCHED_MARKER, True)

    modeling_vla.Qwen3VLVLAModel.get_rope_index = _patched_get_rope_index

    modeling_vla.Qwen3VLVLAModel.from_pretrained_qwen3vl = classmethod(_patched_from_pretrained_qwen3vl)
