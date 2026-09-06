"""§7.1 trex_patch.py -- CPU-only, no GPU/network required.

Proves the two patches are behaviorally equivalent to upstream on real cache objects
(``transformers.cache_utils.DynamicCache``, the class T-Rex actually uses), without needing
gradient checkpointing or a full model forward pass to reproduce the double-append bug.
"""
from __future__ import annotations

import torch
from transformers.cache_utils import DynamicCache

import origami.trex_patch as trex_patch


def test_apply_is_idempotent():
    trex_patch.apply()
    trex_patch.apply()  # must not raise / must not double-patch
    import qwen_vla.modeling_qwen3vl_mot as m
    assert getattr(m.Qwen3VLAttentionMoT, trex_patch._PATCHED_MARKER) is True


def test_upstream_hashes_pass_on_pinned_checkout():
    # If T-Rex/ has drifted from f88e10c this raises -- same invariant as test_upstream_drift.py.
    trex_patch._assert_upstream_hashes()


def test_patch2_assertion_passes_on_pinned_checkout():
    trex_patch._assert_attention_mask_absent_from_cached_prefix_calls()


# ── Patch 1: read_cached_kv ──────────────────────────────────────────────────────────────

def test_read_cached_kv_returns_none_before_any_update():
    cache = DynamicCache()
    assert trex_patch.read_cached_kv(cache, layer_idx=0) is None


def test_read_cached_kv_matches_manual_reference_after_one_update():
    cache = DynamicCache()
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    k_ref, v_ref = cache.update(k, v, layer_idx=0, cache_kwargs={})

    k_read, v_read = trex_patch.read_cached_kv(cache, layer_idx=0)
    assert torch.equal(k_read, k_ref)
    assert torch.equal(v_read, v_ref)


def test_read_cached_kv_does_not_append():
    """The whole point of patch 1: reading must never grow the cache."""
    cache = DynamicCache()
    k = torch.randn(1, 2, 5, 4)
    v = torch.randn(1, 2, 5, 4)
    cache.update(k, v, layer_idx=0, cache_kwargs={})
    seq_len_before = cache.get_seq_length()

    for _ in range(5):
        trex_patch.read_cached_kv(cache, layer_idx=0)

    assert cache.get_seq_length() == seq_len_before


def test_grad_enabled_read_equals_grad_disabled_append_reference():
    """Reproduces the exact scenario patch 1 targets: a "real" no-grad forward that
    genuinely appends (as upstream always does), followed by a "recompute" call under
    grad-enabled that must reproduce the SAME resulting K/V without appending again.

    Reference (what upstream would incorrectly do under grad-enabled recompute): call
    update() a second time -> cache grows to 2x length, WRONG.
    Patched behavior: read the already-appended prefix and concat with the newly (locally)
    computed key/value slice -- must equal the single-append reference exactly.
    """
    B, H, D = 1, 2, 4
    prefix_len, new_len = 5, 3

    prefix_k = torch.randn(B, H, prefix_len, D)
    prefix_v = torch.randn(B, H, prefix_len, D)
    new_k = torch.randn(B, H, new_len, D)
    new_v = torch.randn(B, H, new_len, D)

    # "Real" no-grad forward: genuinely appends once (matches upstream's update() call,
    # taken under torch.is_grad_enabled() == False inside a checkpointed layer).
    cache = DynamicCache()
    with torch.no_grad():
        cache.update(prefix_k, prefix_v, layer_idx=0, cache_kwargs={})
        real_k, real_v = cache.update(new_k, new_v, layer_idx=0, cache_kwargs={})
    # real_k/real_v is the single-true-append reference: [prefix | new], length prefix+new.
    assert real_k.shape[2] == prefix_len + new_len

    # Reset a fresh cache holding just the prefix (mirrors state after the real forward's
    # first update, before its second -- i.e. what the backward recompute call sees).
    cache2 = DynamicCache()
    with torch.no_grad():
        cache2.update(prefix_k, prefix_v, layer_idx=0, cache_kwargs={})

    # "Recompute" call under grad-enabled: patched behavior reads the prefix (no append)
    # and concats with the locally recomputed new_k/new_v.
    with torch.enable_grad():
        cached = trex_patch.read_cached_kv(cache2, layer_idx=0)
        assert cached is not None
        read_prefix_k, read_prefix_v = cached
        patched_k = torch.cat([read_prefix_k, new_k], dim=2)
        patched_v = torch.cat([read_prefix_v, new_v], dim=2)

    assert torch.equal(patched_k, real_k)
    assert torch.equal(patched_v, real_v)
    # And critically, the cache itself was NOT mutated by the read (still length prefix_len).
    assert cache2.get_seq_length() == prefix_len


def test_patched_forward_replaces_class_method():
    trex_patch.apply()
    import qwen_vla.modeling_qwen3vl_mot as m
    assert m.Qwen3VLAttentionMoT.forward is trex_patch._patched_attention_forward


# ── Patch 3: extract_merged_vision_features ──────────────────────────────────────────────
# Found (and only exercisable) against a real forward pass on the real midtrain checkpoint
# (§12 step 10b) -- see origami/PROGRESS.md for the end-to-end verification. These cover the
# unwrap logic itself in isolation, CPU-only, no checkpoint needed.

def test_extract_merged_vision_features_prefers_pooler_output():
    """The actual bug scenario: our installed transformers' Qwen3VLVisionModel.forward always
    returns this ModelOutput (never a bare tuple) -- `.pooler_output` is the real merged
    sequence (`self.merger(hidden_states)`); `.last_hidden_state` is pre-merge and has the
    wrong token count for the `<image_pad>` mask."""
    from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

    last_hidden = torch.randn(40, 8)   # pre-merge: more tokens
    pooled = torch.randn(10, 8)        # post-merge: matches the image-pad count
    out = BaseModelOutputWithDeepstackFeatures(
        last_hidden_state=last_hidden, pooler_output=pooled, deepstack_features=[])
    assert torch.equal(trex_patch.extract_merged_vision_features(out), pooled)


def test_extract_merged_vision_features_falls_back_to_last_hidden_state():
    class _NoPooler:
        last_hidden_state = torch.randn(3, 4)
    obj = _NoPooler()
    assert torch.equal(trex_patch.extract_merged_vision_features(obj), obj.last_hidden_state)


def test_extract_merged_vision_features_falls_back_to_tuple():
    """Upstream's assumed old-transformers-era shape: `(merged_hidden_states, deepstack)`."""
    merged = torch.randn(6, 4)
    deepstack = [torch.randn(6, 4)]
    assert torch.equal(trex_patch.extract_merged_vision_features((merged, deepstack)), merged)


def test_extract_merged_vision_features_falls_back_to_plain_tensor():
    t = torch.randn(5, 4)
    assert trex_patch.extract_merged_vision_features(t) is t


def test_prepare_inputs_embeds_is_patched_by_apply():
    trex_patch.apply()
    import qwen_vla.modeling_vla as modeling_vla
    assert modeling_vla.Qwen3VLVLAModel.prepare_inputs_embeds is trex_patch._patched_prepare_inputs_embeds


# ── Patch 4: Qwen3VLRotaryEmbeddingWrapper.__init__'s rope_parameters shim ───────────────

def test_rope_wrapper_builds_rope_parameters_and_forwards():
    """Reproduces the exact bug: unpatched, this constructor raises
    `AttributeError: '_RopeCfg' object has no attribute 'rope_parameters'` on our pinned
    transformers before a single tensor is touched. Patched, it must both construct AND
    produce correctly-shaped cos/sin for a real `Qwen3VLTextRotaryEmbedding`."""
    from types import SimpleNamespace as _SNS
    trex_patch.apply()
    import qwen_vla.modeling_qwen3vl_mot as m

    head_dim = 32
    config = _SNS(
        hidden_size=64, num_attention_heads=2, head_dim=head_dim,
        max_position_embeddings=128, rope_theta=1000000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [8, 4, 4]},  # sums to head_dim/2
        partial_rotary_factor=1.0,
    )
    wrapper = m.Qwen3VLRotaryEmbeddingWrapper(config)
    x = torch.randn(1, 5, head_dim)
    position_ids = torch.zeros(3, 1, 5, dtype=torch.long)   # M-RoPE [3, B, L]
    cos, sin = wrapper(x, position_ids)
    assert cos.shape == (1, 5, head_dim)
    assert sin.shape == (1, 5, head_dim)


# ── Patch 5: Qwen3VLVLAModel.get_rope_index's mm_token_type_ids forwarding ───────────────
# Full path (including the `_RopeStub`-needs-a-real-`self` wrinkle) is only exercisable
# against the real installed transformers `Qwen3VLModel.get_rope_index` with a real image --
# verified end-to-end against the real midtrain checkpoint (§12 step 10b, see PROGRESS.md).
# These test the dispatch logic in `_patched_get_rope_index` itself in isolation.

def test_get_rope_index_forwards_mm_token_type_ids_when_supported():
    from types import SimpleNamespace as _SNS

    captured = {}

    def fake_rope_fn(input_ids, mm_token_type_ids, image_grid_thw=None, attention_mask=None):
        captured["mm"] = mm_token_type_ids
        return "position_ids", "deltas"

    fake_self = _SNS(_rope_index_fn=fake_rope_fn, image_token_id=999)
    input_ids = torch.tensor([[1, 999, 2, 999]])
    out = trex_patch._patched_get_rope_index(fake_self, input_ids)
    assert out == ("position_ids", "deltas")
    assert torch.equal(captured["mm"], (input_ids == 999).to(torch.int64))


def test_get_rope_index_falls_back_when_fn_has_no_bound_self_and_no_mm_param():
    from types import SimpleNamespace as _SNS

    def fake_rope_fn(input_ids, image_grid_thw=None, attention_mask=None):
        return "ok-old-signature"

    fake_self = _SNS(_rope_index_fn=fake_rope_fn, image_token_id=999)
    input_ids = torch.tensor([[1, 2, 3]])
    assert trex_patch._patched_get_rope_index(fake_self, input_ids) == "ok-old-signature"


def test_get_rope_index_none_fallback_is_sequential_positions():
    from types import SimpleNamespace as _SNS

    fake_self = _SNS(_rope_index_fn=None)
    input_ids = torch.zeros((2, 5), dtype=torch.long)
    position_ids, deltas = trex_patch._patched_get_rope_index(fake_self, input_ids)
    assert position_ids.shape == (2, 5)
    assert deltas is None


def test_get_rope_index_is_patched_by_apply():
    trex_patch.apply()
    import qwen_vla.modeling_vla as modeling_vla
    assert modeling_vla.Qwen3VLVLAModel.get_rope_index is trex_patch._patched_get_rope_index
