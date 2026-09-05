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
