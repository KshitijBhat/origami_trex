"""Standalone CPU-only test of build_causal_mask's new kv_exclude_mask
argument (dev/memory Part 2: mask redundant memory-row text tokens out of
attention). Pure tensor-level checks against the mask-building function
itself -- no GPU, no real model needed.
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # T-Rex/scripts/
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)                # T-Rex/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from qwen_vla.modeling_qwen3vl_mot import build_causal_mask  # noqa: E402


def test_no_op_when_none():
    """kv_exclude_mask=None must be bit-identical to today's behavior --
    the no-op guarantee every existing call site relies on."""
    torch.manual_seed(0)
    seq_len, past_len, B = 4, 6, 2
    attn = torch.ones(B, past_len + seq_len, dtype=torch.long)
    baseline = build_causal_mask(seq_len, past_len, "cpu", torch.float32, attention_mask=attn)
    with_none = build_causal_mask(seq_len, past_len, "cpu", torch.float32,
                                   attention_mask=attn, kv_exclude_mask=None)
    assert torch.equal(baseline, with_none)
    print("[PASS] kv_exclude_mask=None is bit-identical to omitting the argument")


def test_exclusion_hits_exact_positions():
    """-inf at exactly the excluded (query, kv) pairs, 0.0 elsewhere in the
    past span -- not just 'no crash'."""
    seq_len, past_len, B = 3, 5, 1
    kv_exclude_mask = torch.zeros(B, past_len, dtype=torch.bool)
    excluded_positions = [1, 3]  # simulate 2 text-token positions among 5 memory tokens
    kv_exclude_mask[:, excluded_positions] = True

    mask = build_causal_mask(seq_len, past_len, "cpu", torch.float32,
                              kv_exclude_mask=kv_exclude_mask)
    assert mask.shape == (1, 1, seq_len, past_len + seq_len)

    for kv_pos in range(past_len):
        col = mask[0, 0, :, kv_pos]
        if kv_pos in excluded_positions:
            assert torch.isinf(col).all() and (col < 0).all(), (
                f"kv_pos={kv_pos} should be -inf for every query, got {col}")
        else:
            # Still governed by the ordinary causal rule for positions before
            # past_len (always visible, since q_pos >= past_len > kv_pos for
            # every live query) -- exclusion must not touch these.
            assert (col == 0.0).all(), f"kv_pos={kv_pos} should be unaffected, got {col}"

    # Live (non-memory) span must never be excluded, only the causal rule applies.
    live_block = mask[0, 0, :, past_len:]
    causal_expected = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    assert torch.equal(live_block, causal_expected), (
        "kv_exclude_mask must never touch the live (non-memory) span")
    print("[PASS] kv_exclude_mask hides exactly the specified past positions "
          "from every query, leaves everything else (including the live span) untouched")


def test_composes_with_padding_mask():
    """kv_exclude_mask and the existing padding mask are independent
    additive terms -- a position can be excluded, padded, both, or neither,
    and the combination must simply be the sum of the two effects."""
    seq_len, past_len, B = 2, 4, 1
    total = past_len + seq_len
    attention_mask = torch.ones(B, total, dtype=torch.long)
    attention_mask[:, 0] = 0  # position 0 padded
    kv_exclude_mask = torch.zeros(B, past_len, dtype=torch.bool)
    kv_exclude_mask[:, 1] = True  # position 1 excluded (not padded)

    mask = build_causal_mask(seq_len, past_len, "cpu", torch.float32,
                              attention_mask=attention_mask, kv_exclude_mask=kv_exclude_mask)
    # Position 0: padded -> -inf for every query except the diagonal self-attend fixup
    # (that fixup only applies to LIVE diagonal positions, past_len=4 > 0, so position 0
    # is never on that diagonal -- every query should see -inf here).
    assert torch.isinf(mask[0, 0, :, 0]).all()
    # Position 1: excluded -> -inf for every query too.
    assert torch.isinf(mask[0, 0, :, 1]).all()
    # Positions 2, 3: neither padded nor excluded -> 0.0 (ordinary causal-visible past).
    assert (mask[0, 0, :, 2] == 0.0).all()
    assert (mask[0, 0, :, 3] == 0.0).all()
    print("[PASS] kv_exclude_mask composes correctly with the existing padding mask "
          "(independent additive terms, as designed)")


def test_kv_exclude_mask_batch_dim():
    """Different batch rows can exclude different positions -- confirms the
    mask isn't accidentally broadcast/shared across the batch dimension."""
    seq_len, past_len, B = 2, 3, 2
    kv_exclude_mask = torch.zeros(B, past_len, dtype=torch.bool)
    kv_exclude_mask[0, 0] = True   # row 0 excludes position 0 only
    kv_exclude_mask[1, 2] = True   # row 1 excludes position 2 only

    mask = build_causal_mask(seq_len, past_len, "cpu", torch.float32,
                              kv_exclude_mask=kv_exclude_mask)
    assert mask.shape[0] == B
    assert torch.isinf(mask[0, 0, :, 0]).all() and not torch.isinf(mask[0, 0, :, 2]).any()
    assert torch.isinf(mask[1, 0, :, 2]).all() and not torch.isinf(mask[1, 0, :, 0]).any()
    print("[PASS] kv_exclude_mask is genuinely per-batch-row, not accidentally shared")


def test_kv_exclude_mask_shorter_than_past_len():
    """Regression test for a real bug caught on a real-GPU run: a multi-step
    Euler loop (forward_flow_action_full/_partial) folds each step's own
    latent/action tokens into the SAME cache that originally held only
    memory content -- crop(-n_act) between steps leaves the memory+latent
    prefix in place, so past_len legitimately grows past kv_exclude_mask's
    original length on step 2+ (e.g. memory=442, then past_len=511 once the
    69-token latent block is folded in). kv_exclude_mask must NOT be assumed
    to cover the whole past span -- only its own (shorter) prefix should be
    excluded; the newly-cached live content beyond it must stay attendable,
    and the call must not crash from a shape mismatch."""
    seq_len, past_len, B = 2, 511, 1
    kv_exclude_mask = torch.zeros(B, 442, dtype=torch.bool)  # shorter than past_len=511
    kv_exclude_mask[:, 5] = True

    mask = build_causal_mask(seq_len, past_len, "cpu", torch.float32,
                              kv_exclude_mask=kv_exclude_mask)
    assert mask.shape == (1, 1, seq_len, past_len + seq_len)
    assert torch.isinf(mask[0, 0, :, 5]).all()          # explicitly excluded, still works
    assert (mask[0, 0, :, 6] == 0.0).all()               # unexcluded position within the mask's own span
    assert (mask[0, 0, :, 442] == 0.0).all()             # first position PAST kv_exclude_mask's length
    assert (mask[0, 0, :, 510] == 0.0).all()             # last past position (just below past_len) -- never excluded
    print("[PASS] kv_exclude_mask shorter than past_len: no crash, only its own "
          "prefix is excluded, newly-cached live content beyond it stays attendable")


if __name__ == "__main__":
    test_no_op_when_none()
    test_exclusion_hits_exact_positions()
    test_composes_with_padding_mask()
    test_kv_exclude_mask_batch_dim()
    test_kv_exclude_mask_shorter_than_past_len()
    print("\nALL TESTS PASSED")
