"""Standalone CPU-only test of the new memory-KV plumbing's pure logic:
`Qwen3VLModelMoT.shift_position_ids_for_memory` and
`Qwen3VLVLAModel._front_pad_attention_mask` -- both are staticmethods with no
model weights involved, so they can be tested directly without instantiating
the real 2B model or touching a GPU.
"""
import os
import sys

import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # T-Rex/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
from qwen_vla.modeling_qwen3vl_mot import Qwen3VLModelMoT  # noqa: E402
from qwen_vla.modeling_vla import Qwen3VLVLAModel  # noqa: E402


def test_shift_position_ids_basic():
    # [3, B=2, L=4], batch item 0 all zeros-based text positions 0..3,
    # batch item 1 a vision-shaped block (t=0 const, h/w vary) starting at 0.
    position_ids = torch.tensor([
        [[0, 1, 2, 3], [0, 0, 0, 0]],
        [[0, 1, 2, 3], [0, 0, 1, 1]],
        [[0, 1, 2, 3], [0, 1, 0, 1]],
    ], dtype=torch.long)
    time_offset = torch.tensor([10, 50], dtype=torch.long)  # [B]

    shifted = Qwen3VLModelMoT.shift_position_ids_for_memory(position_ids, time_offset)
    assert shifted.shape == position_ids.shape

    # batch 0: everything shifted back by 10, internal structure preserved.
    assert torch.equal(shifted[:, 0, :], position_ids[:, 0, :] - 10)
    # batch 1: everything shifted back by 50.
    assert torch.equal(shifted[:, 1, :], position_ids[:, 1, :] - 50)

    # Internal relative structure within a row must be UNCHANGED by the
    # shift (a uniform translation) -- e.g. row 0's own td deltas.
    orig_deltas = position_ids[:, 0, 1:] - position_ids[:, 0, :-1]
    shifted_deltas = shifted[:, 0, 1:] - shifted[:, 0, :-1]
    assert torch.equal(orig_deltas, shifted_deltas)

    # Values must land at or below 0 for any nonzero time_offset (a "past" row).
    assert (shifted[:, 0, :] <= 0).all()
    assert (shifted[:, 1, :] <= 0).all()
    print("[PASS] shift_position_ids_for_memory: uniform per-batch translation, "
          "internal structure preserved, lands at negative/zero positions")


def test_shift_position_ids_zero_offset_is_noop():
    position_ids = torch.arange(12).view(3, 1, 4)
    time_offset = torch.zeros(1, dtype=torch.long)
    shifted = Qwen3VLModelMoT.shift_position_ids_for_memory(position_ids, time_offset)
    assert torch.equal(shifted, position_ids)
    print("[PASS] shift_position_ids_for_memory: zero offset is a true no-op")


def test_front_pad_attention_mask_no_past_kv():
    mask = torch.ones(2, 5, dtype=torch.long)
    out = Qwen3VLVLAModel._front_pad_attention_mask(
        mask, batch_size=2, seq_len=5, past_kv=None, device=torch.device("cpu"))
    assert out is mask, "no past_kv -> attention_mask must pass through unchanged"
    print("[PASS] _front_pad_attention_mask: past_kv=None is a true no-op (identity)")


class _FakePastKV:
    """Minimal stand-in exposing only get_seq_length(), what
    _front_pad_attention_mask actually calls -- avoids needing a real
    DynamicCache/model forward pass for this pure-logic test."""
    def __init__(self, length):
        self._len = length

    def get_seq_length(self):
        return self._len


def test_front_pad_attention_mask_zero_length_past_kv():
    mask = torch.ones(2, 5, dtype=torch.long)
    out = Qwen3VLVLAModel._front_pad_attention_mask(
        mask, batch_size=2, seq_len=5, past_kv=_FakePastKV(0), device=torch.device("cpu"))
    assert out is mask
    print("[PASS] _front_pad_attention_mask: past_kv with length 0 is a true no-op")


def test_front_pad_attention_mask_real_padding():
    B, L, past_len = 2, 5, 7
    # Real padding mask: batch item 1 has 2 pad tokens at the front (0s).
    mask = torch.ones(B, L, dtype=torch.long)
    mask[1, :2] = 0
    out = Qwen3VLVLAModel._front_pad_attention_mask(
        mask, batch_size=B, seq_len=L, past_kv=_FakePastKV(past_len), device=torch.device("cpu"))
    assert out.shape == (B, past_len + L)
    # Front past_len positions are always 1 (memory is never padding).
    assert (out[:, :past_len] == 1).all()
    # The tail exactly reproduces the original mask, unchanged.
    assert torch.equal(out[:, past_len:], mask)
    print("[PASS] _front_pad_attention_mask: prepends past_len ones, "
          "preserves original mask exactly at the tail")


def test_front_pad_attention_mask_none_mask_with_past_kv():
    # attention_mask=None + memory present -> build an all-ones mask for the
    # live seq_len portion, then front-pad it: net result is "everything
    # attends", matching the implicit meaning of attention_mask=None.
    B, L, past_len = 3, 4, 6
    out = Qwen3VLVLAModel._front_pad_attention_mask(
        None, batch_size=B, seq_len=L, past_kv=_FakePastKV(past_len), device=torch.device("cpu"))
    assert out.shape == (B, past_len + L)
    assert (out == 1).all()
    print("[PASS] _front_pad_attention_mask: attention_mask=None + memory present "
          "-> synthesized all-ones mask of the right total shape")


def test_end_to_end_total_axis_consistency_with_build_causal_mask():
    """Directly exercises the real build_causal_mask with a front-padded
    mask (as forward_flow_action_full/_partial now produce it) and confirms
    the resulting additive mask places -inf / 0.0 in the positions the
    design intends: memory block always attendable (no -inf from padding),
    live latent's own pad tokens (if any) correctly masked, causal structure
    intact across the whole total axis."""
    from qwen_vla.modeling_qwen3vl_mot import build_causal_mask

    B, L_latent, n_act, past_len = 1, 3, 2, 5
    mask = torch.ones(B, L_latent, dtype=torch.long)
    padded = Qwen3VLVLAModel._front_pad_attention_mask(
        mask, batch_size=B, seq_len=L_latent, past_kv=_FakePastKV(past_len),
        device=torch.device("cpu"))
    assert padded.shape == (B, past_len + L_latent)

    seq_len = L_latent + n_act  # first denoising iteration: latent + act tokens
    causal = build_causal_mask(seq_len=seq_len, past_len=past_len,
                                device=torch.device("cpu"), dtype=torch.float32,
                                attention_mask=padded)
    total = past_len + seq_len
    assert causal.shape == (1, 1, seq_len, total)

    # No padding anywhere (all-ones mask) -> only causal structure should
    # produce -inf, never the padding term. Every query can see the full
    # memory block (kv_pos < past_len) since q_pos >= past_len always here.
    for q in range(seq_len):
        q_pos = past_len + q
        for kv in range(total):
            expected_masked = kv > q_pos  # causal: can't see the future
            is_masked = bool(torch.isinf(causal[0, 0, q, kv]) and causal[0, 0, q, kv] < 0)
            assert is_masked == expected_masked, (
                f"q={q} (abs {q_pos}) kv={kv}: masked={is_masked}, expected={expected_masked}")
    print("[PASS] end-to-end: front-padded mask + build_causal_mask gives the "
          "intended semantics -- full memory block always visible, causal "
          "structure intact across the whole [memory|latent|act] axis")


# ─── Fast-tier tests (build_memory_kv_fast's internals) ─────────────────────
# build_memory_kv_fast itself needs a real instantiated 2B model + vision
# tower (self.visual is None until externally loaded -- too heavy for a
# CPU-only test), so these exercise the same pure staticmethod composition
# it performs internally: _extend_position_ids (action-then-tactile
# ordering) + shift_position_ids_for_memory (uniform backward translation)
# + _front_pad_attention_mask (mask alignment), and the (K-i)*rope_stride
# offset formula in isolation.

def test_extend_then_shift_ordering_matches_content_order():
    """build_memory_kv_fast embeds content as [wrist | action | tactile] and
    calls _extend_position_ids(position_ids, n_action, n_tactile) -- confirm
    that ordering convention (action positions before tactile positions)
    really matches _extend_position_ids's own layout, then confirm the
    subsequent memory shift translates the WHOLE extended block uniformly
    (wrist + action + tactile together), not just the original wrist part.
    """
    from qwen_vla.modeling_qwen3vl_mot import Qwen3VLModelMoT as MoT

    B, L_wrist = 2, 3
    # Fresh get_rope_index()-shaped output for the wrist block: 0-based.
    position_ids = torch.arange(L_wrist).view(1, 1, L_wrist).expand(3, B, L_wrist).clone()
    n_action, n_tactile = 1, 2

    extended = MoT._extend_position_ids(position_ids, n_action, n_tactile)
    assert extended.shape == (3, B, L_wrist + n_action + n_tactile)
    # Wrist block unchanged.
    assert torch.equal(extended[:, :, :L_wrist], position_ids)
    # Action position immediately follows wrist's max (L_wrist - 1 -> L_wrist).
    assert (extended[:, :, L_wrist] == L_wrist).all()
    # Tactile positions follow the action position, still increasing.
    assert (extended[:, :, L_wrist + 1] == L_wrist + 1).all()
    assert (extended[:, :, L_wrist + 2] == L_wrist + 2).all()

    # Now shift the whole extended block as build_memory_kv_fast does --
    # every position (wrist AND appended action/tactile) must move by the
    # SAME per-batch amount, preserving their relative order/spacing.
    time_offset = torch.tensor([20, 20], dtype=extended.dtype)
    shifted = MoT.shift_position_ids_for_memory(extended, time_offset)
    assert torch.equal(shifted, extended - 20)
    # Relative ordering (wrist < action < tactile) must survive the shift.
    assert (shifted[:, :, L_wrist] > shifted[:, :, L_wrist - 1]).all()
    assert (shifted[:, :, L_wrist + 1] > shifted[:, :, L_wrist]).all()
    print("[PASS] _extend_position_ids + shift_position_ids_for_memory compose "
          "correctly for the [wrist | action | tactile] fast-memory row layout")


def test_fast_memory_offset_formula_oldest_first():
    """build_memory_kv_fast uses offset (K - i) * rope_stride for row i of K
    (oldest first, i=0..K-1) -- confirm this gives the oldest row the LARGEST
    offset and decreases monotonically to the newest, matching the dataset's
    own oldest->newest memory_fast ordering (origami_dataset.py)."""
    K, rope_stride = 3, 8.0
    offsets = [(K - i) * rope_stride for i in range(K)]
    assert offsets == [24.0, 16.0, 8.0]
    assert offsets == sorted(offsets, reverse=True)
    print(f"[PASS] fast-memory offset formula: oldest->newest offsets {offsets}, "
          f"monotonically decreasing as expected")


def test_front_pad_attention_mask_chains_across_slow_then_fast():
    """Simulates build_memory_kv_slow's output being fed as build_memory_kv_
    fast's past_kv: the fast tier's own front-pad must account for
    whatever the slow tier already accumulated, and the two front-pads
    compose correctly (total front-pad after both tiers == slow_len +
    fast_row's own past_len at call time)."""
    B, L_slow_total, L_fast_wrist = 2, 12, 5  # e.g. 4 slow rows * 3 tokens each

    # After build_memory_kv_slow finishes, its cache holds L_slow_total
    # tokens. build_memory_kv_fast's first row front-pads against THAT.
    fast_mask = torch.ones(B, L_fast_wrist, dtype=torch.long)
    padded_first_fast_row = Qwen3VLVLAModel._front_pad_attention_mask(
        fast_mask, batch_size=B, seq_len=L_fast_wrist,
        past_kv=_FakePastKV(L_slow_total), device=torch.device("cpu"))
    assert padded_first_fast_row.shape == (B, L_slow_total + L_fast_wrist)
    assert (padded_first_fast_row[:, :L_slow_total] == 1).all()
    assert torch.equal(padded_first_fast_row[:, L_slow_total:], fast_mask)

    # A second fast row calls _front_pad_attention_mask again with the
    # NEW past_len (slow + first fast row's full [wrist|action|tactile]
    # length, e.g. 5+1+2=8) -- confirm it keeps accounting correctly.
    n_action, n_tactile = 1, 2
    past_len_after_row0 = L_slow_total + (L_fast_wrist + n_action + n_tactile)
    padded_second_fast_row = Qwen3VLVLAModel._front_pad_attention_mask(
        fast_mask, batch_size=B, seq_len=L_fast_wrist,
        past_kv=_FakePastKV(past_len_after_row0), device=torch.device("cpu"))
    assert padded_second_fast_row.shape == (B, past_len_after_row0 + L_fast_wrist)
    assert (padded_second_fast_row[:, :past_len_after_row0] == 1).all()
    print("[PASS] _front_pad_attention_mask chains correctly across slow-then-fast "
          "memory tiers accumulating in one combined cache")


if __name__ == "__main__":
    test_shift_position_ids_basic()
    test_shift_position_ids_zero_offset_is_noop()
    test_front_pad_attention_mask_no_past_kv()
    test_front_pad_attention_mask_zero_length_past_kv()
    test_front_pad_attention_mask_real_padding()
    test_front_pad_attention_mask_none_mask_with_past_kv()
    test_end_to_end_total_axis_consistency_with_build_causal_mask()
    test_extend_then_shift_ordering_matches_content_order()
    test_fast_memory_offset_formula_oldest_first()
    test_front_pad_attention_mask_chains_across_slow_then_fast()
    print("\nALL TESTS PASSED")
