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


if __name__ == "__main__":
    test_shift_position_ids_basic()
    test_shift_position_ids_zero_offset_is_noop()
    test_front_pad_attention_mask_no_past_kv()
    test_front_pad_attention_mask_zero_length_past_kv()
    test_front_pad_attention_mask_real_padding()
    test_front_pad_attention_mask_none_mask_with_past_kv()
    test_end_to_end_total_axis_consistency_with_build_causal_mask()
    print("\nALL TESTS PASSED")
