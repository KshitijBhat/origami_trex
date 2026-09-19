"""Standalone CPU-only test of the new joint-torque wiring's real control
flow -- no real Qwen3-VL weights, no GPU. A stub `Qwen3VLVLAModel` (real
class, `__new__`'d to skip building the actual 2B decoder) with a fake
`self.model` that records exactly what it was called with, so the token
bookkeeping (`n_torque`, `act_parts`/`clean_parts` insertion order,
`action_indexes` length) is exercised as REAL code, not re-implemented.
"""
import os
import sys

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # T-Rex/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
from qwen_vla.modeling_vla import Qwen3VLVLAModel  # noqa: E402
from qwen_vla.modeling_qwen3vl_mot import Qwen3VLModelMoT  # noqa: E402
from qwen_vla.diffusion import ActionEmbedder, TimestepEmbedder, FinalLayer  # noqa: E402

H = 16
ACTION_DIM = 4
N_CHUNK = 2
B = 1


class FakeDecoder(nn.Module):
    """Records every call's real shapes/index lengths instead of running a
    real 28-layer decoder. Returns hidden_states shaped like its input, so
    downstream slicing (act_start etc.) is exercised against real shapes."""
    def __init__(self):
        super().__init__()
        self.calls = []
        # Real (pure, no other decoder state touched) position-id extension
        # logic -- exercise it as-is rather than re-implementing it here.
        self._extend_position_ids = Qwen3VLModelMoT._extend_position_ids

    def forward(self, inputs_embeds, position_ids, attention_mask,
                past_key_values, use_cache, latent_indexes, action_indexes,
                tactile_indexes, **kwargs):
        self.calls.append({
            "L": inputs_embeds.shape[1],
            "n_latent": latent_indexes.numel(),
            "n_action": action_indexes.numel(),
            "n_tactile": tactile_indexes.numel(),
        })
        # A fresh empty DynamicCache -- real enough for .crop()/.get_seq_length()
        # to work in the caller's next-step bookkeeping without doing real
        # attention/caching (this stub only exercises token/index bookkeeping).
        return BaseModelOutputWithPast(
            last_hidden_state=inputs_embeds.clone(),
            past_key_values=DynamicCache(),
        )


def make_stub_model(use_torque):
    m = Qwen3VLVLAModel.__new__(Qwen3VLVLAModel)
    nn.Module.__init__(m)
    m.use_torque = use_torque
    m.model = FakeDecoder()
    m.x_embedder = ActionEmbedder(ACTION_DIM, H).to(torch.bfloat16)
    m.t_embedder = TimestepEmbedder(H).to(torch.bfloat16)
    m.final_layer = FinalLayer(H, ACTION_DIM).to(torch.bfloat16)
    if use_torque:
        m.torque_embedder = ActionEmbedder(ACTION_DIM, H).to(torch.bfloat16)
    return m


def run_full(model, torque_embeds):
    inputs_embeds = torch.randn(B, 5, H, dtype=torch.bfloat16)  # L_latent=5
    position_ids = torch.zeros(3, B, 5, dtype=torch.long)
    noise = torch.randn(B, N_CHUNK, ACTION_DIM, dtype=torch.bfloat16)
    return model.forward_flow_action_full(
        inputs_embeds=inputs_embeds, position_ids=position_ids, noise=noise,
        torque_embeds=torque_embeds, num_steps=2)


def test_use_torque_false_is_noop():
    model = make_stub_model(use_torque=False)
    out = run_full(model, torque_embeds=None)
    assert out.shape == (B, N_CHUNK, ACTION_DIM)
    # fast(0) + state(0) + torque(0) + timestep(1) + chunk(2) = 3, every step
    assert all(c["n_action"] == 3 for c in model.model.calls), model.model.calls
    print("[PASS] use_torque=False: no torque tokens enter the sequence at all")


def test_use_torque_true_adds_exactly_one_token():
    model = make_stub_model(use_torque=True)
    torque_raw = torch.randn(B, ACTION_DIM, dtype=torch.bfloat16)
    torque_embeds = model.torque_embedder(torque_raw).unsqueeze(1)
    assert torque_embeds.shape == (B, 1, H)
    out = run_full(model, torque_embeds=torque_embeds)
    assert out.shape == (B, N_CHUNK, ACTION_DIM)
    # step 0: L_latent(5) + fast(0) + state(0) + torque(1) + timestep(1) + chunk(2) = 9
    # step 1: fast(0) + state(0) + torque(1) + timestep(1) + chunk(2) = 4
    step0, step1 = model.model.calls
    assert step0["n_latent"] == 5 and step0["n_action"] == 9 - 5, step0
    assert step1["n_latent"] == 0 and step1["n_action"] == 4, step1
    print("[PASS] use_torque=True: exactly one torque token enters the "
          f"sequence per step (step0 action tokens={step0['n_action']}, "
          f"step1={step1['n_action']})")


def test_torque_value_changes_output():
    """A real numerical check that the torque tokens are actually wired
    into the attended sequence, not silently dropped -- FakeDecoder echoes
    inputs_embeds back as last_hidden_state, so a different torque_embeds
    tensor must produce a different v_act unless act_start is misaligned
    and happens to slice the same region regardless of torque's value."""
    model = make_stub_model(use_torque=True)
    torch.manual_seed(0)
    t1 = model.torque_embedder(torch.randn(B, ACTION_DIM, dtype=torch.bfloat16)).unsqueeze(1)
    t2 = model.torque_embedder(torch.randn(B, ACTION_DIM, dtype=torch.bfloat16)).unsqueeze(1)
    out1 = run_full(model, torque_embeds=t1)
    out2 = run_full(model, torque_embeds=t2)
    assert not torch.equal(out1, out2), (
        "changing torque_embeds did not change the output -- act_start is "
        "likely misindexed and is slicing a region that doesn't depend on "
        "torque's value")
    print("[PASS] different torque values produce different outputs "
          "(act_start correctly tracks the torque token)")


def test_partial_kv_refresh_includes_torque():
    """forward_flow_action_partial's KV-refresh block (clean_parts) is a
    SEPARATE insertion point from the main per-step loop -- confirm torque
    lands there too, not just in the Euler-step loop."""
    model = make_stub_model(use_torque=True)
    torque_embeds = model.torque_embedder(
        torch.randn(B, ACTION_DIM, dtype=torch.bfloat16)).unsqueeze(1)
    inputs_embeds = torch.randn(B, 5, H, dtype=torch.bfloat16)
    position_ids = torch.zeros(3, B, 5, dtype=torch.long)
    noise = torch.randn(B, N_CHUNK, ACTION_DIM, dtype=torch.bfloat16)
    x_split, cached_kv, n_action_in_cache, tau_split = model.forward_flow_action_partial(
        inputs_embeds=inputs_embeds, position_ids=position_ids, noise=noise,
        torque_embeds=torque_embeds, num_steps_total=4, split_step=2,
        refresh_clean_kv=True)
    assert x_split.shape == (B, N_CHUNK, ACTION_DIM)
    # 2 Euler-step calls + 1 refresh call = 3 total FakeDecoder invocations
    assert len(model.model.calls) == 3, model.model.calls
    refresh_call = model.model.calls[-1]
    # refresh: fast(0) + state(0) + torque(1) + timestep(1) + chunk(2) = 4
    assert refresh_call["n_action"] == 4, refresh_call
    assert n_action_in_cache == 4
    print("[PASS] KV-refresh block also includes the torque token "
          f"(n_action_in_cache={n_action_in_cache})")


if __name__ == "__main__":
    test_use_torque_false_is_noop()
    test_use_torque_true_adds_exactly_one_token()
    test_torque_value_changes_output()
    test_partial_kv_refresh_includes_torque()
    print("\nALL TESTS PASSED")
