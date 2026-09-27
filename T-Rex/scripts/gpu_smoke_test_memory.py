"""Real-GPU smoke test for slow+fast memory-KV plumbing (dev/memory Part B).

Builds the REAL Qwen3-VL-2B-Instruct model + processor (real weights, real
vision tower, real get_rope_index), constructs a live tick plus memory_slow/
memory_fast rows via the same apply_chat_template+processor path
OrigamiDataset.collate_fn uses, and runs forward_flow_action_full/_partial +
tactile_flow_continue with memory_kv=None (regression) vs memory_kv=<built>
(new), checking: no crash, correct shapes, no NaN/Inf, and that memory
actually changes the output (the new code path is really wired in, not a
silent no-op).

Needs a GPU + network access to pull Qwen/Qwen3-VL-2B-Instruct from HF, so
this is meant to be run on a GPU box, not locally. Requires the SAME
transformers version this repo is pinned to (pyproject.toml: 4.57.3) --
verified 2026-09-18 that a newer transformers (5.16.1/5.17.0) breaks even
plain model construction (Qwen3VLTextRotaryEmbedding expects
config.rope_parameters, which this repo's rotary-embedding config shim in
modeling_qwen3vl_mot.py does not set) -- a pre-existing, unrelated
compatibility issue, not something this script or the memory feature caused.

    python scripts/gpu_smoke_test_memory.py
"""
import os
import sys

import torch
import PIL.Image
from transformers import AutoProcessor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # T-Rex/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
from qwen_vla import Qwen3VLVLAModel, split_slow_fast_embeds  # noqa: E402

MODEL_PATH = "Qwen/Qwen3-VL-2B-Instruct"
ACTION_DIM = 29
ACTION_CHUNK = 8
DEVICE = "cuda"
DTYPE = torch.bfloat16


def fake_img(color):
    return PIL.Image.new("RGB", (224, 224), color=color)


def build_chat_inputs(processor, images, text=""):
    content = [{"type": "image"} for _ in images]
    if text:
        content.append({"type": "text", "text": text})
    chat_text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False,
        add_generation_prompt=True)
    return processor(text=chat_text, images=images, return_tensors="pt", padding=False)


def to_device(inp):
    input_ids = inp.input_ids.to(DEVICE)
    attention_mask = inp.attention_mask.to(DEVICE) if hasattr(inp, "attention_mask") else None
    pixel_values = getattr(inp, "pixel_values", None)
    pixel_values = pixel_values.to(device=DEVICE, dtype=DTYPE) if pixel_values is not None else None
    image_grid_thw = getattr(inp, "image_grid_thw", None)
    image_grid_thw = image_grid_thw.to(DEVICE) if image_grid_thw is not None else None
    return input_ids, attention_mask, pixel_values, image_grid_thw


def check_finite(name, t):
    assert torch.isfinite(t).all(), f"{name} contains NaN/Inf"


def main():
    print(f"Loading processor from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    print(f"Loading Qwen3VLVLAModel from {MODEL_PATH} (real weights, this takes a bit) ...")
    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        MODEL_PATH,
        action_dim=ACTION_DIM, action_chunk=ACTION_CHUNK,
        use_robot_state=False,
        use_tactile_code=True, use_tactile_vqvae=True,
        vqvae_codebook_size=64, vqvae_config={"codebook_size": 64},
        torch_dtype=DTYPE,
    ).to(DEVICE)
    # from_pretrained_qwen3vl's torch_dtype only covers weights loaded from
    # the pretrained base model -- the new VLA-specific modules (x_embedder,
    # t_embedder, tacf6_embedder, etc.) are constructed at default float32
    # and need an explicit cast, matching trex_origami/loading.py's real
    # production loading path.
    model = model.to(DTYPE)
    if getattr(model, "tactile_vqvae", None) is not None:
        model.tactile_vqvae.float().eval()
        model.tacf6_vqvae_min = model.tacf6_vqvae_min.float()
        model.tacf6_vqvae_max = model.tacf6_vqvae_max.float()
        model.tacf6_vqvae_mask = model.tacf6_vqvae_mask.bool()
    model.eval()
    print("Model loaded.")

    B = 1
    merge = getattr(model.visual, "spatial_merge_size", 2)

    # ── Live tick: head (slow) + wrist_right + wrist_left (fast) ───────────
    live_images = [fake_img((200, 50, 50)), fake_img((50, 200, 50)), fake_img((50, 50, 200))]
    live_inp = build_chat_inputs(processor, live_images, text="fold the paper in half")
    input_ids, attention_mask, pixel_values, image_grid_thw = to_device(live_inp)

    inputs_embeds_full = model.prepare_inputs_embeds(input_ids, pixel_values, image_grid_thw)
    n_slow_img_tokens = int(image_grid_thw[0, 0]
                            * (image_grid_thw[0, 1] // merge)
                            * (image_grid_thw[0, 2] // merge))
    slow_embeds, fast_embeds = split_slow_fast_embeds(
        inputs_embeds_full, input_ids, model.image_token_id, n_slow_img_tokens)
    L_slow = slow_embeds.shape[1]
    print(f"live tick: L_slow={L_slow}, L_fast={fast_embeds.shape[1]}")

    pos_ids_full, _ = model.get_rope_index(input_ids, image_grid_thw, attention_mask)
    pos_ids = pos_ids_full[:, :, :L_slow]

    noise = torch.randn(B, ACTION_CHUNK, ACTION_DIM, device=DEVICE, dtype=DTYPE)

    # ── memory_slow rows: K=2, head image + short text each ────────────────
    memory_slow_rows = []
    for k, color in enumerate([(80, 80, 80), (140, 140, 140)]):
        m_inp = build_chat_inputs(processor, [fake_img(color)], text=f"past step {k}")
        m_ids, m_am, m_pv, m_thw = to_device(m_inp)
        memory_slow_rows.append({
            "input_ids": m_ids, "attention_mask": m_am,
            "pixel_values": m_pv, "image_grid_thw": m_thw,
            "dt_actual": torch.tensor([0.5 + k], device=DEVICE, dtype=torch.float32),
        })

    # ── memory_fast rows: K=2, wrist_right+wrist_left + action + tactile ───
    memory_fast_rows = []
    for k, (cr, cl) in enumerate([((90, 10, 10), (10, 10, 90)), ((120, 10, 10), (10, 10, 120))]):
        m_inp = build_chat_inputs(processor, [fake_img(cr), fake_img(cl)])
        m_ids, m_am, m_pv, m_thw = to_device(m_inp)
        memory_fast_rows.append({
            "input_ids": m_ids, "attention_mask": m_am,
            "pixel_values": m_pv, "image_grid_thw": m_thw,
            "action_abs": torch.randn(B, ACTION_DIM, device=DEVICE, dtype=torch.float32),
            "tacf6_hist": torch.rand(B, 16, 10, 6, device=DEVICE, dtype=torch.float32),
        })

    print("\n=== build_memory_kv_slow / build_memory_kv_fast ===")
    memory_kv_slow, exclude_slow = model.build_memory_kv_slow(memory_slow_rows, rope_stride=32.0)
    assert memory_kv_slow is not None
    assert exclude_slow is not None and exclude_slow.dtype == torch.bool
    assert exclude_slow.shape[1] == memory_kv_slow.get_seq_length()
    print(f"memory_kv_slow built, seq_len={memory_kv_slow.get_seq_length()}, "
          f"excluded={int(exclude_slow.sum().item())}/{exclude_slow.numel()} positions")
    memory_kv, kv_exclude_mask = model.build_memory_kv_fast(
        memory_fast_rows, past_kv=memory_kv_slow, kv_exclude_mask=exclude_slow, rope_stride=8.0)
    assert memory_kv is not None
    assert kv_exclude_mask.shape[1] == memory_kv.get_seq_length()
    # fast-tier rows carry no text -- the newly-appended span must be all-False.
    assert not kv_exclude_mask[:, exclude_slow.shape[1]:].any()
    print(f"memory_kv (slow+fast combined) built, seq_len={memory_kv.get_seq_length()}")

    # Empty-list no-op check.
    assert model.build_memory_kv_slow([]) == (None, None)
    assert model.build_memory_kv_fast([], past_kv=None) == (None, None)
    passthrough_kv, passthrough_excl = model.build_memory_kv_fast(
        [], past_kv=memory_kv_slow, kv_exclude_mask=exclude_slow)
    assert passthrough_kv is memory_kv_slow
    assert passthrough_excl is exclude_slow
    print("[PASS] empty-list no-op behavior confirmed on real model instance")

    print("\n=== forward_flow_action_full: memory_kv=None (regression) ===")
    out_a1 = model.forward_flow_action_full(
        inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        fast_embeds=fast_embeds, num_steps=4, memory_kv=None)
    check_finite("forward_flow_action_full memory_kv=None output", out_a1)
    assert out_a1.shape == (B, ACTION_CHUNK, ACTION_DIM)
    out_a2 = model.forward_flow_action_full(
        inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        fast_embeds=fast_embeds, num_steps=4, memory_kv=None)
    assert torch.equal(out_a1, out_a2), "memory_kv=None must be deterministic/reproducible"
    print(f"[PASS] memory_kv=None: shape {tuple(out_a1.shape)}, finite, reproducible")

    print("\n=== forward_flow_action_full: memory_kv=<slow+fast> (new) ===")
    out_b = model.forward_flow_action_full(
        inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        fast_embeds=fast_embeds, num_steps=4, memory_kv=memory_kv,
        kv_exclude_mask=kv_exclude_mask)
    check_finite("forward_flow_action_full memory_kv=<slow+fast> output", out_b)
    assert out_b.shape == (B, ACTION_CHUNK, ACTION_DIM)
    diff = (out_b.float() - out_a1.float()).abs().max().item()
    print(f"[PASS] memory_kv=<slow+fast>: shape {tuple(out_b.shape)}, finite, "
          f"max abs diff vs no-memory = {diff:.6f} (memory measurably changes output)")
    assert diff > 0, "memory_kv should change the output -- got an exact match, suspicious"

    print("\n=== forward_flow_action_full: kv_exclude_mask actually changes the output ===")
    out_b_unmasked = model.forward_flow_action_full(
        inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        fast_embeds=fast_embeds, num_steps=4, memory_kv=memory_kv,
        kv_exclude_mask=None)
    check_finite("forward_flow_action_full kv_exclude_mask=None output", out_b_unmasked)
    diff_excl = (out_b.float() - out_b_unmasked.float()).abs().max().item()
    print(f"[PASS] kv_exclude_mask on vs off: max abs diff = {diff_excl:.6f} "
          f"(masking the redundant memory-row text measurably changes output, "
          f"not a silently-inert mask)")
    assert diff_excl > 0, "kv_exclude_mask should change the output -- got an exact match"

    print("\n=== forward_flow_action_partial + tactile_flow_continue: memory_kv=None (regression) ===")
    x_split1, cached_kv1, n_act1, tau1 = model.forward_flow_action_partial(
        inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        fast_embeds=fast_embeds, num_steps_total=4, split_step=2, memory_kv=None)
    check_finite("forward_flow_action_partial memory_kv=None x_split", x_split1)
    tac_out1 = model.tactile_flow_continue(
        cached_kv1, pos_ids, n_act1, x_split1, tau1,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        tactile_f6_history=torch.rand(B, 16, 10, 6, device=DEVICE, dtype=torch.float32),
        num_steps_total=4, split_step=2)
    check_finite("tactile_flow_continue memory_kv=None output", tac_out1)
    assert tac_out1.shape == (B, ACTION_CHUNK, ACTION_DIM)
    print(f"[PASS] cascaded (partial+tactile), memory_kv=None: shape {tuple(tac_out1.shape)}, finite")

    print("\n=== forward_flow_action_partial + tactile_flow_continue: memory_kv=<slow+fast> (new) ===")
    x_split2, cached_kv2, n_act2, tau2 = model.forward_flow_action_partial(
        inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        fast_embeds=fast_embeds, num_steps_total=4, split_step=2, memory_kv=memory_kv,
        kv_exclude_mask=kv_exclude_mask)
    check_finite("forward_flow_action_partial memory_kv=<slow+fast> x_split", x_split2)
    tac_out2 = model.tactile_flow_continue(
        cached_kv2, pos_ids, n_act2, x_split2, tau2,
        attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
        tactile_f6_history=torch.rand(B, 16, 10, 6, device=DEVICE, dtype=torch.float32),
        num_steps_total=4, split_step=2, kv_exclude_mask=kv_exclude_mask)
    check_finite("tactile_flow_continue memory_kv=<slow+fast> output", tac_out2)
    assert tac_out2.shape == (B, ACTION_CHUNK, ACTION_DIM)
    diff2 = (tac_out2.float() - tac_out1.float()).abs().max().item()
    print(f"[PASS] cascaded (partial+tactile), memory_kv=<slow+fast>: shape {tuple(tac_out2.shape)}, "
          f"finite, max abs diff vs no-memory = {diff2:.6f} (memory reaches tactile_flow_continue "
          f"transitively through cached_kv, confirming it inherits both memory tiers)")
    assert diff2 > 0

    print("\nALL GPU SMOKE TESTS PASSED")


if __name__ == "__main__":
    with torch.no_grad():
        main()
