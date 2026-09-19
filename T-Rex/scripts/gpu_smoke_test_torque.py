"""Real-GPU smoke test for the joint-torque wiring (feature/joint-torque).

Builds the REAL Qwen3-VL-2B-Instruct model + processor, runs
forward_flow_action_full/_partial with torque_embeds=None (regression) vs a
real torque_embeds tensor (new), checking: no crash, correct shapes, no
NaN/Inf, and that torque measurably changes the output -- same pattern as
gpu_smoke_test_memory.py.

    python scripts/gpu_smoke_test_torque.py
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
ACTION_DIM = 65
ACTION_CHUNK = 16
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

    print(f"Loading Qwen3VLVLAModel from {MODEL_PATH} (real weights, use_torque=True) ...")
    model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
        MODEL_PATH,
        action_dim=ACTION_DIM, action_chunk=ACTION_CHUNK,
        use_robot_state=False, use_torque=True,
        torch_dtype=DTYPE,
    ).to(DEVICE)
    model = model.to(DTYPE)
    model.eval()
    print("Model loaded.")

    B = 1
    merge = getattr(model.visual, "spatial_merge_size", 2)

    live_images = [fake_img((200, 50, 50)), fake_img((50, 200, 50)), fake_img((50, 50, 200))]
    live_inp = build_chat_inputs(processor, live_images, text="fold the paper in half")
    input_ids, attention_mask, pixel_values, image_grid_thw = to_device(live_inp)

    with torch.no_grad():
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
    torque_raw = torch.randn(B, ACTION_DIM, device=DEVICE, dtype=DTYPE)
    torque_embeds = model.torque_embedder(torque_raw).unsqueeze(1)
    assert torque_embeds.shape == (B, 1, model.config.hidden_size)

    print("\n=== forward_flow_action_full: torque_embeds=None (regression) ===")
    with torch.no_grad():
        out_a = model.forward_flow_action_full(
            inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
            attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
            fast_embeds=fast_embeds, num_steps=4, torque_embeds=None)
    check_finite("torque_embeds=None output", out_a)
    assert out_a.shape == (B, ACTION_CHUNK, ACTION_DIM)
    print(f"[PASS] torque_embeds=None: shape {tuple(out_a.shape)}, finite")

    print("\n=== forward_flow_action_full: torque_embeds=<real tensor> (new) ===")
    with torch.no_grad():
        out_b = model.forward_flow_action_full(
            inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
            attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
            fast_embeds=fast_embeds, num_steps=4, torque_embeds=torque_embeds)
    check_finite("torque_embeds=<real> output", out_b)
    assert out_b.shape == (B, ACTION_CHUNK, ACTION_DIM)
    diff = (out_b.float() - out_a.float()).abs().max().item()
    print(f"[PASS] torque_embeds=<real>: shape {tuple(out_b.shape)}, finite, "
          f"max abs diff vs no-torque = {diff:.6f}")
    assert diff > 0, "torque should change the output -- got an exact match, suspicious"

    print("\n=== forward_flow_action_partial: torque_embeds=<real tensor> ===")
    with torch.no_grad():
        x_split, cached_kv, n_action_in_cache, tau_split = model.forward_flow_action_partial(
            inputs_embeds=slow_embeds, position_ids=pos_ids, noise=noise,
            attention_mask=attention_mask[:, :L_slow] if attention_mask is not None else None,
            fast_embeds=fast_embeds, num_steps_total=4, split_step=2,
            torque_embeds=torque_embeds)
    check_finite("forward_flow_action_partial x_split", x_split)
    assert x_split.shape == (B, ACTION_CHUNK, ACTION_DIM)
    print(f"[PASS] forward_flow_action_partial: x_split shape {tuple(x_split.shape)}, "
          f"finite, n_action_in_cache={n_action_in_cache}, tau_split={tau_split:.3f}")

    print("\nALL GPU TORQUE SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
