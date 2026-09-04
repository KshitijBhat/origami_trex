#!/usr/bin/env python3
"""Full-episode ViT self-attention sweep: extract frames at a fixed interval
from a camera video, run each through the frozen ViT (model.visual), capture
ALL blocks' attention-received heatmaps, save into one folder per layer, then
build one GIF per layer showing how that layer's attention evolves over the
episode.

Standalone, read-only w.r.t. training -- loads the checkpoint the same way
inspect_attention.py / vit_probe.py did, no train.py/accelerate involved.
"""
import argparse
import os
import subprocess
import sys
import types

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, "/workspace/origami_trex/T-Rex")
from trex_origami.loading import load_args_from_checkpoint, model_load

# Pure-PIL/numpy jet colormap lookup table (256 entries) -- avoids matplotlib's
# Figure/Axes/colorbar machinery entirely. A first version of this script used
# plt.subplots()+imshow()+savefig() per layer per frame (24 x 189 = 4536 fresh
# figures for one camera) and degraded badly mid-run: GPU utilization dropped
# to 0% and per-frame time went from ~9s to ~55s, almost certainly matplotlib
# Agg-backend figure-churn overhead accumulating despite plt.close(fig) each
# time. This LUT + PIL.Image.blend approach does the same visual job (heatmap
# overlaid on the frame at fixed alpha) with no Figure objects at all.
def _make_jet_lut():
    import matplotlib.cm as cm
    return (cm.jet(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


_JET_LUT = _make_jet_lut()


def save_heatmap_png(image, heat, out_path, alpha=0.45):
    """image: PIL RGB. heat: 2D float array (small grid, e.g. 16x16)."""
    norm = heat - heat.min()
    denom = norm.max()
    norm = norm / denom if denom > 1e-8 else norm
    idx = (norm * 255).astype(np.uint8)
    heat_small_rgb = _JET_LUT[idx]  # [gh, gw, 3]
    heat_img = PIL.Image.fromarray(heat_small_rgb, mode="RGB").resize(image.size, PIL.Image.BILINEAR)
    blended = PIL.Image.blend(image.convert("RGB"), heat_img, alpha=alpha)
    blended.save(out_path)


def extract_frames(video_path, out_dir, fps=1.0):
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, "frame_%05d.jpg")
    subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vf", f"fps={fps}", "-q:v", "2", pattern],
        check=True, capture_output=True,
    )
    frames = sorted(f for f in os.listdir(out_dir) if f.startswith("frame_"))
    return [os.path.join(out_dir, f) for f in frames]


def build_inputs(model, processor, image, instruction, device):
    content = [{"type": "image"}, {"type": "text", "text": instruction}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inp = processor(text=text, images=[image], return_tensors="pt", padding=False)
    pixel_values = inp.pixel_values.to(device, dtype=torch.bfloat16)
    grid_thw = inp.image_grid_thw.to(device)
    return pixel_values, grid_thw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--camera_name", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--frames_dir", required=True, help="where to extract raw frames")
    ap.add_argument("--out_dir", required=True, help="where to write per-layer output")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"[sweep] loading checkpoint {args.checkpoint_path}")
    load_args = load_args_from_checkpoint(args.checkpoint_path)
    model, processor, _ = model_load(load_args)
    model = model.to(device).eval()
    n_layers = len(model.visual.blocks)
    print(f"[sweep] n_layers={n_layers}")

    print(f"[sweep] extracting frames from {args.video_path} at {args.fps}fps")
    frame_paths = extract_frames(args.video_path, args.frames_dir, args.fps)
    print(f"[sweep] {len(frame_paths)} frames extracted")

    for li in range(n_layers):
        os.makedirs(os.path.join(args.out_dir, f"layer{li:02d}"), exist_ok=True)

    for fi, fpath in enumerate(frame_paths):
        image = PIL.Image.open(fpath).convert("RGB")
        image_size = (224, 224)
        if image.size != image_size:
            image = image.resize(image_size, PIL.Image.LANCZOS)
        pixel_values, grid_thw = build_inputs(model, processor, image, "", device)
        # ViT self-attention operates on the RAW pre-merge patch grid -- the
        # spatial_merge_size division only applies after all 24 blocks, inside
        # the merger module. Dividing here would make gh*gw not match the
        # actual attention sequence length (confirmed against vit_probe.py's
        # single-image check, which uses grid_thw directly with no division).
        gh = int(grid_thw[0, 1].item())
        gw = int(grid_thw[0, 2].item())

        captured = {}
        originals = {}

        def make_patched_i(idx):
            def patched(self, hidden_states, cu_seqlens, position_embeddings=None, max_seqlen=None, **kwargs):
                seq_length = hidden_states.shape[0]
                q, k, v = (self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
                           .permute(1, 0, 2, 3).unbind(0))
                cos, sin = position_embeddings
                from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb_vision
                q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
                q = q.transpose(0, 1).unsqueeze(0)
                k = k.transpose(0, 1).unsqueeze(0)
                v = v.transpose(0, 1).unsqueeze(0)
                attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
                attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
                captured[idx] = attn_weights[0].float().mean(dim=0).detach().cpu().numpy()
                attn_output = torch.matmul(attn_weights, v)
                attn_output = attn_output.transpose(0, 1).reshape(seq_length, -1).contiguous()
                return self.proj(attn_output)
            return patched

        for i, block in enumerate(model.visual.blocks):
            originals[i] = block.attn.forward
            block.attn.forward = types.MethodType(make_patched_i(i), block.attn)
        try:
            with torch.no_grad():
                _ = model.visual(pixel_values, grid_thw=grid_thw)
        finally:
            for i, block in enumerate(model.visual.blocks):
                block.attn.forward = originals[i]

        for li, attn in captured.items():
            received = attn.mean(axis=0)
            if received.shape[0] != gh * gw:
                raise RuntimeError(
                    f"frame {fi} layer {li}: attn seq_len={received.shape[0]} "
                    f"!= gh*gw={gh}*{gw}={gh*gw} -- grid assumption is wrong, "
                    f"fix it rather than silently skip (this exact bug already "
                    f"produced an empty-but-'successful'-looking run once)")
            heat = received.reshape(gh, gw).astype(np.float32)
            layer_dir = os.path.join(args.out_dir, f"layer{li:02d}")
            np.save(os.path.join(layer_dir, f"frame_{fi:05d}.npy"), received)
            save_heatmap_png(image, heat, os.path.join(layer_dir, f"frame_{fi:05d}.png"))

        if fi % 20 == 0:
            print(f"[sweep] {args.camera_name}: {fi+1}/{len(frame_paths)} frames done")

    print(f"[sweep] {args.camera_name}: all frames done, building GIFs")
    for li in range(n_layers):
        layer_dir = os.path.join(args.out_dir, f"layer{li:02d}")
        pngs = sorted(f for f in os.listdir(layer_dir) if f.endswith(".png"))
        if not pngs:
            continue
        frames = [PIL.Image.open(os.path.join(layer_dir, p)).convert("RGB") for p in pngs]
        gif_path = os.path.join(args.out_dir, f"{args.camera_name}_layer{li:02d}.gif")
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=150, loop=0, optimize=True)
    print(f"[sweep] {args.camera_name}: done, {n_layers} GIFs written to {args.out_dir}")


if __name__ == "__main__":
    main()
