#!/usr/bin/env python3
"""Full-episode shared-decoder text->vision attention sweep: extract frames
at a fixed interval, run each through the real training-pipeline forward
path (prepare_inputs_embeds + get_rope_index + model.model(output_attentions
=True)), and save per-layer heatmaps for TWO query strategies:
  - last text token (causally sees the whole prefix)
  - each individual WORD of the instruction (BPE-token-aligned)
into one folder per (layer, query), then build one GIF per layer per query.

Standalone, read-only w.r.t. training. Reuses the exact model_load() /
prepare_inputs_embeds() / get_rope_index() path real inference uses (same
as inspect_attention.py, extended here to loop over frames like
vit_episode_sweep.py did for the ViT).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, "/workspace/origami_trex/T-Rex")
from trex_origami.loading import load_args_from_checkpoint, model_load

# Skip these for per-word heatmaps -- function words that aren't individually
# interpretable and would otherwise balloon file count on longer captions
# (e.g. the 22-word insert/coin instruction). last_token always runs
# regardless of this filter.
STOPWORDS = {
    "the", "a", "an", "to", "in", "on", "at", "of", "and", "or", "it", "its",
    "your", "you", "with", "into", "up", "down", "out", "off", "for", "by",
    "is", "are", "be", "as", "that", "this",
}

_JET_LUT = None


def _make_jet_lut():
    import matplotlib.cm as cm
    return (cm.jet(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)


def save_heatmap_png(image, heat, out_path, alpha=0.45):
    global _JET_LUT
    if _JET_LUT is None:
        _JET_LUT = _make_jet_lut()
    norm = heat - heat.min()
    denom = norm.max()
    norm = norm / denom if denom > 1e-8 else norm
    idx = (norm * 255).astype(np.uint8)
    heat_small_rgb = _JET_LUT[idx]
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


def word_token_groups(tokenizer, input_ids_row, text_idx, instruction, device):
    """Return {"NN_word": [positions in input_ids_row]} for each whitespace-
    split word of `instruction` (NN = 1-indexed position in the sentence),
    restricted to the already-identified text_idx positions. Position-indexed
    keys on purpose -- a plain word-string key collides when the same word
    (e.g. "the") appears twice in a sentence, silently averaging two
    different positions' attention into one bucket. Greedy alignment: decode
    text tokens one at a time, accumulate into the current word's buffer
    until it matches (punctuation-insensitive) the next target word, then
    advance. Robust to BPE leading-space markers since we compare on the
    DECODED string, not raw token pieces."""
    words = instruction.split()
    keys = [f"{i+1:02d}_{w}" for i, w in enumerate(words)]
    text_ids = input_ids_row[text_idx].tolist()
    groups = {k: [] for k in keys}
    wi = 0
    acc = ""
    for pos, tid in zip(text_idx.tolist(), text_ids):
        if wi >= len(words):
            break
        piece = tokenizer.decode([tid])
        acc += piece
        groups[keys[wi]].append(pos)
        if acc.strip() == words[wi].strip(".,!?;:"):
            wi += 1
            acc = ""
        elif len(acc.strip()) >= len(words[wi].strip(".,!?;:")):
            # overshot without an exact match (punctuation stripped differently) --
            # accept it anyway rather than silently losing tokens, but flag it
            wi += 1
            acc = ""
    return groups, wi, len(words)


def build_inputs(model, processor, image, instruction, device):
    content = [{"type": "image"}, {"type": "text", "text": instruction}]
    text = processor.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
    inp = processor(text=text, images=[image], return_tensors="pt", padding=False)
    input_ids = inp.input_ids.to(device)
    attention_mask = inp.attention_mask.to(device)
    pixel_values = inp.pixel_values.to(device, dtype=torch.bfloat16)
    grid_thw = inp.image_grid_thw.to(device)
    embeds = model.prepare_inputs_embeds(
        input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw)
    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=grid_thw, attention_mask=attention_mask)
    return input_ids, attention_mask, embeds, position_ids, grid_thw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--video_path", required=True)
    ap.add_argument("--camera_name", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--test_one_frame", action="store_true",
                     help="process only the first extracted frame, print word "
                          "alignment, skip GIF building -- for verification")
    ap.add_argument("--skip_gifs", action="store_true",
                     help="skip in-process GIF building entirely -- measured much "
                          "slower here than in a fresh lightweight process (the "
                          "~11GB-RSS model-loaded process appears to slow down "
                          "plain PIL/CPU work too). Use rebuild_gifs.py separately "
                          "afterward instead.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"[decsweep] loading checkpoint {args.checkpoint_path}")
    load_args = load_args_from_checkpoint(args.checkpoint_path)
    model, processor, _ = model_load(load_args)
    model = model.to(device).eval()
    n_layers = len(model.model.layers)
    merge = model.visual.spatial_merge_size
    print(f"[decsweep] n_layers={n_layers} instruction={args.instruction!r}")

    print(f"[decsweep] extracting frames from {args.video_path} at {args.fps}fps")
    frame_paths = extract_frames(args.video_path, args.frames_dir, args.fps)
    if args.test_one_frame:
        frame_paths = frame_paths[:1]
    print(f"[decsweep] {len(frame_paths)} frames to process")

    queries = None  # set from first frame's word list

    for fi, fpath in enumerate(frame_paths):
        image = PIL.Image.open(fpath).convert("RGB")
        image_size = (224, 224)
        if image.size != image_size:
            image = image.resize(image_size, PIL.Image.LANCZOS)

        input_ids, attention_mask, embeds, position_ids, grid_thw = build_inputs(
            model, processor, image, args.instruction, device)
        image_mask = (input_ids[0] == model.image_token_id)
        text_mask = attention_mask[0].bool() & ~image_mask
        text_idx = text_mask.nonzero(as_tuple=True)[0]
        # text_idx includes PRE-image preamble tokens (e.g. "<|im_start|>user\n
        # <|vision_start|>") ahead of the image, mixed in with the real
        # POST-image instruction text. word_token_groups's greedy matcher must
        # only ever see the post-image span: a preamble token like
        # "<|im_start|>" decodes to a string far longer than a short target
        # word ("north"), so the greedy loop's overshoot branch immediately
        # (and wrongly) "matches" it and advances -- for a short instruction
        # this can exhaust ALL target words against pure preamble garbage
        # before the real instruction tokens are ever reached. Any word
        # aligned to a pre-image position is then causally guaranteed exactly
        # 0.0 attention-to-vision (the image hasn't occurred yet in sequence
        # order), which is what produced the too-good-to-be-true exact zeros.
        last_image_pos = image_mask.nonzero(as_tuple=True)[0].max().item()
        post_image_text_idx = text_idx[text_idx > last_image_pos]

        groups, matched, total = word_token_groups(
            processor.tokenizer, input_ids[0], post_image_text_idx, args.instruction, device)
        groups = {k: v for k, v in groups.items()
                  if k.split("_", 1)[1].lower().strip(".,!?;:") not in STOPWORDS}
        if fi == 0:
            print(f"[decsweep] word alignment: matched {matched}/{total} words")
            for w, idxs in groups.items():
                print(f"    {w!r}: {len(idxs)} tokens")
            queries = {"last_token": None}  # placeholder, filled per-frame below
            for w in groups:
                queries[f"word_{w}"] = None

        with torch.no_grad():
            out = model.model(
                inputs_embeds=embeds, position_ids=position_ids,
                attention_mask=attention_mask, output_attentions=True)

        merge_gh = int(grid_thw[0, 1].item()) // merge
        merge_gw = int(grid_thw[0, 2].item()) // merge

        query_positions = {"last_token": [text_idx[-1].item()]}
        for w, idxs in groups.items():
            if idxs:
                query_positions[f"word_{w}"] = idxs

        for qi, qpos in query_positions.items():
            for li in range(n_layers):
                attn = out.attentions[li][0].float().mean(dim=0)  # [S,S] head-avg
                row = attn[qpos].mean(dim=0)  # avg over the query's token(s)
                vision_row = row[image_mask].cpu().numpy()
                if vision_row.shape[0] != merge_gh * merge_gw:
                    raise RuntimeError(
                        f"frame {fi} layer {li} query {qi}: vision token count "
                        f"{vision_row.shape[0]} != {merge_gh}*{merge_gw}")
                layer_dir = os.path.join(args.out_dir, qi, f"layer{li:02d}")
                os.makedirs(layer_dir, exist_ok=True)
                np.save(os.path.join(layer_dir, f"frame_{fi:05d}.npy"), vision_row)
                heat = vision_row.reshape(merge_gh, merge_gw).astype(np.float32)
                save_heatmap_png(image, heat, os.path.join(layer_dir, f"frame_{fi:05d}.png"))

        if args.test_one_frame:
            print(f"[decsweep] test frame done, query_positions keys: {list(query_positions.keys())}")
            return

        if fi % 10 == 0:
            print(f"[decsweep] {args.camera_name}: {fi+1}/{len(frame_paths)} frames done")

    if args.skip_gifs:
        print(f"[decsweep] {args.camera_name}: all frames done, skipping in-process "
              f"GIF build (--skip_gifs) -- run rebuild_gifs.py separately")
        return
    print(f"[decsweep] {args.camera_name}: all frames done, building GIFs")
    for qi in os.listdir(args.out_dir):
        qdir = os.path.join(args.out_dir, qi)
        if not os.path.isdir(qdir):
            continue
        for li_name in sorted(os.listdir(qdir)):
            layer_dir = os.path.join(qdir, li_name)
            pngs = sorted(f for f in os.listdir(layer_dir) if f.endswith(".png"))
            if not pngs:
                continue
            frames = [PIL.Image.open(os.path.join(layer_dir, p)).convert("RGB") for p in pngs]
            gif_path = os.path.join(args.out_dir, f"{args.camera_name}_{qi}_{li_name}.gif")
            # optimize=True (LZW-optimal palette) measured at ~23s/gif here --
            # with 252 layer x query combos per clip that's ~96 min of GIF-
            # building alone. Dropping it: slightly bigger files, much faster.
            frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                           duration=150, loop=0, optimize=False)
    print(f"[decsweep] {args.camera_name}: done, GIFs written to {args.out_dir}")


if __name__ == "__main__":
    main()
