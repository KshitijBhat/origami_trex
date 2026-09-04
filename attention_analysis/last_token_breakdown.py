#!/usr/bin/env python3
"""For one frame, decode the exact token sequence (to show which real token
'last_token' is), and compute last_token's attention weight distribution over
the TEXT tokens (not vision) at several layers, grouped by word using the
same post-image-only alignment fix as decoder_episode_sweep.py."""
import json
import sys

import numpy as np
import PIL.Image
import torch

sys.path.insert(0, "/workspace/origami_trex/T-Rex")
from trex_origami.loading import load_args_from_checkpoint, model_load

CHECKPOINT = sys.argv[1]
INSTRUCTION = sys.argv[2]
FRAME_PATH = sys.argv[3]
OUT_JSON = sys.argv[4]

device = torch.device("cuda")
load_args = load_args_from_checkpoint(CHECKPOINT)
model, processor, _ = model_load(load_args)
model = model.to(device).eval()

image = PIL.Image.open(FRAME_PATH).convert("RGB").resize((224, 224), PIL.Image.LANCZOS)
content = [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]
text = processor.apply_chat_template([{"role": "user", "content": content}],
                                      tokenize=False, add_generation_prompt=True)
inp = processor(text=text, images=[image], return_tensors="pt", padding=False)
input_ids = inp.input_ids.to(device)
attention_mask = inp.attention_mask.to(device)
pixel_values = inp.pixel_values.to(device, dtype=torch.bfloat16)
grid_thw = inp.image_grid_thw.to(device)
embeds = model.prepare_inputs_embeds(input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=grid_thw)
position_ids, _ = model.get_rope_index(input_ids=input_ids, image_grid_thw=grid_thw, attention_mask=attention_mask)

image_mask = (input_ids[0] == model.image_token_id)
text_mask = attention_mask[0].bool() & ~image_mask
text_idx = text_mask.nonzero(as_tuple=True)[0]
last_image_pos = image_mask.nonzero(as_tuple=True)[0].max().item()
post_image_text_idx = text_idx[text_idx > last_image_pos]

# decode every token in the full sequence, for the "what is last_token really" display
tok = processor.tokenizer
all_ids = input_ids[0].tolist()
token_strs = [tok.decode([t]) for t in all_ids]
last_pos = text_idx[-1].item()

# word groups (post-image only, same fix as decoder_episode_sweep.py)
words = INSTRUCTION.split()
keys = [f"{i+1:02d}_{w}" for i, w in enumerate(words)]
post_ids = input_ids[0][post_image_text_idx].tolist()
groups = {k: [] for k in keys}
wi, acc = 0, ""
for pos, tid in zip(post_image_text_idx.tolist(), post_ids):
    if wi >= len(words):
        break
    piece = tok.decode([tid])
    acc += piece
    groups[keys[wi]].append(pos)
    if acc.strip() == words[wi].strip(".,!?;:"):
        wi += 1; acc = ""
    elif len(acc.strip()) >= len(words[wi].strip(".,!?;:")):
        wi += 1; acc = ""

with torch.no_grad():
    out = model.model(inputs_embeds=embeds, position_ids=position_ids,
                       attention_mask=attention_mask, output_attentions=True)

n_layers = len(model.model.layers)
result = {
    "instruction": INSTRUCTION,
    "full_sequence_tokens": token_strs,
    "last_token_position": last_pos,
    "last_token_string": token_strs[last_pos],
    "image_span": [int(image_mask.nonzero(as_tuple=True)[0].min()), int(last_image_pos)],
    "word_positions": {k: v for k, v in groups.items()},
    "last_token_text_attn_by_layer": {},
}

for li in range(n_layers):
    attn = out.attentions[li][0].float().mean(dim=0)  # [S,S]
    row = attn[last_pos]  # last_token's query row
    per_word = {}
    for k, positions in groups.items():
        if positions:
            per_word[k] = float(row[positions].sum().item())
    text_total = float(row[post_image_text_idx].sum().item())
    vision_total = float(row[image_mask].sum().item())
    result["last_token_text_attn_by_layer"][f"layer{li:02d}"] = {
        "per_word": per_word,
        "text_total": text_total,
        "vision_total": vision_total,
    }

with open(OUT_JSON, "w") as f:
    json.dump(result, f, indent=2)
print(f"wrote {OUT_JSON}")
print(f"last_token pos={last_pos} string={token_strs[last_pos]!r}")
print(f"full sequence ({len(token_strs)} tokens):")
for i, s in enumerate(token_strs):
    tag = " <== LAST_TOKEN" if i == last_pos else ""
    print(f"  [{i:3d}] {s!r}{tag}")
