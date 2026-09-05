"""Step 2 probe #2 — vision token budget.

Implements REDESIGN_PLAN.md §4.5-A / §12 step 2.

Prints ``image_processor.patch_size * image_processor.merge_size`` and the resolved
smart_resize grid for each candidate ``--image_size``, then picks the square size whose
token count is closest to upstream's ``384 288``. No network access: this probes the
transformers library's own class defaults / AutoImageProcessor mapping, not a downloaded
checkpoint's ``preprocessor_config.json`` (that per-checkpoint override is re-checked at
training/deploy time by gate G16, once a real checkpoint exists).

Findings, resolved here (do not re-derive):

* Installed transformers (this repo pins >=4.53.0; resolved to a 5.x release locally
  because ``lerobot``'s ``huggingface-hub>=1.6.0`` requirement conflicts with
  ``transformers==4.57.*``'s ``huggingface-hub<1.0`` cap -- see origami/PROGRESS.md.
  There is no dedicated ``qwen3_vl`` image processor module in this transformers version;
  ``AutoImageProcessor``'s mapping for ``"qwen3_vl"`` resolves to
  ``Qwen2VLImageProcessor`` (confirmed via
  ``transformers.models.auto.image_processing_auto.IMAGE_PROCESSOR_MAPPING_NAMES``).
* ``Qwen2VLImageProcessor`` class defaults: ``patch_size=14``, ``merge_size=2`` ->
  factor 28. ``Qwen3VLVisionConfig`` class defaults: ``patch_size=16``,
  ``spatial_merge_size=2`` -> factor 32. Both match REDESIGN_PLAN.md's table exactly.
* Because the *actual* factor at train/deploy time comes from the checkpoint's
  ``preprocessor_config.json`` (which may set either 14 or 16), REDESIGN_PLAN.md says not
  to assume one -- this probe reports both and picks a size that is closest to upstream's
  token count *at both* factors, so the choice is robust either way.
"""

import math

from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
    smart_resize,
)
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig

UPSTREAM_H, UPSTREAM_W = 384, 288  # T-Rex's own train.sh --image_size
CANDIDATES = [(384, 288), (384, 384), (336, 336), (224, 224)]


def _grid_and_tokens(h: int, w: int, factor: int) -> tuple[int, int, int]:
    h_bar, w_bar = smart_resize(h, w, factor=factor)
    grid_h, grid_w = h_bar // factor, w_bar // factor
    return grid_h, grid_w, grid_h * grid_w


def test_qwen3_vl_maps_to_qwen2_vl_image_processor():
    from transformers.models.auto.image_processing_auto import (
        IMAGE_PROCESSOR_MAPPING_NAMES,
    )

    mapped = IMAGE_PROCESSOR_MAPPING_NAMES["qwen3_vl"]
    # Value is {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}
    # in this transformers version -- there is no dedicated qwen3_vl image processor.
    assert mapped["torchvision"] == "Qwen2VLImageProcessor"


def test_factor_defaults_match_plan():
    assert Qwen2VLImageProcessor.patch_size == 14
    assert Qwen2VLImageProcessor.merge_size == 2

    vision_cfg = Qwen3VLVisionConfig()
    assert vision_cfg.patch_size == 16
    assert vision_cfg.spatial_merge_size == 2


def test_probe_token_budget_and_pick_size():
    upstream_tokens = {}
    grids = {}
    for factor in (28, 32):
        gh, gw, tokens = _grid_and_tokens(UPSTREAM_H, UPSTREAM_W, factor)
        upstream_tokens[factor] = tokens
        grids[(UPSTREAM_H, UPSTREAM_W, factor)] = (gh, gw, tokens)

    print(f"\nupstream {UPSTREAM_H}x{UPSTREAM_W} tokens: f28={upstream_tokens[28]} "
          f"f32={upstream_tokens[32]}")

    results = {}
    for h, w in CANDIDATES:
        for factor in (28, 32):
            gh, gw, tokens = _grid_and_tokens(h, w, factor)
            results[(h, w, factor)] = (gh, gw, tokens)
            print(f"  {h}x{w} @ f={factor}: grid {gh}x{gw} = {tokens} tokens")

    # The real factor is only known once a checkpoint's preprocessor_config.json is read
    # (§4.5-A) -- unavailable at this offline-probe step. So pick per-factor.
    #
    # Selection rule: the SMALLEST square candidate whose token count is >= upstream's
    # (never fewer tokens than the point the frozen ViT was trained at -- undershooting
    # starves the midtrain ViT of resolution it's calibrated for; overshooting just gives
    # it slightly more than it saw in training, which is safe). Plain nearest-by-absolute-
    # distance picks 336x336 at BOTH factors (100 and 144 vs. upstream's 108 and 140), which
    # contradicts REDESIGN_PLAN.md §4.5-A's stated picks (384x384 at f=32, 336x336 at f=28)
    # -- the floor-constrained rule below is the one that reproduces the plan's own answer.
    square_candidates = [(h, w) for h, w in CANDIDATES if h == w]

    def _pick(factor):
        at_or_above = [
            hw for hw in square_candidates
            if results[(hw[0], hw[1], factor)][2] >= upstream_tokens[factor]
        ]
        return min(at_or_above, key=lambda hw: results[(hw[0], hw[1], factor)][2])

    best_per_factor = {factor: _pick(factor) for factor in (28, 32)}
    print(f"chosen square --image_size per factor: {best_per_factor}")

    # Sanity-check against the plan's own worked numbers (§4.5-A table).
    assert results[(384, 384, 32)][2] == 144
    assert results[(384, 384, 28)][2] == 196
    assert results[(336, 336, 28)][2] == 144
    assert results[(384, 288, 32)][2] == 108
    assert results[(384, 288, 28)][2] == 140

    assert best_per_factor[32] == (384, 384)
    assert best_per_factor[28] == (336, 336)
