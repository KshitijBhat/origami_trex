"""Tactile delay curriculum for the LeRobot loader path. REDESIGN_PLAN.md §7.5-B.

`qwen_vla/lerobot_dataset.py:301` hard-wires ``tactile_f6s_delayed = norm_tacf6`` (i.e.
delay ≡ 0), so the LeRobot path never trains the tactile expert on the condition it
actually sees at deploy (§9.1 fires the refine tick at in-chunk offsets ``{0,4,8,12}`` --
`hardware_code/config/default.yaml`'s ``refine_offsets``). `midtrain.py`'s JSON-format
dataset (``MidtrainTacFlareDataset``) already samples ``delay_k ~ U(tactile_delay_offsets)``
per item; this module reproduces that for the LeRobot path by subclassing
``TRexLeRobotDataset`` rather than re-deriving its already-verified image/action/state
collation.

``--tactile_delay_scope``:
  * ``none`` -- identical to upstream ``TRexLeRobotDataset`` (delay ≡ 0).
  * ``f6`` (default) -- delay the F6/VQ-VAE channel only. Free: ``observation.tactile_f6``
    is a numeric parquet feature (§11.4), so extending its ``delta_timestamps`` window from
    ``[-(W-1)/fps .. 0]`` to ``[-(W-1)/fps .. +max(delay)/fps]`` costs a parquet read, not a
    video seek. Deform stays at the anchor frame.
  * ``both`` -- additionally delays deform. Video seeks are the real bottleneck (§7.4), so
    this does NOT fetch one frame per configured delay value (that would need
    ``1 + len(delay_offsets)`` seeks per key); it fetches exactly ``[0, max(delay)/fps]`` --
    2 timestamps per key, 10 keys, 20 seeks/sample (vs. 10 today) -- and uses the
    max-delay frame as an approximation for every *nonzero* per-sample ``delay_k`` (not an
    exact per-value fetch like F6 gets for free). This is a deliberate simplification to
    keep the "expensive" scope's cost bounded and predictable; REDESIGN_PLAN.md §7.5-B
    explicitly flags measuring `both` before paying the 2x decode cost, so exactness here
    matters less than F6's (which the model actually attends to at a finer VQ-VAE
    granularity).

G17 ("delay-curriculum parity: trained offsets == deployed offsets") has two halves. This
module + its test only cover the loader-level half -- the delayed F6 tensor at a given
``delay_k`` equals the corresponding slice of the extended F6 window. The other half
(training offsets == the offsets `serve_zenoh.py` actually fires at deploy) needs
`serve_zenoh.py`, which does not exist yet (§12 step 13) -- not implemented/tested here.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from qwen_vla.lerobot_dataset import TRexLeRobotDataset, _normalize
from utils.lerobot_common import DEFORM_KEYS, KEY_TACF6

VALID_SCOPES = ("none", "f6", "both")


class DelayedTRexLeRobotDataset(TRexLeRobotDataset):
    def __init__(self, config, processor, accelerator, episodes=None, _ds=None, _stats=None):
        self.tactile_delay_offsets = tuple(
            int(k) for k in getattr(config, "tactile_delay_offsets", (0, 4, 8, 12)))
        scope = getattr(config, "tactile_delay_scope", "f6") or "f6"
        assert scope in VALID_SCOPES, f"--tactile_delay_scope must be one of {VALID_SCOPES}, got {scope!r}"
        self.tactile_delay_scope = scope
        self.max_delay = max(self.tactile_delay_offsets) if scope != "none" else 0
        # Attributes above must exist before TRexLeRobotDataset.__init__ builds
        # delta_timestamps (it calls self._f6_offsets()/self._build_delta_timestamps()).
        super().__init__(config, processor, accelerator, episodes=episodes, _ds=_ds, _stats=_stats)

    # ── §7.5-B: extend the F6 window past "now" to cover every configured delay ──
    def _f6_offsets(self):
        W = self.vqvae_window
        return [(i - (W - 1)) / self.fps for i in range(W + self.max_delay)]

    def _build_delta_timestamps(self) -> Dict[str, list]:
        dt = super()._build_delta_timestamps()
        if self.tactile_delay_scope == "both" and self.has_deform and self.use_tactile_deform:
            for key in DEFORM_KEYS:
                dt[key] = [0.0, self.max_delay / self.fps]
        return dt

    def collate_fn(self, batch: List[Dict]) -> Dict:
        out = dict(super().collate_fn(batch))
        if self.tactile_delay_scope == "none":
            return out

        B = len(batch)
        W = self.vqvae_window
        rng = np.random.default_rng()
        delay_ks = [int(rng.choice(self.tactile_delay_offsets)) for _ in range(B)]
        out["delay_k"] = torch.tensor(delay_ks, dtype=torch.long)

        # ── F6 / VQ-VAE: exact per-sample delay, free (extended parquet window) ──
        if self.has_tactile and KEY_TACF6 in batch[0]:
            f6_window = torch.stack([x[KEY_TACF6].float() for x in batch], dim=0)  # [B, W+max_delay, 10, 6]
            if self.use_tactile_vqvae:
                hist = torch.stack(
                    [f6_window[b, k: k + W] for b, k in enumerate(delay_ks)], dim=0)  # [B, W, 10, 6]
                out["tactile_f6_history"] = hist
            if self.use_tactile_vec:
                delayed = torch.stack(
                    [f6_window[b, W - 1 + k] for b, k in enumerate(delay_ks)], dim=0)  # [B, 10, 6]
                flat = delayed.reshape(B, -1).numpy()
                out["tactile_f6s_delayed"] = torch.tensor(
                    _normalize(flat, self.tacf6_mask, self.tacf6_min, self.tacf6_max).reshape(B, -1, 6),
                    dtype=torch.bfloat16)

        # ── Deform: only "both" scope changes it; "f6" scope keeps super()'s anchor-only
        # tactile_deforms_delayed (documented deliberately -- keep deform at the anchor).
        if self.tactile_delay_scope == "both" and self.use_tactile_deform and self.has_deform:
            per_sample_anchor, per_sample_delayed = [], []
            for b, x in enumerate(batch):
                anchor_fingers = [x[k][0][0] for k in DEFORM_KEYS]        # 10 x [H, W]
                per_sample_anchor.append(torch.stack(anchor_fingers, dim=0))
                use_delayed = delay_ks[b] > 0
                delayed_fingers = [x[k][1 if use_delayed else 0][0] for k in DEFORM_KEYS]
                per_sample_delayed.append(torch.stack(delayed_fingers, dim=0))
            out["tactile_deforms"] = torch.stack(per_sample_anchor, dim=0).unsqueeze(2).float()
            out["tactile_deforms_delayed"] = torch.stack(per_sample_delayed, dim=0).unsqueeze(2).float()

        return out
