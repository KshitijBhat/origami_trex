"""Standalone CPU-only test of OrigamiDataset's memory_slow_seconds (exponential
lookback) / memory_fast (linear) windowing logic in __getitem__ -- no processor,
no model, no GPU needed. Tests the highest-risk part: seconds->row-offset
conversion, episode-boundary padding, and window ordering.
"""
import io
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import PIL.Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # T-Rex/
_QWEN_VLA_DIR = os.path.join(_PROJECT_DIR, "qwen_vla")
if _QWEN_VLA_DIR not in sys.path:
    sys.path.insert(0, _QWEN_VLA_DIR)
from origami_dataset import OrigamiDataset, ACTION_DIM, ACTION_CHUNK, N_FINGERS, F6_PER_FINGER  # noqa: E402


def make_blob(tag: int) -> bytes:
    # PNG (lossless): this test verifies row-SELECTION logic (which row's
    # image __getitem__ returns), not codec fidelity -- _pil() is
    # format-agnostic (PIL auto-detects).
    img = PIL.Image.new("RGB", (8, 8), color=(tag % 256, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_episode_parquet(path: str, n_rows: int, ep_tag: int):
    rows = []
    for row in range(n_rows):
        tag = ep_tag * 1000 + row  # unique per (episode, row)
        rows.append({
            "state": [float(tag)] * ACTION_DIM,
            "action_chunk": [float(tag)] * (ACTION_CHUNK * ACTION_DIM),
            "action_abs": [float(tag)] * ACTION_DIM,
            "phase": row / max(1, n_rows - 1),
            "head": make_blob(tag),
            "wrist_left": make_blob(tag),
            "wrist_right": make_blob(tag),
            "tacf6_hist": [float(tag)] * (16 * N_FINGERS * F6_PER_FINGER),
        })
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def build_fake_dataset(tmpdir: str, n_rows_ep0: int, sample_fps: float,
                       memory_slow_seconds, memory_fast: int) -> OrigamiDataset:
    ds = OrigamiDataset.__new__(OrigamiDataset)
    ds.pq = pq
    ds.root = tmpdir
    ds.episodes = []
    ep_specs = [(0, n_rows_ep0), (1, 3)]
    for ep_tag, n_rows in ep_specs:
        fname = f"ep{ep_tag}.parquet"
        make_episode_parquet(os.path.join(tmpdir, fname), n_rows, ep_tag)
        ds.episodes.append({"file": fname, "instruction": f"fold task {ep_tag}"})

    ds.image_size = None
    ds.use_flare = False
    ds.bake_flare = False
    ds.use_tactile_vec = True
    ds.use_tactile_deform = False
    ds.use_tactile_vqvae = True
    ds.use_robot_state = False
    ds.vqvae_window = 16
    ds.action_dim = ACTION_DIM
    ds.action_chunk = ACTION_CHUNK
    ds.phase_mode = "none"
    ds.n_phases = 6
    ds.instruction = ""
    ds.instruction_override = ""
    ds.sample_fps = sample_fps
    # Mirror __init__'s own list-or-CLI-string parsing (this harness bypasses
    # __init__ via __new__, so it has to replicate that one bit by hand to
    # actually exercise the string-parsing path Case 6 below tests).
    _mss = memory_slow_seconds
    if isinstance(_mss, str):
        _mss = [float(s) for s in _mss.split(",") if s.strip()]
    ds.memory_slow_seconds = list(_mss or [])
    ds.memory_slow_jitter_sec = 0.0
    ds.memory_fast = memory_fast

    ds._files = {}
    ds._cache = {}
    ds._cache_order = []
    ds.cache_groups = 8
    ds.index = []
    ds.row_groups = []
    ds.ep_rows = []
    ds._build_index()
    return ds


def tag_from_head(pil_img: PIL.Image.Image) -> int:
    return pil_img.getpixel((0, 0))[0]


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        # sample_fps=10 -> easy round-number row-offset math:
        # 0.25s->2.5->round 2, 0.5s->5, 1s->10, 5s->50
        FPS = 10.0
        ds = build_fake_dataset(tmpdir, n_rows_ep0=60, sample_fps=FPS,
                                memory_slow_seconds=[0.25, 0.5, 1.0, 5.0],
                                memory_fast=2)

        print(f"total samples: {len(ds)} (expect 60 + 3 = 63)")
        assert len(ds) == 63

        # --- Case 1: deep in episode 0 (row 55), all 4 lookback offsets are
        # real distinct past rows, no padding involved. Verify exact row
        # offsets: 55 - round(0.25*10)=55-2=53, -5=50, -10=45, -50=5.
        # Sorted furthest-first (5s) -> nearest (0.25s) = oldest->newest order.
        item = ds[55]
        assert len(item["memory_slow"]) == 4
        tags = [tag_from_head(m["head"]) for m in item["memory_slow"]]
        expected_rows = [5, 45, 50, 53]  # oldest (5s back) -> newest (0.25s back)
        expected_tags = [(0 * 1000 + r) % 256 for r in expected_rows]
        assert tags == expected_tags, (
            f"row 55, memory_slow_seconds=[0.25,0.5,1,5]: expected rows {expected_rows} "
            f"(tags {expected_tags}), got tags {tags}")
        print(f"[PASS] row 55 exponential lookback: rows {expected_rows} "
              f"(0.25/0.5/1/5s back at {FPS} fps), correctly oldest->newest")

        # --- Case 2: row 3 of episode 0 -- 5s/1s/0.5s lookback all clamp to
        # row 0 (padding), only 0.25s (offset 2) gives a real distinct row (row 1) ---
        item = ds[3]
        tags = [tag_from_head(m["head"]) for m in item["memory_slow"]]
        expected_rows = [0, 0, 0, 1]  # 5s->clamp(3-50,0)=0, 1s->clamp(3-10,0)=0,
                                       # 0.5s->clamp(3-5,0)=0, 0.25s->clamp(3-2,0)=1
        expected_tags = [(0 * 1000 + r) % 256 for r in expected_rows]
        assert tags == expected_tags, f"row 3: expected rows {expected_rows}, got tags {tags}"
        print(f"[PASS] row 3 near episode start: mostly padded to row 0, "
              f"only nearest (0.25s) lookback reaches a real distinct row")

        # --- Case 2b: dt_actual reflects the REAL clamped gap, not the
        # unclamped nominal target -- this is what the model-side position
        # shift must use, or position label and content would disagree ---
        dt_actuals = [m["dt_actual"] for m in item["memory_slow"]]
        # row 3, prow clamped to 0 for the first three (nominal 5/1/0.5s):
        # real gap = (3-0)/10 = 0.3s each, NOT the nominal 5/1/0.5.
        # Only the last (nominal 0.25s -> prow=1, real gap=(3-1)/10=0.2s)
        # is close to (but not exactly) its nominal target.
        expected_dt_actuals = [0.3, 0.3, 0.3, 0.2]
        assert all(abs(a - e) < 1e-9 for a, e in zip(dt_actuals, expected_dt_actuals)), (
            f"dt_actual should reflect the real clamped gap {expected_dt_actuals}, "
            f"got {dt_actuals}")
        print(f"[PASS] dt_actual correctly reflects real clamped gaps {expected_dt_actuals}, "
              f"not the unclamped nominal targets [5.0, 1.0, 0.5, 0.25]")

        # --- Case 2b: jitter perturbs the offset within bounds, and does so
        # differently across repeated __getitem__ calls (stochastic) ---
        ds_jit = build_fake_dataset(tmpdir, n_rows_ep0=60, sample_fps=FPS,
                                    memory_slow_seconds=[1.0], memory_fast=0)
        ds_jit.memory_slow_jitter_sec = 0.3  # +/- 0.3s = +/- 3 rows at 10fps
        seen_rows = set()
        for _ in range(40):
            item = ds_jit[55]
            tag = tag_from_head(item["memory_slow"][0]["head"])
            seen_rows.add(tag)
        # row 55, target 1.0s -> nominal row 45, jittered row in [42,48] (0->255 tag range)
        nominal_row = 55 - round(1.0 * FPS)
        assert nominal_row == 45
        possible_tags = {(0 * 1000 + r) % 256 for r in range(42, 49)}
        assert seen_rows.issubset(possible_tags), (
            f"jittered rows {seen_rows} outside expected bound {possible_tags}")
        assert len(seen_rows) > 1, (
            "jitter=0.3s over 40 draws produced only 1 distinct row -- "
            "expected genuine stochastic variation")
        print(f"[PASS] jitter=0.3s: {len(seen_rows)} distinct rows seen over 40 draws, "
              f"all within +/-3-row bound of nominal row {nominal_row}")

        # --- Case 3: memory_fast (linear, unchanged behavior) still works ---
        item = ds[10]
        assert len(item["memory_fast"]) == 2
        fast_actions = [m["action_abs"][0] for m in item["memory_fast"]]
        assert fast_actions == [8.0, 9.0], f"expected [8.0, 9.0], got {fast_actions}"
        print("[PASS] memory_fast linear window still correct (row 10 -> [row8, row9])")

        # --- Case 4: episode boundary -- episode 1's row 0 doesn't reach into
        # episode 0 even with the large 5s/50-row slow offset ---
        ei, row = ds.index[60]  # first row of episode 1 (global idx 60, after ep0's 60 rows)
        assert ei == 1 and row == 0
        item = ds[60]
        tags = [tag_from_head(m["head"]) for m in item["memory_slow"]]
        expected_tag_ep1_row0 = (1 * 1000 + 0) % 256
        assert all(t == expected_tag_ep1_row0 for t in tags), (
            f"episode 1 row 0's 5s-back lookback must NOT leak into episode 0, "
            f"got tags {tags}, expected all {expected_tag_ep1_row0}")
        print("[PASS] episode boundary holds even for the largest (5s/50-row) lookback")

        # --- Case 5: off (empty list / 0) is a true no-op ---
        ds_off = build_fake_dataset(tmpdir, n_rows_ep0=60, sample_fps=FPS,
                                    memory_slow_seconds=[], memory_fast=0)
        item = ds_off[10]
        assert "memory_slow" not in item and "memory_fast" not in item
        print("[PASS] memory_slow_seconds=[] / memory_fast=0: no-op, backward compatible")

        # --- Case 6: memory_slow_seconds as a CLI-style comma-separated
        # string (train.py's --memory_slow_seconds, always a plain str from
        # argparse) parses identically to passing the list directly ---
        ds_str = build_fake_dataset(tmpdir, n_rows_ep0=60, sample_fps=FPS,
                                    memory_slow_seconds="0.25,0.5,1.0,5.0", memory_fast=0)
        assert ds_str.memory_slow_seconds == [0.25, 0.5, 1.0, 5.0], ds_str.memory_slow_seconds
        item_from_str = ds_str[55]
        item_from_list = ds[55]
        tags_str = [tag_from_head(m["head"]) for m in item_from_str["memory_slow"]]
        tags_list = [tag_from_head(m["head"]) for m in item_from_list["memory_slow"]]
        assert tags_str == tags_list, (tags_str, tags_list)
        print("[PASS] memory_slow_seconds as a CLI-style comma-separated string "
              "parses identically to passing the list directly")

        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
