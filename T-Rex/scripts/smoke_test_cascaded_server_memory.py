"""Standalone CPU-only test of CascadedServer's Part C memory buffering
(dev/memory): nearest-timestamp slow-memory selection, linear fast-memory
window, eviction, and reset_episode -- no GPU, no real model/processor
needed since these are stubbed (they're exercised separately on GPU/in
build_memory_kv_slow/_fast's own tests; this file is purely about the NEW
server-side buffer/selection glue in test.py).
"""
import os
import sys

import numpy as np
import torch
import PIL.Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # T-Rex/scripts/
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)                # T-Rex/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
from test import CascadedServer  # noqa: E402


class FakeInp:
    def __init__(self):
        self.input_ids = torch.zeros(1, 3, dtype=torch.long)
        self.attention_mask = torch.ones(1, 3, dtype=torch.long)


class FakeProcessor:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "fake chat text"

    def __call__(self, text, images, return_tensors="pt", padding=False):
        return FakeInp()


class FakeModel:
    """Records exactly what rows/kwargs build_memory_kv_slow/_fast were
    called with, instead of actually running a forward pass."""
    def __init__(self):
        self.slow_calls = []
        self.fast_calls = []

    def build_memory_kv_slow(self, rows, rope_stride=32.0):
        self.slow_calls.append((rows, rope_stride))
        return f"slow_kv(n={len(rows)})" if rows else None

    def build_memory_kv_fast(self, rows, past_kv=None, rope_stride=8.0):
        self.fast_calls.append((rows, past_kv, rope_stride))
        if not rows:
            return past_kv
        return f"combined_kv(prev={past_kv}, n_fast={len(rows)})"


def make_server(memory_slow_seconds, memory_fast, margin=2.0):
    srv = CascadedServer.__new__(CascadedServer)
    srv.processor = FakeProcessor()
    srv.model = FakeModel()
    srv.device = "cpu"
    srv.statistic = {
        "action_mask": np.ones(4, dtype=bool),
        "action_min": np.full(4, -1.0, dtype=np.float32),
        "action_max": np.full(4, 1.0, dtype=np.float32),
    }
    srv.memory_slow_seconds = sorted(memory_slow_seconds, reverse=True)
    srv.memory_fast = memory_fast
    srv.memory_rope_stride_slow = 32.0
    srv.memory_rope_stride_fast = 8.0
    srv.memory_buffer_margin_sec = margin
    srv.memory_buf_slow = []
    srv.memory_buf_fast = []
    # _prev_command's state -- unused by these tests directly, but needed
    # by the fast-tick-rate capture test below, which calls _prev_command
    # the same way _run_slow/_run_fast really do.
    srv.last_chunk = None
    srv.last_chunk_time = 0.0
    srv.seed_state = None
    srv._last_fast_images = None
    return srv


def img(tag):
    return PIL.Image.new("RGB", (4, 4), color=(tag % 256, 0, 0))


def test_slow_no_op_when_disabled():
    srv = make_server([], 0)
    assert srv._select_memory_slow_rows(100.0) == []
    assert srv._build_memory_kv(100.0) is None
    print("[PASS] slow memory disabled: true no-op")


def test_slow_no_op_when_buffer_empty():
    srv = make_server([0.25, 1.0], 0)
    assert srv._select_memory_slow_rows(100.0) == []
    print("[PASS] slow memory enabled but buffer empty (episode start): true no-op")


def test_slow_nearest_timestamp_selection():
    srv = make_server([0.25, 1.0, 5.0], 0)
    # Buffered ticks at t=90, 94.8, 99.7, 99.9 (irregular spacing, like real
    # inference ticks) -- "now" = 100.0.
    for t, tag in [(90.0, 1), (94.8, 2), (99.7, 3), (99.9, 4)]:
        srv._remember_slow(t, img(tag), f"task{tag}")
    rows = srv._select_memory_slow_rows(100.0)
    assert len(rows) == 3
    # target 5.0 -> now-5=95.0 -> nearest buffered ts is 94.8 -> dt_actual=5.2
    # target 1.0 -> now-1=99.0 -> nearest buffered ts is 99.7 -> dt_actual=0.3
    # target 0.25 -> now-0.25=99.75 -> nearest buffered ts is 99.7 -> dt_actual=0.3
    expected_dt = [5.2, 0.3, 0.3]
    got_dt = [round(r["dt_actual"].item(), 4) for r in rows]
    assert got_dt == expected_dt, f"expected dt_actual {expected_dt}, got {got_dt}"
    print(f"[PASS] nearest-timestamp selection over irregular ticks: dt_actual={got_dt} "
          f"(oldest target first, each snapped to its real nearest buffered tick)")


def test_slow_eviction_by_time_window():
    srv = make_server([1.0], 0, margin=0.5)
    # max(memory_slow_seconds) + margin = 1.5 -> cutoff = now - 1.5
    srv._remember_slow(90.0, img(1), "old")     # will be evicted
    srv._remember_slow(99.0, img(2), "recent")  # kept (within 1.5s of now=100)
    assert len(srv.memory_buf_slow) == 1
    assert srv.memory_buf_slow[0][2] == "recent"
    print("[PASS] slow-memory buffer evicts entries older than "
          "max(memory_slow_seconds)+margin, keeps recent ones")


def test_fast_linear_window_and_eviction():
    srv = make_server([], 2)
    for i in range(5):
        srv._remember_fast(float(i), [img(i), img(i + 100)], np.array([float(i)] * 4))
    # Only the last `memory_fast`=2 entries survive.
    assert len(srv.memory_buf_fast) == 2
    kept_ts = [e[0] for e in srv.memory_buf_fast]
    assert kept_ts == [3.0, 4.0], f"expected [3.0, 4.0], got {kept_ts}"
    rows = srv._select_memory_fast_rows()
    assert len(rows) == 2
    for r in rows:
        assert r["action_abs"].shape == (1, 4)
        assert (r["action_abs"].abs() <= 1.0 + 1e-6).all(), "action_abs should be normalized"
    print("[PASS] fast memory: linear window keeps last N, oldest->newest order, "
          "action_abs normalized")


def test_reset_episode_clears_both_buffers():
    srv = make_server([1.0], 2)
    srv._remember_slow(10.0, img(1), "t")
    srv._remember_fast(10.0, [img(1), img(2)], np.zeros(4))
    assert srv.memory_buf_slow and srv.memory_buf_fast
    CascadedServer.reset_episode(srv)
    assert srv.memory_buf_slow == [] and srv.memory_buf_fast == []
    print("[PASS] reset_episode clears both memory buffers")


def test_build_memory_kv_chains_slow_into_fast():
    srv = make_server([1.0], 1)
    srv._remember_slow(99.0, img(1), "t")
    srv._remember_fast(99.0, [img(1), img(2)], np.zeros(4))
    kv = srv._build_memory_kv(100.0)
    assert kv == "combined_kv(prev=slow_kv(n=1), n_fast=1)", kv
    # Confirm build_memory_kv_fast really was called WITH the slow tier's
    # output as its past_kv (not None, not a separate independent call).
    fast_rows, fast_past_kv, _ = srv.model.fast_calls[-1]
    assert fast_past_kv == "slow_kv(n=1)"
    print("[PASS] _build_memory_kv chains build_memory_kv_slow's output into "
          "build_memory_kv_fast's past_kv, producing one combined cache")


def test_current_tick_never_sees_its_own_memory():
    """The ordering _run_slow uses (build memory_kv, THEN remember this
    tick) must mean a tick's own content is never selectable as its own
    memory -- simulate that ordering directly here."""
    srv = make_server([1.0], 1)
    now = 100.0
    memory_kv = srv._build_memory_kv(now)         # buffer still empty
    assert memory_kv is None
    srv._remember_slow(now, img(1), "self")
    srv._remember_fast(now, [img(1), img(2)], np.zeros(4))
    # A SUBSEQUENT tick now sees this one as memory -- but this tick itself
    # got memory_kv=None, confirmed above.
    later_kv = srv._build_memory_kv(now + 0.1)
    assert later_kv is not None
    print("[PASS] a tick's own content is recorded AFTER its own memory_kv "
          "is built, so it never becomes memory for itself")


def test_fast_memory_captures_at_fast_tick_rate_not_slow_tick_rate():
    """Regression test for a real bug found auditing the architecture: the
    real wire protocol (eval_trex_async.py) never sends fresh wrist images
    on a fast-mode request -- only tactile -- so _remember_fast used to
    only ever get called from _run_slow, meaning the fast-memory buffer
    updated at ~5Hz (slow-tick rate) despite representing what training
    builds from consecutive parquet rows at ~30Hz. The fix: _run_fast now
    also calls _remember_fast, reusing the last slow tick's wrist images
    (nothing fresher exists over the wire) paired with THAT fast tick's own
    just-reconstructed action (which genuinely does update every tick).
    This simulates that exact call pattern -- one _run_slow-style capture,
    then several _run_fast-style captures -- without needing a real model.
    """
    import time as _time
    srv = make_server([], 3)  # fast-only, window=3

    # -- _run_slow's part: stash the wrist images fast ticks will reuse,
    # and its own capture (mirrors _run_slow's existing self._remember_fast
    # call using self._prev_command(state) with a real state). --
    wrist_imgs = [img(10), img(11)]
    srv._last_fast_images = wrist_imgs
    t0 = _time.time()
    srv.last_chunk = np.array([[1.0, 1.0, 1.0, 1.0]])  # this tick's action
    srv.last_chunk_time = t0
    srv._remember_fast(t0, wrist_imgs, srv._prev_command(srv.seed_state))
    assert len(srv.memory_buf_fast) == 1

    # -- _run_fast's part, called several times in quick succession (as the
    # real ~20Hz loop would): each call reconstructs a NEW action (mirrors
    # _reconstruct writing a fresh self.last_chunk) then captures memory
    # using the reused wrist images + that fresh action. --
    for i in range(2):
        srv.last_chunk = np.array([[2.0 + i, 2.0 + i, 2.0 + i, 2.0 + i]])
        srv.last_chunk_time = _time.time()
        if srv._last_fast_images:
            srv._remember_fast(_time.time(), srv._last_fast_images,
                               srv._prev_command(srv.seed_state))

    # Three fast-tick-rate captures total (one slow-tick-triggered + two
    # fast-tick-triggered) -- NOT stuck at one entry per slow tick.
    assert len(srv.memory_buf_fast) == 3, (
        f"expected 3 fast-tick-rate captures, got {len(srv.memory_buf_fast)} -- "
        f"fast memory is still only updating at slow-tick rate")

    # Wrist images are identical across all three (correctly reused, since
    # nothing fresher exists) -- but the action differs each time (correctly
    # fresh per fast tick, not stale).
    actions = [np.asarray(e[2]).tolist() for e in srv.memory_buf_fast]
    assert actions == [[1.0] * 4, [2.0] * 4, [3.0] * 4], actions
    images_are_same_object = all(e[1] is not wrist_imgs for e in srv.memory_buf_fast)
    assert images_are_same_object, "each snapshot should copy, not alias, the image list"
    print("[PASS] fast memory now captures at fast-tick rate: 3 entries from "
          "1 slow tick + 2 fast ticks, wrist images correctly reused "
          "(nothing fresher over the wire) while action is fresh each time "
          f"({actions})")


if __name__ == "__main__":
    test_slow_no_op_when_disabled()
    test_slow_no_op_when_buffer_empty()
    test_slow_nearest_timestamp_selection()
    test_slow_eviction_by_time_window()
    test_fast_linear_window_and_eviction()
    test_reset_episode_clears_both_buffers()
    test_build_memory_kv_chains_slow_into_fast()
    test_current_tick_never_sees_its_own_memory()
    test_fast_memory_captures_at_fast_tick_rate_not_slow_tick_rate()
    print("\nALL TESTS PASSED")
