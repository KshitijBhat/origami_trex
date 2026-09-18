"""Real-processor check that collate_fn's memory_fast action_abs is
normalized the same way the live tick's flow target (norm_actions) is --
regression test for a bug found while tracing the normalization path:
build_memory_kv_fast feeds action_abs straight into x_embedder, which
everywhere else only ever sees normalized [-1,1]-ish values.
"""
import io
import os
import sys
import tempfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import PIL.Image
import torch
from transformers import AutoProcessor

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)  # T-Rex/
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
from qwen_vla.origami_dataset import OrigamiDataset, ACTION_DIM, ACTION_CHUNK, N_FINGERS, F6_PER_FINGER  # noqa: E402


def make_blob(tag):
    img = PIL.Image.new("RGB", (32, 32), color=(tag % 256, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

    with tempfile.TemporaryDirectory() as tmpdir:
        n_rows = 20
        # action_abs deliberately RAW/large (e.g. joint radians ~[-3,3], far
        # outside a normalized [-1,1] range) so a missed normalization would
        # be obvious in the output rather than accidentally looking plausible.
        rows = []
        for row in range(n_rows):
            tag = row
            rows.append({
                "state": [float(tag)] * ACTION_DIM,
                "action_chunk": [float(tag)] * (ACTION_CHUNK * ACTION_DIM),
                "action_abs": [3.0 * np.sin(row + d) for d in range(ACTION_DIM)],
                "phase": row / max(1, n_rows - 1),
                "head": make_blob(tag),
                "wrist_left": make_blob(tag),
                "wrist_right": make_blob(tag),
                "tacf6_hist": [float(tag)] * (16 * N_FINGERS * F6_PER_FINGER),
            })
        table = pa.Table.from_pylist(rows)
        path = os.path.join(tmpdir, "ep0.parquet")
        pq.write_table(table, path)

        ds = OrigamiDataset.__new__(OrigamiDataset)
        ds.pq = pq
        ds.root = tmpdir
        ds.episodes = [{"file": "ep0.parquet", "instruction": "fold"}]
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
        ds.sample_fps = 10.0
        ds.memory_slow_seconds = []
        ds.memory_slow_jitter_sec = 0.0
        ds.memory_fast = 2
        ds.processor = processor
        # Real-ish action normalization stats: q01/q99 = [-3.5, 3.5], so the
        # raw values above (amplitude 3.0) land inside the normalizable range
        # and a correct normalize() call produces values clearly in [-1, 1].
        ds.action_mask = np.ones(ACTION_DIM, dtype=bool)
        ds.action_min = np.full(ACTION_DIM, -3.5, dtype=np.float32)
        ds.action_max = np.full(ACTION_DIM, 3.5, dtype=np.float32)
        ds.state_mask = np.ones(ACTION_DIM, dtype=bool)
        ds.state_min = np.full(ACTION_DIM, -3.5, dtype=np.float32)
        ds.state_max = np.full(ACTION_DIM, 3.5, dtype=np.float32)
        ds.tacf6_mask = np.ones(N_FINGERS * F6_PER_FINGER, dtype=bool)
        ds.tacf6_min = np.zeros(N_FINGERS * F6_PER_FINGER, dtype=np.float32)
        ds.tacf6_max = np.ones(N_FINGERS * F6_PER_FINGER, dtype=np.float32)

        ds._files = {}
        ds._cache = {}
        ds._cache_order = []
        ds.cache_groups = 8
        ds.index = []
        ds.row_groups = []
        ds.ep_rows = []
        ds._build_index()

        batch = [ds[i] for i in range(10, 14)]
        out = ds.collate_fn(batch)

        assert out["memory_fast"] is not None and len(out["memory_fast"]) == 2
        for k, entry in enumerate(out["memory_fast"]):
            a = entry["action_abs"]
            assert a.shape == (len(batch), ACTION_DIM), f"unexpected shape {a.shape}"
            amax = a.abs().max().item()
            assert amax <= 1.0 + 1e-4, (
                f"memory_fast[{k}]['action_abs'] max abs value {amax:.3f} exceeds 1.0 -- "
                f"looks UNnormalized (raw amplitude was ~3.0), normalization fix did not apply")
            print(f"[PASS] memory_fast[{k}]['action_abs']: shape {tuple(a.shape)}, "
                  f"max abs {amax:.4f} (within normalized [-1,1] range)")

        print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
