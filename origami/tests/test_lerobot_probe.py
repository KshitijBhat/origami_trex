"""Step 2 probe #1 — LeRobot v3.0 API surface.

Implements REDESIGN_PLAN.md §11.4 / §12 step 2.

Writes a tiny (3-frame, 1-episode) LeRobot v3.0 dataset using the **full**
``utils.lerobot_common.build_trex_features`` schema (including the 10
``observation.tactile_deform.{l,r}{0..4}`` video keys, which are not under the
``observation.images.*`` prefix), reopens it with ``delta_timestamps``, and asserts
every key round-trips. Resolves the three open questions from §11.4 for the pinned
lerobot version:

1. ``LeRobotDataset.create`` / ``add_frame`` / ``save_episode`` / ``finalize``
   signatures (recorded below; used by origami/convert.py).
2. Whether non-``observation.images.*`` keys are accepted as ``dtype: "video"``.
   **Yes** — confirmed no key-name pattern is enforced anywhere in
   ``lerobot.datasets.{feature_utils,dataset_writer,dataset_metadata}``; dtype alone
   decides video vs. numeric handling.
3. Whether ``lerobot.datasets.aggregate.aggregate_datasets`` exists. **Yes** —
   ``aggregate_datasets(repo_ids, aggr_repo_id, roots=..., aggr_root=..., ...)``
   is present in the pinned lerobot version, so REDESIGN_PLAN.md §5.3 says to use it
   and delete the hand-rolled merger. origami/merge.py does so.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from utils.lerobot_common import (
    ACTION_CHUNK,
    ACTION_DIM,
    DEFORM_KEYS,
    KEY_ACTION,
    KEY_ACTION_ABS,
    KEY_HEAD,
    KEY_STATE,
    KEY_TACF6,
    KEY_WRIST_L,
    KEY_WRIST_R,
    build_trex_features,
)

FPS = 30
N_FRAMES = 3
HEAD_SHAPE = (3, 224, 224)
WRIST_SHAPE = (3, 224, 224)
DEFORM_SHAPE = (3, 240, 240)


@pytest.fixture
def tmp_dataset_root():
    parent = Path(tempfile.mkdtemp(prefix="origami_probe_"))
    root = parent / "probe_dataset"  # LeRobotDatasetMetadata.create() requires a non-existing dir
    yield root
    shutil.rmtree(parent, ignore_errors=True)


def _make_frame(rng: np.random.Generator) -> dict:
    frame = {
        "task": "probe task",
        KEY_HEAD: rng.integers(0, 256, size=HEAD_SHAPE, dtype=np.uint8).transpose(1, 2, 0),
        KEY_WRIST_R: rng.integers(0, 256, size=WRIST_SHAPE, dtype=np.uint8).transpose(1, 2, 0),
        KEY_WRIST_L: rng.integers(0, 256, size=WRIST_SHAPE, dtype=np.uint8).transpose(1, 2, 0),
        KEY_STATE: rng.normal(size=(ACTION_DIM,)).astype(np.float32),
        KEY_ACTION: rng.normal(size=(ACTION_CHUNK, ACTION_DIM)).astype(np.float32),
        KEY_ACTION_ABS: rng.normal(size=(ACTION_DIM,)).astype(np.float32),
        KEY_TACF6: rng.normal(size=(10, 6)).astype(np.float32),
    }
    for k in DEFORM_KEYS:
        tile = rng.integers(0, 256, size=DEFORM_SHAPE, dtype=np.uint8).transpose(1, 2, 0)
        frame[k] = tile
    return frame


def test_lerobot_probe_full_schema_round_trips(tmp_dataset_root):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    features = build_trex_features(
        head_shape=HEAD_SHAPE,
        include_wrist=True,
        wrist_shape=WRIST_SHAPE,
        include_tactile=True,
        deform_shape=DEFORM_SHAPE,
        include_action_abs=True,
    )
    # DEFORM_KEYS are NOT under observation.images.* -- this is exactly what §11.4
    # asks us to probe.
    for k in DEFORM_KEYS:
        assert not k.startswith("observation.images.")
        assert features[k]["dtype"] == "video"

    ds = LeRobotDataset.create(
        repo_id="origami/probe",
        fps=FPS,
        features=features,
        root=tmp_dataset_root,
        robot_type="north_poc2_2",
        use_videos=True,
        video_files_size_in_mb=64,
        data_files_size_in_mb=64,
    )

    rng = np.random.default_rng(0)
    written_frames = [_make_frame(rng) for _ in range(N_FRAMES)]
    for frame in written_frames:
        ds.add_frame(frame)
    ds.save_episode()
    ds.finalize()

    assert ds.meta.total_episodes == 1
    assert ds.meta.total_frames == N_FRAMES

    delta_timestamps = {
        KEY_HEAD: [0.0, 1.0 / FPS],
        KEY_WRIST_R: [0.0, 1.0 / FPS],
        KEY_WRIST_L: [0.0, 1.0 / FPS],
        KEY_TACF6: [-1.0 / FPS, 0.0],
        **{k: [0.0] for k in DEFORM_KEYS},
    }

    reopened = LeRobotDataset(
        repo_id="origami/probe",
        root=tmp_dataset_root,
        delta_timestamps=delta_timestamps,
    )
    assert len(reopened) == N_FRAMES

    sample = reopened[1]  # middle frame: both delta_timestamps offsets stay in-range

    assert sample[KEY_HEAD].shape == (2, *HEAD_SHAPE)
    assert sample[KEY_WRIST_R].shape == (2, *WRIST_SHAPE)
    assert sample[KEY_WRIST_L].shape == (2, *WRIST_SHAPE)
    assert sample[KEY_TACF6].shape == (2, 10, 6)
    # A single-offset delta_timestamps entry (e.g. deform's [0.0]) comes back WITHOUT a
    # leading temporal dim -- the pinned lerobot version squeezes length-1 windows, unlike
    # the >1-offset keys above which keep [T, ...]. Probed behavior, not assumed.
    for k in DEFORM_KEYS:
        assert sample[k].shape == DEFORM_SHAPE

    assert sample[KEY_STATE].shape == (ACTION_DIM,)
    assert sample[KEY_ACTION].shape == (ACTION_CHUNK, ACTION_DIM)
    assert sample[KEY_ACTION_ABS].shape == (ACTION_DIM,)

    np.testing.assert_allclose(
        sample[KEY_STATE].numpy(), written_frames[1][KEY_STATE], atol=1e-5
    )
    np.testing.assert_allclose(
        sample[KEY_ACTION].numpy(), written_frames[1][KEY_ACTION], atol=1e-5
    )
    np.testing.assert_allclose(
        sample[KEY_ACTION_ABS].numpy(), written_frames[1][KEY_ACTION_ABS], atol=1e-5
    )
    np.testing.assert_allclose(
        sample[KEY_TACF6].numpy()[-1], written_frames[1][KEY_TACF6], atol=1e-5
    )


def test_aggregate_datasets_exists():
    from lerobot.datasets.aggregate import aggregate_datasets

    import inspect

    sig = inspect.signature(aggregate_datasets)
    assert "roots" in sig.parameters
    assert "aggr_root" in sig.parameters
