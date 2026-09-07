"""§7.2 -- origami/train_origami.py is a vendored copy of
`T-Rex@full-pipeline:scripts/midtrain.py` with a small, precisely marked delta set.

CPU-only, network-free: `full-pipeline` was already fetched into the T-Rex submodule during
step 1 (see origami/PROGRESS.md), so `git show origin/full-pipeline:...` is a local ref
lookup, not a network call.
"""
from __future__ import annotations

import difflib
import subprocess
from pathlib import Path

import pytest

from origami.constants import REPO_ROOT
from origami.train_origami import _episode_cum_frames

TREX_ROOT = REPO_ROOT / "T-Rex"
VENDORED_PATH = REPO_ROOT / "origami" / "train_origami.py"
B23EAFE = "b23eafe564a1457cd4eacb889aaf6fbf29a29034"

MARKER = "ORIGAMI-DELTA"
CONTEXT_LINES = 3  # how many surrounding lines of an inserted/replaced hunk we search


def _git_show(ref: str, path: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        cwd=TREX_ROOT,
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout


def _upstream_lines() -> list[str]:
    return _git_show(B23EAFE, "scripts/midtrain.py").splitlines(keepends=True)


def _vendored_lines() -> list[str]:
    return VENDORED_PATH.read_text().splitlines(keepends=True)


def test_vendored_file_exists_and_is_nontrivially_different():
    upstream = _upstream_lines()
    vendored = _vendored_lines()
    assert vendored != upstream
    assert len(vendored) > 0.9 * len(upstream)  # a vendor+small-delta, not a rewrite


def test_every_changed_hunk_carries_origami_delta_marker():
    upstream = _upstream_lines()
    vendored = _vendored_lines()
    sm = difflib.SequenceMatcher(a=upstream, b=vendored, autojunk=False)

    unmarked_hunks = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        # Search a window around the changed region in the VENDORED file (insertions and
        # replacements land there) for the marker. For pure deletions (tag == "delete",
        # j1 == j2), there's nothing to mark in the vendored file by construction -- check
        # the marker is present just before/after the deletion point instead.
        lo = max(0, j1 - CONTEXT_LINES)
        hi = min(len(vendored), j2 + CONTEXT_LINES)
        window = "".join(vendored[lo:hi])
        if MARKER not in window:
            unmarked_hunks.append({
                "tag": tag,
                "upstream_lines": "".join(upstream[i1:i2]),
                "vendored_lines": "".join(vendored[j1:j2]),
            })

    assert not unmarked_hunks, (
        f"{len(unmarked_hunks)} changed hunk(s) vs. full-pipeline@{B23EAFE}'s "
        f"scripts/midtrain.py have no nearby `# {MARKER}:` marker:\n"
        + "\n---\n".join(
            f"upstream:\n{h['upstream_lines']}\nvendored:\n{h['vendored_lines']}"
            for h in unmarked_hunks
        )
    )


def test_vendored_file_is_valid_python():
    import ast
    ast.parse(VENDORED_PATH.read_text())


def _make_fake_trex_dataset(cum_frames):
    """A stand-in for TRexLeRobotDataset/DelayedTRexLeRobotDataset exposing exactly the
    `.ds.meta.episodes` / `.ds.num_episodes` surface `_episode_cum_frames` reads -- the real
    dataset classes carry no `_cum_frames`/`_num_episodes` of their own (see the bug this
    covers below)."""
    import numpy as np
    from types import SimpleNamespace

    cum_frames = np.asarray(cum_frames, dtype=np.int64)
    total = int(cum_frames[-1])
    meta = SimpleNamespace(episodes={
        "episode_index": list(range(len(cum_frames))),
        "dataset_to_index": cum_frames.tolist(),
    })

    class _FakeDataset:
        ds = SimpleNamespace(meta=meta, num_episodes=len(cum_frames))

        def __len__(self):
            return total

    return _FakeDataset()


def test_episode_cum_frames_reads_dataset_to_index_from_lerobot_meta():
    """Real bug found running the first real `train_origami.py` invocation against the
    full-scale eef62_train root (§12 step 11's pilot, pulled forward into verification):
    `EpisodeGroupedSampler.__init__` read `dataset._cum_frames`/`dataset._num_episodes` --
    attributes that never existed on the real `TRexLeRobotDataset`/`DelayedTRexLeRobotDataset`
    classes (verified by reading T-Rex/qwen_vla/lerobot_dataset.py's actual `__init__` --
    it stores only `self.ds`, the wrapped `lerobot.datasets.lerobot_dataset.LeRobotDataset`).
    No test had ever constructed `EpisodeGroupedSampler` against a real dataset instance
    before this bug surfaced for real, hence it went uncaught since step 9. Fix: derive
    episode boundaries from `dataset.ds.meta.episodes["dataset_to_index"]` (confirmed against
    a real merged root: `dataset_to_index`'s last value equals the dataset's total frame
    count, and rows come pre-sorted by `episode_index`) instead.
    """
    fake_dataset = _make_fake_trex_dataset([5, 12, 20])
    cum_frames, num_episodes = _episode_cum_frames(fake_dataset)
    assert num_episodes == 3
    assert list(cum_frames) == [5, 12, 20]


def test_episode_grouped_sampler_works_without_a_process_group():
    """`EpisodeGroupedSampler(dataset, shuffle=True, ...)` is called at its one real call site
    (train()'s dataloader construction) with num_replicas/rank left at their None defaults,
    which torch's own `DistributedSampler.__init__` resolves via bare `dist.get_world_size()`/
    `dist.get_rank()` -- raising `ValueError: Default process group has not been initialized`
    on any non-distributed launch, i.e. every invocation of our §7.3 recipe
    (`accelerate launch --num_processes 1`, no process group). Asserts the fix
    (`_world_size()`/`_rank()` defaults, mirroring the existing `TrainingMetrics.world_size`
    pattern) makes single-process construction succeed and resolve to a 1-replica, rank-0
    sampler -- without needing `torch.distributed` initialized at all, exactly the CI/test
    environment this suite already runs in.
    """
    from origami.train_origami import EpisodeGroupedSampler
    import torch.distributed as dist

    assert not dist.is_initialized(), "test assumes no process group -- the real-world failure mode"

    fake_dataset = _make_fake_trex_dataset([5, 12, 20])
    sampler = EpisodeGroupedSampler(fake_dataset, shuffle=True, seed=0, drop_last=True)
    assert sampler.num_replicas == 1
    assert sampler.rank == 0
    # __iter__ must also run end-to-end (exercises the episode-grouping logic downstream
    # of the fixed __init__, not just construction).
    indices = list(sampler)
    assert len(indices) == sampler.num_samples
    assert all(0 <= i < 20 for i in indices)
