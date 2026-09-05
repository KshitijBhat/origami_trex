"""Season-shard concatenation into one root. REDESIGN_PLAN.md §5.3.

§5.3 asks to *first* probe for ``lerobot.datasets.aggregate.aggregate_datasets`` in the
pinned lerobot version and use it if it works, falling back to a hand-rolled merger
otherwise. It exists in the pinned version (confirmed, step 2), but it cannot actually merge
our shards: ``aggregate_data`` always rewrites each data parquet via
``to_parquet_one_row_group_per_episode``, which round-trips through
``pa.Table.from_pandas(df, preserve_index=False)`` with no explicit schema. That round-trip
loses the fixed-shape-array typing of our 2D ``action`` feature ([16,62] per row) and pyarrow
raises ``ArrowTypeError: Conversion failed for column action with type array[float32]`` --
confirmed independent of anything in this repo by reading straight from a real 2-shard merge
attempt. So this module is the hand-rolled merger, made simple by §5.2's
``ONE_EPISODE_PER_FILE_MB`` writer setting: every source data/video file holds *exactly one*
episode, so merging is a pure copy-and-renumber -- read each data parquet with ``pyarrow``
directly (never through pandas, so the 2D array column is never re-inferred), patch only the
``index``/``episode_index`` int columns, and byte-copy video files untouched.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from origami.stats import StreamingNormStats

logger = logging.getLogger(__name__)


def merge_shards(
    shard_roots: list[str | Path],
    out_root: str | Path,
    repo_id: str = "origami/eef62",
) -> Path:
    """Concatenate per-season shards (each a season-named directory produced by
    ``convert_season``) into one LeRobot dataset root at ``out_root``, renumbering
    episode/chunk/file indices into one global sequence, and merge each shard's
    ``meta/trex_norm_stats.json`` (§5.4) + ``meta/origami_prep.json`` the same way.
    """
    shard_roots = [Path(p) for p in shard_roots]
    assert shard_roots, "merge_shards: no shards given"
    out_root = Path(out_root)

    infos = [json.loads((r / "meta" / "info.json").read_text()) for r in shard_roots]
    features, fps, codebase_version = infos[0]["features"], infos[0]["fps"], infos[0]["codebase_version"]
    for r, info in zip(shard_roots[1:], infos[1:]):
        assert info["features"] == features, f"{r}: features mismatch"
        assert info["fps"] == fps, f"{r}: fps mismatch"
        assert info["codebase_version"] == codebase_version, f"{r}: codebase_version mismatch"

    task_dfs = [pd.read_parquet(r / "meta" / "tasks.parquet") for r in shard_roots]
    for r, df in zip(shard_roots[1:], task_dfs[1:]):
        assert df.equals(task_dfs[0]), f"{r}: tasks.parquet differs from {shard_roots[0]} (§5.7)"

    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]
    (out_root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for vk in video_keys:
        (out_root / "videos" / vk / "chunk-000").mkdir(parents=True, exist_ok=True)

    global_ep = 0
    global_frame = 0
    data_file_ctr = 0
    video_file_ctr = {vk: 0 for vk in video_keys}
    out_episode_rows = []

    for shard_root in shard_roots:
        episodes_meta = pd.read_parquet(shard_root / "meta" / "episodes").sort_values("episode_index")
        for _, row in episodes_meta.iterrows():
            src_ep_idx = int(row["episode_index"])
            src_data_path = (
                shard_root / "data" / f"chunk-{int(row['data/chunk_index']):03d}"
                / f"file-{int(row['data/file_index']):03d}.parquet"
            )
            table = pq.read_table(src_data_path)
            ep_col = table.column("episode_index").to_pylist()
            assert set(ep_col) == {src_ep_idx}, (
                f"{src_data_path}: expected exactly one episode per data file (§5.2's "
                f"ONE_EPISODE_PER_FILE_MB), found episode_index values {set(ep_col)}"
            )
            n = table.num_rows

            table = table.set_column(
                table.schema.get_field_index("index"), "index",
                pa.array(range(global_frame, global_frame + n), type=pa.int64()),
            )
            table = table.set_column(
                table.schema.get_field_index("episode_index"), "episode_index",
                pa.array([global_ep] * n, type=pa.int64()),
            )

            dst_data_path = out_root / "data" / "chunk-000" / f"file-{data_file_ctr:03d}.parquet"
            pq.write_table(table, dst_data_path)

            new_row = row.to_dict()
            new_row["episode_index"] = global_ep
            new_row["dataset_from_index"] = global_frame
            new_row["dataset_to_index"] = global_frame + n
            new_row["data/chunk_index"] = 0
            new_row["data/file_index"] = data_file_ctr

            for vk in video_keys:
                src_chunk, src_file = int(row[f"videos/{vk}/chunk_index"]), int(row[f"videos/{vk}/file_index"])
                src_video_path = shard_root / "videos" / vk / f"chunk-{src_chunk:03d}" / f"file-{src_file:03d}.mp4"
                dst_video_path = out_root / "videos" / vk / "chunk-000" / f"file-{video_file_ctr[vk]:03d}.mp4"
                shutil.copy2(src_video_path, dst_video_path)
                new_row[f"videos/{vk}/chunk_index"] = 0
                new_row[f"videos/{vk}/file_index"] = video_file_ctr[vk]
                # from_timestamp/to_timestamp stay unchanged: the video file is copied whole.
                video_file_ctr[vk] += 1

            out_episode_rows.append(new_row)
            data_file_ctr += 1
            global_frame += n
            global_ep += 1

    merged_episodes_df = pd.DataFrame(out_episode_rows)
    (out_root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    merged_episodes_df.to_parquet(out_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet", index=False)

    task_dfs[0].to_parquet(out_root / "meta" / "tasks.parquet")

    merged_info = dict(infos[0])
    merged_info["total_episodes"] = global_ep
    merged_info["total_frames"] = global_frame
    merged_info["total_tasks"] = len(task_dfs[0])
    merged_info["splits"] = {"all": f"0:{global_ep}"}
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    (out_root / "meta" / "info.json").write_text(json.dumps(merged_info, indent=2))

    _merge_norm_stats(shard_roots, out_root)
    _merge_origami_prep_meta(shard_roots, out_root)
    return out_root


def _merge_norm_stats(shard_roots: list[Path], out_root: Path) -> None:
    merged_stats = StreamingNormStats()
    n_merged = 0
    for shard_root in shard_roots:
        dump_path = shard_root / "meta" / "trex_norm_stats.pkl"
        if dump_path.exists():
            merged_stats.merge(StreamingNormStats.load(dump_path))
            n_merged += 1
        else:
            logger.warning(
                "merge_shards: %s has no trex_norm_stats.pkl; skipping it in the merged stats "
                "(convert_season should dump() the accumulator, not just write() the JSON).",
                shard_root,
            )
    if n_merged > 0:
        merged_stats.write(str(out_root))
    else:
        logger.warning("merge_shards: no shard had a trex_norm_stats.pkl; %s has no merged stats", out_root)


def _merge_origami_prep_meta(shard_roots: list[Path], out_root: Path) -> None:
    """Merge each shard's ``meta/origami_prep.json`` (truncation logs, LockedConfig digest,
    etc. -- written by ``convert_season``) into one record at the merged root."""
    seasons, truncated, locked_digests = [], [], set()
    for shard_root in shard_roots:
        prep_path = shard_root / "meta" / "origami_prep.json"
        if not prep_path.exists():
            continue
        prep = json.loads(prep_path.read_text())
        seasons.append(shard_root.name)
        truncated.extend(prep.get("truncated", []))
        if "locked_digest" in prep:
            locked_digests.add(prep["locked_digest"])

    if len(locked_digests) > 1:
        raise ValueError(
            f"merge_shards: shards were converted under different LockedConfig digests "
            f"{sorted(locked_digests)} -- refusing to merge (§3.2: the absolute state frame "
            f"is only consistent within one LockedConfig)."
        )

    out_path = out_root / "meta" / "origami_prep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"seasons": seasons, "truncated": truncated, "locked_digest": next(iter(locked_digests), None)},
        indent=2,
    ))
