"""Pipelined driver for `prepare`: fetch the next seasons while converting this one.

`prepare.main` walks seasons strictly serially -- download, convert, delete,
repeat -- so the network sits idle for the whole conversion and the CPU and GPU
sit idle for the whole download.  On a 126-season sweep that is most of the
wall clock, because a season is ~1 GB to fetch and well under a minute to
convert once the bytes are local.

Here downloads run on a thread pool and conversions on a process pool, so the
two overlap.  A semaphore caps how many seasons may be on disk at once
(downloaded but not yet converted), which is what keeps the stream-and-delete
disk guarantee: peak stays at `output-so-far + disk_budget seasons` rather than
growing without bound just because the downloaders got ahead.

Conversion runs in processes, not threads, because `prepare` is CPU-bound
between ffmpeg calls; downloads run in threads because they are socket-bound and
`huggingface_hub` releases the GIL.

Codec note: the `lerobot3.0` exports are *mostly AV1*, with some h264 seasons
(the dataset card puts it as "lerobot3.0 is primarily AV1, lerobotv2.1 is
primarily H.264"), so the decode budget is set by AV1 rather than h264 -- roughly
2-4x more CPU per frame through libdav1d, and NVDEC only decodes AV1 on Ampere
and later.  `accel.usable_decoders()` settles that per codec at startup and this
module logs the answer, so a run on a box that cannot GPU-decode AV1 says so on
line 2 instead of emitting one fallback warning per video file.  When most
seasons land on the CPU, `--converters` is the knob that matters: it is the only
thing keeping the cores busy.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import get_context
from typing import Dict, List, Optional, Sequence, Tuple

from .anchoring import ANCHOR_MODES, anchor_spec, describe as describe_anchor
from .prepare import (
    EXIT_PARTIAL,
    PrepConfig,
    _merge_entries,
    check_config_compatible,
    prepare_season,
    season_already_done,
    write_dataset_meta,
)
from .seasons import select_seasons

logger = logging.getLogger(__name__)


def _convert(season: str, src_root: str, out_root: str, cfg: PrepConfig,
             overwrite: bool) -> Tuple[str, List[dict], Optional[str]]:
    """Convert one already-downloaded season.  Runs in a worker process."""
    try:
        from . import accel
        accel.install()
        entries = prepare_season(season, src_root, out_root, cfg,
                                 overwrite=overwrite)
        return season, entries, None
    except Exception as exc:                       # keep the sweep going
        return season, [], f"{type(exc).__name__}: {exc}"


def run(seasons: Sequence[str], out_root: str, cache_root: str, cfg: PrepConfig,
        *, downloaders: int, converters: int, disk_budget: int,
        keep_source: bool, overwrite: bool, src_root: str,
        hf_token: Optional[str]) -> List[Tuple[str, str]]:
    """Fetch and convert `seasons`, overlapping the two.  Returns failures."""
    from .fetch import download_season, drop_season, have_season

    todo = [s for s in seasons
            if overwrite or not season_already_done(out_root, s)]
    skipped = len(seasons) - len(todo)
    if skipped:
        logger.info("[fast] %d/%d seasons already converted, not fetching",
                    skipped, len(seasons))
    if not todo:
        return []

    # Bounds seasons resident on disk so the downloaders cannot outrun the
    # converters and fill the filesystem.  A permit is taken before a download
    # starts and returned once that season's conversion has been retired.
    room = threading.Semaphore(disk_budget)
    state = threading.Lock()
    all_entries: List[dict] = []
    failures: List[Tuple[str, str]] = []
    done = 0
    started = time.time()

    ctx = get_context("spawn")
    pool = ctx.Pool(processes=converters)

    def retire(result: Tuple[str, List[dict], Optional[str]]) -> None:
        """Record a finished conversion and free its disk slot.

        Runs on the pool's result-handler thread, deliberately: the permit has
        to come back without help from the main thread, which spends most of
        its time parked in `as_completed` waiting for the next download.  When
        the release depended on the main loop, a full `room` deadlocked the
        whole sweep -- every downloader blocked on `acquire` meant no future
        ever completed, so nothing ever drained.
        """
        nonlocal done
        name, entries, error = result
        try:
            with state:
                if error:
                    failures.append((name, error))
                else:
                    all_entries.extend(entries)
                    write_dataset_meta(
                        out_root, _merge_entries(out_root, all_entries), cfg)
                done += 1
                progress = (done, sum(e["n_samples"] for e in all_entries))
            if not src_root and not keep_source:
                drop_season(cache_root, name)
        finally:
            room.release()
        logger.info("[fast] %d/%d seasons | %d samples | %.1f min",
                    progress[0], len(todo), progress[1],
                    (time.time() - started) / 60)

    def crashed(exc: BaseException) -> None:
        """Only reached if the worker itself dies -- `_convert` catches its own."""
        with state:
            failures.append(("<worker>", f"{type(exc).__name__}: {exc}"))
        room.release()

    def fetch(season: str) -> Optional[str]:
        room.acquire()
        try:
            if src_root:
                return season
            if not have_season(cache_root, season):
                download_season(season, cache_root, token=hf_token)
            return season
        except Exception as exc:
            room.release()
            with state:
                failures.append((season, f"download: {exc}"))
            # A truncated tree would fail conversion forever; drop it so a
            # rerun refetches instead of retrying the same missing file.
            if not src_root and not keep_source:
                drop_season(cache_root, season)
            return None

    try:
        with ThreadPoolExecutor(max_workers=downloaders,
                                thread_name_prefix="fetch") as fetcher:
            futures = [fetcher.submit(fetch, season) for season in todo]
            for future in as_completed(futures):
                season = future.result()
                if season is None:
                    continue
                pool.apply_async(
                    _convert,
                    (season, src_root or cache_root, out_root, cfg, overwrite),
                    callback=retire, error_callback=crashed)
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()

    write_dataset_meta(out_root, _merge_entries(out_root, all_entries), cfg)
    return failures


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pipelined, GPU-accelerated origami-flat conversion.")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seasons", nargs="*", default=None)
    parser.add_argument("--cache-root", default="")
    parser.add_argument("--src-root", default="")
    parser.add_argument("--keep-source", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--sample-stride", type=int, default=PrepConfig.sample_stride)
    parser.add_argument("--chunk-stride", type=int, default=PrepConfig.chunk_stride)
    parser.add_argument("--action-chunk", type=int, default=PrepConfig.action_chunk)
    parser.add_argument("--vqvae-window", type=int, default=PrepConfig.vqvae_window)
    parser.add_argument("--image-size", type=int, default=PrepConfig.image_size)
    parser.add_argument("--rgb-quality", type=int, default=PrepConfig.rgb_quality)
    parser.add_argument("--deform-quality", type=int, default=PrepConfig.deform_quality)
    parser.add_argument("--phase-mode", choices=["none", "progress"], default="none")
    parser.add_argument("--anchor-mode", choices=list(ANCHOR_MODES),
                        default=PrepConfig.anchor_mode,
                        help="see trex_origami.anchoring; 'hybrid' anchors the arms "
                             "to the previous command and leaves hands/motor absolute")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", "") or None)
    parser.add_argument("--downloaders", type=int, default=3,
                        help="concurrent season downloads")
    parser.add_argument("--converters", type=int, default=2,
                        help="worker processes converting seasons")
    parser.add_argument("--disk-budget", type=int, default=4,
                        help="max seasons resident on disk (~1 GB each)")
    parser.add_argument("--cpu-only", action="store_true",
                        help="disable GPU decoding (same as ORIGAMI_GPU=0)")
    parser.add_argument("--probe-codecs", type=int, default=1,
                        help="probe NVDEC once per codec before starting (the release "
                             "mixes AV1 and h264 and not every GPU decodes AV1)")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--stats-subsample", type=int, default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    if args.cpu_only:
        os.environ["ORIGAMI_GPU"] = "0"

    cfg = PrepConfig(
        sample_stride=args.sample_stride, chunk_stride=args.chunk_stride,
        action_chunk=args.action_chunk, vqvae_window=args.vqvae_window,
        image_size=args.image_size, rgb_quality=args.rgb_quality,
        deform_quality=args.deform_quality, phase_mode=args.phase_mode,
        anchor_mode=args.anchor_mode)

    seasons = args.seasons or select_seasons(args.split, args.limit)
    cache_root = args.cache_root or os.path.join(
        os.path.dirname(os.path.abspath(args.out_root)), "_src")
    os.makedirs(args.out_root, exist_ok=True)
    check_config_compatible(args.out_root, cfg, args.overwrite)

    from . import accel
    # `lerobot3.0` is primarily AV1 with some h264 seasons, and NVDEC only gained
    # AV1 on Ampere -- an A100/T4/V100 box decodes the h264 seasons on the GPU and
    # must fall back to libdav1d for the rest.  Probing once here turns that into
    # one startup line instead of a per-file warning from every worker.
    live = accel.usable_decoders() if args.probe_codecs else {}
    if args.probe_codecs:
        gpu_desc = (", ".join(f"{codec}->{dec}" for codec, dec in sorted(live.items()))
                    if live else "off (cpu for every codec)")
    else:
        gpu_desc = "on (rgb only)" if accel.gpu_decode_available() else "off (cpu)"
    logger.info("[fast] %s: %d seasons -> %s", args.split, len(seasons), args.out_root)
    logger.info("[fast] nvdec: %s  |  deform strip always cpu", gpu_desc)
    logger.info("[fast] downloaders=%d  converters=%d  disk_budget=%d seasons",
                args.downloaders, args.converters, args.disk_budget)
    logger.info("[fast] action anchoring %s", describe_anchor(anchor_spec(cfg.anchor_mode)))

    failures = run(seasons, args.out_root, cache_root, cfg,
                   downloaders=args.downloaders, converters=args.converters,
                   disk_budget=args.disk_budget, keep_source=args.keep_source,
                   overwrite=args.overwrite, src_root=args.src_root,
                   hf_token=args.hf_token)

    entries = _merge_entries(args.out_root, [])
    logger.info("[fast] %d episodes / %d samples / %d seasons",
                len(entries), sum(e["n_samples"] for e in entries),
                len({e["season"] for e in entries}))

    if args.stats:
        from .stats import compute_and_write
        compute_and_write(args.out_root, subsample=args.stats_subsample)

    if failures:
        logger.error("[fast] %d season(s) failed:", len(failures))
        for season, message in failures:
            logger.error("         %s: %s", season, message)
        # See prepare.EXIT_PARTIAL: incomplete-but-usable must not abort the
        # caller, or one bad download costs the whole rest of the pipeline.
        return EXIT_PARTIAL if entries else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
