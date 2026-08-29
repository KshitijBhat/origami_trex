"""Heterogeneous (GPU + CPU) frame decoding for `prepare`.

The release is *mixed*: some seasons ship h264 and some ship AV1, sometimes with
different codecs in the same split, so the decoder has to be chosen per file
rather than assumed.  Measured on this box (RTX 4060 Laptop + i7-13620H):

    stream            path                        wall     throughput
    h264, uncontended, stride 20:
    head_left 480x480 cpu lanczos                 10.7s      1949 fps
    head_left 480x480 h264_cuvid -resize 224       4.1s      5085 fps   <- 2.6x
    head_left 480x480 cuda + scale_cuda            6.6s      3159 fps
    deform  1200x480  cpu (no scaling)            30.9s      3199 fps   <- best
    deform  1200x480  cuda + hwdownload           41.8s      2365 fps
    av1, 3000 frames, measured under load (a conversion was running):
    head_left 480x480 cpu libdav1d+lanczos         2.1s      1457 fps
    head_left 480x480 av1_cuvid -resize 224        1.7s      1777 fps   <- 1.2x
    deform  1200x480  cpu                          4.0s       745 fps   <- best
    deform  1200x480  av1_cuvid                    5.6s       531 fps

The AV1 pair is contended and so understates the GPU, but the *shape* is the
same for both codecs: the GPU wins on the three RGB streams, which downscale
480->224 inside the decoder so only 224x224 frames cross the PCIe bus, and
*loses* on the deform strip, which is not downscaled at all -- there the
device-to-host copy of full 1200x480 frames costs more than NVDEC saves.
`decode_frames` therefore routes per stream rather than flipping one global
switch.

Which codecs the GPU can actually take is a *second* per-box question, and it
does not follow from "NVDEC works here": NVDEC only gained AV1 on Ampere, so an
A100 / V100 / T4 decodes the h264 seasons on the GPU and has to send the AV1
ones to libdav1d.  `usable_decoders()` probes each codec end-to-end once at
startup, and `install()` additionally retires a decoder for the rest of the
process the first time a real file fails on it — otherwise every AV1 file in the
sweep pays a failed NVDEC attempt before falling back.

Fidelity: `-resize` uses the decoder's own scaler, not lanczos.  Against the CPU
path, mean absolute error is ~1.0/255 with a pixel correlation of 0.9993 (frames
0/500/1042 of a 1043-frame sample), i.e. below JPEG quantisation noise.  Frame
counts are identical, so `_split_jpegs`' count assertion still holds.  Set
`ORIGAMI_GPU=0` to reproduce the byte-exact CPU output.
"""
from __future__ import annotations

import functools
import logging
import os
import subprocess
import threading
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


#: NVDEC decoder per source codec.  Anything absent here stays on the CPU.
CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "av1": "av1_cuvid",
    "hevc": "hevc_cuvid",
    "vp9": "vp9_cuvid",
}

#: Encoders used to synthesise a throwaway clip when probing a codec end-to-end,
#: best-effort in order.  A codec with no available encoder cannot be probed
#: ahead of time; `install()` then finds out on the first real file and retires
#: the decoder if it fails.
PROBE_ENCODERS = {
    "h264": ("libx264",),
    "av1": ("libsvtav1", "librav1e", "libaom-av1"),
    "hevc": ("libx265",),
    "vp9": ("libvpx-vp9",),
}

#: Decoders that failed on a real file in this process.  Per-process by design:
#: conversion runs in a `spawn` pool, so each worker learns once and the cost of
#: learning is one failed decode rather than one per file.
_RETIRED: set = set()
_RETIRED_LOCK = threading.Lock()


@functools.lru_cache(maxsize=512)
def probe_codec(video_path: str) -> str:
    """Codec of the first video stream, or "" if ffprobe cannot say.

    Necessary because the dataset mixes h264 and AV1 seasons.  Hardcoding
    `h264_cuvid` sent every AV1 file straight into "No start code is found",
    which the fallback caught -- correct output, but the whole season then
    decoded on the CPU while an idle NVDEC sat next to it.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=nw=1:nk=1", video_path],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
        return out.stdout.decode("utf-8", "replace").strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


@functools.lru_cache(maxsize=1)
def _compiled_decoders() -> frozenset:
    """Decoder names this ffmpeg build knows about."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-decoders"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=60)
        names = set()
        for line in out.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0][:1] in "VAS":
                names.add(parts[1])
        return frozenset(names)
    except (OSError, subprocess.SubprocessError):
        return frozenset()


def cuvid_decoder(video_path: str) -> Optional[str]:
    """The NVDEC decoder to use for this file, or None to stay on the CPU.

    Three conditions: we map the file's codec to a cuvid decoder, this box
    provably (or plausibly — see `usable_decoders`) decodes that codec on the
    GPU, and the decoder has not already been retired after failing on a real
    file in this process.  `-decoders` alone is not proof the *device* works,
    which is why the end-to-end probes gate the path and `decode_frames` keeps
    its CPU fallback regardless.
    """
    codec = probe_codec(video_path)
    name = CUVID_DECODERS.get(codec)
    if not name or name in _RETIRED:
        return None
    return name if codec_decode_available(codec) is not False else None


def retire_decoder(decoder: str, reason: str) -> None:
    """Stop trying `decoder` in this process after it failed on a real file."""
    with _RETIRED_LOCK:
        if decoder in _RETIRED:
            return
        _RETIRED.add(decoder)
    logger.warning("[accel] retiring %s for this process after: %s", decoder, reason)


@functools.lru_cache(maxsize=1)
def _compiled_encoders() -> frozenset:
    """Encoder names this ffmpeg build knows about (used only for probing)."""
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             timeout=60)
        names = set()
        for line in out.stdout.decode("utf-8", "replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0][:1] in "VAS":
                names.add(parts[1])
        return frozenset(names)
    except (OSError, subprocess.SubprocessError):
        return frozenset()


def _encode_probe_clip(codec: str, path: str) -> bool:
    """Write ~3 frames of `codec` to `path`.  False if no encoder can do it."""
    available = _compiled_encoders()
    for encoder in PROBE_ENCODERS.get(codec, ()):
        if encoder not in available:
            continue
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y",
               "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=0.1",
               "-c:v", encoder, "-pix_fmt", "yuv420p"]
        # libaom is glacial at its default speed and this clip is throwaway.
        if encoder == "libaom-av1":
            cmd += ["-cpu-used", "8", "-strict", "experimental"]
        elif encoder == "libsvtav1":
            cmd += ["-preset", "12"]
        try:
            made = subprocess.run(cmd + [path], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL, timeout=180)
        except (OSError, subprocess.SubprocessError):
            continue
        if made.returncode == 0 and os.path.getsize(path) > 0:
            return True
    return False


@functools.lru_cache(maxsize=8)
def codec_decode_available(codec: str) -> Optional[bool]:
    """Can NVDEC decode `codec` *on this box*, through the exact command we run?

    Returns True/False, or None when the question could not be settled offline
    because this ffmpeg has no encoder for the codec.  The distinction matters:
    None means "try it on the first real file", False means "do not bother".
    """
    decoder = CUVID_DECODERS.get(codec)
    if not decoder or decoder not in _compiled_decoders():
        return False
    if os.environ.get("ORIGAMI_GPU", "1") == "0":
        return False
    tmp = None
    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        if not _encode_probe_clip(codec, tmp):
            return None
        out = subprocess.run(
            build_cmd(tmp, "gte(n\\,0)", scale=224, quality=3, decoder=decoder),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
        return out.returncode == 0 and out.stdout.startswith(b"\xff\xd8\xff")
    except (OSError, subprocess.SubprocessError, ImportError):
        return False
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def usable_decoders() -> Dict[str, str]:
    """codec -> cuvid decoder, for the codecs this box provably decodes on the GPU.

    Codecs whose probe was inconclusive (no local encoder to make a test clip)
    are included optimistically: `install()`'s per-file fallback still covers
    them, and excluding them would forfeit NVDEC on a box that can do it.
    """
    out: Dict[str, str] = {}
    for codec, decoder in CUVID_DECODERS.items():
        verdict = codec_decode_available(codec)
        if verdict is None:
            logger.info("[accel] %s: cannot probe (no local encoder); will try "
                        "%s on the first file and fall back if it fails",
                        codec, decoder)
            out[codec] = decoder
        elif verdict:
            out[codec] = decoder
    return out


@functools.lru_cache(maxsize=1)
def gpu_decode_available() -> bool:
    """True if NVDEC can serve *at least one* of the codecs in this release.

    Probed end-to-end per codec (`codec_decode_available`) -- encode a throwaway
    clip, decode it through `<codec>_cuvid -resize`, require real JPEG output --
    rather than by parsing `-decoders` or poking `-init_hw_device`.  Both of
    those report success on this box even with no usable device
    (`-init_hw_device cuda` exits 0 and prints nothing), so a lighter check would
    send every worker down a path that only fails once it reaches the first real
    video.

    "At least one" is the right bar because the release mixes codecs: on a box
    whose NVDEC predates AV1, the h264 seasons should still take the GPU while
    the AV1 ones go to libdav1d, and `cuvid_decoder` makes that call per file.
    """
    if os.environ.get("ORIGAMI_GPU", "1") == "0":
        return False
    return bool(usable_decoders())


# A wedged GPU decode would otherwise hold a converter process forever; the CPU
# path has no timeout because that is the fallback and matches `prepare`'s own
# behaviour.  Generous enough that real files never trip it -- the largest
# stream here decodes in well under a minute.
GPU_TIMEOUT = float(os.environ.get("ORIGAMI_GPU_TIMEOUT", "900"))


def _run(cmd: Sequence[str], video_path: str,
         timeout: Optional[float] = None) -> bytes:
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed on {video_path}: "
            f"{proc.stderr.decode('utf-8', 'replace')[:2000]}")
    return proc.stdout


def build_cmd(video_path: str, select: str, *, scale: Optional[int],
              quality: int, decoder: Optional[str] = None) -> List[str]:
    """The ffmpeg argv for one decode pass.

    `decoder` is a cuvid decoder name to run on the GPU, or None for the CPU
    path.  A GPU decode is only ever built when `scale` is set: the resize
    happens inside NVDEC via `-resize`, which is what keeps the readback small
    enough to pay off.
    """
    if decoder and scale:
        return [
            "ffmpeg", "-nostdin", "-v", "error",
            "-c:v", decoder, "-resize", f"{scale}x{scale}",
            "-i", video_path,
            "-vf", f"select='{select}'",
            "-vsync", "0",
            "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", str(quality),
            "pipe:1",
        ]
    vf = f"select='{select}'"
    if scale:
        vf += f",scale={scale}:{scale}:flags=lanczos"
    return [
        "ffmpeg", "-nostdin", "-v", "error",
        "-i", video_path,
        "-vf", vf,
        "-vsync", "0",
        "-f", "image2pipe", "-c:v", "mjpeg", "-q:v", str(quality),
        "pipe:1",
    ]


def install() -> bool:
    """Point `prepare.decode_frames` at the heterogeneous implementation.

    `FrameSource._decode_file` resolves `decode_frames` as a module global, so
    rebinding the attribute is enough; no call sites need to change.  Returns
    whether the GPU path is live, for logging.
    """
    from . import prepare as prep

    use_gpu = gpu_decode_available()

    def decode_frames(video_path, ranges, expected, *, scale, quality):
        select = prep._select_expr(ranges)
        context = os.path.basename(video_path)
        decoder = cuvid_decoder(video_path) if (use_gpu and scale) else None
        if decoder:
            try:
                blob = _run(build_cmd(video_path, select, scale=scale,
                                      quality=quality, decoder=decoder),
                            video_path, timeout=GPU_TIMEOUT)
                return prep._split_jpegs(blob, expected, context)
            except (RuntimeError, subprocess.TimeoutExpired) as exc:
                # A per-file GPU failure (odd dimensions, exhausted decoder
                # sessions) must not lose the season -- redo it on the CPU.
                logger.warning("[accel] %s failed on %s, using CPU: %s",
                               decoder, context, str(exc)[:200])
                # A decoder this GPU does not implement (the AV1-on-A100 case)
                # fails on *every* file of that codec.  Retire it rather than
                # paying the failed attempt once per file for the whole sweep.
                retire_decoder(decoder, f"{context}: {str(exc)[:160]}")
        blob = _run(build_cmd(video_path, select, scale=scale,
                              quality=quality), video_path)
        return prep._split_jpegs(blob, expected, context)

    prep.decode_frames = decode_frames
    return use_gpu
