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
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


#: NVDEC decoder per source codec.  Anything absent here stays on the CPU.
CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "av1": "av1_cuvid",
    "hevc": "hevc_cuvid",
    "vp9": "vp9_cuvid",
}


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

    Two conditions: we map the codec to a cuvid decoder, and this ffmpeg build
    actually has it compiled in.  `-decoders` alone is not proof the *device*
    works (see `gpu_decode_available`), which is why that end-to-end probe
    still gates the whole path and `decode_frames` keeps its CPU fallback.
    """
    name = CUVID_DECODERS.get(probe_codec(video_path))
    return name if name and name in _compiled_decoders() else None


@functools.lru_cache(maxsize=1)
def gpu_decode_available() -> bool:
    """True if ffmpeg can actually complete the exact GPU decode we intend to run.

    Probed end-to-end -- encode a throwaway h264 clip, then decode it through
    `h264_cuvid -resize` and require real JPEG output -- rather than by parsing
    `-decoders` or poking `-init_hw_device`.  Both of those report success on
    this box even with no usable device (`-init_hw_device cuda` exits 0 and
    prints nothing), so a lighter check would send every worker down a path
    that only fails once it reaches the first real video.
    """
    if os.environ.get("ORIGAMI_GPU", "1") == "0":
        return False
    tmp = None
    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        made = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y",
             "-f", "lavfi", "-i", "testsrc=size=320x240:rate=30:duration=1",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", tmp],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        if made.returncode != 0:
            return False
        out = subprocess.run(
            build_cmd(tmp, "not(mod(n\\,10))", scale=224, quality=3,
                      decoder="h264_cuvid"),
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=120)
        return out.returncode == 0 and out.stdout.startswith(b"\xff\xd8\xff")
    except (OSError, subprocess.SubprocessError, ImportError):
        return False
    finally:
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


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
        blob = _run(build_cmd(video_path, select, scale=scale,
                              quality=quality), video_path)
        return prep._split_jpegs(blob, expected, context)

    prep.decode_frames = decode_frames
    return use_gpu
