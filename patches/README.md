# patches/

Local patches for third-party packages that `uv sync` will silently overwrite
(they aren't in `pyproject.toml`/the lockfile, so a fresh environment loses
them unless reapplied).

## lerobot-0.6.1-clamp-torchcodec-frame-index.patch

Fixes `IndexError: Invalid frame index=N for streamIndex=0; must be less
than N` crashing training when a video-frame query timestamp (built from the
dataset's nominal fps) drifts from a file's *measured* fps and rounds up to
one past the last decodable frame. Hit ~8.6% of `eef62_train` episodes
(TRAINING.md's pilot run) because their video files' real fps drifts
slightly from the nominal 30fps the delta_timestamps math assumes.

This is the exact fix from lerobot's own (unmerged, as of `lerobot==0.6.1`)
upstream PR: https://github.com/huggingface/lerobot/pull/3884 -- clamps the
torchcodec frame index into `[0, num_frames - 1]` instead of letting it
overrun, turning the hard crash into lerobot's own recoverable
`FrameTimestampError` (whose message says "we advise to ignore this item
during training" -- handled by `TRexLeRobotDataset.__getitem__` in
`T-Rex/qwen_vla/lerobot_dataset.py`, which catches it and skips to the next
episode).

Reapply after every `uv sync` (or any fresh environment / new machine):

```bash
cd .venv/lib/python3.12/site-packages
patch -p1 < ../../../../patches/lerobot-0.6.1-clamp-torchcodec-frame-index.patch
```
(adjust the relative path to `patches/` and the `python3.12` version if your
venv differs)

Check if it's still needed on a `lerobot` upgrade: try `patch -p1 --dry-run`
first -- if it fails to apply (context mismatch) or PR #3884 has since been
merged/released, drop the patch and re-verify with a short training run
instead.
