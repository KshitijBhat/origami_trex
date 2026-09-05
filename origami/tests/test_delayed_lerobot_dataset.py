"""§7.5-B delayed_lerobot_dataset.py -- CPU-only, no GPU/network required.

Builds a tiny real converted shard (reusing origami.convert.convert_season with
max_frames_per_episode truncation, exactly like test_convert.py/test_merge.py) and checks
the F6-delay half of gate G17 ("delay-curriculum parity") at the loader level: the delayed
F6 tensor at a given ``delay_k`` equals the corresponding slice of the extended F6 window
(``f6_window[:, W-1+k]``).

The other half of G17 ("training offsets == deployed offsets") needs ``serve_zenoh.py``,
which does not exist yet (§12 step 13) -- not covered here; see the module docstring of
``origami/delayed_lerobot_dataset.py``.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from origami.constants import INSTRUCTION, REPO_ROOT
from origami.convert import ConvertConfig, convert_season
from origami.delayed_lerobot_dataset import DelayedTRexLeRobotDataset
from origami.kinematics import LockedConfig, OrigamiKinematics
from qwen_vla.lerobot_dataset import _normalize
from utils.lerobot_common import KEY_TACF6

FIXTURE_SEASON = REPO_ROOT / "season_POC22061_2026_05_23_19_21_25_train"
N_FRAMES = 80          # > MIN_EPISODE_LENGTH (64), > W + max_delay with margin
VQVAE_WINDOW = 4
DELAY_OFFSETS = (0, 2, 4)


class _FakeOut(dict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError as e:
            raise AttributeError(item) from e


class _FakeProcessor:
    """Just enough surface for TRexLeRobotDataset.collate_fn's image-processing calls --
    this test exercises the tactile-delay loader logic, not real image processing."""
    class tokenizer:
        pad_token_id = 0

    def __init__(self):
        self.image_processor = self._image_processor

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text, images, return_tensors="pt", padding=False):
        return _FakeOut(input_ids=torch.zeros((1, 3), dtype=torch.long), pixel_values=None)

    @staticmethod
    def _image_processor(imgs, return_tensors="pt"):
        n = len(imgs)
        return _FakeOut(pixel_values=torch.zeros((n, 3, 4, 4)),
                         image_grid_thw=torch.zeros((n, 3), dtype=torch.long))


class _FakeAccelerator:
    def print(self, *args, **kwargs):
        pass


@pytest.fixture(scope="module")
def converted_root(tmp_path_factory):
    pytest.importorskip("pyarrow")
    out_root = tmp_path_factory.mktemp("delayed_lerobot_smoke")
    kin = OrigamiKinematics(locked=LockedConfig.zeros())
    cfg = ConvertConfig(instruction=INSTRUCTION)
    convert_season(FIXTURE_SEASON, out_root, kin, cfg, max_episodes=1, max_frames_per_episode=N_FRAMES)
    return out_root / FIXTURE_SEASON.name


def _make_dataset(converted_root, scope: str) -> DelayedTRexLeRobotDataset:
    config = SimpleNamespace(
        lerobot_root=str(converted_root),
        lerobot_repo_id="",
        image_size=(224, 224),
        # use_flare=1/n_flare_steps=1 sidesteps a pre-existing upstream quirk (§11.4):
        # a SINGLE-offset delta_timestamps entry is squeezed (no leading temporal dim), so
        # KEY_HEAD needs >=2 offsets for TRexLeRobotDataset.collate_fn's `head_seq[0]`
        # frame-indexing to be valid -- true of upstream's own real recipe too
        # (--use_flare 1 --n_flare_steps 8 always), not something this subclass changes.
        use_flare=1, n_flare_steps=1, flare_frame_stride=1,
        use_tactile_vec=1, use_tactile_deform=0, use_tactile_vqvae=1,
        use_robot_state=0,
        vqvae_window=VQVAE_WINDOW,
        action_dim=62,
        tactile_delay_offsets=list(DELAY_OFFSETS),
        tactile_delay_scope=scope,
    )
    return DelayedTRexLeRobotDataset(config, _FakeProcessor(), _FakeAccelerator())


def test_f6_offsets_extend_past_now_by_max_delay(converted_root):
    ds = _make_dataset(converted_root, scope="f6")
    offsets = ds._f6_offsets()
    assert len(offsets) == VQVAE_WINDOW + max(DELAY_OFFSETS)
    # offset[W-1] must be exactly "now" (0.0); offsets are evenly spaced at 1/fps.
    assert offsets[VQVAE_WINDOW - 1] == pytest.approx(0.0)
    assert offsets[-1] == pytest.approx(max(DELAY_OFFSETS) / ds.fps)


def test_scope_none_matches_upstream_offsets(converted_root):
    ds = _make_dataset(converted_root, scope="none")
    assert len(ds._f6_offsets()) == VQVAE_WINDOW
    assert ds._f6_offsets()[-1] == pytest.approx(0.0)


def test_delayed_f6_tensor_equals_window_slice_for_every_configured_delay(converted_root):
    """G17 (loader half): tactile_f6s_delayed / tactile_f6_history at a forced delay_k must
    equal f6_window[:, W-1+k] / f6_window[:, k:k+W] exactly, for every configured delay."""
    ds = _make_dataset(converted_root, scope="f6")
    W = VQVAE_WINDOW
    idx = W - 1 + max(DELAY_OFFSETS) + 2  # comfortably clear of the episode's start boundary

    item = ds[idx]
    window = item[KEY_TACF6].float()  # [W + max_delay, 10, 6]
    assert window.shape[0] == W + max(DELAY_OFFSETS)

    for k in DELAY_OFFSETS:
        expected_delayed = window[W - 1 + k]          # [10, 6]
        expected_history = window[k: k + W]           # [W, 10, 6]

        batch = [item]
        with pytest.MonkeyPatch.context() as mp:
            # np.random.default_rng() inside collate_fn is a fresh generator each call, so
            # patch the module-level factory to force a deterministic choice of k.
            mp.setattr(np.random, "default_rng", lambda: SimpleNamespace(choice=lambda seq: k))
            out = ds.collate_fn(batch)

        assert out["delay_k"].item() == k
        got_delayed_norm = out["tactile_f6s_delayed"][0].reshape(-1).float()
        expected_delayed_norm = torch.tensor(
            _normalize(expected_delayed.numpy().reshape(-1), ds.tacf6_mask, ds.tacf6_min, ds.tacf6_max),
            dtype=torch.float32,
        )
        # bfloat16 has ~3 significant decimal digits -- tactile_f6s_delayed is stored at
        # that precision (matches upstream TRexLeRobotDataset.collate_fn's norm_tacf6).
        torch.testing.assert_close(got_delayed_norm.float(), expected_delayed_norm, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(out["tactile_f6_history"][0].float(), expected_history)


def test_scope_none_leaves_delay_at_zero(converted_root):
    ds = _make_dataset(converted_root, scope="none")
    item = ds[10]
    out = ds.collate_fn([item])
    # base TRexLeRobotDataset behavior: tactile_f6s_delayed == tactile_f6s (delay == 0)
    torch.testing.assert_close(out["tactile_f6s_delayed"], out["tactile_f6s"])
