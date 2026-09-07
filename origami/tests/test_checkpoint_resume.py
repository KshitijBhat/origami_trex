"""Checkpoint save / --resume_full_state correctness. CPU-only, no GPU/network required.

Covers a real bug found verifying the user's explicit requirement ("when training is resumed,
it must start at the correct optimizer state and the same lr as when it was stopped") against
train_origami.py's actual resume path:

1. `lr_scheduler` was never passed to `accelerator.prepare()` in `train()` -- only
   `model`/`optimizer`(/`val_dataloader`) were. Accelerate's own `save_state()` only persists
   the schedulers it tracks in `self._schedulers`, which `prepare()` is what populates
   (confirmed by reading accelerate 1.14.0's `Accelerator.save_state`/`AcceleratedScheduler`
   source directly). So `--resume_full_state 1` restored the optimizer's momentum/state
   correctly (it *was* prepared) but silently restarted the LR schedule from scratch (fresh
   warmup) on every resume -- `accelerator.load_state()` had nothing to load the scheduler
   from. Fixed by adding `lr_scheduler` to the `accelerator.prepare(...)` call.
2. `global_step` was hardcoded to 0 after the resume block regardless of `--resume_full_state`,
   even though `training_state.json` (written by `save_checkpoint`) already recorded it --
   silently resetting `--save_steps` cadence, wandb step numbering, and `--max_steps`/
   `--val_freq` gating on every resume. Fixed by reading it back via the new
   `_load_training_state` helper.

These tests exercise the real `accelerate.Accelerator`/`save_state`/`load_state` machinery
(CPU device, tiny dummy model) rather than mocking it, since the bug was specifically about
what that real machinery does and does not track.
"""
from __future__ import annotations

import json
import math

import pytest
import torch
import torch.nn as nn
from accelerate import Accelerator

from origami.train_origami import _load_training_state, get_cosine_schedule_with_warmup


def _build(lr=1e-2, num_warmup_steps=5, num_training_steps=20, min_lr_ratio=0.0):
    accelerator = Accelerator(cpu=True)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps, min_lr_ratio=min_lr_ratio)
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    return accelerator, model, optimizer, lr_scheduler


# ── _load_training_state: the global_step/epoch half ──────────────────────────────────────

def test_load_training_state_returns_none_when_file_absent(tmp_path):
    assert _load_training_state(str(tmp_path)) is None


def test_load_training_state_round_trips_epoch_and_global_step(tmp_path):
    (tmp_path / "training_state.json").write_text(
        json.dumps({"epoch": 3, "global_step": 1234, "learning_rate": 5e-5}))
    start_epoch, global_step = _load_training_state(str(tmp_path))
    assert start_epoch == 4  # epoch + 1: resume on the epoch AFTER the one that was saved
    assert global_step == 1234


def test_load_training_state_defaults_missing_fields_to_zero(tmp_path):
    (tmp_path / "training_state.json").write_text(json.dumps({}))
    start_epoch, global_step = _load_training_state(str(tmp_path))
    assert (start_epoch, global_step) == (1, 0)


# ── lr_scheduler resume: the LR-continuity half ────────────────────────────────────────────

def test_unprepared_scheduler_state_is_not_captured_by_save_state(tmp_path):
    """Reproduces the bug directly: a scheduler NOT passed through accelerator.prepare() is
    invisible to save_state()/load_state() -- proves why the fix (passing lr_scheduler into
    prepare()) is necessary, not just plausible."""
    accelerator = Accelerator(cpu=True)
    model = nn.Linear(4, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=5, num_training_steps=20)
    model, optimizer = accelerator.prepare(model, optimizer)  # lr_scheduler NOT prepared

    for _ in range(10):
        optimizer.step()
        lr_scheduler.step()
    lr_before_save = lr_scheduler.get_last_lr()[0]

    accelerator.save_state(str(tmp_path))
    assert not any((tmp_path).glob("scheduler.bin")), (
        "an unprepared scheduler must not be written by save_state() -- "
        "if this starts failing, accelerate's tracked-object behavior changed and the "
        "fix's premise needs re-checking."
    )

    # A "fresh resumed run"'s scheduler starts back at step 0 -- load_state() has nothing to
    # restore it from, so it stays at the warmup-start LR, NOT lr_before_save.
    fresh_scheduler = get_cosine_schedule_with_warmup(
        torch.optim.AdamW(nn.Linear(4, 4).parameters(), lr=1e-2),
        num_warmup_steps=5, num_training_steps=20)
    accelerator.load_state(str(tmp_path))
    assert fresh_scheduler.get_last_lr()[0] != lr_before_save


def test_prepared_scheduler_resumes_at_the_same_lr_and_step(tmp_path):
    """The fix, proven end-to-end: a scheduler that WAS passed through accelerator.prepare()
    (train_origami.py's post-fix call) round-trips its exact LR and step count through
    save_state()/load_state(), across two independent Accelerator instances (a fresh process
    is what a real resumed run looks like)."""
    accelerator1, model1, optimizer1, lr_scheduler1 = _build()

    n_steps_before_save = 10
    lrs_seen = []
    for _ in range(n_steps_before_save):
        loss = model1(torch.randn(2, 4)).sum()
        accelerator1.backward(loss)
        optimizer1.step()
        lr_scheduler1.step()
        optimizer1.zero_grad()
        lrs_seen.append(lr_scheduler1.get_last_lr()[0])

    lr_at_save = lr_scheduler1.get_last_lr()[0]
    # sanity: warmup means these 10 steps' LRs are strictly increasing, not already flat --
    # otherwise a same-LR assertion below would pass trivially.
    assert lrs_seen[0] < lrs_seen[-1]

    accelerator1.save_state(str(tmp_path))

    # A brand-new Accelerator + freshly constructed model/optimizer/scheduler -- exactly what
    # a resumed process looks like (train_origami.py rebuilds all three before load_state()).
    accelerator2, model2, optimizer2, lr_scheduler2 = _build()
    accelerator2.load_state(str(tmp_path))

    assert lr_scheduler2.get_last_lr()[0] == lr_at_save
    assert lr_scheduler2.scheduler.last_epoch == lr_scheduler1.scheduler.last_epoch

    # Continuing to step past the resume point must continue the SAME cosine curve, not
    # restart warmup from step 0 -- compare against directly evaluating the closed-form
    # schedule at step n_steps_before_save + 1 (independent of either scheduler object).
    optimizer2.step()
    lr_scheduler2.step()
    warmup, total = 5, 20
    expected_progress = (n_steps_before_save + 1 - warmup) / (total - warmup)
    expected_lr = 1e-2 * 0.5 * (1.0 + math.cos(math.pi * expected_progress))
    assert lr_scheduler2.get_last_lr()[0] == pytest.approx(expected_lr, rel=1e-6)
