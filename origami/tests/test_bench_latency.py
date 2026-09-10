"""Tests for origami/bench_latency.py::feasible_action_horizon. REDESIGN_PLAN.md §8.3, G13."""
from __future__ import annotations

from origami.bench_latency import feasible_action_horizon


def test_feasible_action_horizon_fast_enough_for_T4():
    assert feasible_action_horizon(0.1, command_hz=30) == 4  # 100ms < 133ms budget


def test_feasible_action_horizon_needs_T8():
    assert feasible_action_horizon(0.2, command_hz=30) == 8  # 200ms > 133ms, <= 267ms


def test_feasible_action_horizon_needs_T16():
    assert feasible_action_horizon(0.29, command_hz=30) == 16  # matches this project's real p99


def test_feasible_action_horizon_caps_at_16_even_when_impossible():
    assert feasible_action_horizon(10.0, command_hz=30) == 16
