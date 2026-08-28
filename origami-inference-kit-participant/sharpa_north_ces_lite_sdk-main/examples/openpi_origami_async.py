#!/usr/bin/env python3
"""Self-contained async (temporal-aggregation) inference runtime for the kit.

This mirrors the rtac1-style asynchronous control loop, but is packaged fully
inside ``examples/`` and depends only on a small environment interface. It does
not import or include a robot transport implementation.

Two decoupled threads:

* inference thread (``inference_hz``): read the latest observation, call the policy
  to get an action chunk ``{"/action/<part>/<type>": (T, D)}``, then push it into a
  ``TemporalEnsembler`` with a delay-compensating ``offset_steps`` (instead of
  dropping leading steps). It does NOT publish directly.
* publish thread (``control_hz``): keep a monotonic ``global_step`` and, each tick,
  pop one *fused* step from the ensembler (per-step ensembling over recent
  overlapping chunks) and publish it.

Publishing uses ``env.publish_single_action(step)`` when available (clean per-step
publish that bypasses the buffer). If the supplied runtime does not have it, it
falls back to ``env.step({key: (1, D)})`` (a one-row chunk drained by the client's
background sender), so the loop still runs on older runtime packages.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("openpi_origami_async")

ActionChunk = Dict[str, np.ndarray]
ActionStep = Dict[str, np.ndarray]
PolicyFn = Callable[[dict], Optional[ActionChunk]]


@dataclass
class ChunkRecord:
    start_step: int
    actions: ActionChunk  # {key: (H, D)} float32
    created_ts: float

    @property
    def horizon(self) -> int:
        h = 0
        for v in self.actions.values():
            if v.ndim > 0:
                h = max(h, int(v.shape[0]))
        return h

    @property
    def end_step_exclusive(self) -> int:
        return int(self.start_step + self.horizon)


class TemporalEnsembler:
    """Thread-safe chunk store + per-step temporal aggregation (dict actions).

    For a given publish step, gather every chunk that covers it, keep the newest
    ``agg_n`` candidates, and exponentially weight them (newer = higher weight) so
    overlapping predictions from consecutive inferences fuse into a smoother, more
    robust single step. Optional cross-step EMA (``smooth_alpha``) and a
    ``hold_last`` fallback (repeat the last action when nothing covers the step)
    round out the behavior.
    """

    def __init__(
        self,
        *,
        max_chunks: int = 16,
        agg_n: int = 4,
        exp_k: float = 0.01,
        hold_last: bool = True,
        smooth_alpha: float = 0.0,
    ):
        self.max_chunks = int(max_chunks)
        self.agg_n = int(agg_n)
        self.exp_k = float(exp_k)
        self.hold_last = bool(hold_last)
        self.smooth_alpha = float(smooth_alpha)
        if not (0.0 <= self.smooth_alpha <= 1.0):
            raise ValueError(f"smooth_alpha must be in [0, 1], got {self.smooth_alpha}")

        self._lock = threading.Lock()
        self._chunks: List[ChunkRecord] = []
        self._current_step: int = 0
        self._last_sent: Optional[ActionStep] = None

    def set_current_step(self, step: int) -> None:
        with self._lock:
            self._current_step = int(step)
            self._chunks = [c for c in self._chunks if c.end_step_exclusive > self._current_step]

    def clear(self) -> None:
        with self._lock:
            self._chunks.clear()
            self._last_sent = None

    def push_chunk(self, actions: ActionChunk, *, offset_steps: int = 0) -> None:
        """Push an inference action chunk, aligned as start_step = current - offset."""
        created_ts = float(time.time())
        norm: ActionChunk = {}
        for key, values in actions.items():
            arr = np.asarray(values, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr[None, :]
            norm[key] = arr
        with self._lock:
            start_step = int(self._current_step - int(offset_steps))
            self._chunks.append(ChunkRecord(start_step=start_step, actions=norm, created_ts=created_ts))
            self._chunks.sort(key=lambda r: r.created_ts)
            if len(self._chunks) > self.max_chunks:
                self._chunks = self._chunks[-self.max_chunks :]

    def _gather_candidates(self, key: str, global_step: int) -> List[Tuple[float, np.ndarray]]:
        cands: List[Tuple[float, np.ndarray]] = []
        for c in self._chunks:
            arr = c.actions.get(key)
            if arr is None:
                continue
            idx = global_step - c.start_step
            if 0 <= idx < arr.shape[0]:
                cands.append((c.created_ts, arr[int(idx)]))
        cands.sort(key=lambda x: x[0])  # oldest -> newest
        return cands

    @staticmethod
    def _weights(n: int, exp_k: float) -> np.ndarray:
        w = np.exp(-exp_k * np.arange(n, dtype=np.float32))
        return (w / float(np.sum(w)))[::-1]  # newest gets the highest weight

    def _all_keys(self) -> List[str]:
        keys: List[str] = []
        seen = set()
        for c in self._chunks:
            for k in c.actions.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        return keys

    def pop_step(self, global_step: int) -> Optional[ActionStep]:
        with self._lock:
            self._chunks = [c for c in self._chunks if c.end_step_exclusive > global_step]
            if not self._chunks:
                if self.hold_last and self._last_sent is not None:
                    return {k: v.copy() for k, v in self._last_sent.items()}
                return None

            out: ActionStep = {}
            any_cand = False
            for key in self._all_keys():
                cands = self._gather_candidates(key, global_step)
                if self.agg_n > 0 and len(cands) > self.agg_n:
                    cands = cands[-self.agg_n :]
                if not cands:
                    if self.hold_last and self._last_sent is not None and key in self._last_sent:
                        out[key] = self._last_sent[key].copy()
                    continue
                any_cand = True
                w = self._weights(len(cands), self.exp_k)
                vecs = np.stack([v for (_ts, v) in cands], axis=0)  # (M, D)
                out_raw = np.sum(vecs * w[:, None], axis=0).astype(np.float32)
                a = self.smooth_alpha
                if a > 0.0 and self._last_sent is not None and key in self._last_sent:
                    out_raw = (a * out_raw + (1.0 - a) * self._last_sent[key]).astype(
                        np.float32, copy=False
                    )
                out[key] = out_raw

            if not any_cand:
                if self.hold_last and self._last_sent is not None:
                    return {k: v.copy() for k, v in self._last_sent.items()}
                return None

            self._last_sent = {k: v.copy() for k, v in out.items()}
            return out


class AsyncTimeAggregationInferencer:
    """Async control loop: inference thread pushes chunks, publish thread streams
    per-step fused actions at ``control_hz`` (rtac1-style temporal ensembling)."""

    def __init__(
        self,
        env,
        policy: PolicyFn,
        *,
        inference_hz: float = 10.0,
        control_hz: float = 30.0,
        compensation_steps="auto",
        extra_compensation_steps: int = 0,
        exec_steps: int = 0,
        mode_validator: Optional[Callable[[dict], bool]] = None,
        clear_on_invalid_mode: bool = True,
        send_actions: bool = True,
        require_new_obs: bool = True,
        ta_agg_n: int = 4,
        ta_exp_k: float = 0.01,
        ta_max_chunks: int = 16,
        ta_hold_last: bool = True,
        ta_smooth_alpha: float = 0.0,
        log_interval_s: float = 2.0,
        logger_: Optional[logging.Logger] = None,
    ):
        if isinstance(compensation_steps, str) and compensation_steps != "auto":
            raise ValueError("compensation_steps must be 'auto' or an int")
        self.env = env
        self.policy = policy
        self.inference_hz = float(inference_hz)
        self.inference_dt = 1.0 / inference_hz if inference_hz and inference_hz > 0 else 0.0
        self.control_hz = float(control_hz)
        self.pub_dt = 1.0 / self.control_hz if self.control_hz and self.control_hz > 0 else 0.0
        if self.pub_dt <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.compensation_steps = compensation_steps
        self.extra_compensation_steps = int(extra_compensation_steps)
        self.exec_steps = max(0, int(exec_steps or 0))
        self.mode_validator = mode_validator
        self.clear_on_invalid_mode = clear_on_invalid_mode
        self.send_actions = send_actions
        self.require_new_obs = require_new_obs
        self.log_interval_s = log_interval_s
        self.logger = logger_ or logger

        self._scheduler = TemporalEnsembler(
            max_chunks=ta_max_chunks,
            agg_n=ta_agg_n,
            exp_k=ta_exp_k,
            hold_last=ta_hold_last,
            smooth_alpha=ta_smooth_alpha,
        )
        self._publish_single = getattr(env, "publish_single_action", None)
        if not callable(self._publish_single):
            self.logger.warning(
                "env has no publish_single_action(); falling back to per-step env.step(). "
                "Prefer a runtime that provides publish_single_action for clean per-step publishing."
            )

        self._stop = threading.Event()
        self._publish_stop = threading.Event()
        self._publish_thread: Optional[threading.Thread] = None
        self._global_step = 0

        self._stats_lock = threading.Lock()
        self._steps = 0
        self._latency_sum = 0.0
        self._last_latency = 0.0
        self._last_offset = 0
        self._last_chunk_steps = 0
        self._last_log = 0.0

    def stop(self) -> None:
        self._stop.set()

    def compensation_for(self, latency_s: float) -> int:
        if self.compensation_steps == "auto":
            return max(0, int(round(latency_s * self.control_hz)) + self.extra_compensation_steps)
        return max(0, int(self.compensation_steps))

    @staticmethod
    def _action_keys_chunk(action_chunk: Dict[str, Any]) -> ActionChunk:
        return {
            k: np.asarray(v)
            for k, v in action_chunk.items()
            if str(k).startswith("/action/") and np.asarray(v).ndim > 0
        }

    def _limit_exec_steps(self, action_chunk: Dict[str, Any]) -> Dict[str, Any]:
        if self.exec_steps <= 0:
            return action_chunk
        return {
            k: (np.asarray(v)[: self.exec_steps] if np.asarray(v).ndim > 0 else v)
            for k, v in action_chunk.items()
        }

    def _publish_step(self, step: ActionStep) -> None:
        if not self.send_actions:
            return
        if callable(self._publish_single):
            self._publish_single(step)
        else:
            self.env.step({k: np.asarray(v)[None, :] for k, v in step.items()})

    def _start_publish_thread(self) -> None:
        self._publish_stop.clear()
        self._global_step = 0
        self._scheduler.set_current_step(0)

        def _loop() -> None:
            next_t = time.time()
            while not self._publish_stop.is_set() and not self._stop.is_set():
                now = time.time()
                if now < next_t:
                    self._publish_stop.wait(min(0.001, next_t - now))
                    continue
                next_t = now + self.pub_dt
                self._scheduler.set_current_step(self._global_step)
                step = self._scheduler.pop_step(self._global_step)
                if step is not None:
                    self._publish_step(step)
                self._global_step += 1

        self._publish_thread = threading.Thread(target=_loop, name="ta-publish", daemon=True)
        self._publish_thread.start()
        self.logger.info("time-aggregation publish thread started at %.3gHz", self.control_hz)

    def _record(self, latency_s: float, offset: int, chunk_steps: int) -> None:
        with self._stats_lock:
            self._steps += 1
            self._latency_sum += latency_s
            self._last_latency = latency_s
            self._last_offset = offset
            self._last_chunk_steps = chunk_steps

    def _maybe_log(self) -> None:
        if self.log_interval_s <= 0:
            return
        now = time.time()
        if now - self._last_log < self.log_interval_s:
            return
        self._last_log = now
        with self._stats_lock:
            steps = self._steps
            avg_lat = (self._latency_sum / steps) if steps else 0.0
            last_lat = self._last_latency
            last_offset = self._last_offset
            last_chunk_steps = self._last_chunk_steps
        self.logger.info(
            "[async+ta] steps=%d infer=%.3gHz control=%.3gHz last_latency=%.1fms avg_latency=%.1fms "
            "last_offset=%d last_chunk_steps=%d compensation=%s",
            steps, self.inference_hz, self.control_hz, last_lat * 1000.0, avg_lat * 1000.0,
            last_offset, last_chunk_steps, self.compensation_steps,
        )

    def run(self, duration: Optional[float] = None, max_steps: Optional[int] = None) -> dict:
        self._stop.clear()
        self._scheduler.clear()
        with self._stats_lock:
            self._steps = 0
            self._latency_sum = 0.0
        run_start = time.time()
        self._last_log = run_start
        last_obs_ts = None

        self._start_publish_thread()
        try:
            while not self._stop.is_set():
                if duration is not None and time.time() - run_start >= duration:
                    break
                with self._stats_lock:
                    steps_done = self._steps
                if max_steps is not None and steps_done >= max_steps:
                    break

                loop_start = time.time()
                obs = self.env.wait_for_observation(
                    timeout=self.inference_dt or None,
                    after_ts=last_obs_ts if self.require_new_obs else None,
                )
                if obs is None:
                    continue
                last_obs_ts = obs.get("into_buffer_ts", last_obs_ts)

                if self.mode_validator is not None and not self.mode_validator(obs):
                    if self.clear_on_invalid_mode:
                        self._scheduler.clear()
                    continue

                t0 = time.time()
                action_chunk = self.policy(obs)
                latency = time.time() - t0

                if action_chunk:
                    action_chunk = self._limit_exec_steps(action_chunk)
                    offset_steps = self.compensation_for(latency)
                    if "output_text" in action_chunk:
                        try:
                            self.env.output_text = action_chunk["output_text"]
                        except Exception:  # noqa: BLE001 - best-effort language passthrough
                            pass
                    sched_chunk = self._action_keys_chunk(action_chunk)
                    chunk_steps = 0
                    for v in sched_chunk.values():
                        chunk_steps = max(chunk_steps, int(v.shape[0]))
                    if sched_chunk:
                        self._scheduler.push_chunk(sched_chunk, offset_steps=offset_steps)
                    self._record(latency, offset_steps, chunk_steps)

                self._maybe_log()

                elapsed = time.time() - loop_start
                if self.inference_dt and elapsed < self.inference_dt:
                    self._stop.wait(self.inference_dt - elapsed)
        except KeyboardInterrupt:
            self.logger.info("interrupted; stopping async inference loop")
        finally:
            self._stop.set()
            self._publish_stop.set()
            if self._publish_thread is not None:
                self._publish_thread.join(timeout=2.0)
            self._scheduler.clear()
            try:
                self.env.clear_action_and_history()
            except Exception:  # noqa: BLE001 - cleanup is best-effort
                self.logger.debug("clear_action_and_history failed", exc_info=True)

        with self._stats_lock:
            steps = self._steps
            avg_lat = (self._latency_sum / steps) if steps else 0.0
        return {
            "mode": "async",
            "time_aggregation": True,
            "steps": steps,
            "avg_latency_ms": avg_lat * 1000.0,
            "seconds": time.time() - run_start,
        }
