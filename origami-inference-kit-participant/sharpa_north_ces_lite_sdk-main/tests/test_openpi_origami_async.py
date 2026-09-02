from __future__ import annotations

import pathlib
import sys
import threading
import time
import unittest

import numpy as np


EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from openpi_origami_async import (  # noqa: E402
    AsyncTimeAggregationInferencer,
    TemporalEnsembler,
)


ACTION_KEY = "/action/all/joint_position"


class TemporalEnsemblerTest(unittest.TestCase):
    def test_overlapping_chunks_are_fused(self) -> None:
        scheduler = TemporalEnsembler(agg_n=2, exp_k=0.0, hold_last=False)
        scheduler.set_current_step(0)
        scheduler.push_chunk(
            {ACTION_KEY: np.asarray([[0.0], [2.0]], dtype=np.float32)}
        )
        scheduler.push_chunk(
            {ACTION_KEY: np.asarray([[2.0], [4.0]], dtype=np.float32)}
        )

        step = scheduler.pop_step(0)

        self.assertIsNotNone(step)
        np.testing.assert_allclose(step[ACTION_KEY], np.asarray([1.0], np.float32))

    def test_delay_compensation_aligns_chunk_to_current_step(self) -> None:
        scheduler = TemporalEnsembler(hold_last=False)
        scheduler.set_current_step(3)
        scheduler.push_chunk(
            {ACTION_KEY: np.arange(5, dtype=np.float32)[:, None]},
            offset_steps=2,
        )

        step = scheduler.pop_step(3)

        self.assertIsNotNone(step)
        np.testing.assert_array_equal(step[ACTION_KEY], np.asarray([2.0], np.float32))

    def test_hold_last_repeats_last_available_action(self) -> None:
        scheduler = TemporalEnsembler(hold_last=True)
        scheduler.set_current_step(0)
        scheduler.push_chunk({ACTION_KEY: np.asarray([[7.0]], dtype=np.float32)})
        first = scheduler.pop_step(0)
        scheduler.set_current_step(1)
        repeated = scheduler.pop_step(1)

        self.assertIsNotNone(first)
        self.assertIsNotNone(repeated)
        np.testing.assert_array_equal(repeated[ACTION_KEY], first[ACTION_KEY])


class FakeEnvironment:
    def __init__(self) -> None:
        self.published: list[dict[str, np.ndarray]] = []
        self._observation_index = 0
        self._lock = threading.Lock()

    def wait_for_observation(self, *, timeout: float | None, after_ts: int | None):
        del timeout, after_ts
        with self._lock:
            self._observation_index += 1
            index = self._observation_index
        time.sleep(0.002)
        return {"into_buffer_ts": index}

    def publish_single_action(self, action: dict[str, np.ndarray]) -> None:
        self.published.append(action)

    def clear_action_and_history(self) -> None:
        return None


class AsyncInferencerTest(unittest.TestCase):
    def test_inference_and_publish_loops_are_decoupled(self) -> None:
        environment = FakeEnvironment()

        def policy(_observation):
            time.sleep(0.01)
            return {
                ACTION_KEY: np.repeat(
                    np.asarray([[1.0]], dtype=np.float32), 8, axis=0
                )
            }

        runner = AsyncTimeAggregationInferencer(
            environment,
            policy,
            inference_hz=100.0,
            control_hz=200.0,
            compensation_steps=0,
            send_actions=True,
            require_new_obs=True,
            ta_hold_last=True,
            log_interval_s=0.0,
        )

        result = runner.run(max_steps=3)

        self.assertEqual(result["mode"], "async")
        self.assertEqual(result["steps"], 3)
        self.assertGreater(len(environment.published), 0)
        self.assertTrue(
            all(ACTION_KEY in published for published in environment.published)
        )


if __name__ == "__main__":
    unittest.main()
