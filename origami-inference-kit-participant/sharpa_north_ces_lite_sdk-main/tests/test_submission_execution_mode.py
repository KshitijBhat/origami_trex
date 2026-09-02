from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from unittest import mock

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "openpi-base-main" / "scripts"
sys.path.insert(0, str(SCRIPTS))


class FakePolicy:
    def reset(self) -> None:
        return None

    def infer(self, _observation):
        return {"actions": np.zeros((25, 65), dtype=np.float32)}


class SubmissionExecutionModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fake_base = mock.MagicMock()
        fake_base.Args = object
        fake_base.Checkpoint = object
        fake_config = mock.MagicMock()
        fake_tyro = types.SimpleNamespace(cli=mock.MagicMock())
        fake_openpi = types.ModuleType("openpi")
        fake_training = types.ModuleType("openpi.training")
        fake_training.config = fake_config
        fake_openpi.training = fake_training
        with mock.patch.dict(
            sys.modules,
            {
                "serve_policy": fake_base,
                "openpi": fake_openpi,
                "openpi.training": fake_training,
                "openpi.training.config": fake_config,
                "tyro": fake_tyro,
            },
        ):
            spec = importlib.util.spec_from_file_location(
                "serve_policy_zenoh_under_test", SCRIPTS / "serve_policy_zenoh.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
        cls.module = module

    def test_metadata_advertises_async_and_source_kit(self) -> None:
        server = self.module.OpenPIZenohServer(
            FakePolicy(),
            endpoint="tcp/127.0.0.1:7447",
            session_id="test-session",
            action_horizon=25,
            execution_mode="async",
        )

        self.assertEqual(server.metadata["execution_mode"], "async")
        self.assertEqual(
            server.metadata["inference_kit"], "origami-inference-kit-async"
        )

    def test_invalid_execution_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "execution_mode"):
            self.module.OpenPIZenohServer(
                FakePolicy(),
                endpoint="tcp/127.0.0.1:7447",
                session_id="test-session",
                action_horizon=25,
                execution_mode="streaming",
            )

    def test_dockerfile_build_arg_and_source_label(self) -> None:
        dockerfile = (
            SCRIPTS / "docker" / "submission-zenoh-bundled.Dockerfile"
        ).read_text()
        self.assertIn("ARG EXECUTION_MODE=async", dockerfile)
        self.assertIn("EXECUTION_MODE=${EXECUTION_MODE}", dockerfile)
        self.assertIn(
            'LABEL org.opencontainers.image.source-kit="origami-inference-kit-async"',
            dockerfile,
        )


if __name__ == "__main__":
    unittest.main()
