"""Tests for origami/eval_shadow.py. REDESIGN_PLAN.md §8.2, §12 step 14. Gates G10, G11.

The full integration test (real checkpoint, real subprocess ``serve_zenoh.py``, real Zenoh
wire protocol via an in-process router) is ``skipif``-gated on a real checkpoint + prep root
being present locally -- it is slow (~30s checkpoint load) but exercises the actual submission
wire path end-to-end, matching what was manually verified when this module was built (see
``origami/PROGRESS.md``).

G11 (shadow replay against a real recorded season) additionally needs a local
``season_*/lerobot3.0`` directory, which this environment does not have (the fixture season is
gitignored, project data, never committed) -- that half is exercised structurally via
``run_protocol_conformance`` only here; see PROGRESS.md for the documented gap.
"""
from __future__ import annotations

import uuid

import pytest

from origami.constants import REPO_ROOT
from origami.eval_shadow import LocalZenohRouter, _free_tcp_port

CHECKPOINT_DIR = REPO_ROOT / "checkpoints" / "sept9_ckpt"
LOCKED_CONFIG_SOURCE = REPO_ROOT / "data" / "meta" / "origami_prep.json"


def test_free_tcp_port_returns_a_usable_port():
    port = _free_tcp_port()
    assert 1 <= port <= 65535


def test_local_zenoh_router_open_close():
    router = LocalZenohRouter()
    try:
        assert router.endpoint == f"tcp/127.0.0.1:{router.port}"
    finally:
        router.close()


@pytest.mark.skipif(
    not (CHECKPOINT_DIR.is_dir() and LOCKED_CONFIG_SOURCE.is_file()),
    reason="needs the real checkpoint + a real prep root locally",
)
def test_g10_protocol_conformance_against_real_running_server():
    from origami.eval_shadow import run_protocol_conformance, start_policy_server

    router = LocalZenohRouter()
    session_id = uuid.uuid4().hex
    proc = None
    try:
        proc = start_policy_server(
            str(CHECKPOINT_DIR), router.endpoint, session_id, action_horizon=4,
            locked_config_source=str(LOCKED_CONFIG_SOURCE), urdf_path=None, cuda="0",
            extra_args=[], startup_timeout=600.0,
        )
        result = run_protocol_conformance(router.endpoint, session_id, timeout=30.0, requests=3)
        assert result["pass"], result
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
        router.close()
