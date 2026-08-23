import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)

_COMMAND_KEY = "__origami_command__"


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._request_count = 0
        logging.getLogger("websockets.server").setLevel(logging.WARNING)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())

                if isinstance(obs, dict) and _COMMAND_KEY in obs:
                    command = obs[_COMMAND_KEY]
                    if command != "reset":
                        raise ValueError(f"Unsupported policy command: {command!r}")
                    self._policy.reset()
                    await websocket.send(packer.pack({"ok": True, "command": "reset"}))
                    prev_total_time = None
                    continue

                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time
                self._request_count += 1

                if self._metadata.get("protocol_version") == "origami-v1":
                    actions = np.ascontiguousarray(action["actions"], dtype=np.float32)
                    expected_shape = (
                        self._metadata.get("action_horizon"),
                        self._metadata.get("action_dim"),
                    )
                    if actions.shape != expected_shape:
                        raise ValueError(f"Policy returned actions with shape {actions.shape}, expected {expected_shape}")
                    action["actions"] = actions
                    policy_timing = action.get("policy_timing", {})
                    model_ms = policy_timing.get("infer_ms") if isinstance(policy_timing, dict) else None
                    state = obs.get("observation/state") if isinstance(obs, dict) else None
                    image_shapes = (
                        {
                            key: tuple(np.asarray(value).shape)
                            for key, value in obs.items()
                            if key.startswith("observation/image/")
                        }
                        if isinstance(obs, dict)
                        else {}
                    )
                    logger.info(
                        "inference request=%d state_shape=%s image_shapes=%s "
                        "action_shape=%s dtype=%s finite=%s min=%.5f max=%.5f "
                        "model_ms=%s server_infer_ms=%.1f",
                        self._request_count,
                        tuple(np.asarray(state).shape) if state is not None else None,
                        image_shapes,
                        actions.shape,
                        actions.dtype,
                        bool(np.isfinite(actions).all()),
                        float(actions.min()),
                        float(actions.max()),
                        f"{float(model_ms):.1f}" if model_ms is not None else "n/a",
                        infer_time * 1000,
                    )

                action["server_timing"] = {
                    "infer_ms": infer_time * 1000,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
