"""Bounded MessagePack/NumPy codec; never uses pickle or object arrays."""

from __future__ import annotations

import math
from typing import Any

import msgpack
import numpy as np

MAX_PAYLOAD_BYTES = 64 * 1024 * 1024


def pack_payload(value: Any) -> bytes:
    payload = msgpack.packb(value, default=_pack_numpy, use_bin_type=True)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("encoded payload exceeds 64 MiB")
    return payload


def unpack_payload(value: bytes | bytearray | memoryview) -> Any:
    payload = bytes(value)
    if len(payload) > MAX_PAYLOAD_BYTES:
        raise ValueError("reply payload exceeds 64 MiB")
    try:
        return msgpack.unpackb(
            payload,
            object_hook=_unpack_numpy,
            raw=False,
            strict_map_key=False,
            max_bin_len=MAX_PAYLOAD_BYTES,
            max_array_len=1_000_000,
            max_map_len=10_000,
            max_str_len=1_000_000,
        )
    except (msgpack.UnpackException, ValueError, TypeError) as error:
        raise ValueError(f"invalid MessagePack payload: {error}") from error


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"O", "V", "c"} or value.dtype.hasobject:
            raise ValueError(f"unsafe NumPy dtype: {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": array.tobytes(),
            b"dtype": array.dtype.str,
            b"shape": array.shape,
        }
    if isinstance(value, np.generic):
        if value.dtype.kind in {"O", "V", "c"}:
            raise ValueError(f"unsafe NumPy dtype: {value.dtype}")
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    raise TypeError(f"cannot MessagePack-encode {type(value).__name__}")


def _field(value: dict[Any, Any], name: str) -> Any:
    return value[name] if name in value else value.get(name.encode("ascii"))


def _parse_dtype(value: Any) -> np.dtype:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise ValueError("invalid NumPy dtype descriptor") from error
    if not isinstance(value, str):
        raise ValueError("NumPy dtype descriptor must be a string")
    try:
        dtype = np.dtype(value)
    except (TypeError, ValueError) as error:
        raise ValueError("invalid NumPy dtype descriptor") from error
    if dtype.kind in {"O", "V", "c"} or dtype.hasobject:
        raise ValueError(f"unsafe NumPy dtype: {dtype}")
    return dtype


def _unpack_numpy(value: dict[Any, Any]) -> Any:
    if _field(value, "__ndarray__") is True:
        data = _field(value, "data")
        shape = _field(value, "shape")
        dtype = _parse_dtype(_field(value, "dtype"))
        if not isinstance(data, bytes):
            raise ValueError("NumPy array data must be MessagePack binary")
        if not isinstance(shape, (list, tuple)) or len(shape) > 8:
            raise ValueError("NumPy array shape must contain at most 8 dimensions")
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise ValueError("invalid NumPy array shape")
        expected = math.prod(shape) * dtype.itemsize
        if expected > MAX_PAYLOAD_BYTES or len(data) != expected:
            raise ValueError("NumPy array payload size mismatch")
        return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))
    if _field(value, "__npgeneric__") is True:
        dtype = _parse_dtype(_field(value, "dtype"))
        try:
            return dtype.type(_field(value, "data"))
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("invalid NumPy scalar") from error
    return value
