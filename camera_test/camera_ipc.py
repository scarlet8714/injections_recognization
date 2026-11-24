"""IPC client for the IC Imaging Control 3.5 C++ camera server."""

from __future__ import annotations

import mmap
import socket
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

HOST = "127.0.0.1"
PORT = 8765
SHARED_MEMORY_NAME = r"Local\MedicineCameraFrame"
SHARED_MEMORY_SIZE = 64 * 1024 * 1024
FRAME_OFFSET = 64
HEADER = struct.Struct("<6I")


class CameraServerError(RuntimeError):
    pass


def command(text: str, timeout: float = 5.0) -> list[str]:
    try:
        with socket.create_connection((HOST, PORT), timeout=timeout) as client:
            client.sendall(text.encode("utf-8") + b"\n")
            response = bytearray()
            while True:
                block = client.recv(65536)
                if not block:
                    break
                response.extend(block)
    except OSError as exc:
        raise CameraServerError(
            "Cannot connect to camera_server.exe on 127.0.0.1:8765"
        ) from exc
    lines = response.decode("utf-8", errors="replace").splitlines()
    if not lines or not lines[0].startswith("OK"):
        message = lines[0][4:] if lines and lines[0].startswith("ERR ") else "No response"
        raise CameraServerError(message)
    return [line for line in lines[1:] if line != "END"]


def server_ready() -> bool:
    try:
        command("STATUS", timeout=0.3)
        return True
    except CameraServerError:
        return False


@dataclass
class VideoMode:
    index: int
    width: int
    height: int
    name: str
    frame_rates: list[float]


def get_modes() -> list[VideoMode]:
    modes = []
    for line in command("MODES"):
        if not line.startswith("MODE="):
            continue
        index, width, height, name, rates = line[5:].split("|", 4)
        modes.append(
            VideoMode(
                int(index),
                int(width),
                int(height),
                name,
                [float(value) for value in rates.split(",") if value],
            )
        )
    return modes


def get_status() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in command("STATUS"):
        name, raw = line.split("=", 1)
        values = raw.split(",")
        if name in {"EXPOSURE", "GAIN", "FOCUS"}:
            result[name.lower()] = {
                "supported": bool(int(values[0])),
                "auto": bool(int(values[1])),
                "auto_enabled": bool(int(values[2])),
                "one_push": bool(int(values[3])),
                "minimum": float(values[4]),
                "maximum": float(values[5]),
                "value": float(values[6]),
                "unit": values[7] if len(values) > 7 else "",
            }
        elif name == "WB":
            result["white_balance"] = {
                "supported": bool(int(values[0])),
                "auto": bool(int(values[1])),
                "auto_enabled": bool(int(values[2])),
                "red": tuple(map(int, values[3:6])),
                "green": tuple(map(int, values[6:9])),
                "blue": tuple(map(int, values[9:12])),
            }
        elif name == "AF_ROI":
            result["af_roi"] = {
                "supported": bool(int(values[0])),
                "enable_supported": bool(int(values[1])),
                "enabled": bool(int(values[2])),
                "left": tuple(map(float, values[3:6])),
                "top": tuple(map(float, values[6:9])),
            }
    return result


class SharedFrameReader:
    def __init__(self) -> None:
        try:
            self.memory = mmap.mmap(
                -1,
                SHARED_MEMORY_SIZE,
                tagname=SHARED_MEMORY_NAME,
                access=mmap.ACCESS_READ,
            )
        except (OSError, TypeError) as exc:
            raise CameraServerError("Cannot open camera shared memory") from exc
        self.last_sequence = 0

    def close(self) -> None:
        self.memory.close()

    def read(self) -> np.ndarray | None:
        for _ in range(3):
            header = HEADER.unpack_from(self.memory, 0)
            sequence, width, height, stride, data_size, bottom_up = header
            if sequence == 0 or sequence & 1 or sequence == self.last_sequence:
                time.sleep(0.001)
                continue
            if not width or not height or stride < width * 3:
                return None
            raw = self.memory[FRAME_OFFSET : FRAME_OFFSET + data_size]
            sequence_after = HEADER.unpack_from(self.memory, 0)[0]
            if sequence == sequence_after and not sequence_after & 1:
                rows = np.frombuffer(raw, dtype=np.uint8).reshape(height, stride)
                frame = rows[:, : width * 3].reshape(height, width, 3).copy()
                if bottom_up:
                    frame = np.flipud(frame).copy()
                self.last_sequence = sequence
                return frame
        return None
