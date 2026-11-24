"""DirectShow focus range and manual control through the bundled native helper."""

from __future__ import annotations

import json
import queue
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

HELPER = Path(__file__).resolve().parent / "native/build/Release/focus_control.exe"
AUTOFOCUS_METHODS = ("autofocus", "fast-autofocus")
FOCUS_SEARCH_METHODS = ("full", "range", *AUTOFOCUS_METHODS)


@dataclass(frozen=True)
class FocusRange:
    minimum: int
    maximum: int
    step: int

    def __post_init__(self):
        if self.step <= 0 or self.maximum < self.minimum:
            raise ValueError("Camera reports an invalid focus range or step.")

    @property
    def count(self) -> int:
        return (self.maximum - self.minimum) // self.step + 1

    def value(self, index: int) -> int:
        return self.minimum + max(0, min(self.count - 1, index)) * self.step


class NativeFocusControl:
    def __init__(self, index: int):
        if not HELPER.is_file():
            raise RuntimeError(f"Missing {HELPER}. Build focus_test/native first (see README).")
        self.process = subprocess.Popen([str(HELPER), str(index)], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                        text=True, encoding="utf-8",
                                        creationflags=subprocess.CREATE_NO_WINDOW)
        self.responses: queue.Queue = queue.Queue()

        def receive():
            try:
                for line in self.process.stdout:
                    self.responses.put(line)
            finally:
                self.responses.put(None)

        self.receiver = threading.Thread(target=receive, daemon=True)
        self.receiver.start()
        try:
            self.initial = self._response()
            self.focus_range = FocusRange(self.initial["minimum"], self.initial["maximum"], self.initial["step"])
        except BaseException:
            self.close()
            raise

    def _response(self) -> dict:
        try:
            line = self.responses.get(timeout=8)
        except queue.Empty as exc:
            raise RuntimeError("Focus control helper timed out.") from exc
        if line is None:
            raise RuntimeError("Focus control helper exited without a response.")
        reply = json.loads(line)
        if not reply.get("ok"):
            raise RuntimeError(reply.get("error", "Focus control failed"))
        return reply

    def _request(self, command: str) -> dict:
        try:
            self.process.stdin.write(command + "\n")
            self.process.stdin.flush()
        except (OSError, ValueError) as exc:
            raise RuntimeError("Focus control helper is no longer available.") from exc
        return self._response()

    def get(self) -> dict:
        return self._request("GET")

    def set_manual(self, value: int) -> dict:
        result = self._request(f"SET {value} 2")
        if result["value"] != value or result["flags"] != 2:
            raise RuntimeError(f"Focus readback mismatch: requested {value}, "
                               f"received {result['value']}, flags={result['flags']}")
        return result

    def set_auto(self, value: int) -> dict:
        result = self._request(f"SET {value} 1")
        if not result["flags"] & 1:
            raise RuntimeError("Camera did not enter autofocus mode.")
        return result

    def restore(self, state: dict) -> dict:
        if state["flags"] & 1:
            return self.set_auto(state["value"])
        return self.set_manual(state["value"])

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self.process.stdin.write("QUIT\n")
                self.process.stdin.flush()
                self.process.wait(timeout=2)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                self.process.terminate()
                self.process.wait(timeout=2)
        self.process.stdin.close()
        self.receiver.join(timeout=1)
        self.process.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
