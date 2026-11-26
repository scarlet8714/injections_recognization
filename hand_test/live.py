"""以現有 USB 相機即時偵測手部，並可記錄原始影格供回放。"""

import argparse
import math
import sys
import time
from collections import Counter
from contextlib import ExitStack
from fractions import Fraction
from pathlib import Path
from typing import TextIO

import av
import cv2
import numpy as np

from focus_test.main import UsbFrameReader
from hand_test.detector import Detector, add_detector_arguments, open_detector
from hand_test.replay import (WINDOW, Point, VideoFrame, inspect_frame,
                              parse_point, render_preview, write_record)


class VideoRecording:
    def __init__(self, output: av.container.OutputContainer):
        self.output = output
        self.stream = None
        self.timestamp_ms = -1

    def write(self, image: np.ndarray, timestamp_ms: float) -> None:
        if self.stream is None:
            height, width = image.shape[:2]
            self.stream = self.output.add_stream("libx264", rate=30,
                                                 options={"preset": "ultrafast"})
            self.stream.width = width
            self.stream.height = height
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.time_base = Fraction(1, 1000)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        self.timestamp_ms = max(self.timestamp_ms + 1, round(timestamp_ms))
        frame.pts = self.timestamp_ms
        frame.time_base = Fraction(1, 1000)
        self.output.mux(self.stream.encode(frame))

    def finish(self) -> None:
        if self.stream is not None:
            self.output.mux(self.stream.encode())


def observe(reader: UsbFrameReader, detector: Detector, log: TextIO,
            points: list[Point], seconds: float | None, preview: bool,
            recording: VideoRecording | None = None) -> int:
    counts = Counter()
    started = time.perf_counter()
    latest_frame = started
    first_frame = None
    window_open = False
    status = "completed"
    try:
        while seconds is None or time.perf_counter() - started < seconds:
            try:
                image = reader.read()
                error = "未取得相機影格。" if image is None else None
            except cv2.error as exc:
                image, error = None, str(exc)
            now = time.perf_counter()
            if image is not None:
                if first_frame is None:
                    first_frame = now
                latest_frame = now
            timestamp_ms = (now - first_frame) * 1000 if first_frame is not None else None
            frame = VideoFrame(sum(counts.values()) + 1, timestamp_ms, image, error=error)
            record = inspect_frame(frame, detector)
            write_record(log, record)
            counts[record["status"]] += 1

            if image is not None:
                if recording is not None:
                    recording.write(image, timestamp_ms)
                if preview:
                    if not window_open:
                        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                        window_open = True
                    label = f"{timestamp_ms / 1000:.3f} s | {record['status']} | Q/Esc"
                    cv2.imshow(WINDOW, render_preview(image, points, record["hands"], label,
                                                     record["hand_regions"]))
            if window_open:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    status = "stopped"
                    break
            if image is None:
                if now - latest_frame >= 5:
                    status = "read_error"
                    break
                time.sleep(0.01)
    except KeyboardInterrupt:
        status = "stopped"
    finally:
        if window_open:
            cv2.destroyAllWindows()
        if recording is not None:
            recording.finish()

    received = sum(counts.values()) - counts["read_error"]
    elapsed = latest_frame - first_frame if first_frame is not None else 0
    processed_fps = round((received - 1) / elapsed, 3) if elapsed > 0 else None
    if received == 0:
        status = "read_error"
    elif status == "completed" and (counts["read_error"] or counts["processing_error"]):
        status = "completed_with_errors"
    write_record(log, {"type": "summary", "status": status,
                       "counts": dict(counts), "processed_fps": processed_fps})
    return 1 if status == "read_error" or counts["processing_error"] else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="從 USB 相機即時偵測手部。")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    add_detector_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="新建的 JSONL 紀錄路徑")
    parser.add_argument("--record", type=Path, help="同時保存無標示的原始 MP4 錄影")
    parser.add_argument("--seconds", type=float, help="指定經過幾秒後結束")
    parser.add_argument("--point", type=parse_point, action="append", default=[], metavar="X,Y")
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args(argv)
    if args.camera_index < 0 or min(args.width, args.height) < 1:
        parser.error("相機索引須為非負整數，影像尺寸須為正整數。")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds 必須為有限的正數。")
    if args.record is not None and args.record.suffix.lower() != ".mp4":
        parser.error("--record 請指定 .mp4 檔案。")

    try:
        with ExitStack() as resources:
            detector, settings = open_detector(args, parser)
            resources.callback(detector.close)
            reader = UsbFrameReader(args.camera_index, args.width, args.height)
            resources.callback(reader.close)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            log = resources.enter_context(args.output.open("x", encoding="utf-8"))
            recording = None
            if args.record is not None:
                args.record.parent.mkdir(parents=True, exist_ok=True)
                target = resources.enter_context(args.record.open("xb"))
                recording = VideoRecording(resources.enter_context(av.open(target, "w", format="mp4")))
            write_record(log, {
                "type": "run",
                "camera_index": args.camera_index,
                "requested_size": [args.width, args.height],
                "opencv_version": cv2.__version__,
                "reference_points": args.point,
                "recording": str(args.record.resolve()) if args.record else None,
                **settings,
            })
            print("實驗開始。", flush=True)
            code = observe(reader, detector, log, args.point, args.seconds, not args.no_preview, recording)
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error, av.FFmpegError) as exc:
        print(f"手部相機實驗失敗。{exc}", file=sys.stderr)
        return 1
    print(f"逐影格紀錄已保存於 {args.output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
