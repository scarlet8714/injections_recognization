"""回放影片，顯示手部位置並輸出逐影格紀錄。"""

import argparse
import json
import sys
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import av
import cv2
import numpy as np

from hand_test.detector import (Box, ColorHandDetector, Detector, Hands,
                                add_detector_arguments, open_detector)

Point = tuple[int, int]
WINDOW = "Hand detection"


@dataclass
class VideoFrame:
    index: int
    timestamp_ms: float | None
    image: np.ndarray | None = None
    duration_ms: float = 0
    error: str | None = None


def read_video(path: Path) -> Iterator[VideoFrame]:
    index = 0
    origin = None
    try:
        with av.open(str(path)) as video:
            if not video.streams.video:
                yield VideoFrame(0, None, error="檔案沒有影片軌。")
                return
            for index, frame in enumerate(video.decode(video=0), start=1):
                if frame.time is None:
                    yield VideoFrame(index, None, error="影格缺少顯示時間。")
                    continue
                if origin is None:
                    origin = frame.time
                timestamp_ms = (frame.time - origin) * 1000
                if frame.is_corrupt:
                    yield VideoFrame(index, timestamp_ms, error="影片解碼器回報影格損壞。")
                    continue
                try:
                    image = frame.to_ndarray(format="bgr24")
                except av.FFmpegError as exc:
                    yield VideoFrame(index, timestamp_ms, error=str(exc))
                    continue
                duration_ms = float(frame.duration * frame.time_base) * 1000
                yield VideoFrame(index, timestamp_ms, image, duration_ms)
    except (av.FFmpegError, OSError) as exc:
        yield VideoFrame(index + 1, None, error=str(exc))
        return
    if index == 0:
        yield VideoFrame(0, None, error="影片沒有可解碼的影格。")


def inspect_frame(frame: VideoFrame, detector: Detector) -> dict:
    record = {
        "type": "frame",
        "frame_index": frame.index,
        "timestamp_ms": round(frame.timestamp_ms, 3) if frame.timestamp_ms is not None else None,
        "hands": [],
        "hand_regions": [],
        "processing_ms": None,
    }
    if frame.image is None or frame.timestamp_ms is None:
        return dict(record, status="read_error", error=frame.error or "影格或顯示時間不可用。")

    started = time.perf_counter()
    try:
        if isinstance(detector, ColorHandDetector):
            box, area = detector.detect(frame.image)
            record["hand_regions"] = [box] if box is not None else []
            record["region_area_px"] = area
        else:
            record["hands"] = detector.detect(frame.image, frame.timestamp_ms)
        record["status"] = "detected" if record["hands"] or record["hand_regions"] else "no_hand"
    except (RuntimeError, ValueError, cv2.error) as exc:
        record.update(status="processing_error", error=str(exc))
    record["processing_ms"] = round((time.perf_counter() - started) * 1000, 3)
    return record


def render_preview(image: np.ndarray, points: list[Point], hands: Hands, label: str,
                   hand_regions: tuple[Box, ...] | list[Box] = ()) -> np.ndarray:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    for hand in hands:
        for x, y in hand:
            if 0 <= x < 1 and 0 <= y < 1:
                cv2.circle(annotated, (int(x * width), int(y * height)), 4, (64, 240, 80), -1)
    for x1, y1, x2, y2 in hand_regions:
        cv2.rectangle(annotated, (round(x1 * width), round(y1 * height)),
                      (round(x2 * width) - 1, round(y2 * height) - 1), (64, 240, 80), 2)
    for index, point in enumerate(points, start=1):
        if point[0] >= width or point[1] >= height:
            continue
        cv2.drawMarker(annotated, point, (0, 200, 255), cv2.MARKER_CROSS, 17, 2)
        cv2.putText(annotated, str(index), (point[0] + 9, point[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
    cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4)
    cv2.putText(annotated, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return annotated


def write_record(log: TextIO, record: dict) -> None:
    log.write(json.dumps(record, ensure_ascii=False) + "\n")


def replay(path: Path, detector: Detector, log: TextIO, points: list[Point], preview: bool) -> int:
    counts = Counter()
    stopped = False
    window_open = False
    frames = read_video(path)
    try:
        for frame in frames:
            started = time.perf_counter()
            record = inspect_frame(frame, detector)
            write_record(log, record)
            counts[record["status"]] += 1
            if not preview or frame.image is None:
                continue
            if not window_open:
                cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
                window_open = True
            label = f"{frame.index} | {frame.timestamp_ms / 1000:.3f} s | {record['status']}"
            cv2.imshow(WINDOW, render_preview(frame.image, points, record["hands"], label,
                                             record["hand_regions"]))
            wait_ms = max(1, round(frame.duration_ms - (time.perf_counter() - started) * 1000))
            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                stopped = True
                break
    finally:
        frames.close()
        if window_open:
            cv2.destroyAllWindows()

    errors = counts["read_error"] + counts["processing_error"]
    status = "completed_with_errors" if errors else "completed"
    if stopped:
        status = "stopped"
    write_record(log, {"type": "summary", "status": status, "counts": dict(counts)})
    return 1 if errors else 0


def parse_point(value: str) -> Point:
    try:
        x, y = map(int, value.split(","))
        if min(x, y) >= 0:
            return x, y
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("放置參考點請用非負整數 X,Y，例如 540,720。")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="回放影片並記錄手部位置。")
    parser.add_argument("video", type=Path, help="來源影片")
    add_detector_arguments(parser)
    parser.add_argument("--output", type=Path, required=True, help="新建的 JSONL 紀錄路徑")
    parser.add_argument("--point", type=parse_point, action="append", default=[], metavar="X,Y",
                        help="放置參考點的原始影格像素座標，可重複指定")
    parser.add_argument("--no-preview", action="store_true", help="直接處理所有影格並輸出紀錄")
    args = parser.parse_args(argv)
    if not args.video.is_file():
        parser.error(f"找不到影片 {args.video}")

    try:
        detector, settings = open_detector(args, parser)
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as log:
                write_record(log, {
                    "type": "run",
                    "video": str(args.video.resolve()),
                    "pyav_version": av.__version__,
                    "opencv_version": cv2.__version__,
                    "reference_points": args.point,
                    **settings,
                })
                code = replay(args.video, detector, log, args.point, not args.no_preview)
        finally:
            detector.close()
    except (ImportError, OSError, RuntimeError, ValueError, cv2.error) as exc:
        print(f"手部回放實驗失敗。{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    print(f"逐影格紀錄已保存於 {args.output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
