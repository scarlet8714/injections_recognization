"""Bottle ROI detection, fixed crops, and per-ROI focus measurements."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
RESULT_ROOT = Path(__file__).resolve().parent / "result"
DEFAULT_MODEL = PROJECT_ROOT / "segmentation_roi/model/best.pt"
Box = tuple[int, int, int, int]  # x1, y1, x2, y2, exclusive right/bottom


def read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def save_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode(path.suffix, image)
    if not ok:
        raise OSError(f"Cannot encode image: {path}")
    data.tofile(path)


def detect_rois(frame: np.ndarray, background: np.ndarray, *, threshold: int = 25,
                min_area_ratio: float = 0.001, detect_width: int = 1280,
                morph_size: int = 5) -> tuple[list[Box], np.ndarray]:
    """Find all sufficiently large components; never force the expected count."""
    if frame.shape != background.shape:
        raise ValueError("Background and frame dimensions differ. Capture background again.")
    height, width = frame.shape[:2]
    scale = min(1.0, detect_width / max(width, height))
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    current = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    empty = cv2.resize(background, size, interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(cv2.GaussianBlur(current, (3, 3), 0),
                             cv2.GaussianBlur(empty, (3, 3), 0))
    # Channel maximum also detects color changes with similar grayscale intensity.
    _, mask = cv2.threshold(difference.max(axis=2), threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = map(int, stats[index])
        if area < mask.size * min_area_ratio:
            continue
        boxes.append((max(0, int(np.floor(x * width / size[0]))),
                      max(0, int(np.floor(y * height / size[1]))),
                      min(width, int(np.ceil((x + w) * width / size[0]))),
                      min(height, int(np.ceil((y + h) * height / size[1])))))
    return sorted(boxes, key=lambda box: (box[0], box[1])), mask


def focus_box(box: Box, region: list[float] | tuple[float, ...]) -> Box:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    left, top, right, bottom = region
    return (x1 + int(left * width), y1 + int(top * height),
            x1 + int(right * width), y1 + int(bottom * height))


def focus_scores(patch: np.ndarray) -> dict[str, float]:
    """Measure original pixels, excluding a filter border; no resizing/sharpening."""
    if patch.size == 0 or min(patch.shape[:2]) < 5:
        raise ValueError("Focus patch must be at least 5 x 5 pixels.")
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=1)[1:-1, 1:-1]
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)[1:-1, 1:-1]
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)[1:-1, 1:-1]
    return {"laplacian_variance": float(laplacian.var()),
            "tenengrad": float(np.mean(dx * dx + dy * dy))}


def measure(frame: np.ndarray, boxes: list[Box], region: list[float]) -> list[dict]:
    results = []
    for index, box in enumerate(boxes, 1):
        inner = focus_box(box, region)
        x1, y1, x2, y2 = inner
        scores = focus_scores(frame[y1:y2, x1:x2])
        results.append({"id": index, "box": list(box), "focus_box": list(inner), **scores})
    return results


def load_boxes(path: Path, frame: np.ndarray) -> list[Box]:
    config = json.loads(path.read_text(encoding="utf-8"))
    height, width = frame.shape[:2]
    if config["image_size"] != [width, height]:
        raise ValueError("ROI configuration dimensions do not match the image.")
    boxes = []
    for values in config["boxes"]:
        if len(values) != 4 or any(type(v) is not int for v in values):
            raise ValueError("ROI coordinates must be four integers.")
        x1, y1, x2, y2 = values
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"Invalid ROI: {values}")
        boxes.append(tuple(values))
    return sorted(boxes, key=lambda box: (box[0], box[1]))


def annotate(frame: np.ndarray, measurements: list[dict]) -> np.ndarray:
    result = frame.copy()
    for item in measurements:
        x1, y1, x2, y2 = item["box"]
        cv2.rectangle(result, (x1, y1), (x2 - 1, y2 - 1), (0, 220, 0), 3)
        a, b, c, d = item["focus_box"]
        cv2.rectangle(result, (a, b), (c - 1, d - 1), (0, 220, 255), 3)
        label = f"#{item['id']} Lap={item['laplacian_variance']:.1f} Ten={item['tenengrad']:.1f}"
        font_scale = max(0.5, frame.shape[1] / 2400)
        origin = (x1, max(25, y1 - 12))
        cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 6)
        cv2.putText(result, label, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 220, 0), 2)
    return result


def save_capture(frame: np.ndarray, boxes: list[Box], args: argparse.Namespace,
                 source: str, background: np.ndarray | None = None,
                 mask: np.ndarray | None = None, detections: list[dict] | None = None) -> Path:
    measurements = measure(frame, boxes, args.focus_region)
    run_dir = args.result_dir / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    save_image(run_dir / "capture.png", frame)
    save_image(run_dir / "annotated.png", annotate(frame, measurements))
    if background is not None:
        save_image(run_dir / "background.png", background)
    if mask is not None:
        save_image(run_dir / "mask.png", mask)
    for item in measurements:
        x1, y1, x2, y2 = item["box"]
        a, b, c, d = item["focus_box"]
        item["image"] = f"rois/{item['id']:02d}.png"
        item["focus_image"] = f"focus/{item['id']:02d}.png"
        save_image(run_dir / item["image"], frame[y1:y2, x1:x2])
        save_image(run_dir / item["focus_image"], frame[b:d, a:c])
    report = {"captured_at": datetime.now().astimezone().isoformat(),
              "status": "measured" if len(boxes) == args.expected else "needs_roi_review",
              "roi_source": source, "expected_count": args.expected, "roi_count": len(boxes),
              "focus_region": args.focus_region, "sharpness_threshold": None,
              "detection": ({"model": str(args.model.resolve()), "image_size": args.imgsz,
                             "confidence": args.confidence, "results": detections}
                            if source == "yolo" else
                            {"threshold": args.threshold, "min_area_ratio": args.min_area_ratio,
                             "detect_width": args.detect_width, "morph_size": args.morph_size}),
              "rois": measurements,
              "note": "Scores are uncalibrated; compare the same fixed ROI across frames. "
                      "Matching count alone does not verify one component per bottle. "
                      "Focus region is a geometric inset, not detected label text."}
    (run_dir / "focus_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    config = {"image_size": [frame.shape[1], frame.shape[0]], "boxes": boxes}
    (run_dir / "roi_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(f"Saved {len(boxes)}/{args.expected} ROI(s): {run_dir}")
    for item in measurements:
        print(f"  #{item['id']} Laplacian={item['laplacian_variance']:.2f} "
              f"Tenengrad={item['tenengrad']:.2f}")
    return run_dir


def get_bottle_model(model_path: Path):
    if not model_path.is_file():
        raise ValueError(f"Bottle model does not exist: {model_path}")
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from segmentation_roi.main import load_model
    return load_model(str(model_path.resolve()))


def detect_model_rois(frame: np.ndarray, args: argparse.Namespace) -> tuple[list[Box], list[dict]]:
    model = get_bottle_model(args.model)
    prediction = model(frame, imgsz=args.imgsz, conf=args.confidence, iou=args.iou, verbose=False)[0]
    height, width = frame.shape[:2]
    detections = []
    for detected in prediction.boxes:
        raw = detected.xyxy[0].cpu().numpy()
        x1, y1 = max(0, int(np.floor(raw[0]))), max(0, int(np.floor(raw[1])))
        x2, y2 = min(width, int(np.ceil(raw[2]))), min(height, int(np.ceil(raw[3])))
        if x2 <= x1 or y2 <= y1:
            continue
        class_id = int(detected.cls[0])
        detections.append({"box": [x1, y1, x2, y2], "class_id": class_id,
                           "class_name": str(model.names[class_id]),
                           "confidence": float(detected.conf[0])})
    detections.sort(key=lambda item: (item["box"][0], item["box"][1]))
    return [tuple(item["box"]) for item in detections], detections


def detect(frame: np.ndarray, background: np.ndarray | None, args: argparse.Namespace):
    if args.roi_method == "yolo":
        print(f"Detecting bottles with YOLO (imgsz={args.imgsz})...", flush=True)
        boxes, detections = detect_model_rois(frame, args)
        return boxes, None, detections
    if background is None:
        raise ValueError("Background mode: remove bottles and press B first, or use M for manual ROIs.")
    boxes, mask = detect_rois(frame, background, threshold=args.threshold,
                              min_area_ratio=args.min_area_ratio, detect_width=args.detect_width,
                              morph_size=args.morph_size)
    height, width = frame.shape[:2]
    if any((x2 - x1) * (y2 - y1) > width * height * 0.85 for x1, y1, x2, y2 in boxes):
        raise ValueError("Background difference covers most of the image. Lighting may have changed "
                         "or bottles merged. Use --roi-method yolo or M for manual ROIs.")
    return boxes, mask, None


def camera_startup_error(log_path: Path, exit_code: int | None) -> str:
    # A file avoids blocking a long-lived server when stdout fills a pipe.
    with log_path.open("rb") as log:
        log.seek(0, 2)
        log.seek(max(0, log.tell() - 8192))
        detail = log.read().decode("utf-8", errors="replace").strip()
    message = (f"Camera server startup failed (exit={exit_code}).\n"
               f"Log: {log_path.resolve()}\n{detail or '(no server output)'}")
    if "No IC Imaging Control 3.5 camera found" in detail:
        message += ("\nThe SDK could not enumerate a camera. Check camera power/USB and "
                    "driver availability; also check whether this execution environment "
                    "has access to the camera.")
    return message


def ensure_camera_server(log_dir: Path = RESULT_ROOT):
    from camera_test.camera_ipc import CameraServerError, SharedFrameReader, server_ready

    def stream_alive(timeout: float) -> bool:
        reader = SharedFrameReader()
        try:
            deadline = time.monotonic() + timeout
            seen = False
            while time.monotonic() < deadline:
                if reader.read() is not None:
                    if seen:
                        return True
                    seen = True
                time.sleep(0.01)
            return False
        finally:
            reader.close()

    if server_ready():
        if stream_alive(2):
            return None
        raise CameraServerError("Camera stream is frozen; restart camera_server.exe.")
    executable = PROJECT_ROOT / "cpp_camera/build/Release/camera_server.exe"
    if not executable.is_file():
        raise CameraServerError(f"Missing camera server: {executable}")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / datetime.now().strftime("camera_server_%Y%m%d_%H%M%S_%f.log")
    with log_path.open("wb") as log:
        process = subprocess.Popen([str(executable)], cwd=executable.parent,
                                   stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                exit_code = process.poll()
                if exit_code is not None:
                    raise CameraServerError(camera_startup_error(log_path, exit_code))
                if server_ready() and stream_alive(1):
                    return process
                time.sleep(0.1)
            raise CameraServerError("Camera startup timed out.\n" +
                                    camera_startup_error(log_path, process.poll()))
        except BaseException:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=5)
            raise


def preview_image(frame: np.ndarray) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(1.0, 1280 / max(width, height))
    return cv2.resize(frame, (round(width * scale), round(height * scale)),
                      interpolation=cv2.INTER_AREA)


class UsbFrameReader:
    """Read a standard USB webcam without the vendor camera server."""

    def __init__(self, index: int, width: int = 1920, height: int = 1080):
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        self.capture = cv2.VideoCapture(index, backend)
        if not self.capture.isOpened():
            self.capture.release()
            raise RuntimeError(f"Cannot open USB camera index {index}. "
                               "Close other camera applications or try --camera-index 1.")
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        print(f"USB camera {index}: {int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    def read(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        return frame if ok else None

    def close(self) -> None:
        self.capture.release()


def run_scan(reader, boxes: list[Box], frame: np.ndarray, args: argparse.Namespace,
             source: str, detections=None, progress=None) -> Path:
    if args.backend != "usb":
        raise ValueError("Focus scanning currently supports the USB backend only.")
    from focus_test.focus_control import NativeFocusControl
    from focus_test.focus_scan import render_curves, scan_bottles
    with NativeFocusControl(args.camera_index) as controller:
        info = controller.get()
        print(f"{info['name']}: focus {info['minimum']}..{info['maximum']}, "
              f"step={info['step']}, current={info['value']}, flags={info['flags']}", flush=True)
        report_path = scan_bottles(reader, controller, boxes, args, frame.shape, measure, save_image,
                                   progress, source, detections, reference_frame=frame)
    try:
        render_curves(report_path)
    except Exception as exc:
        print(f"Focus scan saved; curve preview could not be rendered: {exc}", file=sys.stderr)
    return report_path


def run_headless_scan(args: argparse.Namespace) -> Path:
    from focus_test.focus_scan import capture_settled
    reader = UsbFrameReader(args.camera_index, args.width, args.height)
    try:
        deadline = time.monotonic() + 8
        frame = reader.read()
        while frame is None:
            if time.monotonic() > deadline:
                raise RuntimeError("No camera frame for focus scan.")
            frame = reader.read()
        frame = capture_settled(reader, frame.shape, 1, 3, 1, None, {})[0]
        if args.rois:
            boxes, source, detections = load_boxes(args.rois, frame), "roi_config", None
        else:
            background = read_image(args.background) if args.background else None
            boxes, _, detections = detect(frame, background, args)
            source = args.roi_method
        return run_scan(reader, boxes, frame, args, source, detections)
    finally:
        reader.close()


def run_live(args: argparse.Namespace) -> None:
    background = read_image(args.background) if args.background else None
    boxes: list[Box] = []
    source = args.roi_method
    mask = None
    detections = None
    measurements: list[dict] = []
    launched = None
    reader = None
    window = "Focus test"
    try:
        if args.backend == "usb":
            reader = UsbFrameReader(args.camera_index, args.width, args.height)
        else:
            from camera_test.camera_ipc import SharedFrameReader
            launched = ensure_camera_server(args.result_dir)
            reader = SharedFrameReader()
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
        frame = None
        scored_at = 0.0
        received_at = time.monotonic()
        while True:
            fresh = reader.read()
            if fresh is not None:
                received_at = time.monotonic()
                if frame is not None and fresh.shape != frame.shape:
                    boxes, measurements, mask, background = [], [], None, None
                    detections = None
                    print("Resolution changed: background and ROIs cleared.")
                frame = fresh
                if args.rois is not None:
                    boxes = load_boxes(args.rois, frame)
                    source = "roi_config"
                    args.rois = None
                if time.monotonic() - scored_at >= args.interval:
                    measurements = measure(frame, boxes, args.focus_region)
                    scored_at = time.monotonic()
                preview = preview_image(annotate(frame, measurements))
                lines = [f"Mode: {args.roi_method} | D: detect + lock | F: focus each bottle | M: manual ROIs",
                         "SPACE: save | C: clear | Q/Esc: quit | B: empty background (background mode)",
                         f"ROIs: {len(boxes)}/{args.expected} | Green: bottle, yellow: focus area"]
                for index, text in enumerate(lines):
                    y = 24 + index * 27
                    cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 4)
                    cv2.putText(preview, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                cv2.imshow(window, preview)
            if time.monotonic() - received_at > 5:
                raise RuntimeError("No fresh camera frame for 5 seconds.")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if frame is None:
                continue
            try:
                if key in (ord("b"), ord("B")):
                    background = frame.copy()
                    boxes, measurements, mask = [], [], None
                    detections = None
                    save_image(args.result_dir / "background.png", background)
                    print("Empty background saved. Place bottles, then press D.")
                elif key in (ord("c"), ord("C")):
                    boxes, measurements, mask = [], [], None
                    detections = None
                elif key in (ord("d"), ord("D")):
                    candidate_boxes, candidate_mask, candidate_detections = detect(frame, background, args)
                    candidate_measurements = measure(frame, candidate_boxes, args.focus_region)
                    boxes, mask, detections = candidate_boxes, candidate_mask, candidate_detections
                    measurements = candidate_measurements
                    source = "yolo" if args.roi_method == "yolo" else "background_difference"
                    print(f"Locked {len(boxes)}/{args.expected} ROI(s). Verify one box per bottle.")
                    if mask is not None:
                        cv2.imshow("Foreground mask (detection frame)", mask)
                elif key in (ord("m"), ord("M")):
                    selection = preview_image(frame)
                    name = "Select bottles - ENTER: accept each, ESC: finish"
                    selected = cv2.selectROIs(name, selection, showCrosshair=True, fromCenter=False)
                    cv2.destroyWindow(name)
                    if len(selected):
                        sx, sy = frame.shape[1] / selection.shape[1], frame.shape[0] / selection.shape[0]
                        candidate_boxes = sorted([(round(x * sx), round(y * sy), round((x + w) * sx),
                                                   round((y + h) * sy)) for x, y, w, h in selected])
                        candidate_measurements = measure(frame, candidate_boxes, args.focus_region)
                        boxes, measurements = candidate_boxes, candidate_measurements
                        source, mask = "manual", None
                        detections = None
                        print(f"Locked {len(boxes)} manual ROI(s).")
                    received_at = time.monotonic()
                elif key in (ord("f"), ord("F")):
                    if not boxes:
                        candidate_boxes, candidate_mask, candidate_detections = detect(frame, background, args)
                        candidate_measurements = measure(frame, candidate_boxes, args.focus_region)
                        boxes, mask, detections = candidate_boxes, candidate_mask, candidate_detections
                        measurements, source = candidate_measurements, args.roi_method

                    def scan_progress(scan_frame, info):
                        from focus_test.focus_scan import ScanCancelled
                        preview = preview_image(scan_frame)
                        sx, sy = preview.shape[1] / scan_frame.shape[1], preview.shape[0] / scan_frame.shape[0]
                        for i, (x1, y1, x2, y2) in enumerate(boxes, 1):
                            color = (0, 255, 255) if i == info["bottle_id"] else (0, 200, 0)
                            cv2.rectangle(preview, (round(x1 * sx), round(y1 * sy)),
                                          (round(x2 * sx), round(y2 * sy)), color, 2)
                        text = (f"Bottle {info['bottle_id']}/{info['bottle_count']} | Focus {info['focus']} | "
                                f"Phase {info['phase']} | Q/Esc: cancel")
                        cv2.putText(preview, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 0, 0), 4)
                        cv2.putText(preview, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 255, 255), 1)
                        cv2.imshow(window, preview)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27) or \
                                cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                            raise ScanCancelled("Focus scan cancelled; completed bottle images are preserved.")

                    try:
                        run_scan(reader, boxes, frame, args, source, detections, scan_progress)
                    finally:
                        received_at = time.monotonic()
                elif key == ord(" "):
                    if not boxes:
                        raise ValueError("Press D or M to select and verify ROIs first.")
                    save_capture(frame, boxes, args, source, background, mask, detections)
            except (ValueError, RuntimeError, OSError, cv2.error) as exc:
                print(f"Action failed: {exc}", file=sys.stderr)
    finally:
        if reader is not None:
            reader.close()
        cv2.destroyAllWindows()
        if launched is not None and launched.poll() is None:
            launched.terminate()
            launched.wait(timeout=5)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from focus_test.focus_control import FOCUS_SEARCH_METHODS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Offline input; omit for live camera")
    parser.add_argument("--scan", action="store_true", help="Run one headless per-bottle focus scan and exit")
    parser.add_argument("--focus-info", action="store_true", help="Print USB focus range/current value and exit")
    parser.add_argument("--backend", choices=("usb", "ic35"), default="usb",
                        help="usb: OpenCV webcam (default); ic35: existing vendor camera server")
    parser.add_argument("--camera-index", type=int, default=0, help="USB camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1920,
                        help="Requested USB image width (default: 1920); device may negotiate another size")
    parser.add_argument("--height", type=int, default=1080, help="Requested USB image height (default: 1080)")
    parser.add_argument("--background", type=Path, help="Empty scene with identical camera settings")
    parser.add_argument("--rois", type=Path, help="Reuse a saved roi_config.json")
    parser.add_argument("--roi-method", choices=("yolo", "background"), default="yolo",
                        help="Bottle localization method (default: yolo)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size (default: 1280)")
    parser.add_argument("--confidence", type=float, default=0.5, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="YOLO overlap threshold for suppressing duplicate boxes")
    parser.add_argument("--result-dir", type=Path, default=RESULT_ROOT)
    parser.add_argument("--expected", type=int, default=5)
    parser.add_argument("--threshold", type=int, default=25)
    parser.add_argument("--min-area-ratio", type=float, default=0.001)
    parser.add_argument("--detect-width", type=int, default=1280, help="Longest detection edge; crops stay full resolution")
    parser.add_argument("--morph-size", type=int, default=5, help="Odd closing kernel size in detection pixels")
    parser.add_argument("--interval", type=float, default=0.25, help="Live scoring interval in seconds")
    parser.add_argument("--scan-samples", type=int, help="Sample count per scan level (fast default 9, otherwise 17)")
    parser.add_argument("--focus-search", choices=FOCUS_SEARCH_METHODS, default="full",
                        help="Use the full range, a fixed window, or one of two autofocus-seeded searches")
    parser.add_argument("--focus-range", type=int, nargs=2, metavar=("MIN", "MAX"),
                        help="Focus device units for --focus-search range")
    parser.add_argument("--focus-settle", type=float, help="Motor settling time (fast default 0.15 s, otherwise 0.25 s)")
    parser.add_argument("--focus-discard", type=int, help="Buffered frames discarded after settling (0 to use the next frame)")
    parser.add_argument("--focus-samples", type=int, default=3, help="Fresh frames per focus value; rank by median")
    parser.add_argument("--focus-retries", type=int, help="Retry count for unstable focus samples")
    parser.add_argument("--focus-stability", type=float, default=0.2, help="Maximum relative sample score spread")
    parser.add_argument("--focus-metric", choices=("laplacian_variance", "tenengrad"), default="laplacian_variance")
    parser.add_argument("--focus-region", type=float, nargs=4, default=[0.15, 0.45, 0.85, 0.90],
                        metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"),
                        help="Relative measurement inset inside each bottle; not label detection")
    args = parser.parse_args(argv)
    fast = args.focus_search == "fast-autofocus"
    if args.scan_samples is None:
        args.scan_samples = 9 if fast else 17
    if args.focus_settle is None:
        args.focus_settle = 0.15 if fast else 0.25
    if args.focus_discard is None:
        args.focus_discard = 1 if fast else 3
    if args.focus_retries is None:
        args.focus_retries = 0 if fast else 2
    l, t, r, b = args.focus_region
    if not (0 <= l < r <= 1 and 0 <= t < b <= 1):
        parser.error("--focus-region must satisfy 0 <= left < right <= 1 and 0 <= top < bottom <= 1")
    if not (0 <= args.threshold <= 255 and 0 < args.min_area_ratio < 1):
        parser.error("Invalid threshold or min-area-ratio")
    if args.expected < 1 or args.detect_width < 16 or args.interval <= 0:
        parser.error("expected, detect-width and interval must be positive (detect-width >= 16)")
    if args.morph_size < 1 or args.morph_size % 2 != 1:
        parser.error("--morph-size must be a positive odd integer")
    if args.camera_index < 0 or args.width <= 0 or args.height <= 0:
        parser.error("camera-index must be nonnegative; width and height must be positive")
    if args.imgsz < 32 or not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        parser.error("imgsz must be >= 32, and confidence and iou must be in (0, 1]")
    if args.image and args.roi_method == "background" and not (args.background or args.rois):
        parser.error("Offline background mode requires --background or --rois")
    if sum((args.image is not None, args.scan, args.focus_info)) > 1:
        parser.error("Choose only one of --image, --scan or --focus-info")
    if (args.scan or args.focus_info) and args.backend != "usb":
        parser.error("--scan and --focus-info require --backend usb")
    if (args.focus_search == "range") != (args.focus_range is not None):
        parser.error("--focus-search range requires --focus-range MIN MAX, and vice versa")
    if args.focus_range and args.focus_range[0] >= args.focus_range[1]:
        parser.error("The lower focus limit must be below the upper limit")
    if args.scan_samples < 5 or args.focus_settle < 0 or args.focus_discard < 0 or args.focus_samples < 1:
        parser.error("scan-samples must be >= 5, settle and discard >= 0, focus-samples >= 1")
    if args.focus_retries < 0 or args.focus_stability <= 0:
        parser.error("focus-retries must be >= 0 and focus-stability must be positive")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.focus_info:
            from focus_test.focus_control import NativeFocusControl
            with NativeFocusControl(args.camera_index) as controller:
                print(json.dumps(controller.initial, ensure_ascii=False, indent=2))
        elif args.scan:
            run_headless_scan(args)
        elif args.image:
            frame = read_image(args.image)
            background = read_image(args.background) if args.background else None
            mask = None
            detections = None
            if args.rois:
                boxes, source = load_boxes(args.rois, frame), "roi_config"
            else:
                boxes, mask, detections = detect(frame, background, args)
                source = "yolo" if args.roi_method == "yolo" else "background_difference"
            save_capture(frame, boxes, args, source, background, mask, detections)
        else:
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            run_live(args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Focus test failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
