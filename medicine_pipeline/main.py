"""Detect and focus bottles, recognize labels, and look up the results in SQLite."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from focus_test.focus_control import FOCUS_SEARCH_METHODS
from medicine_pipeline.lookup import DEFAULT_DATABASE, run_lookup
from medicine_pipeline.train import LIVE_CONFIDENCE, LIVE_NMS_IOU, MODELS, RUNS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    live = subcommands.add_parser("live", help="Detect, focus and perform OCR with a USB camera")
    live.add_argument("--model", type=Path, required=True, help="Trained bottle detector checkpoint")
    live.add_argument("--expected", type=int, required=True, help="Number of bottles visible in the camera")
    live.add_argument("--camera-index", type=int, default=0)
    live.add_argument("--imgsz", type=int, default=640)
    live.add_argument("--confidence", type=float, default=LIVE_CONFIDENCE)
    live.add_argument("--iou", type=float, default=LIVE_NMS_IOU, help="Overlap threshold for suppressing duplicate boxes")
    live.add_argument("--focus-metric", choices=("laplacian_variance", "tenengrad"), default="laplacian_variance")
    live.add_argument("--focus-settle", type=float, help="Motor settling time (fast default 0.15 s, otherwise 0.35 s)")
    live.add_argument("--focus-search", choices=FOCUS_SEARCH_METHODS, default="full")
    live.add_argument("--focus-range", type=int, nargs=2, metavar=("MIN", "MAX"))
    live.add_argument("--scan-samples", type=int, help="Sample count per scan level (fast default 9, otherwise 17)")
    live.add_argument("--focus-region", type=float, nargs=4, default=[0.15, 0.45, 0.85, 0.90],
                      metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    offline = subcommands.add_parser("ocr", help="Perform OCR on a completed focus scan")
    offline.add_argument("scan_dir", type=Path, help="Directory containing focus_scan.json and ROI images")
    lookup = subcommands.add_parser("lookup", help="Look up an existing OCR summary without using the camera or OCR")
    lookup.add_argument("summary", type=Path, help="OCR summary.json")
    lookup.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="Medicine SQLite file")
    for subcommand in (live, offline):
        subcommand.add_argument("--ocr-device", default="cpu", help="Paddle device such as cpu or gpu:0")
        subcommand.add_argument("--include-review", action="store_true", help="Also OCR crops flagged for review")
        database_options = subcommand.add_mutually_exclusive_group()
        database_options.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="Medicine SQLite file")
        database_options.add_argument("--skip-lookup", action="store_true", help="Stop after OCR")
    args = parser.parse_args()
    try:
        if args.command == "lookup":
            run_lookup(args.summary.resolve(), args.database)
            return 0
        if args.command == "live":
            if not args.model.is_file():
                parser.error(f"Bottle detector not found: {args.model}")
            config = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "ultralytics")))
            (config / "Ultralytics").mkdir(parents=True, exist_ok=True)
            from focus_test.main import parse_args, run_headless_scan

            settle = args.focus_settle
            if settle is None and args.focus_search != "fast-autofocus":
                settle = 0.35
            focus_args = parse_args([
                "--scan", "--model", str(args.model.resolve()), "--expected", str(args.expected),
                "--camera-index", str(args.camera_index), "--imgsz", str(args.imgsz),
                "--confidence", str(args.confidence), "--iou", str(args.iou), "--focus-metric", args.focus_metric,
                "--result-dir", str(RUNS), "--focus-search", args.focus_search,
                *(["--focus-settle", str(settle)] if settle is not None else []),
                *(["--scan-samples", str(args.scan_samples)] if args.scan_samples is not None else []),
                *(["--focus-range", *map(str, args.focus_range)] if args.focus_range else []),
                "--focus-region", *map(str, args.focus_region),
            ])
            scan_path = run_headless_scan(focus_args)
        else:
            scan_path = args.scan_dir.resolve() / "focus_scan.json"

        from medicine_pipeline.ocr import run_ocr

        output = scan_path.parent / "ocr" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
        report = run_ocr(scan_path, output, args.ocr_device, include_review=args.include_review)
        status = json.loads(report.read_text(encoding="utf-8"))["status"]
        print(f"OCR {status}: {report}")
        if not args.skip_lookup:
            run_lookup(report, args.database)
        return 1 if status == "ocr_error" else 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Bottle pipeline failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
