"""Coarse-to-fine search with shared frames and a separate focus score per bottle."""

from __future__ import annotations

import json
import csv
import math
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from focus_test.focus_control import AUTOFOCUS_METHODS, FocusRange

AUTOFOCUS_WAIT_SECONDS = 1.0
AUTOFOCUS_RADIUS = 55  # Device focus units around the autofocus readback.
FAST_AUTOFOCUS_RADIUS = 72


class ScanCancelled(RuntimeError):
    pass


def grid_indices(low: int, high: int, samples: int) -> list[int]:
    if low == high:
        return [low]
    return sorted(set(int(round(v)) for v in np.linspace(low, high, min(samples, high - low + 1))))


def write_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Windows can briefly lock a just-written JSON file during indexing.
    for attempt in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.02 * (attempt + 1))


def capture_settled(reader, expected_shape: tuple, settle: float, discard: int, samples: int,
                    progress, info: dict) -> list[np.ndarray]:
    """Keep draining capture during motor settling, then drop buffered frames."""
    deadline = time.monotonic() + settle
    timeout = time.monotonic() + settle + 8
    discarded = 0
    frames = []
    while len(frames) < samples:
        if time.monotonic() > timeout:
            raise RuntimeError("Timed out waiting for fresh focus-scan frames.")
        frame = reader.read()
        if frame is None:
            time.sleep(.005)
            continue
        if frame.shape != expected_shape:
            raise RuntimeError("Camera resolution changed during focus scan.")
        if progress is not None:
            progress(frame, info)
        if time.monotonic() < deadline:
            continue
        if discarded < discard:
            discarded += 1
            continue
        frames.append(frame.copy())
    return frames


def evaluate_rois(reader, frame_shape, boxes, args, measure_fn, progress, info):
    """Score each ROI from the same exposures and retry only unstable ROIs."""
    pending = list(enumerate(boxes))
    outcomes = [None] * len(boxes)
    for attempt in range(args.focus_retries + 1):
        frames = capture_settled(reader, frame_shape, args.focus_settle, args.focus_discard,
                                 args.focus_samples, progress, info)
        retry = []
        for index, box in pending:
            measurements = [measure_fn(frame, [box], args.focus_region)[0] for frame in frames]
            values = [row[args.focus_metric] for row in measurements]
            score = float(np.median(values))
            # Keep small fluctuations in fully defocused patches from triggering retries.
            spread = float(np.ptp(values) / max(abs(score), 50))
            stable = spread <= args.focus_stability
            if stable or attempt == args.focus_retries:
                outcomes[index] = frames, measurements, score, spread, attempt, stable
            else:
                retry.append((index, box))
        if not retry:
            break
        pending = retry
    return outcomes


def evaluate_samples(reader, frame_shape, box, args, measure_fn, progress, info):
    return evaluate_rois(reader, frame_shape, [box], args, measure_fn, progress, info)[0]


def median_frame(frames, measurements, metric):
    order = np.argsort([measurement[metric] for measurement in measurements])
    index = int(order[len(order) // 2])
    return frames[index], measurements[index]


def scan_bottles(reader, controller, boxes: list[tuple], args, frame_shape: tuple,
                 measure_fn, save_fn, progress=None, roi_source="unknown", detections=None,
                 reference_frame=None) -> Path:
    if len(boxes) != args.expected:
        raise ValueError(f"Focus scan requires {args.expected} verified ROIs; found {len(boxes)}. Press D or M first.")
    initial = controller.get()
    if not initial["capabilities"] & 2:
        raise RuntimeError("This camera does not report manual focus support.")
    focus_range: FocusRange = controller.focus_range
    if focus_range.count < 2:
        raise RuntimeError("Camera focus range has fewer than two positions.")
    fast = args.focus_search == "fast-autofocus"
    use_autofocus = args.focus_search in AUTOFOCUS_METHODS
    if use_autofocus and not initial["capabilities"] & 1:
        raise RuntimeError("This camera does not report autofocus support.")
    scan_low, scan_high = 0, focus_range.count - 1
    first_index = None
    if args.focus_search == "range":
        requested_low, requested_high = args.focus_range
        scan_low = max(scan_low, math.ceil((requested_low - focus_range.minimum) / focus_range.step))
        scan_high = min(scan_high, (requested_high - focus_range.minimum) // focus_range.step)
        if scan_high <= scan_low:
            raise ValueError("The requested window contains fewer than two camera focus positions.")
        first_index = (scan_low + scan_high) // 2
    # Validate every measurement window before changing any camera controls.
    for x1, y1, x2, y2 in boxes:
        l, t, r, b = args.focus_region
        if int(r * (x2 - x1)) - int(l * (x2 - x1)) < 5 or \
                int(b * (y2 - y1)) - int(t * (y2 - y1)) < 5:
            raise ValueError("A focus measurement ROI is smaller than 5 x 5 pixels.")
    run_dir = args.result_dir / datetime.now().strftime("scan_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    path = run_dir / "focus_scan.json"
    started = time.perf_counter()
    report = {"started_at": datetime.now().astimezone().isoformat(), "status": "scanning",
              "camera": initial, "image_size": [frame_shape[1], frame_shape[0]],
              "roi_source": roi_source, "detections": detections, "boxes": boxes,
              "focus_region": args.focus_region, "metric": args.focus_metric,
              "search": {"strategy": "shared_sparse_frames" if fast else "shared_frames",
                         "method": args.focus_search, "scan_focus_commands": 0,
                         "verification_focus_commands": 0, "reused_scores": 0,
                         "coarse_samples": args.scan_samples, "settle_seconds": args.focus_settle,
                         "discard_frames": args.focus_discard, "samples_per_value": args.focus_samples,
                         "max_sample_spread": args.focus_stability, "max_retries": args.focus_retries},
              "rois": [], "restored": False,
              "note": "Scan samples are shared by focus setting; each bottle selects its own best focus. "
                      "Fixed ROIs require stationary objects. "
                      "Coarse-to-fine search does not guarantee a global optimum. "
                      "Focus values are device units, not calibrated distance."}
    if roi_source == "yolo":
        report["detector"] = {"model": str(args.model.resolve()), "imgsz": args.imgsz,
                              "confidence": args.confidence, "iou": args.iou}
    write_report(path, report)
    if reference_frame is not None:
        save_fn(run_dir / "detection_frame.png", reference_frame)
    config = {"image_size": report["image_size"], "boxes": boxes}
    (run_dir / "roi_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    failure = None
    search = report["search"]
    shared_scores = {}
    best_captures = {}
    try:
        if args.focus_search == "range":
            search["requested_range"] = args.focus_range
        if use_autofocus:
            auto_started = time.perf_counter()
            controller.set_auto(initial["value"])
            info = {"bottle_id": 1, "bottle_count": len(boxes), "focus": initial["value"], "phase": "autofocus"}
            capture_settled(reader, frame_shape, AUTOFOCUS_WAIT_SECONDS, 0, 1, progress, info)
            auto = controller.get()
            if not auto["flags"] & 1:
                raise RuntimeError("Camera left autofocus mode before its focus value could be read.")
            first_index = min(scan_high, max(scan_low, round((auto["value"] - focus_range.minimum) / focus_range.step)))
            radius = FAST_AUTOFOCUS_RADIUS if fast else AUTOFOCUS_RADIUS
            radius_steps = max(1, radius // focus_range.step)
            scan_low = max(scan_low, first_index - radius_steps)
            scan_high = min(scan_high, first_index + radius_steps)
            search["autofocus"] = {"value": auto["value"], "elapsed_seconds": time.perf_counter() - auto_started,
                                   "radius_steps": radius_steps}
        search["initial_range"] = [focus_range.value(scan_low), focus_range.value(scan_high)]
        search["start_focus"] = focus_range.value(scan_low if first_index is None else first_index)
        for bottle_id, box in enumerate(boxes, 1):
            item = {"id": bottle_id, "box": list(box), "status": "scanning", "curve": []}
            report["rois"].append(item)
            seen = set()
            best_index = None
            best_score = -math.inf
            lower_limit, upper_limit = scan_low, scan_high
            low, high = lower_limit, upper_limit
            phase = 0
            while True:
                indices = grid_indices(low, high, args.scan_samples)
                scan_order = indices
                if phase == 0 and first_index is not None:
                    scan_order = [first_index, *(index for index in indices if index != first_index)]
                for index in scan_order:
                    if index in seen:
                        continue
                    reused = index in shared_scores
                    if reused:
                        search["reused_scores"] += 1
                    else:
                        value = focus_range.value(index)
                        readback = controller.set_manual(value)
                        search["scan_focus_commands"] += 1
                        info = {"bottle_id": bottle_id, "bottle_count": len(boxes), "focus": value,
                                "phase": phase, "best_focus": None if best_index is None else focus_range.value(best_index),
                                "best_score": None if best_index is None else best_score}
                        samples = evaluate_rois(reader, frame_shape, boxes, args, measure_fn, progress, info)
                        shared_scores[index] = [
                            {"focus": value, "readback": readback["value"], "score": score,
                             "sample_scores": [row[args.focus_metric] for row in measurements],
                             "sample_spread": spread, "retries": retries, "stable": stable}
                            for _, measurements, score, spread, retries, stable in samples
                        ]
                        if fast:
                            # Keep only the median image at each bottle's highest stable score.
                            for roi_index, (frames, measurements, score, spread, retries, stable) in enumerate(samples):
                                previous = best_captures.get(roi_index)
                                if stable and (previous is None or score > previous[0]):
                                    frame, measurement = median_frame(frames, measurements, args.focus_metric)
                                    best_captures[roi_index] = score, spread, retries, frame, measurement
                    entry = {**shared_scores[index][bottle_id - 1], "phase": phase, "reused": reused}
                    seen.add(index)
                    item["curve"].append(entry)
                    if entry["stable"] and entry["score"] > best_score:
                        best_index, best_score = index, entry["score"]
                    write_report(path, report)
                if best_index is None:
                    raise RuntimeError("Focus scores did not stabilize. Check lighting/motion or increase --focus-settle.")
                if fast:
                    break
                if args.focus_search == "autofocus":
                    if best_index == lower_limit and lower_limit > 0:
                        high = lower_limit
                        low = lower_limit = max(0, lower_limit - radius_steps)
                        phase += 1
                        continue
                    if best_index == upper_limit and upper_limit < focus_range.count - 1:
                        low = upper_limit
                        high = upper_limit = min(focus_range.count - 1, upper_limit + radius_steps)
                        phase += 1
                        continue
                spacing = max((b - a for a, b in zip(indices, indices[1:])), default=1)
                if spacing <= 1:
                    break
                low, high = max(lower_limit, best_index - spacing), min(upper_limit, best_index + spacing)
                phase += 1
            best_focus = focus_range.value(best_index)
            if fast:
                _, spread, retries, frame, measurement = best_captures.pop(bottle_id - 1)
            else:
                controller.set_manual(best_focus)
                search["verification_focus_commands"] += 1
                info = {"bottle_id": bottle_id, "bottle_count": len(boxes), "focus": best_focus,
                        "phase": "verify", "best_focus": best_focus, "best_score": best_score}
                frames, measurements, _, spread, retries, stable = evaluate_samples(
                    reader, frame_shape, box, args, measure_fn, progress, info)
                if not stable:
                    raise RuntimeError(f"Bottle {bottle_id} did not stabilize at the selected focus; increase --focus-settle.")
                frame, measurement = median_frame(frames, measurements, args.focus_metric)
            x1, y1, x2, y2 = box
            a, b, c, d = measurement["focus_box"]
            item.update({"status": "completed", "best_focus": best_focus,
                         "best_scan_score": best_score, "verified_score": measurement[args.focus_metric],
                         "laplacian_variance": measurement["laplacian_variance"],
                         "tenengrad": measurement["tenengrad"], "focus_box": measurement["focus_box"],
                         "verification_spread": spread, "verification_retries": retries,
                         "capture_method": "scan_frame" if fast else "verification_recapture",
                         "image": f"rois/{bottle_id:02d}.png", "focus_image": f"focus/{bottle_id:02d}.png",
                         "frame": f"frames/{bottle_id:02d}.png",
                         "searched_range": [focus_range.value(lower_limit), focus_range.value(upper_limit)],
                         "best_at_range_edge": best_index in (lower_limit, upper_limit)})
            if measurement[args.focus_metric] < best_score * .8:
                item["warning"] = "Verification score fell over 20% below the scan peak; inspect lighting, motion and motor settling."
            save_fn(run_dir / item["image"], frame[y1:y2, x1:x2])
            save_fn(run_dir / item["focus_image"], frame[b:d, a:c])
            save_fn(run_dir / item["frame"], frame)
            write_report(path, report)
            print(f"Bottle {bottle_id}/{len(boxes)}: best focus={best_focus}, "
                  f"{args.focus_metric}={measurement[args.focus_metric]:.2f}", flush=True)
        report["status"] = "completed"
    except BaseException as exc:
        failure = exc
        report["status"] = "cancelled" if isinstance(exc, (ScanCancelled, KeyboardInterrupt)) else "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            report["restored_state"] = controller.restore(initial)
            report["restored"] = True
        except Exception as exc:
            report["restore_error"] = str(exc)
            if failure is None:
                report["status"] = "restore_failed"
                failure = RuntimeError(f"Images saved, but original focus mode could not be restored: {exc}")
        report["finished_at"] = datetime.now().astimezone().isoformat()
        report["elapsed_seconds"] = time.perf_counter() - started
        write_report(path, report)
        with (run_dir / "focus_curve.csv").open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["bottle_id", "focus", "phase", args.focus_metric])
            for item in report["rois"]:
                for entry in item["curve"]:
                    writer.writerow([item["id"], entry["focus"], entry["phase"], entry["score"]])
    print(f"Focus scan {report['status']}: {path.resolve()}", flush=True)
    if failure is not None:
        raise failure
    return path


def render_curves(report_path: Path) -> Path:
    """Render recorded measurements; interpolation is only a visual guide."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rois"]
    figure = Figure(figsize=(11, 2.1 * len(rows) + 1), layout="constrained")
    axes = figure.subplots(len(rows), 1, sharex=True, squeeze=False)
    if "initial_range" in report["search"]:
        ranges = [item.get("searched_range", report["search"]["initial_range"]) for item in rows]
        limits = (min(window[0] for window in ranges), max(window[1] for window in ranges))
    else:
        limits = (report["camera"]["minimum"], report["camera"]["maximum"])
    for ax, item in zip(axes[:, 0], rows):
        curve = sorted(item["curve"], key=lambda row: row["focus"])
        ax.plot([row["focus"] for row in curve], [row["score"] for row in curve],
                "o-", markersize=3, linewidth=1, color="#2563eb")
        if "best_focus" in item:
            ax.axvline(item["best_focus"], color="#dc2626", linestyle="--", linewidth=1)
            ax.set_title(f"Bottle {item['id']} - selected focus {item['best_focus']}", loc="left", fontsize=10)
        ax.set_ylabel("Focus score")
        ax.grid(alpha=.2)
    axes[-1, 0].set_xlim(*limits)
    axes[-1, 0].set_xlabel("Focus setting (device units; not distance)")
    figure.suptitle(f"Focus scores by bottle - {report['metric']}\n"
                    f"Median of {report['search']['samples_per_value']} frames at each sampled setting", fontsize=12)
    path = report_path.parent / "focus_curves.png"
    FigureCanvasAgg(figure).print_png(str(path))
    return path
