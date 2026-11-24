"""Read bottle crops selected by a focus scan using PP-OCRv6 small."""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path

import cv2
import numpy as np

from focus_test.focus_scan import write_report
from focus_test.main import measure, read_image, save_image
from medicine_pipeline.train import MODELS

DETECTION_MODEL = "PP-OCRv6_small_det"
RECOGNITION_MODEL = "PP-OCRv6_small_rec"


def assess_focus(scan: dict, item: dict) -> dict:
    """Compare focus measurements within the same bottle; flag ambiguous results."""
    metric = scan["metric"]
    coarse_scores = [entry["score"] for entry in item["curve"]
                     if entry["phase"] == 0 and entry["stable"]]
    coarse_median = statistics.median(coarse_scores) if coarse_scores else None
    peak = item["best_scan_score"]
    verified = item["verified_score"]
    reasons = []
    if peak <= 0 or coarse_median is None or peak <= coarse_median * 1.1:
        reasons.append("weak_focus_peak")
    if item.get("best_at_range_edge"):
        reasons.append("focus_range_edge")
    if verified < peak * 0.8:
        reasons.append("verification_below_peak")
    return {
        "status": "review" if reasons else "passed",
        "reasons": reasons,
        "metric": metric,
        "best_focus": item["best_focus"],
        "coarse_median": coarse_median,
        "scan_peak": peak,
        "verified_score": verified,
        "verification_spread": item["verification_spread"],
        "laplacian_variance": item["laplacian_variance"],
        "tenengrad": item["tenengrad"],
    }


def load_ocr(device: str):
    model_paths = [MODELS / f"{name}_infer" for name in (DETECTION_MODEL, RECOGNITION_MODEL)]
    for path in model_paths:
        if not all((path / file).is_file() for file in ("inference.json", "inference.pdiparams", "inference.yml")):
            raise FileNotFoundError(f"Missing PP-OCRv6 model files in {path}. See medicine_pipeline/README.md.")

    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODELS / "paddlex_cache"))
    # Reuse the project's Torch-before-Paddle DLL initialization on Windows.
    from ppocrv5_server_test.main import inspect_environment

    inspect_environment()
    from paddleocr import PaddleOCR

    return PaddleOCR(
        text_detection_model_name=DETECTION_MODEL, text_detection_model_dir=str(model_paths[0]),
        text_recognition_model_name=RECOGNITION_MODEL, text_recognition_model_dir=str(model_paths[1]),
        use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
        device=device, cpu_threads=4, enable_mkldnn=False,
    )


def recognize(ocr, image_path: Path, output: Path) -> list[dict]:
    image = read_image(image_path)
    (result,) = ocr.predict(image, text_rec_score_thresh=0.0)
    data = result.json["res"]
    lines = [
        {"text": text, "confidence": float(score), "polygon": polygon}
        for text, score, polygon in zip(data["rec_texts"], data["rec_scores"], data["rec_polys"], strict=True)
        if text.strip()
    ]
    output.mkdir()
    result.save_to_json(str(output / "paddle_result.json"), ensure_ascii=False)
    for number, line in enumerate(lines, 1):
        polygon = np.asarray(line["polygon"], dtype=np.int32)
        cv2.polylines(image, [polygon], True, (0, 190, 255), 2, cv2.LINE_AA)
        x, y = map(int, polygon[0])
        cv2.putText(image, str(number), (max(0, x), max(12, y)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 130, 0), 1, cv2.LINE_AA)
    save_image(output / "annotated.png", image)
    (output / "recognized.txt").write_text(
        "\n".join(f"{number}\t{line['confidence']:.3f}\t{line['text']}"
                  for number, line in enumerate(lines, 1)), encoding="utf-8",
    )
    return lines


def run_ocr(scan_path: Path, output: Path, device: str, include_review: bool = False) -> Path:
    scan = json.loads(scan_path.read_text(encoding="utf-8"))
    if scan["status"] != "completed" or not scan["restored"]:
        raise ValueError(f"Expected a completed focus scan with restored camera state: {scan_path}")

    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "summary.json"
    report = {"status": "processing", "source_scan": str(scan_path.resolve()), "device": device,
              "models": [DETECTION_MODEL, RECOGNITION_MODEL], "rois": []}
    write_report(report_path, report)
    ocr = None
    try:
        reference_path = scan_path.parent / "detection_frame.png"
        reference = read_image(reference_path) if reference_path.is_file() else None
        for item in scan["rois"]:
            quality = assess_focus(scan, item)
            if reference is not None:
                baseline = measure(reference, [item["box"]], scan["focus_region"])[0]
                quality["reference_score"] = baseline[scan["metric"]]
            image_path = scan_path.parent / item["image"]
            bottle_dir = output / f"bottle_{item['id']:02d}"
            row = {"id": item["id"], "box": item["box"], "image": str(image_path.resolve()),
                   "focus": quality, "status": "skipped_focus_review", "lines": []}
            report["rois"].append(row)
            if quality["status"] == "passed" or include_review:
                row["status"] = "processing"
                if ocr is None:
                    ocr = load_ocr(device)
                try:
                    row["lines"] = recognize(ocr, image_path, bottle_dir)
                    row["ocr_dir"] = bottle_dir.name
                    row["status"] = "recognized" if row["lines"] else "no_text"
                except Exception as exc:
                    row["status"] = "ocr_error"
                    row["error"] = f"{type(exc).__name__}: {exc}"
            write_report(report_path, report)
            print(f"Bottle {item['id']}: {row['status']}; {len(row['lines'])} OCR lines", flush=True)
        report["status"] = "completed" if all(
            row["status"] == "recognized" and row["focus"]["status"] == "passed" for row in report["rois"]
        ) else "needs_review"
        if any(row["status"] == "ocr_error" for row in report["rois"]):
            report["status"] = "ocr_error"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        write_report(report_path, report)
    return report_path
