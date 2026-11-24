"""Fine-tune a COCO-pretrained YOLO11n detector on the bottle dataset."""

from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
DATASET = ROOT.parent / "dataset"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
LIVE_CONFIDENCE = 0.40
LIVE_NMS_IOU = 0.45


def evaluate_run(work_dir: Path, device: str) -> None:
    from ultralytics import YOLO

    fit_dir = work_dir / "fit"
    fit = yaml.safe_load((fit_dir / "args.yaml").read_text(encoding="utf-8"))
    run_data = work_dir / "data.yaml"
    options = {"data": str(run_data), "imgsz": fit["imgsz"], "batch": fit["batch"],
               "device": device, "workers": 0, "project": str(work_dir)}
    candidates = []
    for name in ("best", "last"):
        checkpoint = fit_dir / "weights" / f"{name}.pt"
        model = YOLO(str(checkpoint))
        ranked = model.val(**options, name=f"validation_{name}", plots=True, exist_ok=True)
        deployed = model.val(**options, name=f"operating_point_{name}", plots=False, exist_ok=True,
                             conf=LIVE_CONFIDENCE, iou=LIVE_NMS_IOU)
        counts = {key: sum(row[key] for row in deployed.box.image_metrics.values()) for key in ("tp", "fp", "fn")}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        operating = {**counts, "precision": tp / (tp + fp) if tp + fp else 0.0,
                     "recall": tp / (tp + fn) if tp + fn else 0.0,
                     "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}
        candidates.append({"checkpoint": str(checkpoint), "operating_point": operating,
                           "metrics": {key: float(value) for key, value in ranked.results_dict.items()}})

    selected = max(candidates, key=lambda row: (row["operating_point"]["f1"], row["metrics"]["metrics/mAP50-95(B)"]))
    with (fit_dir / "results.csv").open(encoding="utf-8", newline="") as training_log:
        epochs = sum(1 for _ in csv.DictReader(training_log))
    dataset = yaml.safe_load(run_data.read_text(encoding="utf-8"))["path"]
    report = {"checkpoint": selected["checkpoint"], "pretrained": fit["model"],
              "dataset": dataset, "image_size": fit["imgsz"], "device": device, "epochs_completed": epochs,
              "confidence": LIVE_CONFIDENCE, "nms_iou": LIVE_NMS_IOU, "match_iou": 0.5,
              "candidates": candidates}
    report_path = work_dir / "training_summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected checkpoint: {selected['checkpoint']}\nTraining summary: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-run", type=Path, help="Reevaluate checkpoints in an existing training directory")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--weights", type=Path, default=MODELS / "yolo11n.pt")
    parser.add_argument("--device", help="cpu or CUDA index such as 0; defaults to auto")
    parser.add_argument("--epochs", type=int, default=45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.threads) < 1 or args.imgsz < 32 or args.imgsz % 32:
        parser.error("Use positive epochs, batch, and threads, and an image size divisible by 32.")
    config_root = Path(os.environ.setdefault("YOLO_CONFIG_DIR", str(MODELS / "ultralytics")))
    (config_root / "Ultralytics").mkdir(parents=True, exist_ok=True)
    import torch
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    torch.set_num_threads(args.threads)
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    if args.evaluate_run:
        evaluate_run(args.evaluate_run.resolve(), device)
        return
    if not args.weights.is_file():
        parser.error(f"Pretrained weights missing: {args.weights}")

    dataset = args.dataset.resolve()
    data = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8-sig"))
    data["path"] = str(dataset)
    for split in ("train", "val"):
        if not (dataset / data[split]).is_dir():
            parser.error(f"Image directory for {split} is missing: {dataset / data[split]}")

    work_dir = RUNS / datetime.now().strftime("train_%Y%m%d_%H%M%S_%f")
    work_dir.mkdir(parents=True)
    run_data = work_dir / "data.yaml"
    run_data.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

    model = YOLO(str(args.weights.resolve()))
    model.train(
        data=str(run_data), project=str(work_dir), name="fit", epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=device, workers=0, cache="ram",
        freeze=10, optimizer="AdamW", lr0=0.002, lrf=0.05, patience=12,
        seed=42, deterministic=True, amp=False, plots=True,
        hsv_h=0.008, hsv_s=0.30, hsv_v=0.25, degrees=4.0, translate=0.08,
        scale=0.25, fliplr=0.0, flipud=0.0, mosaic=0.2, close_mosaic=10,
        mixup=0.0, copy_paste=0.0, erasing=0.0,
    )
    evaluate_run(work_dir, device)


if __name__ == "__main__":
    main()
