"""Run PP-OCRv5 server detection and recognition models on an image or PDF."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

TEST_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = TEST_ROOT / "result"
DETECTION_MODEL = "PP-OCRv5_server_det"
RECOGNITION_MODEL = "PP-OCRv5_server_rec"


def distribution_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def inspect_environment() -> dict[str, Any]:
    """Return package and Paddle device information without requiring Paddle."""
    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "torch": distribution_version("torch"),
        "paddleocr": distribution_version("paddleocr"),
        "paddlepaddle": distribution_version("paddlepaddle"),
        "paddlepaddle_gpu": distribution_version("paddlepaddle-gpu"),
        "cuda_compiled": False,
        "cuda_device_count": 0,
        "cuda_version": None,
    }
    if importlib.util.find_spec("paddle") is None:
        return report

    # PaddleOCR/PaddleX can import Torch internally. On Windows, importing Paddle
    # first can make Torch's shm.dll fail with WinError 127 because both runtimes
    # load native libraries into the same process. Establish the working order.
    if sys.platform == "win32" and importlib.util.find_spec("torch") is not None:
        try:
            import torch

            report["torch_module_version"] = torch.__version__
        except Exception as exc:
            report["torch_import_error"] = f"{type(exc).__name__}: {exc}"

    import paddle

    report["paddle_module_version"] = paddle.__version__
    try:
        report["cuda_compiled"] = bool(paddle.is_compiled_with_cuda())
        if report["cuda_compiled"]:
            report["cuda_device_count"] = int(paddle.device.cuda.device_count())
            report["cuda_version"] = paddle.version.cuda()
    except Exception as exc:
        report["device_probe_error"] = f"{type(exc).__name__}: {exc}"
    return report


def print_environment(report: dict[str, Any]) -> None:
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["paddleocr"]:
        print(
            "\nMissing paddleocr. Install requirements.txt in this test's venv.",
            file=sys.stderr,
        )
    if not report["paddlepaddle"] and not report["paddlepaddle_gpu"]:
        print(
            "\nMissing PaddlePaddle inference engine. Install paddlepaddle (CPU) "
            "or paddlepaddle-gpu (GPU) manually.",
            file=sys.stderr,
        )


def resolve_device(requested: str, report: dict[str, Any]) -> str:
    if requested != "auto":
        return requested
    if report["cuda_compiled"] and report["cuda_device_count"] > 0:
        return "gpu:0"
    return "cpu"


def unique_output_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    candidate = root / base
    suffix = 1
    while candidate.exists():
        candidate = root / f"{base}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def run_ocr(
    input_path: Path,
    output_root: Path,
    device: str,
    score_threshold: float,
) -> Path:
    from paddleocr import PaddleOCR

    output_dir = unique_output_dir(output_root)
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print("Loading PP-OCRv5 server models (the first run may download weights)...")

    ocr = PaddleOCR(
        text_detection_model_name=DETECTION_MODEL,
        text_recognition_model_name=RECOGNITION_MODEL,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device=device,
    )

    count = 0
    for result in ocr.predict(
        str(input_path),
        text_rec_score_thresh=score_threshold,
    ):
        result.print(ensure_ascii=False)
        result.save_to_json(str(output_dir), ensure_ascii=False)
        result.save_to_img(str(output_dir))
        count += 1

    if count == 0:
        raise RuntimeError("PaddleOCR returned no result")
    print(f"Completed {count} result(s): {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Input image or PDF path. Not required with --check.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, gpu, or gpu:<index> (default: auto)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="Minimum recognition confidence from 0 to 1 (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Output root (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print Python, PaddleOCR, PaddlePaddle, and GPU status, then exit.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = inspect_environment()

    if args.check:
        print_environment(report)
        dependencies_ready = bool(
            report["paddleocr"]
            and (report["paddlepaddle"] or report["paddlepaddle_gpu"])
        )
        return 0 if dependencies_ready else 1

    if args.input is None:
        print("error: input is required unless --check is used", file=sys.stderr)
        return 2
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        print(f"error: input file does not exist: {input_path}", file=sys.stderr)
        return 2
    if not 0.0 <= args.score_threshold <= 1.0:
        print("error: --score-threshold must be between 0 and 1", file=sys.stderr)
        return 2

    if not report["paddleocr"]:
        print_environment(report)
        return 1
    if not report["paddlepaddle"] and not report["paddlepaddle_gpu"]:
        print_environment(report)
        return 1

    device = resolve_device(args.device, report)
    try:
        run_ocr(input_path, args.output_dir.resolve(), device, args.score_threshold)
        return 0
    except Exception as exc:
        print(f"OCR failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
