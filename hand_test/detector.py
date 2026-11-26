"""從影格取得手部關鍵點或色彩候選區塊。"""

import argparse
from pathlib import Path

import cv2
import numpy as np

Hands = list[list[tuple[float, float]]]
Box = tuple[float, float, float, float]


class HandDetector:
    def __init__(self, model: Path, max_hands: int):
        import mediapipe as mp

        self._mediapipe = mp
        self._timestamp_ms = -1
        self.version = mp.__version__
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=max_hands,
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def detect(self, frame: np.ndarray, timestamp_ms: float) -> Hands:
        # 模型需要嚴格遞增的整數毫秒，紀錄仍保留原始影格時間。
        self._timestamp_ms = max(self._timestamp_ms + 1, round(timestamp_ms))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = self._mediapipe.Image(image_format=self._mediapipe.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, self._timestamp_ms)
        return [[(point.x, point.y) for point in hand] for hand in result.hand_landmarks]

    def close(self) -> None:
        self._landmarker.close()


class ColorHandDetector:
    size = (480, 270)
    lower = (0, 138, 76)
    upper = (255, 178, 127)
    minimum_area = 648

    def __init__(self):
        self.kernel = np.ones((3, 3), np.uint8)

    def detect(self, frame: np.ndarray) -> tuple[Box | None, int]:
        small = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        ycrcb = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb)
        mask = cv2.inRange(ycrcb, self.lower, self.upper)
        cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        if count <= 1:
            return None, 0
        index = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
        x, y, width, height, area = map(int, stats[index])
        if area < self.minimum_area:
            return None, area
        image_width, image_height = self.size
        return (x / image_width, y / image_height,
                (x + width) / image_width, (y + height) / image_height), area

    def close(self) -> None:
        pass


Detector = HandDetector | ColorHandDetector


def add_detector_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--detector", choices=("mediapipe", "color"), default="mediapipe",
                        help="手部關鍵點或色彩候選實驗")
    parser.add_argument("--model", type=Path, help="MediaPipe 手部模型")
    parser.add_argument("--max-hands", type=int, default=2, help="MediaPipe 的手部數量上限")


def open_detector(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Detector, dict]:
    if args.detector == "color":
        if args.model is not None:
            parser.error("--model 適用於 MediaPipe 手部關鍵點模式。")
        detector = ColorHandDetector()
        return detector, {
            "detector": "color",
            "candidate_size": detector.size,
            "candidate_ycrcb_lower": detector.lower,
            "candidate_ycrcb_upper": detector.upper,
            "candidate_minimum_area_px": detector.minimum_area,
        }
    if args.max_hands < 1:
        parser.error("--max-hands 必須至少為 1。")
    if args.model is None:
        parser.error("請用 --model 指定 MediaPipe 手部模型。")
    if not args.model.is_file():
        parser.error(f"找不到手部模型 {args.model}")
    detector = HandDetector(args.model, args.max_hands)
    return detector, {
        "detector": "mediapipe",
        "model": str(args.model.resolve()),
        "mediapipe_version": detector.version,
        "max_hands": args.max_hands,
    }
