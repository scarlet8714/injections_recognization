"""驗證影片入口輸出的時間軸、預覽及失敗狀態。"""

import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import av
import numpy as np

from hand_test import replay


class SampleDetector:
    version = "test"

    def __init__(self, model: Path, max_hands: int):
        pass

    def detect(self, frame: np.ndarray, timestamp_ms: float) -> replay.Hands:
        level = frame[0, 0, 0]
        if level == 120:
            raise RuntimeError("影格偵測失敗")
        return [[(0.5, 0.5)]] if level > 0 else []

    def close(self) -> None:
        pass


class ReplayTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.video = self.root / "sample.mkv"
        self.output = self.root / "sample.jsonl"
        self.model = self.root / "sample.task"
        self.model.touch()

    def write_sample(self):
        with av.open(str(self.video), "w") as video:
            stream = video.add_stream("ffv1", rate=25)
            stream.width = 96
            stream.height = 64
            stream.pix_fmt = "bgr0"
            stream.codec_context.time_base = Fraction(1, 1000)
            for pts, level in ((0, 0), (40, 255), (130, 120)):
                image = np.full((64, 96, 3), level, dtype=np.uint8)
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts = pts
                frame.time_base = Fraction(1, 1000)
                video.mux(stream.encode(frame))
            video.mux(stream.encode())

    def run_experiment(self, *args: str) -> tuple[int, list[dict]]:
        with patch("hand_test.detector.HandDetector", SampleDetector):
            result = replay.main([str(self.video), "--model", str(self.model),
                                  "--output", str(self.output), "--point", "25,42", *args])
        records = [json.loads(line) for line in self.output.read_text(encoding="utf-8").splitlines()]
        return result, records

    def test_source_times_and_processing_error_are_reported(self):
        self.write_sample()
        result, records = self.run_experiment("--no-preview")
        self.assertEqual(result, 1)
        self.assertEqual(records[0]["reference_points"], [[25, 42]])
        self.assertEqual([record["timestamp_ms"] for record in records[1:-1]], [0, 40, 130])
        self.assertEqual([record["status"] for record in records[1:-1]],
                         ["no_hand", "detected", "processing_error"])
        self.assertEqual(records[2]["hands"], [[[0.5, 0.5]]])
        self.assertEqual(records[-1]["counts"]["processing_error"], 1)

    def test_invalid_video_is_not_counted_as_no_hand(self):
        self.video.write_bytes(b"invalid video")
        result, records = self.run_experiment("--no-preview")
        self.assertEqual(result, 1)
        self.assertEqual(records[1]["status"], "read_error")
        self.assertIsNone(records[1]["timestamp_ms"])
        self.assertEqual(records[-1]["counts"], {"read_error": 1})

    def test_preview_displays_hand_and_reference_point(self):
        self.write_sample()
        with patch.object(replay.cv2, "namedWindow"), patch.object(replay.cv2, "destroyAllWindows"), \
                patch.object(replay.cv2, "getWindowProperty", return_value=1), \
                patch.object(replay.cv2, "waitKey", side_effect=[-1, ord("q")]), \
                patch.object(replay.cv2, "imshow") as show:
            result, records = self.run_experiment()
        self.assertEqual(result, 0)
        rendered = show.call_args.args[1]
        self.assertEqual(tuple(rendered[42, 25]), (0, 200, 255))
        self.assertEqual(tuple(rendered[32, 48]), (64, 240, 80))
        self.assertEqual(records[-1]["status"], "stopped")


if __name__ == "__main__":
    unittest.main()
