"""驗證相機的偵測紀錄能對照同步錄影。"""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import av
import numpy as np

from hand_test import live


class LiveTest(unittest.TestCase):
    def test_read_failure_is_logged_without_creating_a_video_frame(self):
        image = np.zeros((64, 96, 3), dtype=np.uint8)
        camera = Mock()
        camera.read.side_effect = [image, None, image]
        detector = Mock()
        detector.detect.side_effect = [[], [[(0.5, 0.5)]]]
        log = io.StringIO()

        with tempfile.TemporaryDirectory() as temp:
            clip = Path(temp) / "capture.mp4"
            with av.open(str(clip), "w") as output:
                with patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "destroyAllWindows"), \
                        patch.object(live.cv2, "getWindowProperty", return_value=1), \
                        patch.object(live.cv2, "waitKey", side_effect=[-1, -1, ord("q")]), \
                        patch.object(live.cv2, "imshow"):
                    result = live.observe(camera, detector, log, [], None, True, live.VideoRecording(output))
            with av.open(str(clip)) as source:
                recorded_times = [frame.time * 1000 for frame in source.decode(video=0)]

        rows = [json.loads(line) for line in log.getvalue().splitlines()]
        self.assertEqual(result, 0)
        self.assertEqual([row["status"] for row in rows[:-1]], ["no_hand", "read_error", "detected"])
        self.assertEqual(len(recorded_times), 2)
        self.assertAlmostEqual(recorded_times[0], rows[0]["timestamp_ms"], delta=1)
        self.assertAlmostEqual(recorded_times[1], rows[2]["timestamp_ms"], delta=1)
        self.assertEqual(rows[-1]["status"], "stopped")

    def test_color_preview_marks_connected_hand_region(self):
        background = np.full((270, 480, 3), 180, dtype=np.uint8)
        scattered = background.copy()
        scattered[45:65, 60:80] = (90, 126, 183)
        scattered[80:100, 60:80] = (90, 126, 183)
        hand = background.copy()
        hand[100:124, 80:120] = (90, 126, 183)
        camera = Mock()
        camera.read.side_effect = [background, scattered, hand, background, KeyboardInterrupt]

        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "color.jsonl"
            with patch.object(live, "UsbFrameReader", return_value=camera), \
                    patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "destroyAllWindows"), \
                    patch.object(live.cv2, "getWindowProperty", return_value=1), \
                    patch.object(live.cv2, "waitKey", return_value=-1), \
                    patch.object(live.cv2, "imshow") as show:
                result = live.main(["--detector", "color", "--output", str(log), "--point", "200,190"])
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(result, 0)
        self.assertEqual(rows[0]["candidate_minimum_area_px"], 648)
        frames = rows[1:-1]
        self.assertEqual([row["status"] for row in frames], ["no_hand", "no_hand", "detected", "no_hand"])
        self.assertEqual(frames[1]["region_area_px"], 400)
        self.assertEqual(frames[2]["region_area_px"], 960)
        self.assertEqual(frames[2]["hand_regions"], [[80 / 480, 100 / 270, 120 / 480, 124 / 270]])
        preview = show.call_args_list[2].args[1]
        self.assertEqual(tuple(preview[100, 80]), (64, 240, 80))
        self.assertEqual(tuple(preview[190, 200]), (0, 200, 255))
        camera.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
