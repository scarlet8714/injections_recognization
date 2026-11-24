"""Offline checks of segmentation failure behavior and focus measurement."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from focus_test.main import (UsbFrameReader, detect, detect_model_rois, detect_rois,
                             ensure_camera_server, focus_scores, load_boxes, parse_args, save_capture)


class FocusTest(unittest.TestCase):
    def scene(self):
        background = np.full((400, 1000, 3), 230, np.uint8)
        frame = background.copy()
        for index in range(5):
            cv2.rectangle(frame, (30 + index * 190, 100), (120 + index * 190, 350),
                          (30, 70, 120), -1)
        return frame, background

    def test_separate_components_and_mapping(self):
        frame, background = self.scene()
        boxes, _ = detect_rois(frame, background, detect_width=500)
        self.assertEqual(len(boxes), 5)
        for index, (x1, y1, x2, y2) in enumerate(boxes):
            self.assertLessEqual(abs(x1 - (30 + index * 190)), 4)
            self.assertLessEqual(abs(y1 - 100), 4)
            self.assertGreaterEqual(x2, 120 + index * 190)
            self.assertGreaterEqual(y2, 350)

    def test_touching_bottles_remain_merged(self):
        frame, background = self.scene()
        cv2.rectangle(frame, (110, 150), (230, 300), (30, 70, 120), -1)
        boxes, _ = detect_rois(frame, background)
        self.assertEqual(len(boxes), 4)  # Must not invent a fifth bottle.

    def test_empty_scene_and_wrong_background(self):
        frame, background = self.scene()
        boxes, _ = detect_rois(background, background)
        self.assertEqual(boxes, [])
        with self.assertRaises(ValueError):
            detect_rois(frame, background[:200])

    def test_global_lighting_change_is_rejected_in_background_mode(self):
        background = np.full((200, 400, 3), 100, np.uint8)
        current = np.full_like(background, 140)
        args = parse_args(["--roi-method", "background"])
        with self.assertRaisesRegex(ValueError, "covers most of the image"):
            detect(current, background, args)

    def test_model_boxes_are_clipped_sorted_and_keep_confidence(self):
        class Coordinates:
            def __init__(self, values):
                self.values = np.array(values)

            def cpu(self):
                return self

            def numpy(self):
                return self.values

        def detection(box, confidence):
            return SimpleNamespace(xyxy=[Coordinates(box)], cls=[0], conf=[confidence])

        with patch("focus_test.main.get_bottle_model") as get_model:
            model = get_model.return_value
            model.names = {0: "ampoule"}
            model.return_value = [SimpleNamespace(boxes=[
                detection([200.4, 10.2, 410.8, 205.1], .9),
                detection([-2.1, -3.2, 100.3, 180.2], .8),
                detection([500, 0, 510, 100], .7)])]
            frame = np.zeros((200, 400, 3), np.uint8)
            boxes, mask, details = detect(frame, None, parse_args([]))
            self.assertEqual(boxes, [(0, 0, 101, 181), (200, 10, 400, 200)])
            self.assertIsNone(mask)
            self.assertEqual([item["confidence"] for item in details], [.8, .9])

    def test_blur_decreases_both_scores_for_fixed_content(self):
        image = np.full((200, 400, 3), 240, np.uint8)
        cv2.putText(image, "Focus 123", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 2)
        sharp = focus_scores(image)
        blurred = focus_scores(cv2.GaussianBlur(image, (15, 15), 3))
        for name in sharp:
            self.assertGreater(sharp[name], blurred[name] * 2)

    def test_capture_saves_original_pixels_and_mismatch_status(self):
        frame, background = self.scene()
        boxes, mask = detect_rois(frame, background)
        with tempfile.TemporaryDirectory() as temp:
            args = parse_args(["--result-dir", temp])
            run = save_capture(frame, boxes[:4], args, "background_difference", background, mask)
            report = json.loads((run / "focus_results.json").read_text())
            self.assertEqual(report["status"], "needs_roi_review")
            self.assertEqual(report["roi_count"], 4)
            self.assertEqual(load_boxes(run / "roi_config.json", frame), boxes[:4])
            x1, y1, x2, y2 = boxes[0]
            np.testing.assert_array_equal(cv2.imread(str(run / "rois/01.png")), frame[y1:y2, x1:x2])
            for item in report["rois"]:
                self.assertTrue((run / item["focus_image"]).is_file())

    def test_camera_startup_failure_preserves_sdk_error(self):
        from camera_test.camera_ipc import CameraServerError

        class FailedServer:
            def poll(self):
                return 1

            def wait(self, timeout):
                return 1

        def launch(*args, **kwargs):
            kwargs["stdout"].write(b"Camera server error: No IC Imaging Control 3.5 camera found\n")
            kwargs["stdout"].flush()
            return FailedServer()

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root / "cpp_camera/build/Release/camera_server.exe"
            executable.parent.mkdir(parents=True)
            executable.touch()
            with patch("focus_test.main.PROJECT_ROOT", root), \
                    patch("camera_test.camera_ipc.server_ready", return_value=False), \
                    patch("focus_test.main.subprocess.Popen", side_effect=launch):
                with self.assertRaises(CameraServerError) as raised:
                    ensure_camera_server(root / "logs")
            self.assertIn("No IC Imaging Control 3.5 camera found", str(raised.exception))
            self.assertIn("exit=1", str(raised.exception))
            self.assertEqual(len(list((root / "logs").glob("camera_server_*.log"))), 1)

    def test_usb_is_default_and_failed_device_is_released(self):
        self.assertEqual(parse_args([]).backend, "usb")
        with patch("focus_test.main.cv2.VideoCapture") as video_capture:
            capture = video_capture.return_value
            capture.isOpened.return_value = False
            with self.assertRaisesRegex(RuntimeError, "USB camera index 2"):
                UsbFrameReader(2)
            capture.release.assert_called_once()

    def test_usb_delivers_frame_and_releases_device(self):
        image, _ = self.scene()
        with patch("focus_test.main.cv2.VideoCapture") as video_capture:
            capture = video_capture.return_value
            capture.isOpened.return_value = True
            capture.get.return_value = 1000
            capture.read.return_value = (True, image)
            reader = UsbFrameReader(0)
            self.assertIs(reader.read(), image)
            capture.read.return_value = (False, None)
            self.assertIsNone(reader.read())
            reader.close()
            capture.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()
