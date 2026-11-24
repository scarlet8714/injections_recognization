"""Focus search checks with a camera that has five known focal planes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from focus_test.focus_control import FocusRange, NativeFocusControl
from focus_test.focus_scan import ScanCancelled, evaluate_rois, evaluate_samples, scan_bottles
from focus_test.main import parse_args, save_image


class SimulatedCamera:
    def __init__(self):
        self.focus_range = FocusRange(10, 210, 2)
        self.state = {"minimum": 10, "maximum": 210, "step": 2, "value": 50,
                      "flags": 1, "capabilities": 3, "name": "simulated camera"}
        self.restored = False
        self.focus_commands = []

    def get(self):
        return self.state.copy()

    def set_manual(self, value):
        if not 10 <= value <= 210 or (value - 10) % 2:
            raise RuntimeError("Illegal focus step")
        self.focus_commands.append(value)
        self.state.update(value=value, flags=2)
        return self.get()

    def set_auto(self, _value):
        self.state.update(value=110, flags=1)
        return self.get()

    def restore(self, state):
        self.state = state.copy()
        self.restored = True
        return self.get()

    def read(self):
        return np.full((100, 200, 3), self.state["value"], np.uint8)


PEAKS = [30, 74, 118, 162, 204]
BOXES = [(i * 40, 10, i * 40 + 30, 90) for i in range(5)]


def simulated_measure(frame, boxes, _region):
    box = boxes[0]
    target = PEAKS[box[0] // 40]
    score = float(50000 - (int(frame[0, 0, 0]) - target) ** 2)
    return [{"focus_box": list(box), "laplacian_variance": score, "tenengrad": score * 2}]


class FocusScanTest(unittest.TestCase):
    def args(self, result_dir, *options):
        return parse_args(["--result-dir", result_dir, "--focus-settle", "0", *options])

    def test_shared_sweep_finds_five_distinct_focus_planes_and_saves_matching_frames(self):
        camera = SimulatedCamera()
        initial = camera.get()
        with tempfile.TemporaryDirectory() as temp:
            report_path = scan_bottles(camera, camera, BOXES, self.args(temp), (100, 200, 3),
                                       simulated_measure, save_image)
            report = json.loads(report_path.read_text())
            self.assertEqual(report["status"], "completed")
            self.assertEqual([item["best_focus"] for item in report["rois"]], PEAKS)
            for item in report["rois"]:
                image = cv2.imread(str(report_path.parent / item["image"]))
                self.assertEqual(int(image[0, 0, 0]), item["best_focus"])
                self.assertGreaterEqual(len(item["curve"]), 17)
            points = [sample for item in report["rois"] for sample in item["curve"]]
            unique = len({sample["focus"] for sample in points})
            self.assertEqual(report["search"]["scan_focus_commands"], unique)
            self.assertEqual(report["search"]["reused_scores"], len(points) - unique)
            self.assertEqual(len(camera.focus_commands), unique + len(BOXES))
            self.assertTrue(report["restored"])
            self.assertEqual(camera.get(), initial)

    def test_fixed_window_starts_at_legal_midpoint_and_marks_capped_peaks(self):
        camera = SimulatedCamera()
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(temp, "--focus-search", "range", "--focus-range", "31", "161")
            path = scan_bottles(camera, camera, BOXES, args, (100, 200, 3), simulated_measure, save_image)
            report = json.loads(path.read_text())
        self.assertEqual(report["search"]["initial_range"], [32, 160])
        self.assertEqual(camera.focus_commands[0], 96)
        self.assertTrue(all(32 <= value <= 160 for value in camera.focus_commands))
        self.assertEqual([row["best_focus"] for row in report["rois"]], [32, 74, 118, 160, 160])
        self.assertEqual([row["best_at_range_edge"] for row in report["rois"]], [True, False, False, True, True])

    def test_autofocus_seed_expands_when_bottles_peak_at_either_window_edge(self):
        camera = SimulatedCamera()
        original = camera.get()
        with tempfile.TemporaryDirectory() as temp, patch("focus_test.focus_scan.AUTOFOCUS_WAIT_SECONDS", 0):
            args = self.args(temp, "--focus-search", "autofocus")
            path = scan_bottles(camera, camera, BOXES, args, (100, 200, 3), simulated_measure, save_image)
            report = json.loads(path.read_text())
        self.assertEqual(report["search"]["autofocus"]["value"], 110)
        self.assertEqual(report["search"]["initial_range"], [56, 164])
        self.assertEqual(camera.focus_commands[0], 110)
        self.assertEqual([row["best_focus"] for row in report["rois"]], PEAKS)
        self.assertEqual(camera.get(), original)

    def test_fast_autofocus_saves_best_captured_frame_without_a_second_move(self):
        camera = SimulatedCamera()
        original = camera.get()
        with tempfile.TemporaryDirectory() as temp, patch("focus_test.focus_scan.AUTOFOCUS_WAIT_SECONDS", 0):
            args = self.args(temp, "--focus-search", "fast-autofocus", "--expected", "3")
            path = scan_bottles(camera, camera, BOXES[1:4], args, (100, 200, 3), simulated_measure, save_image)
            report = json.loads(path.read_text())
            self.assertEqual(camera.focus_commands, [110, 38, 56, 74, 92, 128, 146, 164, 182])
            self.assertEqual(report["search"]["verification_focus_commands"], 0)
            for item, value in zip(report["rois"], [74, 110, 164], strict=True):
                crop = cv2.imread(str(path.parent / item["image"]))
                self.assertEqual(item["capture_method"], "scan_frame")
                self.assertEqual(item["best_focus"], value)
                self.assertEqual(int(crop[0, 0, 0]), value)
        self.assertEqual(camera.get(), original)

    def test_cancellation_restores_autofocus_and_keeps_report(self):
        camera = SimulatedCamera()
        initial = camera.get()

        def cancel(_frame, _info):
            raise ScanCancelled("test cancellation")

        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ScanCancelled):
                scan_bottles(camera, camera, BOXES, self.args(temp), (100, 200, 3),
                             simulated_measure, save_image, progress=cancel)
            path = next(Path(temp).glob("scan_*/focus_scan.json"))
            report = json.loads(path.read_text())
            self.assertEqual(report["status"], "cancelled")
            self.assertTrue(report["restored"])
            self.assertEqual(camera.get(), initial)

    def test_capture_failure_restores_focus(self):
        camera = SimulatedCamera()
        initial = camera.get()
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(camera, "read", side_effect=RuntimeError("disconnected")):
            with self.assertRaisesRegex(RuntimeError, "disconnected"):
                scan_bottles(camera, camera, BOXES, self.args(temp), (100, 200, 3),
                             simulated_measure, save_image)
            self.assertTrue(camera.restored)
            self.assertEqual(camera.get(), initial)

    def test_driver_ignoring_manual_setting_is_not_accepted(self):
        controller = NativeFocusControl.__new__(NativeFocusControl)
        with patch.object(controller, "_request", return_value={"value": 50, "flags": 1}):
            with self.assertRaisesRegex(RuntimeError, "readback mismatch"):
                controller.set_manual(100)

    def test_unstable_samples_are_retried_before_ranking(self):
        camera = SimulatedCamera()
        values = iter([10, 500, 20, 100, 101, 99])

        def measure(_frame, _boxes, _region):
            return [{"laplacian_variance": next(values)}]

        with tempfile.TemporaryDirectory() as temp:
            _, _, score, _, retries, stable = evaluate_samples(
                camera, (100, 200, 3), BOXES[0], self.args(temp), measure, None, {})
        self.assertTrue(stable)
        self.assertEqual(retries, 1)
        self.assertEqual(score, 100)

    def test_retry_keeps_other_bottles_first_stable_score(self):
        camera = SimulatedCamera()
        scores = {BOXES[0]: iter([101, 99, 100]), BOXES[1]: iter([10, 500, 20, 190, 200, 210])}

        def measure(_frame, boxes, _region):
            return [{"laplacian_variance": next(scores[boxes[0]])}]

        with tempfile.TemporaryDirectory() as temp:
            outcomes = evaluate_rois(camera, (100, 200, 3), BOXES[:2], self.args(temp), measure, None, {})
        self.assertEqual([(score, retries, stable) for _, _, score, _, retries, stable in outcomes],
                         [(100, 0, True), (200, 1, True)])

    def test_wrong_bottle_count_does_not_change_focus(self):
        camera = SimulatedCamera()
        initial = camera.get()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "requires 5 verified ROIs"):
                scan_bottles(camera, camera, BOXES[:4], self.args(temp), (100, 200, 3),
                             simulated_measure, save_image)
            self.assertEqual(list(Path(temp).iterdir()), [])
            self.assertEqual(camera.get(), initial)


if __name__ == "__main__":
    unittest.main()
