import csv
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tracking.app import draw_output, parser, run
from tracking.core import Config
from tracking.detection import DetectionTracker


def detection(x=20, size=8, class_id=0):
    return dict(bbox=[x, 60, size, size], confidence=.9, class_id=class_id)


class SequenceDetector:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return next(self.batches)


class ContinuousTrackingTests(unittest.TestCase):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)

    def test_short_gap_recovers_and_next_object_has_fresh_motion_and_class(self):
        detector = SequenceDetector([[detection()], [detection(24)], [], [detection(28)],
                                     [], [detection(90, 50, class_id=1)]])
        tracker = DetectionTracker(detector, Config(coast_seconds=.2),
                                   strict_motion_gate=True, continuous=True)
        for i, stamp in enumerate((0., .04, .08, .12)):
            row = tracker.update(self.frame, stamp, i, "test")
            self.assertEqual(row["segment"], 1)
        self.assertEqual(row["state"], "TRACKING")
        row = tracker.update(self.frame, .33, 4, "test")
        self.assertEqual(row["state"], "LOST")
        self.assertIsNone(tracker.last_box)
        self.assertIsNone(tracker.last_visible_time)
        self.assertIsNone(tracker.class_id)
        row = tracker.update(self.frame, .4, 5, "test")
        self.assertEqual(row["state"], "WAITING")
        self.assertIsNone(row["segment"])
        self.assertIsNone(row["center"])
        self.assertIsNone(row["estimated_center"])
        row = tracker.update(self.frame, 2., 6, "test")
        self.assertEqual(row["segment"], 2)
        self.assertEqual(row["class_id"], 1)
        self.assertEqual(row["state"], "TRACKING")
        self.assertIsNone(row["predicted_center"])
        np.testing.assert_array_equal(tracker.motion.x[2:], [0., 0.])
        self.assertFalse(tracker.velocity_ready)
        first = tracker.segments[0]
        self.assertEqual(first["observed_frames"], 3)
        self.assertEqual(first["last_observed_frame"], 3)
        self.assertEqual(first["end_frame"], 4)
        self.assertEqual(first["end_reason"], "tracking_timeout")
        tracker.finish_segment(6, 2., "user_quit")
        tracker.finish_segment(7, 2.1, "end_of_readable_video")
        self.assertEqual(tracker.segments[1]["end_reason"], "user_quit")
        self.assertEqual(tracker.segments[1]["observed_frames"], 1)

    def test_waiting_searches_every_frame_even_with_sparse_tracking(self):
        detector = SequenceDetector([[], [], [detection()], [detection(24)]])
        tracker = DetectionTracker(detector, interval=3, continuous=True)
        rows = [tracker.update(self.frame, i*.04, i, "test") for i in range(6)]
        self.assertEqual([r["detection_ran"] for r in rows], [True, True, True, False, False, True])
        self.assertEqual(rows[2]["segment"], 1)
        self.assertEqual(rows[3]["state"], "PREDICTED")
        self.assertEqual(rows[5]["segment"], 1)

    def test_frame_limit_conflict_is_rejected_before_model_loading(self):
        args = parser().parse_args(["missing.mp4", "--weights", "missing.pt",
                                    "--mode", "continuous-track", "--max-frames", "16"])
        with patch("tracking.detection.YoloDetector") as detector:
            with self.assertRaisesRegex(ValueError, "continuous-track conflicts with --max-frames"):
                run(args)
        detector.assert_not_called()

    def make_video(self, root, frames):
        path = root / "source.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
        self.assertTrue(writer.isOpened())
        try:
            for index in range(frames):
                writer.write(np.full((120, 160, 3), index*5, np.uint8))
        finally:
            writer.release()
        return path

    def test_continues_through_two_objects_and_saves_all_frames_until_eof(self):
        for second_disappears in (False, True):
            with self.subTest(second_disappears=second_disappears), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                video = self.make_video(root, 30)

                class SceneDetector:
                    def __init__(self, *args):
                        self.metadata = {"backend": "pt"}
                        self.last_timing = {}

                    def warmup(self, frame, count):
                        pass

                    def detect(self, frame):
                        index = round(float(frame.mean())/5)
                        self.last_timing = dict(detector_ms=0., inference_ms=0.)
                        if index in (2, 3, 5, 6):
                            return [detection(20+index)]
                        if 15 <= index <= (18 if second_disappears else 29):
                            return [detection(90, 50)]
                        return []

                rendered = {}

                def render(frame, result, history, estimates, size):
                    rendered[result["frame_index"]] = (list(history), list(estimates))
                    return draw_output(frame, result, history, estimates, size)

                args = parser().parse_args([
                    str(video), "--weights", "unused.pt", "--mode", "continuous-track",
                    "--headless", "--decoder", "opencv", "--encoder", "opencv",
                    "--coast-seconds", "0.14", "--strict-motion-gate", "--warmup", "0",
                    "--output", str(root/"out")])
                with patch("tracking.detection.YoloDetector", SceneDetector), \
                        patch("tracking.app.draw_output", side_effect=render), redirect_stdout(io.StringIO()):
                    output = run(args)
                rows = [json.loads(line) for line in (output/"track.jsonl").read_text().splitlines()]
                summary = json.loads((output/"summary.json").read_text())
                self.assertEqual(len(rows), 30)
                self.assertEqual(summary["frames"], 30)
                self.assertEqual(summary["end_reason"], "end_of_readable_video")
                self.assertEqual(summary["segment_count"], 2)
                self.assertEqual(rows[4]["state"], "LOST_PENDING")
                self.assertEqual(rows[5]["segment"], 1)
                self.assertEqual(rows[10]["state"], "LOST")
                for index in range(11, 15):
                    self.assertEqual(rows[index]["state"], "WAITING")
                    self.assertTrue(rows[index]["detection_ran"])
                    self.assertIsNone(rows[index]["center"])
                    self.assertIsNone(rows[index]["segment"])
                self.assertEqual(rows[15]["segment"], 2)
                self.assertIsNone(rows[15]["predicted_center"])
                self.assertEqual(rendered[10], ([], []))
                self.assertEqual(rendered[15], ([], []))
                first, second = summary["segments"]
                self.assertEqual((first["start_frame"], first["last_observed_frame"], first["end_frame"]),
                                 (2, 6, 10))
                self.assertEqual(first["observed_frames"], 4)
                self.assertEqual(first["end_reason"], "tracking_timeout")
                self.assertAlmostEqual(first["start_timestamp"], .08)
                self.assertAlmostEqual(first["last_observed_timestamp"], .24)
                self.assertAlmostEqual(first["end_timestamp"], .4)
                self.assertEqual(second["end_frame"], 22 if second_disappears else 29)
                self.assertEqual(second["observed_frames"], 4 if second_disappears else 15)
                self.assertEqual(second["end_reason"], "tracking_timeout" if second_disappears
                                 else "end_of_readable_video")
                for name in ("track.csv", "timing.csv"):
                    with (output/name).open(encoding="utf-8-sig", newline="") as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), 30)
                capture = cv2.VideoCapture(str(output/"annotated.mp4"))
                try:
                    decoded = 0
                    while capture.read()[0]:
                        decoded += 1
                    self.assertEqual(decoded, 30)
                finally:
                    capture.release()

    def test_empty_scene_finishes_at_eof_without_creating_a_segment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.make_video(root, 8)
            args = parser().parse_args([str(video), "--weights", "unused.pt", "--headless", "--no-video",
                                        "--mode", "continuous-track", "--decoder", "opencv",
                                        "--detect-interval", "3", "--warmup", "0", "--output", str(root/"out")])
            with patch("tracking.detection.YoloDetector") as detector, redirect_stdout(io.StringIO()):
                detector.return_value.metadata = {"backend": "pt"}
                detector.return_value.last_timing = {}
                detector.return_value.detect.return_value = []
                output = run(args)
            rows = [json.loads(line) for line in (output/"track.jsonl").read_text().splitlines()]
            summary = json.loads((output/"summary.json").read_text())
            self.assertEqual(summary["segment_count"], 0)
            self.assertEqual(summary["segments"], [])
            self.assertEqual(summary["frames"], 8)
            self.assertTrue(all(r["state"] == "WAITING" and r["detection_ran"] for r in rows))


if __name__ == "__main__":
    unittest.main()
