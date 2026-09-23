import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from tracking.app import parser, run, draw_output
from tracking.benchmark import command_for, compare
from tracking.performance import stats


class FakeDetector:
    def __init__(self, *args):
        self.metadata = {"backend": "pt"}
        self.last_timing = {}

    def warmup(self, frame, count):
        self.last_timing = {}

    def detect(self, frame):
        started = time.perf_counter()
        time.sleep(.001)
        self.last_timing = dict(detector_ms=(time.perf_counter()-started)*1000,
                                inference_ms=1., preprocess_ms=0., postprocess_ms=0.,
                                result_transfer_ms=0., detector_overhead_ms=0.)
        return []


class PerformanceTests(unittest.TestCase):
    def make_video(self, root, frames=8):
        path = root / "source.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (160, 120))
        self.assertTrue(writer.isOpened())
        for _ in range(frames):
            writer.write(np.zeros((120, 160, 3), np.uint8))
        writer.release()
        return path

    def test_benchmark_infers_on_all_frames_and_skips_all_drawing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.make_video(root)
            args = parser().parse_args([str(video), "--weights", "unused.pt", "--mode", "detect-benchmark",
                                        "--headless", "--no-video", "--output", str(root/"out")])
            with patch("tracking.detection.YoloDetector", FakeDetector), patch("tracking.app.draw_output") as draw:
                output = run(args)
            draw.assert_not_called()
            perf = json.loads((output/"performance.json").read_text())
            self.assertEqual(perf["detection_calls"], 8)
            self.assertEqual(perf["stages"]["detector_ms"]["count"], 8)
            self.assertEqual(perf["stages"]["draw_ms"]["total_ms"], 0)
            self.assertIn("encoder_finalize_ms", perf["startup_and_finalize_ms"])
            with (output/"timing.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 8)
            self.assertEqual(rows[0]["read_cached"], "True")
            self.assertEqual(rows[1]["read_cached"], "False")

    def test_skipped_frames_have_empty_inference_time_and_correct_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.make_video(root)
            args = parser().parse_args([str(video), "--weights", "unused.pt", "--detect-interval", "3",
                                        "--headless", "--no-video", "--output", str(root/"out")])
            with patch("tracking.detection.YoloDetector", FakeDetector):
                output = run(args)
            perf = json.loads((output/"performance.json").read_text())
            self.assertEqual(perf["detection_calls"], 3)
            self.assertEqual(perf["stages"]["inference_ms"]["count"], 3)
            self.assertEqual(perf["stages"]["inference_ms"]["mean_ms"], 1.)
            with (output/"timing.csv").open(encoding="utf-8-sig", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[1]["inference_ms"], "")

    def test_manual_roi_still_initializes_yolo_tracking(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.make_video(root)
            args = parser().parse_args([str(video), "--weights", "unused.pt", "--roi", "20", "30", "20", "20",
                                        "--headless", "--no-video", "--output", str(root/"out")])
            with patch("tracking.detection.YoloDetector", FakeDetector):
                output = run(args)
            rows = [json.loads(line) for line in (output/"track.jsonl").read_text().splitlines()]
            self.assertEqual(rows[0]["source"], "manual")
            self.assertEqual(rows[0]["center"], [30, 40])
            self.assertTrue(rows[1]["detection_ran"])
            self.assertEqual(rows[1]["state"], "LOST_PENDING")

    def test_progress_reports_processed_frames_each_video_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = self.make_video(root, frames=26)
            args = parser().parse_args([str(video), "--weights", "unused.pt", "--headless", "--no-video",
                                        "--output", str(root/"out")])
            printed = io.StringIO()
            with patch("tracking.detection.YoloDetector", FakeDetector), redirect_stdout(printed):
                run(args)
            output = printed.getvalue()
            self.assertIn("已处理视频 25 帧，模型检测 25 帧", output)
            self.assertIn("视频处理完成：处理帧数=26，模型检测帧数=26", output)
            self.assertNotIn("已处理视频 26 帧", output)

    def test_weights_are_required_without_old_backend(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser().parse_args(["input.mp4"])

    def test_output_resize_preserves_observation_coordinates(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        result = dict(center=[80, 60], bbox=[60, 40, 40, 40], predicted_center=None,
                      segment=1, frame_index=1, timestamp=.04, state="TRACKING", source="test",
                      reason="test", mahalanobis_squared=None)
        image = draw_output(frame, result, [], [], (80, 60))
        self.assertEqual(image.shape, (60, 80, 3))
        self.assertEqual(result["bbox"], [60, 40, 40, 40])

    def test_commands_keep_spaced_paths_and_explicit_backend(self):
        args = parser().parse_args(["a video.mp4", "--weights", "a model.pt", "--backend", "tensorrt"])
        command = command_for(args)
        self.assertIn("a video.mp4", command)
        self.assertEqual(command[command.index("--backend")+1], "tensorrt")

    def test_comparison_counts_missing_boxes_and_rejects_frame_mismatch(self):
        a = dict(frame_index=1, timestamp=.04, detections=[dict(bbox=[0, 0, 20, 20], class_id=0)])
        b = dict(a, detections=[])
        self.assertEqual(compare([a], [b])["unmatched_pt_boxes"], 1)
        self.assertEqual(compare([a], [a])["mean_matched_iou"], 1.)
        with self.assertRaises(ValueError):
            compare([a], [dict(b, frame_index=2)])

    def test_empty_stage_is_not_zero_latency(self):
        self.assertIsNone(stats([None, None])["mean_ms"])
