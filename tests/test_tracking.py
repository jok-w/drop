import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from tracking.app import VideoClock, config_for_frame, parser
from tracking.core import Config, MotionFilter, SingleTargetTracker
from tracking.appearance import ContrastLocator


ROOT = Path(__file__).resolve().parents[1]


def synthetic_video(path, frames=40):
    """Known visible motion followed by full disappearance; deterministic pixels."""
    rng = np.random.default_rng(7)
    background = rng.integers(20, 65, (240, 320, 3), dtype=np.uint8)
    patch = rng.integers(90, 255, (32, 32, 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (320, 240))
    if not writer.isOpened():
        raise RuntimeError("Synthetic video codec unavailable")
    truth = []
    try:
        for i in range(frames):
            frame = background.copy()
            if i < 25:
                x, y = 40 + i * 2, 100
                frame[y:y + 32, x:x + 32] = patch
                truth.append([x + 16, y + 16])
            else:
                truth.append(None)
            writer.write(frame)
    finally:
        writer.release()
    return truth


class MotionTests(unittest.TestCase):
    def test_4k_startup_and_turning_candidates_are_not_rejected(self):
        # Recorded image candidates from the user's failing frame 100 onwards.
        boxes = [(1620,1656,76,84), (1642,1640,76,84), (1648,1605,78,86),
                 (1663,1572,79,87), (1667,1550,82,91), (1670,1544,81,89),
                 (1679,1544,78,86), (1685,1548,73,81), (1682,1542,75,82)]
        times = [3.995,4.035,4.075,4.115,4.154,4.194,4.234,4.274,4.314]

        class RecordedTracker:
            def init(self, frame, box):
                self.index = 0

            def update(self, frame):
                self.index += 1
                return True, boxes[self.index]

        frame = np.zeros((2160,3840,3), dtype=np.uint8)
        config = config_for_frame(parser().parse_args(["example.mp4"]), frame.shape)
        self.assertEqual(config.initial_velocity_std, 600)
        tracker = SingleTargetTracker(config, RecordedTracker)
        tracker.initialize(frame, boxes[0], times[0], 100, "test")
        for i, stamp in enumerate(times[1:], 1):
            result = tracker.update(frame, stamp, 100+i, "test")
            self.assertEqual(result["state"], "TRACKING", result)

    def test_explicit_noise_parameters_override_resolution_defaults(self):
        args = parser().parse_args(["example.mp4", "--measurement-std", "3",
                                   "--acceleration-std", "700", "--initial-velocity-std", "150"])
        config = config_for_frame(args, (2160,3840,3))
        self.assertEqual((config.measurement_std, config.acceleration_std,
                          config.initial_velocity_std), (3,700,150))

    def test_retry_uses_last_trusted_frame_and_can_recover(self):
        calls, initialized = [], []

        class BrieflyWrongTracker:
            def init(self, frame, box):
                initialized.append((int(frame[0,0,0]), tuple(box)))

            def update(self, frame):
                calls.append(1)
                return True, (220,180,20,20) if len(calls)==1 else (14,10,20,20)

        tracker = SingleTargetTracker(tracker_factory=BrieflyWrongTracker)
        initial = np.zeros((240,320,3), dtype=np.uint8)
        tracker.initialize(initial, (10,10,20,20), 0, 0, "test")
        rejected = tracker.update(initial + 30, .04, 1, "test")
        self.assertIsNone(rejected["center"])
        recovered = tracker.update(initial + 60, .08, 2, "test")
        self.assertEqual(recovered["reason"], "recovered")
        self.assertEqual(recovered["center"], [24,20])
        self.assertEqual(initialized, [(0,(10,10,20,20)), (0,(10,10,20,20))])

    def test_anisotropic_gate(self):
        kf = MotionFilter((100, 200), Config(measurement_std=1))
        kf.P = np.diag([99., 24., 1., 1.])
        self.assertAlmostEqual(kf.distance_squared((120, 205)), 5)
        self.assertAlmostEqual(kf.distance_squared((105, 220)), 16.25)

    def test_constant_velocity_with_irregular_timestamps(self):
        kf = MotionFilter((0, 0), Config())
        time = 0
        for dt in [.04, .08, .12, .04] * 10:
            time += dt
            kf.predict(dt)
            kf.correct((50 * time, 20 * time))
        np.testing.assert_allclose(kf.x, [50 * time, 20 * time, 50, 20], atol=.1)
        self.assertGreaterEqual(np.linalg.eigvalsh(kf.P).min(), -1e-9)

    def test_rejected_measurement_does_not_correct_and_can_reinitialize(self):
        class WrongTracker:
            def init(self, frame, box):
                pass

            def update(self, frame):
                return True, (220, 180, 20, 20)

        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        tracker = SingleTargetTracker(tracker_factory=WrongTracker)
        tracker.initialize(frame, (10, 10, 20, 20), 0, 0, "test")
        result = tracker.update(frame, .04, 1, "test")
        self.assertEqual(result["reason"], "motion_gate")
        self.assertIsNone(result["center"])
        np.testing.assert_allclose(tracker.motion.x[:2], [20, 20])
        self.assertIsNone(tracker.image_tracker)
        result = tracker.update(frame, .60, 2, "test")
        self.assertEqual(result["state"], "LOST")
        self.assertIsNone(result["predicted_center"])
        result = tracker.initialize(frame, (100, 100, 20, 20), .60, 2, "test")
        self.assertEqual(result["segment"], 2)
        self.assertEqual(result["center"], [110, 110])

    def test_timestamp_fallback_is_monotonic(self):
        clock = VideoClock(25)
        stamps = [clock.get(ms, i) for i, ms in enumerate([0, 40, 40, 0, 160])]
        np.testing.assert_allclose([s[0] for s in stamps], [0, .04, .08, .12, .16])
        self.assertEqual(stamps[2][1], "fps_fallback")


class IntegrationTests(unittest.TestCase):
    def test_real_csrt_cli_exports_and_disappearance(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            video, output = folder / "input.avi", folder / "result"
            truth = synthetic_video(video)
            command = [sys.executable, str(ROOT / "track_video.py"), str(video),
                       "--headless", "--appearance", "off", "--roi", "40", "100", "32", "32", "--output", str(output)]
            proc = subprocess.run(command, capture_output=True, text=True, timeout=90)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rows = [json.loads(line) for line in (output / "track.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 40)
            visible = [r for r in rows[:25] if r["center"] is not None]
            self.assertGreaterEqual(len(visible), 23)
            errors = [np.linalg.norm(np.array(r["center"]) - truth[r["frame_index"]]) for r in visible]
            self.assertLess(float(np.mean(errors)), 5.0)
            self.assertIsNone(rows[-1]["center"])
            self.assertIn(rows[-1]["state"], ["LOST", "LOST_PENDING"])
            with (output / "track.csv").open(encoding="utf-8-sig", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 40)
            cap = cv2.VideoCapture(str(output / "annotated.mp4"))
            decoded = 0
            while cap.read()[0]:
                decoded += 1
            cap.release()
            self.assertEqual(decoded, 40)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["frames"], 40)
            # Repeating an output path must not overwrite an earlier run.
            before = (output / "track.jsonl").read_bytes()
            repeat = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeat.returncode, 0)
            self.assertEqual(before, (output / "track.jsonl").read_bytes())


class AppearanceTests(unittest.TestCase):
    @staticmethod
    def frame(center=None, angle=0):
        frame = np.full((240,320,3),180,dtype=np.uint8)
        if center is not None:
            cv2.ellipse(frame,center,(12,8),angle,0,360,(20,20,20),-1)
        return frame

    def test_rotation_direction_change_and_disappearance(self):
        centers=[(80,140),(98,125),(117,105),(116,95),(100,100),(90,115)]
        tracker=SingleTargetTracker()
        tracker.initialize(self.frame(centers[0]),(64,124,32,32),0,0,"test")
        self.assertTrue(tracker.appearance.enabled)
        reasons=[]
        for i,center in enumerate(centers[1:],1):
            result=tracker.update(self.frame(center,i*35),i*.04,i,"test")
            self.assertIsNotNone(result["center"],result)
            self.assertLess(np.linalg.norm(np.asarray(result["center"])-center),3)
            self.assertEqual(result["source"],"local_contrast")
            reasons.append(result["reason"])
        self.assertIn("motion_reset",reasons)
        for i in range(6,21):
            result=tracker.update(self.frame(),i*.04,i,"test")
            self.assertIsNone(result["center"])
        self.assertEqual(result["state"],"LOST")

    def test_short_occlusion_recovers_but_timeout_does_not(self):
        tracker=SingleTargetTracker()
        tracker.initialize(self.frame((80,140)),(64,124,32,32),0,0,"test")
        result=tracker.update(self.frame(),.04,1,"test")
        self.assertIsNone(result["center"])
        result=tracker.update(self.frame((82,140)),.08,2,"test")
        self.assertEqual(result["reason"],"appearance_recovered")
        tracker.update(self.frame(),.12,3,"test")
        result=tracker.update(self.frame((82,140)),.70,4,"test")
        self.assertEqual(result["state"],"LOST")
        self.assertIsNone(result["center"])

    def test_similar_candidates_are_ambiguous(self):
        locator=ContrastLocator(self.frame((100,100)),(84,84,32,32))
        frame=self.frame((70,100))
        cv2.ellipse(frame,(130,100),(12,8),0,0,360,(20,20,20),-1)
        box,details=locator.locate(frame,(100,100),(84,84,32,32),np.eye(2)*16)
        self.assertIsNone(box)
        self.assertEqual(details["appearance_reason"],"ambiguous")

    def test_low_contrast_roi_keeps_csrt_backend(self):
        tracker=SingleTargetTracker()
        tracker.initialize(self.frame(),(64,124,32,32),0,0,"test")
        self.assertFalse(tracker.appearance.enabled)


if __name__ == "__main__":
    unittest.main()
