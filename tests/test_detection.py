import unittest
import math

import numpy as np

from tracking.core import Config
from tracking.detection import DetectionTracker
from tracking.app import csv_row


def detection(x=40, confidence=.8, class_id=0):
    return dict(bbox=[x, 80, 20, 20], confidence=confidence, class_id=class_id)


class FakeDetector:
    def __init__(self, batches):
        self.batches = iter(batches)
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return next(self.batches)


class DetectionTests(unittest.TestCase):
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    def test_sparse_detection_exports_predictions_separately(self):
        detector = FakeDetector([[detection()], [detection(46)]])
        tracker = DetectionTracker(detector, interval=3)
        first = tracker.update(self.frame, 0., 0, "test")
        self.assertEqual(first["source"], "yolo")
        for i in (1, 2):
            row = tracker.update(self.frame, i*.04, i, "test")
            self.assertEqual(row["state"], "PREDICTED")
            self.assertIsNone(row["center"])
            self.assertIsNotNone(row["estimated_center"])
            self.assertIsNone(csv_row(row)["center_x"])
            self.assertFalse(row["detection_ran"])
        row = tracker.update(self.frame, .12, 3, "test")
        self.assertEqual(row["center"], [56, 90])
        self.assertEqual(detector.calls, 2)
        rate = tracker.config.deceleration_rate
        expected = 6 * math.exp(-rate * .12) * rate / (1 - math.exp(-rate * .12))
        np.testing.assert_allclose(tracker.motion.x[2:], [expected, 0])

    def test_ambiguous_start_waits_for_unique_detection(self):
        tracker = DetectionTracker(FakeDetector([[detection(), detection(80)], [detection()]]))
        result = tracker.update(self.frame, 0., 0, "test")
        self.assertEqual(result["state"], "WAITING")
        self.assertIsNone(result["center"])
        self.assertEqual(tracker.update(self.frame, .04, 1, "test")["segment"], 1)

    def test_ambiguous_match_does_not_corrupt_filter(self):
        tracker = DetectionTracker(FakeDetector([[detection()], [detection(38), detection(42)]]))
        tracker.update(self.frame, 0., 0, "test")
        result = tracker.update(self.frame, .04, 1, "test")
        self.assertEqual(result["reason"], "ambiguous_detections")
        self.assertIsNone(result["center"])
        np.testing.assert_allclose(tracker.motion.x, [50, 90, 0, 0])

    def test_class_lock_and_distant_distractor_are_rejected(self):
        tracker = DetectionTracker(FakeDetector([[detection()], [detection(42)],
                                                [detection(class_id=1), detection(280)]]), strict_motion_gate=True)
        tracker.update(self.frame, 0., 0, "test")
        tracker.update(self.frame, .04, 1, "test")
        result = tracker.update(self.frame, .08, 2, "test")
        self.assertEqual(result["reason"], "association_rejected")
        self.assertIsNone(result["center"])

    def test_missing_recovers_then_reinitializes_after_timeout(self):
        detector = FakeDetector([[detection()], [], [detection(42)],
                                 [detection(class_id=1), detection(80)], [detection(82)]])
        tracker = DetectionTracker(detector, Config(coast_seconds=.2))
        tracker.update(self.frame, 0., 0, "test")
        self.assertEqual(tracker.update(self.frame, .04, 1, "test")["state"], "LOST_PENDING")
        self.assertEqual(tracker.update(self.frame, .08, 2, "test")["state"], "TRACKING")
        result = tracker.update(self.frame, .3, 3, "test")
        self.assertEqual(result["state"], "LOST")
        self.assertIsNone(result["estimated_center"])
        self.assertEqual(detector.calls, 3)
        result = tracker.update(self.frame, .34, 4, "test")
        self.assertEqual(result["segment"], 2)
        self.assertEqual(result["reason"], "detected_reinitialization")
        self.assertEqual(result["center"], [90, 90])
        self.assertTrue(result["detection_ran"])
        self.assertEqual(tracker.class_id, 0)
        self.assertEqual(tracker.update(self.frame, .38, 5, "test")["segment"], 2)

    def test_bad_interval_and_timestamp_fail(self):
        with self.assertRaises(ValueError):
            DetectionTracker(FakeDetector([]), interval=0)
        tracker = DetectionTracker(FakeDetector([[]]))
        tracker.update(self.frame, 0., 0, "test")
        with self.assertRaises(ValueError):
            tracker.update(self.frame, 0., 1, "test")

    def test_two_observations_initialize_fast_motion_without_zero_speed_gate(self):
        tracker = DetectionTracker(FakeDetector([[detection(20)], [detection(80)], [detection(140)]]))
        tracker.update(self.frame, 0., 0, "test")
        result = tracker.update(self.frame, .04, 1, "test")
        self.assertEqual(result["state"], "TRACKING")
        dt = .04
        rate = tracker.config.deceleration_rate
        expected = 60 * math.exp(-rate * dt) * rate / (1 - math.exp(-rate * dt))
        np.testing.assert_allclose(tracker.motion.x[2:], [expected, 0])
        self.assertGreaterEqual(np.linalg.eigvalsh(tracker.motion.P).min(), 0)
        self.assertEqual(tracker.update(self.frame, .08, 2, "test")["state"], "TRACKING")

    def test_unique_observation_survives_bad_motion_prior_with_diagnostic(self):
        tracker = DetectionTracker(FakeDetector([[detection()], [detection(42)], [detection(200)]]))
        tracker.update(self.frame, 0., 0, "test")
        tracker.update(self.frame, .04, 1, "test")
        result = tracker.update(self.frame, .08, 2, "test")
        self.assertEqual(result["reason"], "detected_motion_disagreement")
        self.assertEqual(result["center"], [210, 90])
