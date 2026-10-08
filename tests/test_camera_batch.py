import io
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

from tracking.camera_batch_app import parser, run, validate


class FakeStore:
    def counts(self):
        return {0: 1}

    def close(self):
        pass


class FakeIngest:
    def __init__(self, camera, output, args):
        self.frames = queue.Queue()
        self.frames.put((np.zeros((120, 160, 3), np.uint8), dict(
            session_id="session_00001", source_size=[160, 120], pts_ns=40_000_000,
            index=0, received_monotonic=time.monotonic())))
        self.finished = threading.Event()
        self.finished.set()
        self.store = FakeStore()
        self.failure = None
        self.sessions = self.closed_sessions = 1
        self.reconnects = 0
        self.archived = 1
        self.decoded_frames = 1

    def start(self):
        pass

    def close(self):
        pass


class FakeBatchDetector:
    instance = None

    def __init__(self, *args, **kwargs):
        self.batch_size = kwargs["batch_size"]
        self.metadata = {"backend": "fake", "batch_size": self.batch_size}
        self.last_timing = {}
        self.warmups = []
        self.distinct_inputs = None
        FakeBatchDetector.instance = self

    def warmup_batch(self, frames, count):
        self.warmups.append((len(frames), [frame.shape for frame in frames], count))

    def detect_batch(self, frames):
        self.distinct_inputs = len({id(frame) for frame in frames})
        self.last_timing = dict(detector_ms=20., preprocess_ms=2., inference_ms=6.,
                                postprocess_ms=1., result_transfer_ms=1., detector_overhead_ms=1.)
        return [[dict(bbox=[10, 20, 8, 12], confidence=.9, class_id=0)]
                for _ in frames]


class CameraBatchTests(unittest.TestCase):
    def test_requires_live_latest(self):
        self.assertEqual(parser().parse_args([]).seconds, 60.)
        self.assertEqual(parser().parse_args(["--batch-size", "6"]).batch_size, 6)
        with self.assertRaisesRegex(ValueError, "latest"):
            validate(parser().parse_args(["--policy", "complete"]))
        with self.assertRaisesRegex(ValueError, "engine"):
            validate(parser().parse_args(["--backend", "pt", "--engine", "x.engine"]))

    def test_one_source_frame_produces_independent_logical_results(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "out"
            args = parser().parse_args([
                "--backend", "pt", "--batch-size", "2", "--warmup", "3",
                "--output", str(output), "--seconds", "1",
            ])
            with patch("tracking.detection.YoloDetector", FakeBatchDetector), \
                    patch("tracking.camera_batch_app.CameraIngest", FakeIngest), \
                    redirect_stdout(io.StringIO()):
                result = run(args)
            self.assertEqual(result, output)
            detector = FakeBatchDetector.instance
            self.assertEqual(detector.warmups, [(2, [(640, 640, 3), (640, 640, 3)], 3)])
            self.assertEqual(detector.distinct_inputs, 1)
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["physical_frames"], 1)
            self.assertEqual(summary["logical_frames"], 2)
            self.assertEqual(summary["batch_calls"], 1)
            self.assertEqual(summary["timings"]["batch_detector_ms"]["mean_ms"], 20.)
            self.assertEqual(set(summary["logical_cameras"]), {"virtual_01", "virtual_02"})
            records = [json.loads(line) for line in (output / "track.jsonl").read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual({record["camera_id"] for record in records},
                             {"virtual_01", "virtual_02"})
            self.assertEqual({record["physical_frame_index"] for record in records}, {0})


if __name__ == "__main__":
    unittest.main()
