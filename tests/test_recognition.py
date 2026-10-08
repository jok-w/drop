import unittest

import numpy as np

from tracking.recognition import FramePacket, RecognitionEngine, TrackingConfig


class BatchDetector:
    batch_size = 2
    metadata = {"backend": "fake"}

    def __init__(self):
        self.warmup_shapes = None
        self.input_ids = None

    def warmup_batch(self, frames, count):
        self.warmup_shapes = ([frame.shape for frame in frames], count)

    def detect_batch(self, frames):
        self.input_ids = [id(frame) for frame in frames]
        return [[dict(bbox=[10, 20, 8, 12], confidence=.9, class_id=0)]
                for _ in frames]


def packet(camera_id, frame, index=0, session="one", pts_ns=40_000_000):
    return FramePacket(
        camera_id=camera_id,
        frame=frame,
        frame_index=index,
        source_frame_index=index,
        pts_ns=pts_ns,
        session_id=session,
        source_size=(160, 120),
        received_monotonic=1.0,
    )


class RecognitionTests(unittest.TestCase):
    def test_fixed_batch_routes_results_to_independent_camera_state(self):
        detector = BatchDetector()
        engine = RecognitionEngine(detector, TrackingConfig())
        engine.warmup((640, 640, 3), 3)
        self.assertEqual(detector.warmup_shapes,
                         ([(640, 640, 3), (640, 640, 3)], 3))

        first = np.zeros((120, 160, 3), np.uint8)
        second = np.ones((120, 160, 3), np.uint8)
        results = engine.process_batch([packet("a", first), packet("b", second)])
        self.assertEqual([result["camera_id"] for result in results], ["a", "b"])
        self.assertEqual([result["segment"] for result in results], [1, 1])
        self.assertEqual(detector.input_ids, [id(first), id(second)])

        results = engine.process_batch([
            packet("a", first, 1, pts_ns=80_000_000),
            packet("b", second, 1, pts_ns=80_000_000),
        ])
        self.assertEqual([result["segment"] for result in results], [1, 1])
        engine.finish("system_stop")
        self.assertEqual({segment["camera_id"] for segment in engine.segments}, {"a", "b"})
        self.assertTrue(all(segment["end_reason"] == "system_stop"
                            for segment in engine.segments))

    def test_batch_contract_rejects_wrong_count_and_duplicate_camera(self):
        engine = RecognitionEngine(BatchDetector(), TrackingConfig())
        frame = np.zeros((120, 160, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "batch=2"):
            engine.process_batch([packet("a", frame)])
        with self.assertRaisesRegex(ValueError, "重复 camera_id"):
            engine.process_batch([packet("a", frame), packet("a", frame)])

    def test_packet_rejects_invalid_source_metadata(self):
        frame = np.zeros((120, 160, 3), np.uint8)
        with self.assertRaisesRegex(ValueError, "camera_id"):
            packet("", frame)
        with self.assertRaisesRegex(ValueError, "source_size"):
            FramePacket("a", frame, 0, 0, 0, "one", (0, 120), 1.0)


if __name__ == "__main__":
    unittest.main()
