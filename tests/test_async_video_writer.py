import tempfile
import threading
import unittest
from pathlib import Path

import cv2
import numpy as np

from tracking.video_writer import AsyncVideoWriter


class ControlledWriter:
    """Deterministic worker blocking, without timing a real encoder."""

    def __init__(self, fail_write=False, fail_release=False):
        self.entered = threading.Event()
        self.proceed = threading.Event()
        self.rows = []
        self.released = self.aborted = False
        self.fail_write, self.fail_release = fail_write, fail_release
        self.owners = []

    def write(self, frame):
        self.owners.append(threading.get_ident())
        self.entered.set()
        if not self.proceed.wait(3):
            raise RuntimeError("test worker stalled")
        if self.fail_write:
            raise RuntimeError("broken encoder")
        self.rows.append(frame.copy())

    def release(self):
        self.owners.append(threading.get_ident())
        if self.fail_release:
            raise RuntimeError("invalid output metadata")
        self.released = True

    def cancel_pending(self):
        self.proceed.set()

    def abort(self):
        self.owners.append(threading.get_ident())
        self.aborted = True


class AsyncWriterTests(unittest.TestCase):
    def test_bounded_backpressure_copy_ownership_order_and_drain(self):
        raw = ControlledWriter()
        writer = AsyncVideoWriter(raw, capacity=1)
        self.addCleanup(writer.abort)
        producer = None
        try:
            frame = np.full((2, 4, 3), 10, np.uint8)
            writer.write(frame)
            self.assertTrue(raw.entered.wait(1))
            frame[:] = 20
            writer.write(frame)
            frame[:] = 30
            trying, finished = threading.Event(), threading.Event()
            errors = []

            def submit():
                trying.set()
                try:
                    writer.write(frame)
                except Exception as error:
                    errors.append(error)
                finally:
                    finished.set()

            producer = threading.Thread(target=submit)
            producer.start()
            self.assertTrue(trying.wait(1))
            self.assertFalse(finished.wait(.05))  # Full queue must block, never discard a frame.
            raw.proceed.set()
            self.assertTrue(finished.wait(1))
            producer.join(1)
            self.assertEqual(errors, [])
            frame[:] = 99
            writer.release()
            writer.release()
            self.assertEqual([int(f[0, 0, 0]) for f in raw.rows], [10, 20, 30])
            self.assertTrue(raw.released)
            self.assertFalse(raw.aborted)
            self.assertEqual(len(set(raw.owners)), 1)
            self.assertNotEqual(raw.owners[0], threading.get_ident())
            info = writer.statistics()
            self.assertEqual(info["submitted_frames"], info["completed_frames"])
            self.assertEqual(info["completed_frames"], 3)
            self.assertEqual(info["peak_queue_depth"], 1)
            self.assertEqual(info["background_write_ms"]["count"], 3)
            with self.assertRaisesRegex(RuntimeError, "关闭"):
                writer.write(frame)
        finally:
            raw.proceed.set()
            if producer is not None:
                producer.join(1)

    def test_worker_error_unblocks_a_full_queue_and_is_not_hidden_on_release(self):
        raw = ControlledWriter(fail_write=True)
        writer = AsyncVideoWriter(raw, capacity=1)
        self.addCleanup(writer.abort)
        frame = np.zeros((2, 4, 3), np.uint8)
        writer.write(frame)
        self.assertTrue(raw.entered.wait(1))
        writer.write(frame)
        errors, attempted = [], threading.Event()

        def blocked_submit():
            attempted.set()
            try:
                writer.write(frame)
            except RuntimeError as error:
                errors.append(str(error))

        producer = threading.Thread(target=blocked_submit)
        producer.start()
        try:
            self.assertTrue(attempted.wait(1))
            raw.proceed.set()
            producer.join(1)
            self.assertFalse(producer.is_alive())
            self.assertEqual(len(errors), 1)
            self.assertIn("broken encoder", errors[0])
            with self.assertRaisesRegex(RuntimeError, "broken encoder"):
                writer.release()
            self.assertTrue(raw.aborted)
            self.assertFalse(raw.released)
        finally:
            raw.proceed.set()
            producer.join(1)

    def test_encoder_finalization_error_reaches_caller(self):
        raw = ControlledWriter(fail_release=True)
        writer = AsyncVideoWriter(raw)
        with self.assertRaisesRegex(RuntimeError, "invalid output metadata"):
            writer.release()
        self.assertTrue(raw.aborted)
        writer.abort()

    def test_release_waits_for_pending_output(self):
        raw = ControlledWriter()
        writer = AsyncVideoWriter(raw)
        self.addCleanup(writer.abort)
        writer.write(np.zeros((2, 4, 3), np.uint8))
        self.assertTrue(raw.entered.wait(1))
        finished = threading.Event()
        errors = []

        def finish():
            try:
                writer.release()
            except Exception as error:
                errors.append(error)
            finally:
                finished.set()

        thread = threading.Thread(target=finish)
        thread.start()
        try:
            self.assertFalse(finished.wait(.05))
            self.assertFalse(raw.released)
            raw.proceed.set()
            self.assertTrue(finished.wait(1))
            thread.join(1)
            self.assertEqual(errors, [])
            self.assertTrue(raw.released)
            self.assertEqual(writer.statistics()["completed_frames"], 1)
        finally:
            raw.proceed.set()
            thread.join(1)

    def test_full_queue_has_a_bounded_wait_and_can_be_cancelled(self):
        raw = ControlledWriter()
        writer = AsyncVideoWriter(raw, capacity=1, timeout=.05)
        frame = np.zeros((2, 4, 3), np.uint8)
        try:
            writer.write(frame)
            self.assertTrue(raw.entered.wait(1))
            writer.write(frame)
            with self.assertRaisesRegex(RuntimeError, "队列超时"):
                writer.write(frame)
        finally:
            writer.abort()
        self.assertFalse(writer._thread.is_alive())
        self.assertTrue(raw.aborted)

    def test_abort_cancels_pending_frames_and_closes_only_on_worker(self):
        raw = ControlledWriter()
        writer = AsyncVideoWriter(raw, capacity=2)
        writer.write(np.zeros((2, 4, 3), np.uint8))
        self.assertTrue(raw.entered.wait(1))
        writer.write(np.ones((2, 4, 3), np.uint8))
        writer.abort()
        writer.abort()
        self.assertFalse(writer._thread.is_alive())
        self.assertTrue(raw.aborted)
        self.assertFalse(raw.released)
        self.assertEqual(len(raw.rows), 1)
        self.assertEqual(len(set(raw.owners)), 1)
        with self.assertRaisesRegex(RuntimeError, "取消"):
            writer.write(np.zeros((2, 4, 3), np.uint8))

    def test_real_opencv_output_preserves_all_frames_and_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "out.avi"
            raw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 25, (64, 48))
            self.assertTrue(raw.isOpened())
            writer = AsyncVideoWriter(raw, capacity=2)
            values = list(range(10, 231, 10))
            try:
                canvas = np.empty((48, 64, 3), np.uint8)
                for value in values:
                    canvas[:] = value
                    writer.write(canvas)
                writer.release()
            finally:
                writer.abort()
            capture = cv2.VideoCapture(str(path))
            decoded = []
            try:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    decoded.append(frame.mean())
            finally:
                capture.release()
            np.testing.assert_allclose(decoded, values, atol=3)


if __name__ == "__main__":
    unittest.main()
