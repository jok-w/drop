import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch

import numpy as np

from tracking.camera_app import parser, run, validate
from tracking.camera_config import CameraConfig, IngestConfig
from tracking.recognition import FramePacket, StreamTracker, TrackingConfig, scale_detections
from tracking.rtsp_reader import ArchiveReplay, CameraIngest, JournalTail, SegmentStore


class Detector:
    metadata = {'backend': 'test'}
    last_timing = {}

    def __init__(self):
        self.warmup_calls = []

    def warmup(self, frame, count):
        self.warmup_calls.append((frame.shape, count))

    def detect(self, frame):
        return [dict(bbox=[10, 20, 8, 12], confidence=.9, class_id=0)]


class CameraTests(unittest.TestCase):
    def test_credentials_are_not_in_public_configuration_or_repr(self):
        c = CameraConfig(password='sensitive@:/value')
        self.assertEqual(c.url, 'rtsp://192.168.1.64:554/Streaming/Channels/101')
        self.assertNotIn(c.password, repr(c))
        self.assertNotIn(c.password, json.dumps(c.public()))
        self.assertEqual(CameraConfig(channel=2, stream='sub').url.split('/')[-1], '202')
        with self.assertRaises(ValueError):
            CameraConfig(host='rtsp://user:password@camera')

    def test_password_file_preserves_spaces_and_strips_only_newline(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'password'
            path.write_text(' a b \n')
            c = CameraConfig.from_args(parser().parse_args(['--password-file', str(path)]))
            self.assertEqual(c.password, ' a b ')

    def test_invalid_durations_and_unsupported_scaling_are_rejected(self):
        for values in [['--seconds', 'nan'], ['--segment-seconds', '0'],
                       ['--decode-width','640'], ['--policy','latest','--decode-width','639']]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                validate(parser().parse_args(values))

    def test_scaled_observations_return_to_source_coordinates(self):
        detections = Detector().detect(np.zeros((360,640,3),np.uint8))
        result = scale_detections(detections, (640, 360), (3840, 2160))
        self.assertEqual(result[0]['bbox'], [60,120,48,72])
        self.assertEqual(detections[0]['bbox'], [10,20,8,12])

    def test_session_change_ends_track_and_uses_fresh_motion_with_unique_id(self):
        tracker = StreamTracker(TrackingConfig.from_args(parser().parse_args([])))
        detector = Detector()
        frame = np.zeros((120,160,3),np.uint8)
        meta = dict(session_id='a', source_size=[160,120], pts_ns=1_000_000_000,index=0)
        packet = lambda values, index: FramePacket.from_meta('camera', frame, index, dict(
            values, received_monotonic=0.))
        first = tracker.update(packet(meta, 0), detector.detect(frame))
        self.assertEqual(first['segment'],1)
        second_meta = dict(meta,index=3,pts_ns=1_120_000_000)
        second = tracker.update(packet(second_meta, 1), detector.detect(frame))
        self.assertEqual(second['skipped_before'],2)
        third_meta = dict(meta,session_id='b',pts_ns=0)
        third = tracker.update(packet(third_meta, 2), detector.detect(frame))
        self.assertEqual(third['segment'],2)
        self.assertIsNone(third['predicted_center'])
        self.assertEqual(tracker.segments[0]['end_reason'],'stream_disconnect')
        self.assertEqual(tracker.segments[0]['session_id'],'a')
        tracker.finish('stop')
        self.assertEqual(len(tracker.segments),2)

    def test_partial_event_write_is_not_lost_or_parsed_early(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'events'
            tail=JournalTail(path)
            self.assertEqual(tail.read(),[])
            path.write_text('{"event":"sta')
            self.assertEqual(tail.read(),[])
            with path.open('a') as f:
                f.write('rt"}\n{"event":"end"}\n')
            self.assertEqual(tail.read(),[{'event':'start'},{'event':'end'}])
            self.assertEqual(tail.read(),[])

    def test_fragment_queue_is_durable_ordered_and_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.db'
            store=SegmentStore(path)
            store.add(dict(path='a',start_ns=0));store.add(dict(path='a',start_ns=0))
            store.add(dict(path='b',start_ns=10))
            self.assertEqual(store.counts(),{0:2})
            item=store.next();self.assertEqual(item['path'],'a')
            store.done(item['database_id']);store.close()
            store=SegmentStore(path)
            self.assertEqual(store.next()['path'],'b')
            self.assertEqual(store.counts(),{0:1,1:1})
            store.close()

    def test_replay_into_new_results_preserves_tracks_across_fragment_boundary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            capture = root/'capture'; capture.mkdir()
            store = SegmentStore(capture/'fragments.sqlite3')
            for i in range(2):
                path = capture/f'{i}.mp4'; path.touch()
                store.add(dict(path=str(path),session_id='one',source_size=[160,120],
                               start_ns=i*120_000_000,end_ns=(i+1)*120_000_000,
                               closed_monotonic=0))
            store.close()

            class Capture:
                def __init__(self):
                    self.index = 0
                def get(self, prop):
                    import cv2
                    return 25 if prop == cv2.CAP_PROP_FPS else (self.index-1)*40
                def read(self):
                    self.index += 1
                    return (True,np.zeros((120,160,3),np.uint8)) if self.index <= 3 else (False,None)
                def release(self):
                    pass

            output=root/'result'
            args=parser().parse_args(['--replay',str(capture),'--output',str(output)])
            detector = Detector()
            with patch('tracking.detection.YoloDetector', return_value=detector), \
                    patch('tracking.camera_app.open_fragment', side_effect=lambda *a: Capture()), \
                    patch.object(ArchiveReplay, 'start',
                                 side_effect=lambda: self.assertTrue(detector.warmup_calls)), \
                    redirect_stdout(io.StringIO()):
                run(args)
            self.assertEqual(detector.warmup_calls, [((640, 640, 3), 5)])
            records=[json.loads(s) for s in (output/'track.jsonl').read_text().splitlines()]
            self.assertEqual(len(records),6)
            self.assertEqual([r['source_frame_index'] for r in records],list(range(6)))
            self.assertEqual([r['timestamp'] for r in records],[0,.04,.08,.12,.16,.2])
            self.assertEqual({r['segment'] for r in records},{1})
            summary=json.loads((output/'summary.json').read_text())
            self.assertEqual(summary['analyzed_fragments'],2)
            self.assertEqual(summary['pending_fragments'],0)
            self.assertEqual(summary['segments'][0]['end_reason'],'capture_finished')
            self.assertIsNone(summary['camera'])
            original=SegmentStore(capture/'fragments.sqlite3')
            self.assertEqual(original.counts(),{0:2})
            original.close()

    def test_replay_rejects_missing_video_without_touching_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); source=root/'source'; source.mkdir(); output=root/'out';output.mkdir()
            store=SegmentStore(source/'fragments.sqlite3')
            store.add(dict(path=str(root/'missing.mp4'),session_id='a'))
            store.close()
            with self.assertRaisesRegex(ValueError,'不存在'):
                ArchiveReplay(source,output)

    def test_supervisor_reconnects_then_stops_on_disk_failure_without_camera(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            args=parser().parse_args(['--seconds','5'])
            calls=[]

            class FakeWorker:
                def __init__(self, camera, path, args):
                    calls.append(path.name)
                    path.mkdir()
                    self.process=self
                    self.returncode=None
                    self.ticks=0
                    self.tail=self
                    self.path=path/'events.jsonl';self.path.touch()
                    self.batch=[dict(event='stream',source_size=[160,120]),
                        dict(event='fragment_opened',path=str(path/'raw.mp4'),running_ns=0),
                        dict(event='fragment_closed',path=str(path/'raw.mp4'),running_ns=100,
                             monotonic=0),
                        dict(event='stopped',decoded_frames=0)]
                    if len(calls)==2:
                        self.batch.append(dict(event='error',code='disk_space_low'))
                def poll(self):
                    self.ticks+=1
                    return None if self.ticks==1 else 1
                def read(self):
                    batch,self.batch=self.batch,[]
                    return batch
                def request_stop(self):
                    pass
                def close(self):
                    self.returncode=1

            ingest=CameraIngest(CameraConfig(),root,IngestConfig.from_args(args))
            with patch('tracking.rtsp_reader.Worker',FakeWorker):
                ingest.start()
                self.assertTrue(ingest.finished.wait(4))
                ingest.close()
            self.assertEqual(ingest.sessions,2)
            self.assertEqual(ingest.reconnects,1)
            self.assertEqual(ingest.failure,'disk_space_low')
            self.assertEqual(ingest.store.counts(),{0:2})
            ingest.store.close()


if __name__ == '__main__':
    unittest.main()
