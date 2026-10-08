"""Reusable frame-to-track pipeline, independent from CLI and result files."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .core import Config
from .detection import DetectionTracker


@dataclass(frozen=True, slots=True)
class FramePacket:
    """One decoded frame and the source identity needed by a recognition system."""

    camera_id: str
    frame: np.ndarray
    frame_index: int
    source_frame_index: int
    pts_ns: int
    session_id: str
    source_size: tuple[int, int]
    received_monotonic: float

    def __post_init__(self):
        if not self.camera_id or not self.session_id:
            raise ValueError("camera_id and session_id are required")
        if not isinstance(self.frame, np.ndarray) or self.frame.ndim != 3:
            raise ValueError("frame must be an HWC numpy image")
        if self.frame_index < 0 or self.source_frame_index < 0 or self.pts_ns < 0:
            raise ValueError("frame indices and pts_ns must be non-negative")
        width, height = self.source_size
        if width < 2 or height < 2:
            raise ValueError("source_size must contain positive width and height")
        if not math.isfinite(self.received_monotonic):
            raise ValueError("received_monotonic must be finite")

    @classmethod
    def from_meta(cls, camera_id: str, frame: np.ndarray, frame_index: int,
                  meta: Mapping[str, Any]):
        return cls(
            camera_id=camera_id,
            frame=frame,
            frame_index=frame_index,
            source_frame_index=int(meta["index"]),
            pts_ns=int(meta["pts_ns"]),
            session_id=str(meta["session_id"]),
            source_size=tuple(meta["source_size"]),
            received_monotonic=float(meta["received_monotonic"]),
        )


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    strict_motion_gate: bool = False
    gate: float = 5.991
    measurement_std: float | None = None
    acceleration_std: float | None = None
    initial_velocity_std: float | None = None
    deceleration_rate: float = 2.0
    coast_seconds: float = 0.5

    @classmethod
    def from_args(cls, args):
        return cls(
            strict_motion_gate=args.strict_motion_gate,
            gate=args.gate,
            measurement_std=args.measurement_std,
            acceleration_std=args.acceleration_std,
            initial_velocity_std=args.initial_velocity_std,
            deceleration_rate=args.deceleration_rate,
            coast_seconds=args.coast_seconds,
        )

    def for_source(self, source_size: tuple[int, int]) -> Config:
        scale = max(1.0, max(source_size) / 1280.0)
        return Config(
            gate=self.gate,
            measurement_std=self.measurement_std if self.measurement_std is not None else 4.0 * scale,
            acceleration_std=self.acceleration_std if self.acceleration_std is not None else 800.0 * scale,
            deceleration_rate=self.deceleration_rate,
            initial_velocity_std=(self.initial_velocity_std
                                  if self.initial_velocity_std is not None else 200.0 * scale),
            coast_seconds=self.coast_seconds,
        )


def scale_detections(detections: Sequence[Mapping[str, Any]], processing_size: tuple[int, int],
                     source_size: tuple[int, int]):
    """Return source-coordinate detections without mutating detector output."""
    processing_width, processing_height = processing_size
    source_width, source_height = source_size
    if min(processing_width, processing_height, source_width, source_height) < 2:
        raise ValueError("processing and source sizes must be positive")
    sx, sy = source_width / processing_width, source_height / processing_height
    scaled = []
    for detection in detections:
        item = dict(detection)
        x, y, width, height = item["bbox"]
        left, top = max(0, round(x * sx)), max(0, round(y * sy))
        right = min(source_width, round((x + width) * sx))
        bottom = min(source_height, round((y + height) * sy))
        if right <= left or bottom <= top:
            continue
        item["bbox"] = [left, top, right - left, bottom - top]
        scaled.append(item)
    return scaled


class StreamTracker:
    """Maintain independent state across frames and reset cleanly across sessions."""

    def __init__(self, config: TrackingConfig):
        self.config = config
        self.tracker = None
        self.session = self.size = None
        self.segments = []
        self.last_index = self.last_stamp = None
        self.offset = 0
        self.last_source_index = -1

    def finish(self, reason):
        if self.tracker is not None:
            self.tracker.finish_segment(self.last_index, self.last_stamp, reason)
            self.segments.extend(dict(segment, session_id=self.session)
                                 for segment in self.tracker.segments)
            self.offset = self.tracker.segment
            self.tracker = None

    def update(self, packet: FramePacket, detections):
        session, size = packet.session_id, packet.source_size
        stamp = packet.pts_ns / 1e9
        if (self.session != session or self.size != size
                or (self.last_stamp is not None and stamp <= self.last_stamp)):
            self.finish("stream_disconnect" if self.session != session else "stream_discontinuity")
            self.session, self.size = session, size
            self.last_source_index = -1
        if self.tracker is None:
            self.tracker = DetectionTracker(
                None,
                self.config.for_source(size),
                interval=1,
                strict_motion_gate=self.config.strict_motion_gate,
                continuous=True,
            )
            self.tracker.segment = self.offset
        source_detections = scale_detections(
            detections,
            (packet.frame.shape[1], packet.frame.shape[0]),
            packet.source_size,
        )
        result = self.tracker.update(
            packet.frame,
            stamp,
            packet.frame_index,
            "stream_pts",
            detections=source_detections,
        )
        result.update(
            camera_id=packet.camera_id,
            session_id=session,
            source_size=list(size),
            processing_size=[packet.frame.shape[1], packet.frame.shape[0]],
            pts_ns=packet.pts_ns,
            source_frame_index=packet.source_frame_index,
            skipped_before=max(0, packet.source_frame_index - self.last_source_index - 1),
        )
        self.last_source_index = packet.source_frame_index
        self.last_index, self.last_stamp = packet.frame_index, stamp
        return result


class RecognitionEngine:
    """Run one fixed inference batch and route results to per-camera trackers."""

    def __init__(self, detector, tracking: TrackingConfig):
        self.detector = detector
        self.tracking = tracking
        self.batch_size = int(getattr(detector, "batch_size", 1))
        self.trackers: dict[str, StreamTracker] = {}

    def warmup(self, image_shape: tuple[int, int, int], count: int):
        frames = [np.zeros(image_shape, dtype=np.uint8) for _ in range(self.batch_size)]
        if self.batch_size == 1 and not hasattr(self.detector, "warmup_batch"):
            self.detector.warmup(frames[0], count)
        else:
            self.detector.warmup_batch(frames, count)

    def process_batch(self, packets: Sequence[FramePacket]):
        if len(packets) != self.batch_size:
            raise ValueError(f"输入帧数量必须等于固定 batch={self.batch_size}")
        camera_ids = [packet.camera_id for packet in packets]
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("同一个 batch 不能包含重复 camera_id")
        frames = [packet.frame for packet in packets]
        if self.batch_size == 1 and not hasattr(self.detector, "detect_batch"):
            batches = [self.detector.detect(frames[0])]
        else:
            batches = self.detector.detect_batch(frames)
        if len(batches) != self.batch_size:
            raise RuntimeError("模型结果数量与输入 batch 不一致")
        results = []
        for packet, detections in zip(packets, batches):
            tracker = self.trackers.setdefault(packet.camera_id, StreamTracker(self.tracking))
            results.append(tracker.update(packet, detections))
        return results

    def finish(self, reason):
        for tracker in self.trackers.values():
            tracker.finish(reason)

    @property
    def segments(self):
        segments = []
        for camera_id, tracker in self.trackers.items():
            segments.extend(dict(segment, camera_id=camera_id) for segment in tracker.segments)
        return segments
