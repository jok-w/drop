"""Image measurements and motion predictions remain separate throughout."""
from dataclasses import dataclass
import math

import cv2
import numpy as np

from .appearance import ContrastLocator


@dataclass
class Config:
    gate: float = 5.991
    measurement_std: float = 4.0  # pixels; empirical tuning required
    acceleration_std: float = 800.0  # pixels / second squared
    initial_velocity_std: float = 200.0
    coast_seconds: float = 0.5
    size_ratio_limit: float = 4.0

    def __post_init__(self):
        for name, value in vars(self).items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.size_ratio_limit <= 1:
            raise ValueError("size_ratio_limit must exceed 1")


class MotionFilter:
    def __init__(self, center, config):
        self.config = config
        self.x = np.array([*center, 0.0, 0.0], dtype=float)
        self.P = np.diag([config.measurement_std**2] * 2
                         + [config.initial_velocity_std**2] * 2)
        self.H = np.eye(2, 4)
        self.R = np.eye(2) * config.measurement_std**2

    def predict(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Frame timestamps must increase")
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt**2 / 2, 0], [0, dt**2 / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.config.acceleration_std**2 * G @ G.T
        return self.x[:2].copy()

    def innovation(self, center):
        residual = np.asarray(center) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        return residual, S

    def distance_squared(self, center):
        residual, S = self.innovation(center)
        return float(residual @ np.linalg.solve(S, residual))

    def correct(self, center):
        residual, S = self.innovation(center)
        K = np.linalg.solve(S, self.H @ self.P).T
        self.x += K @ residual
        A = np.eye(4) - K @ self.H
        self.P = A @ self.P @ A.T + K @ self.R @ K.T
        self.P = (self.P + self.P.T) / 2


def create_csrt():
    factory = getattr(cv2, "TrackerCSRT_create", None)
    if factory is None:
        factory = getattr(getattr(cv2, "legacy", None), "TrackerCSRT_create", None)
    if factory is None:
        raise RuntimeError("CSRT unavailable. Install requirements.txt in the project venv.")
    return factory()


def validate_box(box, shape):
    if len(box) != 4 or not all(math.isfinite(float(v)) for v in box):
        raise ValueError("ROI requires four finite numbers: x y width height")
    x, y, w, h = [int(round(v)) for v in box]
    height, width = shape[:2]
    if w < 2 or h < 2 or x < 0 or y < 0 or x + w > width or y + h > height:
        raise ValueError("ROI must lie inside the frame with width/height >= 2")
    return x, y, w, h


def center_of(box):
    x, y, w, h = box
    return x + w / 2, y + h / 2


class SingleTargetTracker:
    def __init__(self, config=None, tracker_factory=create_csrt, appearance_mode="auto"):
        self.config = config or Config()
        self.factory = tracker_factory
        self.image_tracker = None
        self.motion = None
        self.segment = 0
        self.appearance_mode = appearance_mode
        self.appearance = None

    def initialize(self, frame, box, timestamp, frame_index, timestamp_source):
        box = validate_box(box, frame.shape)
        self.image_tracker = self.factory()
        self.image_tracker.init(frame, box)
        self.motion = MotionFilter(center_of(box), self.config)
        self.appearance = ContrastLocator(frame, box) if self.appearance_mode == "auto" else None
        self.appearance_missing = False
        self.last_box = box
        self.last_visible_frame = frame.copy()
        self.retry_allowed = False
        self.last_time = self.last_visible_time = timestamp
        self.segment += 1
        result = self._record(timestamp, frame_index, timestamp_source)
        result.update(state="TRACKING", source="manual", reason="initialized",
                      bbox=list(box), center=list(center_of(box)))
        return result

    def _record(self, timestamp, frame_index, timestamp_source):
        return dict(frame_index=frame_index, timestamp=timestamp,
                    timestamp_source=timestamp_source, segment=self.segment,
                    state="LOST_PENDING", source="none", reason="",
                    bbox=None, center=None, candidate_bbox=None,
                    predicted_center=None, innovation_covariance=None,
                    mahalanobis_squared=None, appearance_reason=None,
                    appearance_cost=None, appearance_candidates=None)

    def _update_appearance(self, frame, timestamp, result, prediction):
        gap = timestamp - self.last_visible_time
        if self.appearance_missing and gap > self.config.coast_seconds:
            result.update(state="LOST", reason="appearance_timeout", predicted_center=None,
                          innovation_covariance=None)
            return result
        candidate, details = self.appearance.locate(frame, prediction, self.last_box,
                                                     self.motion.P[:2,:2], self.appearance_missing)
        result.update(details)
        if candidate is None:
            result["reason"] = "appearance_" + details["appearance_reason"]
            return result
        result["candidate_bbox"] = list(candidate)
        center = center_of(candidate)
        d2 = self.motion.distance_squared(center)
        result["mahalanobis_squared"] = d2
        motion_reset = d2 > self.config.gate
        # An independent, unique, strong image measurement can expose a wrong
        # motion model (camera movement, a bounce). Do not just widen the gate.
        if motion_reset and (details["appearance_candidates"] != 1 or details["appearance_cost"] > 1.0):
            result["reason"] = "appearance_motion_disagreement"
            return result
        if motion_reset:
            velocity = (np.asarray(center)-np.asarray(center_of(self.last_box))) / gap
            self.motion = MotionFilter(center, self.config)
            self.motion.x[2:] = velocity
        else:
            self.motion.correct(center)
        self.appearance.accept(details)
        recovered = self.appearance_missing
        self.appearance_missing = False
        self.last_box = candidate
        self.last_visible_time = timestamp
        result.update(state="TRACKING", source="local_contrast", bbox=list(candidate),
                      center=list(center), reason="motion_reset" if motion_reset else
                      ("appearance_recovered" if recovered else "appearance_matched"))
        return result

    def update(self, frame, timestamp, frame_index, timestamp_source):
        if self.motion is None:
            raise RuntimeError("Initialize before update")
        prediction = self.motion.predict(timestamp - self.last_time)
        self.last_time = timestamp
        result = self._record(timestamp, frame_index, timestamp_source)
        result["predicted_center"] = prediction.tolist()
        result["innovation_covariance"] = self.motion.innovation(prediction)[1].tolist()
        if self.appearance is not None and self.appearance.enabled:
            result = self._update_appearance(frame, timestamp, result, prediction)
            if result["center"] is None:
                self.appearance_missing = True
            return result
        result["reason"] = "awaiting_manual_reinitialization"
        retrying = self.image_tracker is None
        if retrying and self.retry_allowed and timestamp - self.last_visible_time <= self.config.coast_seconds:
            # Rebuild from the last ACCEPTED frame, never from a rejected candidate.
            self.image_tracker = self.factory()
            self.image_tracker.init(self.last_visible_frame, self.last_box)
        if self.image_tracker is not None:
            ok, candidate = self.image_tracker.update(frame)
            result["reason"] = "tracker_failed"
            if ok:
                try:
                    box = validate_box(candidate, frame.shape)
                except ValueError:
                    result["reason"] = "invalid_box"
                else:
                    result["candidate_bbox"] = list(box)
                    d2 = self.motion.distance_squared(center_of(box))
                    result["mahalanobis_squared"] = d2
                    ratios = np.asarray(box[2:]) / np.asarray(self.last_box[2:])
                    limit = self.config.size_ratio_limit
                    size_ok = bool(np.all((ratios >= 1 / limit) & (ratios <= limit)))
                    if size_ok and math.isfinite(d2) and d2 <= self.config.gate:
                        self.motion.correct(center_of(box))
                        self.last_visible_time = timestamp
                        self.last_box = box
                        self.last_visible_frame = frame.copy()
                        self.retry_allowed = False
                        result.update(state="TRACKING", source="csrt",
                                      reason="recovered" if retrying else "accepted", bbox=list(box),
                                      center=list(center_of(box)))
                        return result
                    result["reason"] = "motion_gate" if size_ok else "size_jump"
            # update() may already have contaminated CSRT's internal template.
            # Retry a model disagreement only. CSRT failure or malformed boxes
            # lack image evidence; rebuilding those can latch onto background.
            self.retry_allowed = result["reason"] == "motion_gate"
            self.image_tracker = None
        if timestamp - self.last_visible_time > self.config.coast_seconds:
            result.update(state="LOST", predicted_center=None,
                          innovation_covariance=None)
        return result
