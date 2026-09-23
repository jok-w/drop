"""YOLO observations and conservative single-instance association."""
from pathlib import Path
import math
import time

import numpy as np

from .core import Config, MotionFilter, center_of, validate_box


class YoloDetector:
    def __init__(self, weights, confidence=.25, image_size=1280, device="cpu", class_id=None, iou=.5,
                 backend="pt", engine=None, onnx=None, profile="throughput"):
        path = Path(weights).resolve()
        if not path.is_file():
            raise ValueError(f"Weights do not exist: {path}")
        if not 0 < confidence <= 1 or image_size <= 0 or not 0 < iou <= 1:
            raise ValueError("confidence/iou must be in (0, 1] and image-size must be positive")
        # Keep runtime settings in the project, including on restricted desktops.
        from .runtime import configure_runtime
        configure_runtime()
        try:
            import ultralytics
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Install project dependencies with 'uv sync --python 3.10' to use --weights") from exc
        if image_size < 32 or image_size % 32:
            raise ValueError("imgsz must be a positive multiple of 32")
        from .model_artifacts import (read_engine_metadata, read_onnx_metadata, validate_source,
                                      validate_engine, runtime_info, file_sha256)
        source_hash = file_sha256(path)
        selected = path
        artifact_metadata = None
        if backend == "tensorrt":
            selected = Path(engine or path.with_name(f"{path.stem}.{image_size}.fp16.engine"))
            artifact_metadata = read_engine_metadata(selected)
            validate_engine(artifact_metadata, source_hash, image_size, runtime_info(device))
        elif backend == "onnx":
            selected = Path(onnx or path.with_suffix(".onnx"))
            artifact_metadata = read_onnx_metadata(selected)
            validate_source(artifact_metadata, source_hash)
        elif backend != "pt":
            raise ValueError("backend must be pt, onnx or tensorrt")
        import torch
        self.torch, self.device, self.profile = torch, device, profile
        self.last_timing = {}
        self.model = YOLO(str(selected), task="detect")
        self.model.overrides.update(device=device, imgsz=image_size, rect=False)
        if self.model.task != "detect":
            raise ValueError("Weights must contain an object detection model")
        # Accessing YOLO.names on an exported model instantiates a throwaway
        # predictor. Read the verified artifact metadata to avoid loading twice.
        names = self.model.names if artifact_metadata is None else {
            int(k): v for k, v in artifact_metadata["names"].items()}
        if class_id is not None and class_id not in names:
            raise ValueError(f"Unknown class {class_id}; model classes: {names}")
        self.options = dict(conf=confidence, iou=iou, imgsz=image_size, device=device, rect=False, nms=None,
                            classes=None if class_id is None else [class_id], verbose=False)
        self.metadata = dict(weights=str(path), sha256=source_hash, model_path=str(selected.resolve()),
                             artifact_sha256=file_sha256(selected),
                             backend=backend, torch=torch.__version__, cuda=torch.version.cuda,
                             names=names, ultralytics=ultralytics.__version__, profile=profile,
                             **self.options)

    def synchronize(self):
        if self.device != "cpu" and self.torch.cuda.is_available():
            device = self.device if str(self.device).startswith("cuda") else int(self.device)
            self.torch.cuda.synchronize(device)

    def warmup(self, frame, count):
        for _ in range(count):
            self.detect(frame)
        self.synchronize()
        self.last_timing = {}

    def detect(self, frame):
        if self.profile == "diagnostic":
            self.synchronize()
        started = time.perf_counter()
        result = self.model.predict(frame, **self.options)[0]
        runtime = self.model.predictor.model
        self.metadata["actual_device"] = str(runtime.device)
        self.metadata["actual_format"] = runtime.format
        if hasattr(runtime, "session"):
            self.metadata["onnx_providers"] = runtime.session.get_providers()
        if self.profile == "diagnostic":
            self.synchronize()
        transfer_started = time.perf_counter()
        detections = []
        values = [] if result.boxes is None else result.boxes.data.cpu().tolist()
        for x1, y1, x2, y2, confidence, class_id in values:
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, confidence, class_id)):
                continue
            left, top = max(0, round(x1)), max(0, round(y1))
            right, bottom = min(frame.shape[1], round(x2)), min(frame.shape[0], round(y2))
            try:
                box = validate_box((left, top, right-left, bottom-top), frame.shape)
            except ValueError:
                continue
            detections.append(dict(bbox=list(box), confidence=confidence, class_id=int(class_id)))
        finished = time.perf_counter()
        self.last_timing = {f"{k}_ms": float(result.speed[k]) for k in
                            ("preprocess", "inference", "postprocess")}
        self.last_timing["result_transfer_ms"] = (finished-transfer_started)*1000
        self.last_timing["detector_ms"] = (finished-started)*1000
        self.last_timing["detector_overhead_ms"] = max(0., self.last_timing["detector_ms"] - sum(
            self.last_timing[k] for k in ("preprocess_ms", "inference_ms", "postprocess_ms", "result_transfer_ms")))
        return detections


class DetectionTracker:
    def __init__(self, detector, config=None, interval=1, strict_motion_gate=False):
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("detect-interval must be a positive integer")
        self.detector, self.config, self.interval = detector, config or Config(), interval
        self.strict_motion_gate = strict_motion_gate
        self.motion = None
        self.segment = 0
        self.class_id = None
        self.last_detection_index = None
        self.last_time = None
        self.velocity_ready = False

    def _record(self, timestamp, frame_index, timestamp_source):
        return dict(frame_index=frame_index, timestamp=timestamp,
                    timestamp_source=timestamp_source, segment=self.segment,
                    state="LOST_PENDING", source="none", reason="",
                    bbox=None, center=None, candidate_bbox=None,
                    predicted_center=None, innovation_covariance=None,
                    mahalanobis_squared=None)

    def _result(self, timestamp, index, source):
        result = self._record(timestamp, index, source)
        result.update(estimated_center=None, detection_ran=False, detections=[],
                      detection_confidence=None, class_id=self.class_id,
                      seconds_since_observation=None)
        return result

    def update(self, frame, timestamp, frame_index, timestamp_source):
        if self.last_time is not None and timestamp <= self.last_time:
            raise ValueError("Frame timestamps must increase")
        result = self._result(timestamp, frame_index, timestamp_source)
        if self.motion is not None:
            gap = timestamp - self.last_visible_time
            result["seconds_since_observation"] = gap
            if gap > self.config.coast_seconds:
                self.motion = None
                self.velocity_ready = False
                self.last_detection_index = None
                self.last_visible_time = None
                self.last_time = timestamp
                result.update(state="LOST", reason="tracking_timeout")
                return result
            prediction = self.motion.predict(timestamp-self.last_time)
            result.update(predicted_center=prediction.tolist(), estimated_center=prediction.tolist(),
                          innovation_covariance=self.motion.innovation(prediction)[1].tolist())
        self.last_time = timestamp
        due = self.last_detection_index is None or frame_index-self.last_detection_index >= self.interval
        if not due:
            result.update(state="PREDICTED" if self.motion is not None else "WAITING",
                          reason="detection_skipped")
            return result
        self.last_detection_index = frame_index
        detections = self.detector.detect(frame)
        result.update(detection_ran=True, detections=detections)
        candidates = []
        for detection in detections:
            box = detection["bbox"]
            if self.class_id is not None and detection["class_id"] != self.class_id:
                continue
            d2 = None
            if self.motion is not None:
                ratios = np.asarray(box[2:]) / np.asarray(self.last_box[2:])
                limit = self.config.size_ratio_limit
                if not np.all((ratios >= 1/limit) & (ratios <= limit)):
                    continue
                d2 = self.motion.distance_squared(center_of(box))
                if not math.isfinite(d2):
                    continue
            candidates.append((detection, d2))
        # With one unique semantic detection, motion is diagnostic by default:
        # an inaccurate motion prior must not suppress all future observations.
        # Multiple candidates still require a unique motion-consistent match.
        if self.velocity_ready and (self.strict_motion_gate or len(candidates) > 1):
            candidates = [(d, d2) for d, d2 in candidates if d2 <= self.config.gate]
        if len(candidates) != 1:
            result.update(state="LOST_PENDING" if self.motion is not None else "WAITING",
                          reason="ambiguous_detections" if len(candidates) > 1 else
                          ("no_detection" if not detections else "association_rejected"))
            return result
        detection, d2 = candidates[0]
        box = detection["bbox"]
        center = center_of(box)
        first = self.motion is None
        if first:
            self.motion = MotionFilter(center, self.config)
            self.segment += 1
        elif not self.velocity_ready:
            # Infer velocity at the second observation under the same drag model.
            # A zero-velocity prior cannot gate an already-moving object.
            dt = timestamp - self.last_visible_time
            rate = self.config.deceleration_rate
            decay = math.exp(-rate * dt)
            travel = -math.expm1(-rate * dt) / rate
            velocity_scale = decay / travel
            velocity = (np.asarray(center)-center_of(self.last_box)) * velocity_scale
            self.motion = MotionFilter(center, self.config)
            self.motion.x[2:] = velocity
            variance = self.config.measurement_std**2
            self.motion.P = np.block([[np.eye(2)*variance, np.eye(2)*variance*velocity_scale],
                                      [np.eye(2)*variance*velocity_scale,
                                       np.eye(2)*2*variance*velocity_scale**2]])
            self.velocity_ready = True
        else:
            self.motion.correct(center)
        self.class_id = detection["class_id"]
        self.last_visible_time = timestamp
        self.last_box = box
        reason = ("detected_initialization" if self.segment == 1 else "detected_reinitialization") if first else (
            "detected_motion_disagreement" if d2 is not None and d2 > self.config.gate else "detected")
        result.update(state="TRACKING", source="yolo", reason=reason,
                      segment=self.segment, bbox=list(box), center=list(center), candidate_bbox=list(box),
                      estimated_center=self.motion.x[:2].tolist(), mahalanobis_squared=d2,
                      detection_confidence=detection["confidence"], class_id=self.class_id,
                      seconds_since_observation=0.)
        return result
