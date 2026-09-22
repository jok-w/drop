import argparse
from collections import Counter, deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .core import Config, SingleTargetTracker, create_csrt, validate_box


WINDOW = "Single target | SPACE play/pause | N next | R reselect | Q quit"
FIELDS = ["frame_index", "timestamp", "timestamp_source", "segment", "state", "source",
          "reason", "x", "y", "width", "height", "center_x", "center_y",
          "predicted_x", "predicted_y", "mahalanobis_squared", "candidate_bbox",
          "appearance_reason", "appearance_cost", "appearance_candidates"]
FIELDS += ["estimated_x", "estimated_y", "detection_ran", "detection_confidence",
           "class_id", "seconds_since_observation"]


class VideoClock:
    def __init__(self, fps):
        self.fps = fps
        self.previous = None

    def get(self, milliseconds, index):
        value = milliseconds / 1000.0
        valid = math.isfinite(value) and value >= 0
        valid = valid and (self.previous is None or value > self.previous)
        valid = valid and not (index > 0 and value == 0)
        source = "video_timestamp" if valid else "fps_fallback"
        if not valid:
            value = index / self.fps if self.previous is None else self.previous + 1 / self.fps
        self.previous = value
        return value, source


def draw(frame, result, history, estimated_history=()):
    canvas = frame.copy()
    estimates = list(estimated_history) + [(result.get("estimated_center"), result["segment"])]
    for (a, sa), (b, sb) in zip(estimates, estimates[1:]):
        if a is not None and b is not None and sa == sb:
            if all(-10000 < v < 10000 for v in (*a, *b)):
                cv2.line(canvas, tuple(np.int32(a)), tuple(np.int32(b)), (255, 200, 0), 1)
    points = list(history) + [(result["center"], result["segment"])]
    for (a, sa), (b, sb) in zip(points, points[1:]):
        if a is not None and b is not None and sa == sb:
            cv2.line(canvas, tuple(np.int32(a)), tuple(np.int32(b)), (0, 210, 0), 1)
    if result["bbox"]:
        x, y, w, h = result["bbox"]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 230, 0), 2)
    # Prediction is a separate orange cross; never extend the measured polyline.
    if result["predicted_center"] is not None:
        px, py = result["predicted_center"]
        if -10000 < px < 10000 and -10000 < py < 10000:
            cv2.drawMarker(canvas, (round(px), round(py)), (0, 180, 255),
                           cv2.MARKER_CROSS, 12, 1)
    lines = [f"Frame {result['frame_index']}  t={result['timestamp']:.3f}s  {result['state']}",
             f"{result['source']} | {result['reason']}",
             "Green: image | Orange: prediction"]
    if result["mahalanobis_squared"] is not None:
        lines.append(f"Motion d^2 = {result['mahalanobis_squared']:.2f}")
    if result.get("appearance_cost") is not None:
        lines.append(f"Appearance cost = {result['appearance_cost']:.2f}")
    if result.get("estimated_center") is not None:
        ex, ey = result["estimated_center"]
        if -10000 < ex < 10000 and -10000 < ey < 10000:
            cv2.circle(canvas, (round(ex), round(ey)), 4, (255, 200, 0), 1)
        lines.append("Cyan: Kalman estimate (may be prediction only)")
    for i, line in enumerate(lines):
        origin = (10, 22 + i * 22)
        text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .5, 1)[0][0]
        scale = .5 * min(1.0, max(1, canvas.shape[1] - 20) / max(1, text_width))
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3)
        cv2.putText(canvas, line, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return canvas


def display_size(frame, bounds):
    height, width = frame.shape[:2]
    scale = min(1.0, bounds[0] / width, bounds[1] / height)
    return max(1, round(width * scale)), max(1, round(height * scale))


def draw_output(frame, result, history, estimated_history, size):
    if size == (frame.shape[1], frame.shape[0]):
        return draw(frame, result, history, estimated_history)
    sx, sy = size[0]/frame.shape[1], size[1]/frame.shape[0]
    def point(p):
        return None if p is None else [p[0]*sx, p[1]*sy]
    scaled = dict(result)
    for key in ("center", "predicted_center", "estimated_center"):
        scaled[key] = point(result.get(key))
    if result["bbox"]:
        scaled["bbox"] = [round(v*s) for v, s in zip(result["bbox"], (sx, sy, sx, sy))]
    return draw(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), scaled,
                [(point(p), segment) for p, segment in history],
                [(point(p), segment) for p, segment in estimated_history])


def select_box(frame, bounds=(960, 540)):
    name = "Select target | ENTER confirm | C cancel"
    size = display_size(frame, bounds)
    preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, *size)
    try:
        x, y, w, h = cv2.selectROI(name, preview, showCrosshair=True, fromCenter=False)
    finally:
        cv2.destroyWindow(name)
    if w <= 0 or h <= 0:
        return None
    # Map preview edges back to source pixels, including rounding at the borders.
    sx, sy = frame.shape[1] / size[0], frame.shape[0] / size[1]
    x1, y1 = round(x * sx), round(y * sy)
    x2, y2 = round((x + w) * sx), round((y + h) * sy)
    return x1, y1, x2 - x1, y2 - y1


def csv_row(result):
    row = {key: result.get(key) for key in FIELDS}
    for key, value in zip(["x", "y", "width", "height"], result["bbox"] or [None] * 4):
        row[key] = value
    for key, value in zip(["center_x", "center_y"], result["center"] or [None] * 2):
        row[key] = value
    for key, value in zip(["predicted_x", "predicted_y"], result["predicted_center"] or [None] * 2):
        row[key] = value
    row["candidate_bbox"] = json.dumps(result["candidate_bbox"]) if result["candidate_bbox"] else ""
    for key, value in zip(["estimated_x", "estimated_y"], result.get("estimated_center") or [None] * 2):
        row[key] = value
    return row


def config_for_frame(args, shape):
    # Default noise parameters describe a <=1280px-long-edge image. In 4K,
    # identical fractional motion spans three times as many pixels.
    scale = max(1.0, max(shape[:2]) / 1280.0)
    return Config(
        gate=args.gate,
        measurement_std=args.measurement_std if args.measurement_std is not None else 4.0 * scale,
        acceleration_std=args.acceleration_std if args.acceleration_std is not None else 800.0 * scale,
        initial_velocity_std=args.initial_velocity_std if args.initial_velocity_std is not None else 200.0 * scale,
        coast_seconds=args.coast_seconds,
    )


def run(args):
    from .performance import Performance, STAGES, TIMING_FIELDS, reader_snapshot, reader_delta
    from .video_reader import create_video_capture, GStreamerVideoCapture, PreparedVideoCapture
    from .video_writer import create_video_writer, GStreamerVideoWriter
    perf = Performance()
    if args.start_frame < 0 or (args.max_frames is not None and args.max_frames <= 0):
        raise ValueError("start-frame must be >= 0 and max-frames must be > 0")
    if args.headless and args.roi is None and args.weights is None:
        raise ValueError("--headless requires --roi or --weights")
    if args.backend != "pt" and not args.weights:
        raise ValueError("Exported backends require original --weights for fingerprint validation")
    if args.engine and args.backend != "tensorrt":
        raise ValueError("--engine requires --backend tensorrt")
    if args.onnx and args.backend != "onnx":
        raise ValueError("--onnx requires --backend onnx")
    if min(args.window_size) <= 0 or args.output_max_width < 0 or args.warmup < 0 or args.log_every < 0:
        raise ValueError("Invalid window size, output width, warmup or log interval")
    if args.mode == "detect-benchmark" and (not args.weights or not args.headless or not args.no_video
                                            or args.roi is not None or args.detect_interval != 1):
        raise ValueError("detect-benchmark requires --weights --headless --no-video, interval=1 and no ROI")
    video = Path(args.video).resolve()
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    output = Path(args.output or (Path("outputs") / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f"))).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    detector = None
    with perf.section("model_load_ms"):
        if args.weights is not None:
            from .detection import YoloDetector, DetectionTracker
            if args.detect_interval < 1:
                raise ValueError("detect-interval must be positive")
            detector = YoloDetector(args.weights, args.conf, args.imgsz, args.device, args.class_id,
                                    args.nms_iou, args.backend, args.engine, args.onnx, args.profile)
        else:
            create_csrt()
    cap = writer = None
    try:
        with perf.section("reader_open_ms"):
            cap = create_video_capture(video, args.decoder, args.decode_prefetch, args.gst_python, args.read_ahead)
        actual_decoder = "gstreamer" if isinstance(cap, (GStreamerVideoCapture, PreparedVideoCapture)) else "opencv"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            if args.fallback_fps is None or not math.isfinite(args.fallback_fps) or args.fallback_fps <= 0:
                raise ValueError("Video FPS unavailable; specify --fallback-fps")
            fps = args.fallback_fps
        with perf.section("start_frame_skip_ms"):
            for _ in range(args.start_frame):
                if not cap.read()[0]:
                    raise ValueError("start-frame lies beyond readable video")
        with perf.section("first_frame_read_ms"):
            ok, frame = cap.read()
            if not ok:
                raise ValueError("No readable frame at start-frame")
        height, width = frame.shape[:2]
        config = config_for_frame(args, frame.shape)
        with perf.section("manual_selection_ms"):
            box = args.roi if args.roi is not None else (None if detector else select_box(frame, args.window_size))
        if box is None and detector is None:
            print("Initialization cancelled; no output created.")
            return None
        if box is not None:
            box = validate_box(box, frame.shape)
        if detector and args.detect_interval / fps > config.coast_seconds:
            raise ValueError("Detection interval exceeds coast-seconds")
        with perf.section("warmup_ms"):
            if detector:
                detector.warmup(frame, args.warmup)
        scale = min(1., args.output_max_width / width) if args.output_max_width else 1.
        output_size = (max(2, int(width*scale)//2*2), max(2, int(height*scale)//2*2))
        output.mkdir(parents=True, exist_ok=False)
        actual_encoder = "none"
        with perf.section("writer_open_ms"):
            if not args.no_video:
                writer = create_video_writer(output / "annotated.mp4", fps, *output_size,
                                             args.encoder, "mp4v", args.video_bitrate)
                actual_encoder = "gstreamer" if isinstance(writer, GStreamerVideoWriter) else "opencv"
        tracker = (DetectionTracker(detector, config, args.detect_interval, args.strict_motion_gate) if detector else
                   SingleTargetTracker(config, appearance_mode=args.appearance))
        clock = VideoClock(fps)
        index = args.start_frame
        paused = True
        history, estimated_history = deque(maxlen=250), deque(maxlen=250)
        counts, reasons = Counter(), Counter()
        interventions = fallbacks = count = 0
        end_reason = "end_of_readable_video"
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WINDOW, *display_size(frame, args.window_size))
        # All frame processing, including the first inference and buffered closes, belongs to this interval.
        pipeline_started = time.perf_counter()
        with (output / "track.csv").open("w", encoding="utf-8-sig", newline="") as csv_file, \
                (output / "track.jsonl").open("w", encoding="utf-8") as json_file, \
                (output / "timing.csv").open("w", encoding="utf-8-sig", newline="") as timing_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=FIELDS)
            timing_writer = csv.DictWriter(timing_file, fieldnames=TIMING_FIELDS)
            csv_writer.writeheader()
            timing_writer.writeheader()
            while True:
                frame_started = time.perf_counter()
                timing = dict.fromkeys(STAGES)
                timing.update(read_cached=count == 0, read_wait_ms=0., draw_ms=0., display_ms=0., video_submit_ms=0.)
                if count:
                    before = reader_snapshot(cap)
                    read_started = time.perf_counter()
                    ok, frame = cap.read()
                    timing["read_wait_ms"] = (time.perf_counter()-read_started)*1000
                    timing.update(reader_delta(before, reader_snapshot(cap)))
                    if not ok:
                        perf.sections["eof_read_ms"] = timing["read_wait_ms"]
                        break
                    index += 1
                if frame.shape[:2] != (height, width):
                    raise ValueError("Frame dimensions changed within video")
                stamp, stamp_source = clock.get(cap.get(cv2.CAP_PROP_POS_MSEC), index)
                if detector:
                    detector.last_timing = {}
                processing_started = time.perf_counter()
                if args.mode == "detect-benchmark":
                    detections = detector.detect(frame)
                    result = tracker._result(stamp, index, stamp_source)
                    result.update(state="BENCHMARK", source="yolo", reason="benchmark_detection",
                                  detection_ran=True, detections=detections)
                elif count == 0 and box is not None:
                    result = tracker.initialize(frame, box, stamp, index, stamp_source)
                else:
                    result = tracker.update(frame, stamp, index, stamp_source)
                processing_ms = (time.perf_counter()-processing_started)*1000
                if detector and result.get("detection_ran"):
                    timing.update(detector.last_timing)
                timing["tracking_ms"] = max(0., processing_ms - (timing.get("detector_ms") or 0.))
                canvas = None
                if writer is not None or not args.headless:
                    started = time.perf_counter()
                    canvas = draw_output(frame, result, history, estimated_history, output_size)
                    timing["draw_ms"] = (time.perf_counter()-started)*1000
                quit_requested = False
                if not args.headless:
                    started = time.perf_counter()
                    while True:
                        cv2.imshow(WINDOW, canvas)
                        key = cv2.waitKey(30 if paused else 1) & 0xFF
                        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                            quit_requested = True
                            break
                        if key == ord("r"):
                            new_box = select_box(frame, args.window_size)
                            if new_box is not None:
                                result = tracker.initialize(frame, new_box, stamp, index, stamp_source)
                                interventions += 1
                                canvas = draw_output(frame, result, history, estimated_history, output_size)
                            paused = True
                            continue
                        if key == ord(" "):
                            paused = not paused
                        if key == ord("n"):
                            paused = True
                            break
                        if not paused:
                            break
                    timing["display_ms"] = (time.perf_counter()-started)*1000
                started = time.perf_counter()
                csv_writer.writerow(csv_row(result))
                json_file.write(json.dumps(result, allow_nan=False) + "\n")
                timing["data_write_ms"] = (time.perf_counter()-started)*1000
                if writer is not None:
                    started = time.perf_counter()
                    writer.write(canvas)
                    timing["video_submit_ms"] = (time.perf_counter()-started)*1000
                history.append((result["center"], result["segment"]))
                estimated_history.append((result.get("estimated_center"), result["segment"]))
                counts[result["state"]] += 1
                reasons[result["reason"]] += 1
                fallbacks += stamp_source == "fps_fallback"
                count += 1
                timing.update(frame_index=index, timestamp=stamp, state=result["state"],
                              detection_ran=bool(detector and timing.get("detector_ms") is not None),
                              reason=result["reason"], frame_wall_ms=(time.perf_counter()-frame_started)*1000)
                perf.rows.append(timing)
                timing_writer.writerow(timing)
                if args.log_every and count % args.log_every == 0:
                    print(f"Processed {count} frames; detector calls={sum(r['detection_ran'] for r in perf.rows)}; "
                          f"elapsed={time.perf_counter()-pipeline_started:.2f}s", flush=True)
                if quit_requested:
                    end_reason = "user_quit"
                    break
                if args.max_frames is not None and count >= args.max_frames:
                    end_reason = "max_frames"
                    break
            with perf.section("data_flush_ms"):
                for stream in (csv_file, json_file, timing_file):
                    stream.flush()
        with perf.section("reader_finalize_ms"):
            cap.release()
            cap = None
        with perf.section("encoder_finalize_ms"):
            if writer is not None:
                writer.release()
                writer = None
        elapsed = time.perf_counter()-pipeline_started
        environment = dict(opencv=cv2.__version__, detector=detector.metadata if detector else None,
                           decoder=actual_decoder, encoder=actual_encoder, requested_decoder=args.decoder,
                           requested_encoder=args.encoder, read_ahead=args.read_ahead if actual_decoder == "gstreamer" else 0,
                           decode_prefetch=args.decode_prefetch if actual_decoder == "gstreamer" else 0,
                           input_size=[width, height], output_size=list(output_size),
                           mode=args.mode, warmup_calls=args.warmup if detector else 0,
                           profile=args.profile, headless=args.headless)
        from .runtime import hardware_info
        environment["hardware"] = hardware_info()
        environment["configuration"] = vars(args)
        report = perf.report(elapsed, environment)
        summary = dict(input=str(video), output=str(output), fps=fps, width=width, height=height,
                       frames=count, start_frame=args.start_frame, end_frame=index,
                       end_reason=end_reason, states=dict(counts), reasons=dict(reasons),
                       manual_reinitializations=interventions, timestamp_fallbacks=fallbacks,
                       wall_seconds=elapsed, throughput_fps=count/max(elapsed, 1e-9),
                       config=vars(config), initial_roi=list(box) if box else None, opencv=cv2.__version__,
                       backend=detector.metadata['backend'] if detector else "legacy",
                       detector=detector.metadata if detector else None,
                       detect_interval=args.detect_interval if detector else None,
                       strict_motion_gate=args.strict_motion_gate if detector else None,
                       appearance_mode="disabled" if detector else args.appearance,
                       appearance_active=bool(tracker.appearance and tracker.appearance.enabled),
                       performance_file="performance.json", notes=["Accepted observations are not ground truth.",
                       "FPS includes output finalization; GUI runs include user waits."])
        (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        (output / "performance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        for name in ("read_wait_ms", "detector_ms", "inference_ms", "tracking_ms", "draw_ms", "data_write_ms", "video_submit_ms"):
            value = report["stages"][name]
            if value['count']:
                print(f"{name}: count={value['count']} mean={value['mean_ms']:.3f} p95={value['p95_ms']:.3f}")
        print(f"Finalize encoder={perf.sections['encoder_finalize_ms']:.3f}ms; "
              f"pipeline FPS including finalize={report['pipeline_fps_with_finalize']:.3f}")
        return output
    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            if isinstance(writer, GStreamerVideoWriter):
                writer.abort()
            else:
                writer.release()
        if not args.headless:
            cv2.destroyAllWindows()



def parser():
    p = argparse.ArgumentParser(description="Offline YOLO or CSRT + Kalman single-target tracking")
    p.add_argument("video", help="Input video path")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"))
    p.add_argument("--headless", action="store_true", help="No windows; requires --roi or --weights")
    p.add_argument("--weights", help="Local YOLO .pt weights; enables detection backend")
    p.add_argument("--backend", choices=("pt", "onnx", "tensorrt"), default="pt")
    p.add_argument("--engine", help="TensorRT engine built by tracking.export_model")
    p.add_argument("--onnx", help="ONNX model exported by tracking.export_model")
    p.add_argument("--mode", choices=("track", "detect-benchmark"), default="track")
    p.add_argument("--profile", choices=("throughput", "diagnostic"), default="throughput")
    p.add_argument("--warmup", type=int, default=5, help="Model warmup calls, excluded from steady-state timings")
    p.add_argument("--log-every", type=int, default=100, help="Progress interval; 0 disables")
    p.add_argument("--decoder", choices=("auto", "opencv", "gstreamer"), default="auto")
    p.add_argument("--encoder", choices=("auto", "opencv", "gstreamer"), default="auto")
    p.add_argument("--decode-prefetch", type=int, choices=(1, 2), default=2)
    p.add_argument("--read-ahead", type=int, choices=(0, 1, 2), default=2)
    p.add_argument("--gst-python", default="/usr/bin/python3")
    p.add_argument("--video-bitrate", type=int, default=8000000)
    p.add_argument("--output-max-width", type=int, default=0, help="Saved/display width limit; 0 preserves original")
    p.add_argument("--detect-interval", type=int, default=1, help="Detect every N decoded frames (default: 1)")
    p.add_argument("--conf", type=float, default=.25, help="YOLO confidence threshold")
    p.add_argument("--nms-iou", type=float, default=.5, help="YOLO overlap suppression threshold")
    p.add_argument("--strict-motion-gate", action="store_true",
                   help="Reject even a unique detection when it disagrees with the motion prior")
    p.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size (default: 1280)")
    p.add_argument("--device", default="cpu", help="YOLO device, e.g. cpu or 0")
    p.add_argument("--class-id", type=int, help="Restrict YOLO to one model class ID")
    p.add_argument("--window-size", nargs=2, type=int, default=(960, 540),
                   metavar=("WIDTH", "HEIGHT"), help="Initial preview bounds (default: 960 540)")
    p.add_argument("--appearance", choices=("auto", "off"), default="auto",
                   help="Use local contrast tracking for separable ROIs (default: auto); off uses CSRT")
    p.add_argument("--output", help="New output directory; existing directories are rejected")
    p.add_argument("--start-frame", type=int, default=0, help="Zero-based initialization frame")
    p.add_argument("--max-frames", type=int)
    p.add_argument("--gate", type=float, default=5.991, help="Squared Mahalanobis threshold")
    p.add_argument("--measurement-std", type=float, help="Image measurement std in pixels (default: resolution-scaled)")
    p.add_argument("--acceleration-std", type=float, help="Motion noise std in px/s^2 (default: resolution-scaled)")
    p.add_argument("--initial-velocity-std", type=float, help="Initial velocity uncertainty in px/s (default: resolution-scaled)")
    p.add_argument("--coast-seconds", type=float, default=0.5, help="Max duration of displayed predictions")
    p.add_argument("--fallback-fps", type=float, help="Used only if video FPS is unavailable")
    p.add_argument("--no-video", action="store_true", help="Export CSV/JSON only")
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        p.exit(2, f"Error: {exc}\n")
