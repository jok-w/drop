"""Network camera tracking with explicit completeness versus latency policies."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import queue
import subprocess
import time

import cv2
import numpy as np

from .app import FIELDS, VideoClock, config_for_frame, csv_row
from .camera_config import CameraConfig
from .image_size import DEFAULT_IMAGE_SIZE, image_shape
from .rtsp_reader import ArchiveReplay, CameraIngest, open_fragment


class SourceCoordinateDetector:
    def __init__(self, detector):
        self.detector = detector
        self.source_size = None

    def detect(self, frame):
        detections = self.detector.detect(frame)
        width, height = self.source_size
        sx, sy = width/frame.shape[1], height/frame.shape[0]
        for d in detections:
            x, y, w, h = d["bbox"]
            left, top = round(x*sx), round(y*sy)
            right, bottom = min(width, round((x+w)*sx)), min(height, round((y+h)*sy))
            d["bbox"] = [left, top, right-left, bottom-top]
        return detections


class CameraTracker:
    """Keep global target numbering but reset filter/time across camera sessions."""
    def __init__(self, detector, args):
        self.detector = SourceCoordinateDetector(detector)
        self.args = args
        self.tracker = None
        self.session = self.size = None
        self.segments = []
        self.last_index = self.last_stamp = None
        self.offset = 0
        self.last_source_index = -1

    def finish(self, reason):
        if self.tracker is not None:
            self.tracker.finish_segment(self.last_index, self.last_stamp, reason)
            self.segments.extend(dict(s, session_id=self.session) for s in self.tracker.segments)
            self.offset = self.tracker.segment
            self.tracker = None

    def update(self, frame, meta, index):
        from .detection import DetectionTracker
        session, size = meta["session_id"], meta["source_size"]
        stamp = meta["pts_ns"]/1e9
        if (self.session != session or self.size != size or
                (self.last_stamp is not None and stamp <= self.last_stamp)):
            self.finish("stream_disconnect" if self.session != session else "stream_discontinuity")
            self.session, self.size = session, size
            self.last_source_index = -1
        if self.tracker is None:
            self.tracker = DetectionTracker(self.detector,
                config_for_frame(self.args, (size[1], size[0], 3)),
                interval=1, strict_motion_gate=self.args.strict_motion_gate, continuous=True)
            self.tracker.segment = self.offset
        self.detector.source_size = size
        result = self.tracker.update(frame, stamp, index, "stream_pts")
        result.update(session_id=session, source_size=size, processing_size=[frame.shape[1], frame.shape[0]],
                      pts_ns=meta["pts_ns"], source_frame_index=meta["index"],
                      skipped_before=max(0, meta["index"]-self.last_source_index-1))
        self.last_source_index = meta["index"]
        self.last_index, self.last_stamp = index, stamp
        return result


def parser():
    p = argparse.ArgumentParser(description="海康 RTSP/TCP 持续单目标追踪；默认原始录像并顺序分析")
    p.add_argument("--host", default=os.getenv("HIKVISION_HOST", "192.168.1.64"))
    p.add_argument("--user", default=os.getenv("HIKVISION_USER", "admin"))
    p.add_argument("--password-file", help="仅含密码的文件；否则读取 HIKVISION_PASSWORD")
    p.add_argument("--channel", type=int, default=int(os.getenv("HIKVISION_CHANNEL", "1")))
    p.add_argument("--stream", choices=("main", "sub", "third"), default="main")
    p.add_argument("--rtsp-port", type=int, default=554)
    p.add_argument("--policy", choices=("complete", "latest"), default="complete",
                   help="complete=分段落盘后顺序分析；latest=直接解码，允许跳旧帧，仍保存原始录像")
    p.add_argument("--seconds", type=float, help="接收时长；完整模式停止接收后继续处理已保存片段")
    p.add_argument("--segment-seconds", type=float, default=5, help="录像目标分段时长，实际按关键帧切分")
    p.add_argument("--timeout", type=float, default=5)
    p.add_argument("--latency-ms", type=int, default=200)
    p.add_argument("--min-free-mb", type=float, default=512)
    p.add_argument("--decode-width", type=int, default=0,
                   help="latest 模式的 VIC 缩放宽度，0 保持原尺寸；坐标映射回原图")
    p.add_argument("--gst-python", default="/usr/bin/python3")
    p.add_argument("--weights", default="models/detect.pt")
    p.add_argument("--backend", choices=("pt", "tensorrt"), default="tensorrt")
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", nargs="+", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--nms-iou", type=float, default=.5)
    p.add_argument("--class-id", type=int)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--strict-motion-gate", action="store_true")
    p.add_argument("--gate", type=float, default=5.991)
    p.add_argument("--measurement-std", type=float)
    p.add_argument("--acceleration-std", type=float)
    p.add_argument("--initial-velocity-std", type=float)
    p.add_argument("--deceleration-rate", type=float, default=2.)
    p.add_argument("--coast-seconds", type=float, default=.5)
    p.add_argument("--output")
    p.add_argument("--replay", help="重放已停止采集的输出目录；在新目录重新分析所有已完成录像片段")
    p.add_argument("--check", action="store_true", help="仅检查代码环境、GStreamer 插件和 engine 指纹，不连接摄像机")
    return p


def validate(args):
    for name in ("segment_seconds", "timeout", "min_free_mb", "coast_seconds"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        raise ValueError("seconds 必须为有限正数")
    if args.decode_width < 0 or args.warmup < 0 or args.latency_ms < 0:
        raise ValueError("缩放宽度、预热和缓冲不能为负数")
    if args.decode_width and (args.policy != "latest" or args.decode_width < 2 or args.decode_width % 2):
        raise ValueError("decode-width 仅用于 latest，必须为正偶数")
    if args.replay and (args.policy != "complete" or args.seconds is not None):
        raise ValueError("replay 使用 complete 策略，不接受摄像机接收时长 seconds")


def run(args):
    from .detection import YoloDetector
    validate(args)
    camera = CameraConfig.from_args(args) if not args.replay else CameraConfig()
    output = Path(args.output or Path("outputs") / datetime.now().strftime("camera_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise ValueError("输出目录已存在")
    detector = YoloDetector(args.weights, args.conf, args.imgsz, args.device, args.class_id,
                            args.nms_iou, backend=args.backend)
    warmup_height, warmup_width = image_shape(args.imgsz)
    warmup_started = time.monotonic()
    detector.warmup(np.zeros((warmup_height, warmup_width, 3), dtype=np.uint8), args.warmup)
    warmup_seconds = time.monotonic()-warmup_started
    output.mkdir(parents=True)
    ingest = ArchiveReplay(args.replay, output) if args.replay else CameraIngest(camera, output, args)
    tracker = CameraTracker(detector, args)
    count = calls = skipped = 0
    timing = Counter()
    states = Counter()
    reason = "capture_finished"
    error = None
    started = time.monotonic()
    next_log = started
    source_indices = Counter()
    source_label = f"重放={args.replay}" if args.replay else f"摄像机={camera.url}，TCP"
    print(f"{source_label}，策略={args.policy}，输出={output}", flush=True)

    try:
        with (output / "track.jsonl").open("w", buffering=1) as records, \
                (output / "track.csv").open("w", newline="", encoding="utf-8-sig") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDS+["session_id", "source_frame_index", "skipped_before"])
            writer.writeheader()

            def process(frame, meta):
                nonlocal count, calls, skipped, next_log
                detector.last_timing = {}
                tick = time.monotonic()
                result = tracker.update(frame, meta, count)
                result["result_monotonic"] = time.monotonic()
                result["frame_ready_to_result_ms"] = (time.monotonic()-meta["received_monotonic"])*1000
                result["archive_file"] = meta.get("archive_file")
                result["archive_wait_ms"] = (max(0, tick-meta["closed_monotonic"])*1000
                                             if "closed_monotonic" in meta and not args.replay else None)
                records.write(json.dumps(result, allow_nan=False)+"\n")
                row = csv_row(result)
                row.update({k: result[k] for k in ("session_id", "source_frame_index", "skipped_before")})
                writer.writerow(row)
                count += 1
                calls += bool(result["detection_ran"])
                skipped += result["skipped_before"]
                states[result["state"]] += 1
                timing["processing_seconds"] += time.monotonic()-tick
                for k, v in detector.last_timing.items():
                    timing[k] += v
                if time.monotonic() >= next_log:
                    backlog = ingest.store.counts().get(0, 0)
                    queue_text = f"待分析录像段={backlog}" if args.policy == "complete" else f"归档录像段={ingest.archived}"
                    print(f"已分析={count}，检测={calls}，跳帧={skipped}，{queue_text}，重连={ingest.reconnects}", flush=True)
                    csv_file.flush()
                    next_log = time.monotonic()+5

            try:
                ingest.start()
                while True:
                    if args.policy == "complete":
                        item = ingest.store.next()
                        if item is None:
                            if ingest.finished.is_set() and ingest.store.next() is None:
                                break
                            time.sleep(.05)
                            continue
                        cap = open_fragment(item, args)
                        try:
                            clock = VideoClock(cap.get(cv2.CAP_PROP_FPS))
                            local_index = 0
                            while True:
                                ok, frame = cap.read()
                                if not ok:
                                    break
                                relative, clock_source = clock.get(cap.get(cv2.CAP_PROP_POS_MSEC), local_index)
                                if clock_source != "video_timestamp":
                                    raise RuntimeError("录像片段缺少有效 PTS，停止以避免伪造时间轴")
                                session = item["session_id"]
                                meta = dict(session_id=session, source_size=item["source_size"],
                                            index=source_indices[session],
                                            pts_ns=item["start_ns"]+round(relative*1e9),
                                            received_monotonic=time.monotonic(), archive_file=item["path"],
                                            closed_monotonic=item["closed_monotonic"])
                                process(frame, meta)
                                source_indices[session] += 1
                                local_index += 1
                        finally:
                            cap.release()
                        ingest.store.done(item["database_id"])
                    else:
                        try:
                            frame, meta = ingest.frames.get(timeout=.1)
                        except queue.Empty:
                            if ingest.finished.is_set() and ingest.frames.empty():
                                break
                            continue
                        process(frame, meta)
            except KeyboardInterrupt:
                reason = "user_stop"
            except Exception as exc:
                reason = "analysis_error"
                error = type(exc).__name__ + ": " + str(exc)
            finally:
                ingest.close()
                if reason == "capture_finished" and count == 0:
                    reason = "no_frames"
                if ingest.failure:
                    reason = "capture_error"
                tracker.finish(reason)
                counts = ingest.store.counts()
                ingest.store.close()
                summary = dict(camera=None if args.replay else camera.public(), replay=args.replay,
                               policy=args.policy, frames=count,
                               detection_calls=calls, skipped_before_processed_frames=skipped,
                               decoded_frames_reported=ingest.decoded_frames if args.policy == "latest" else None,
                               unprocessed_decoded_frames=(max(0, ingest.decoded_frames-count)
                                   if args.policy == "latest" and ingest.closed_sessions == ingest.sessions else None),
                               sessions=ingest.sessions,
                               reconnects=ingest.reconnects, archived_fragments=ingest.archived,
                               pending_fragments=counts.get(0, 0), analyzed_fragments=counts.get(1, 0),
                               states=dict(states), segments=tracker.segments, end_reason=reason,
                               capture_error=ingest.failure, analysis_error=error,
                               warmup_seconds=warmup_seconds,
                               wall_seconds=time.monotonic()-started, totals=dict(timing),
                               wall_fps=count/max(.001, time.monotonic()-started),
                               detector=detector.metadata,
                               notes=["complete processes finalized archive fragments; latency includes segment wait and backlog.",
                                      "latest explicitly skips decoded frames; original received stream is still archived.",
                                      "Frame-ready latency is not camera exposure latency; network outages may lose data."])
                (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    finally:
        ingest.close()
        ingest.store.close()
    print(f"摄像机处理结束：{count} 帧，结果={output}", flush=True)
    if error or ingest.failure or count == 0:
        raise RuntimeError(error or ingest.failure or "未取得可分析的视频帧；查看 connection.jsonl")
    return output


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.check:
            check(args)
            return
        run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(2, f"Error: {exc}\n")


def check(args):
    """Static gate: imports/plugins/artifact shape and hash, no CUDA allocation or camera I/O."""
    from .image_size import image_shape, size_label
    from .model_artifacts import read_engine_metadata, file_sha256, validate_source
    validate(args)
    result = subprocess.run([args.gst_python, "-I", str(Path(__file__).with_name("gst_camera_worker.py")),
                             "--check"], capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError("GStreamer/GI 环境检查失败："+result.stderr[-1500:]+result.stdout[-1500:])
    # Jetson native libraries may write diagnostics before/after the JSON line.
    payload = next((line for line in result.stdout.splitlines() if line.startswith('{')), None)
    if payload is None:
        raise RuntimeError("GStreamer 检查缺少结构化结果")
    report = dict(gstreamer=json.loads(payload), camera_connection_tested=False,
                  cuda_inference_tested=False)
    if args.backend == "tensorrt":
        weights = Path(args.weights).resolve()
        engine = weights.with_name(f"{weights.stem}.{size_label(args.imgsz)}.fp16.engine")
        metadata = read_engine_metadata(engine)
        tracking = validate_source(metadata, file_sha256(weights))
        if image_shape(tracking["image_size"]) != image_shape(args.imgsz):
            raise ValueError("engine 输入尺寸与参数不匹配")
        report["engine"] = dict(path=str(engine), image_size=tracking["image_size"],
                                source_fingerprint_valid=True, build_runtime=tracking["runtime"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
