"""Duplicate one live camera frame into a fixed batch of independent logical cameras."""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path
import queue
import subprocess
import time

import numpy as np

from .app import FIELDS, csv_row
from .camera_app import CameraTracker, parser as camera_parser, validate as validate_camera
from .camera_config import CameraConfig
from .image_size import image_shape, size_label
from .performance import stats
from .rtsp_reader import CameraIngest


class SuppliedDetector:
    """Give a normal per-camera tracker one result from the preceding batch call."""

    def __init__(self):
        self.pending = None

    def supply(self, detections):
        if self.pending is not None:
            raise RuntimeError("上一批逻辑摄像机检测结果未被消费")
        self.pending = detections

    def detect(self, unused_frame):
        if self.pending is None:
            raise RuntimeError("逻辑摄像机没有可用的 batch 检测结果")
        result, self.pending = self.pending, None
        return result

    def discard(self):
        self.pending = None


def parser():
    p = camera_parser()
    p.description = "单路 RTSP 画面复制为多个逻辑摄像机，执行固定 batch 识别与独立跟踪"
    p.set_defaults(policy="latest", seconds=60.)
    p.add_argument("--batch-size", type=int, choices=(1, 2, 4, 6, 8), default=2,
                   help="逻辑摄像机数量及固定 TensorRT batch（默认 2）")
    p.add_argument("--engine", help="匹配 batch-size 的 TensorRT engine；未指定时按标准名称查找")
    p.add_argument("--profile", choices=("throughput", "diagnostic"), default="throughput")
    return p


def validate(args):
    validate_camera(args)
    if args.policy != "latest" or args.replay:
        raise ValueError("单摄像机 batch 测试只支持实时 latest；录像仍由同一接收管线保存")
    if args.seconds is None:
        raise ValueError("batch 测试必须设置有限的 --seconds，避免统计样本无限增长")
    if args.engine and args.backend != "tensorrt":
        raise ValueError("--engine 仅能与 --backend tensorrt 一起使用")


def engine_path(args) -> Path:
    weights = Path(args.weights).resolve()
    label = "" if args.batch_size == 1 else f".b{args.batch_size}"
    return Path(args.engine).resolve() if args.engine else weights.with_name(
        f"{weights.stem}.{size_label(args.imgsz)}{label}.fp16.engine")


def check(args):
    """Static validation only; it does not connect to the camera or allocate CUDA."""
    from .model_artifacts import file_sha256, read_engine_metadata, validate_source

    validate(args)
    result = subprocess.run(
        [args.gst_python, "-I", str(Path(__file__).with_name("gst_camera_worker.py")), "--check"],
        capture_output=True, text=True, timeout=15)
    if result.returncode:
        raise RuntimeError("GStreamer/GI 环境检查失败：" + result.stderr[-1500:] + result.stdout[-1500:])
    payload = next((line for line in result.stdout.splitlines() if line.startswith("{")), None)
    if payload is None:
        raise RuntimeError("GStreamer 检查缺少结构化结果")
    report = dict(gstreamer=json.loads(payload), camera_connection_tested=False,
                  cuda_inference_tested=False, batch_size=args.batch_size)
    if args.backend == "tensorrt":
        path = engine_path(args)
        metadata = read_engine_metadata(path)
        tracking = validate_source(metadata, file_sha256(Path(args.weights).resolve()))
        if (tracking.get("max_batch") != args.batch_size
                or image_shape(tracking.get("image_size")) != image_shape(args.imgsz)):
            raise ValueError("engine 的固定 batch 或输入尺寸与参数不匹配")
        report["engine"] = dict(path=str(path), image_size=tracking["image_size"],
                                batch_size=tracking["max_batch"], source_fingerprint_valid=True,
                                build_runtime=tracking["runtime"])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def run(args):
    from .detection import YoloDetector

    validate(args)
    camera = CameraConfig.from_args(args)
    output = Path(args.output or Path("outputs") / datetime.now().strftime(
        "camera_batch_%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        raise ValueError("输出目录已存在")

    detector = YoloDetector(
        args.weights, args.conf, args.imgsz, args.device, args.class_id, args.nms_iou,
        backend=args.backend, engine=args.engine, profile=args.profile, batch_size=args.batch_size)
    height, width = image_shape(args.imgsz)
    warmup_frames = [np.zeros((height, width, 3), dtype=np.uint8) for _ in range(args.batch_size)]
    warmup_started = time.monotonic()
    detector.warmup_batch(warmup_frames, args.warmup)
    warmup_seconds = time.monotonic() - warmup_started

    output.mkdir(parents=True)
    ingest = CameraIngest(camera, output, args)
    supplied = [SuppliedDetector() for _ in range(args.batch_size)]
    trackers = [CameraTracker(item, args) for item in supplied]
    camera_ids = [f"virtual_{index + 1:02d}" for index in range(args.batch_size)]
    states = {camera_id: Counter() for camera_id in camera_ids}
    metrics = {name: [] for name in (
        "duplicate_ms", "queue_wait_ms", "batch_detector_ms", "detector_per_image_ms",
        "preprocess_per_image_ms", "inference_per_image_ms", "postprocess_per_image_ms",
        "result_transfer_batch_ms", "batch_wall_ms", "frame_ready_to_all_results_ms")}
    physical_frames = skipped = 0
    last_source_index = -1
    last_session = None
    error = None
    reason = "capture_finished"
    started = time.monotonic()
    next_log = started

    print(f"摄像机={camera.url}，TCP，单路复制 batch={args.batch_size}，输出={output}", flush=True)
    try:
        fields = FIELDS + ["camera_id", "session_id", "source_frame_index", "skipped_before",
                           "physical_frame_index"]
        with (output / "track.jsonl").open("w", buffering=1) as records, \
                (output / "track.csv").open("w", newline="", encoding="utf-8-sig") as csv_file, \
                (output / "batch_timing.jsonl").open("w", buffering=1) as timing_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()
            try:
                ingest.start()
                while True:
                    try:
                        frame, meta = ingest.frames.get(timeout=.1)
                    except queue.Empty:
                        if ingest.finished.is_set() and ingest.frames.empty():
                            break
                        continue

                    batch_started = time.monotonic()
                    queue_wait_ms = max(0., (batch_started - meta["received_monotonic"]) * 1000)
                    duplicate_started = time.monotonic()
                    batch_frames = ([frame] if args.batch_size == 1
                                    else [frame.copy() for _ in range(args.batch_size)])
                    duplicate_ms = (time.monotonic() - duplicate_started) * 1000
                    detections = detector.detect_batch(batch_frames)
                    if len(detections) != args.batch_size:
                        raise RuntimeError("batch 检测结果数量错误")

                    if meta["session_id"] != last_session:
                        last_session = meta["session_id"]
                        last_source_index = -1
                    gap = max(0, meta["index"] - last_source_index - 1)
                    skipped += gap
                    last_source_index = meta["index"]

                    for slot, (camera_id, logical_tracker, adapter) in enumerate(
                            zip(camera_ids, trackers, supplied)):
                        adapter.supply(detections[slot])
                        logical_meta = dict(meta, session_id=f"{meta['session_id']}:{camera_id}")
                        try:
                            result = logical_tracker.update(
                                batch_frames[slot], logical_meta, physical_frames)
                        finally:
                            adapter.discard()
                        result.update(camera_id=camera_id, physical_frame_index=physical_frames,
                                      physical_session_id=meta["session_id"],
                                      result_monotonic=time.monotonic(),
                                      frame_ready_to_result_ms=(time.monotonic()
                                          - meta["received_monotonic"]) * 1000)
                        records.write(json.dumps(result, allow_nan=False) + "\n")
                        row = csv_row(result)
                        row.update({key: result[key] for key in (
                            "camera_id", "session_id", "source_frame_index", "skipped_before",
                            "physical_frame_index")})
                        writer.writerow(row)
                        states[camera_id][result["state"]] += 1

                    batch_finished = time.monotonic()
                    timing = detector.last_timing
                    row = dict(
                        physical_frame_index=physical_frames, session_id=meta["session_id"],
                        source_frame_index=meta["index"], batch_size=args.batch_size,
                        skipped_before=gap, duplicate_ms=duplicate_ms, queue_wait_ms=queue_wait_ms,
                        batch_detector_ms=timing["detector_ms"],
                        detector_per_image_ms=timing["detector_ms"] / args.batch_size,
                        preprocess_per_image_ms=timing["preprocess_ms"],
                        inference_per_image_ms=timing["inference_ms"],
                        postprocess_per_image_ms=timing["postprocess_ms"],
                        result_transfer_batch_ms=timing["result_transfer_ms"],
                        batch_wall_ms=(batch_finished - batch_started) * 1000,
                        frame_ready_to_all_results_ms=(batch_finished
                            - meta["received_monotonic"]) * 1000)
                    timing_file.write(json.dumps(row, allow_nan=False) + "\n")
                    for name in metrics:
                        metrics[name].append(row[name])
                    physical_frames += 1

                    if time.monotonic() >= next_log:
                        wall = max(.001, time.monotonic() - started)
                        print(f"batch调用={physical_frames}，逻辑帧={physical_frames * args.batch_size}，"
                              f"逻辑吞吐={physical_frames * args.batch_size / wall:.2f}fps，"
                              f"源跳帧={skipped}，重连={ingest.reconnects}", flush=True)
                        csv_file.flush()
                        next_log = time.monotonic() + 5
            except KeyboardInterrupt:
                reason = "user_stop"
            except Exception as exc:
                reason = "analysis_error"
                error = type(exc).__name__ + ": " + str(exc)
            finally:
                ingest.close()
                if ingest.failure:
                    reason = "capture_error"
                if reason == "capture_finished" and physical_frames == 0:
                    reason = "no_frames"
                for logical_tracker in trackers:
                    logical_tracker.finish(reason)
                counts = ingest.store.counts()
                ingest.store.close()
                wall_seconds = time.monotonic() - started
                summary = dict(
                    mode="single-camera-duplicated-batch", camera=camera.public(),
                    batch_size=args.batch_size, physical_frames=physical_frames,
                    logical_frames=physical_frames * args.batch_size, batch_calls=physical_frames,
                    skipped_before_processed_source_frames=skipped,
                    decoded_frames_reported=ingest.decoded_frames,
                    unprocessed_decoded_frames=(max(0, ingest.decoded_frames - physical_frames)
                        if ingest.closed_sessions == ingest.sessions else None),
                    sessions=ingest.sessions, reconnects=ingest.reconnects,
                    archived_fragments=ingest.archived, pending_fragments=counts.get(0, 0),
                    end_reason=reason, capture_error=ingest.failure, analysis_error=error,
                    warmup_seconds=warmup_seconds, wall_seconds=wall_seconds,
                    physical_fps=physical_frames / max(.001, wall_seconds),
                    logical_fps=physical_frames * args.batch_size / max(.001, wall_seconds),
                    timings={name: stats(values) for name, values in metrics.items()},
                    logical_cameras={camera_id: dict(
                        frames=physical_frames, states=dict(states[camera_id]),
                        segments=trackers[index].segments)
                        for index, camera_id in enumerate(camera_ids)},
                    detector=detector.metadata,
                    notes=[
                        "One RTSP decode is copied in host memory; this measures batch inference, "
                        "per-camera tracking, and output, not multi-RTSP/NVDEC/network load.",
                        "Ultralytics preprocess/inference/postprocess timings are per image; "
                        "batch_detector_ms and batch_wall_ms are per batch.",
                        "latest may replace source frames while processing; source gaps are explicit.",
                    ])
                (output / "summary.json").write_text(
                    json.dumps(summary, indent=2, allow_nan=False) + "\n")
    finally:
        ingest.close()

    print(f"batch 测试结束：调用={physical_frames}，逻辑帧={physical_frames * args.batch_size}，"
          f"结果={output}", flush=True)
    if error or ingest.failure or physical_frames == 0:
        raise RuntimeError(error or ingest.failure or "未取得可分析的视频帧；查看 connection.jsonl")
    return output


def main():
    p = parser()
    args = p.parse_args()
    try:
        if args.check:
            check(args)
        else:
            run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
