"""Shared static validation for live-camera entry points."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from .image_size import image_shape, size_label
from .model_artifacts import file_sha256, read_engine_metadata, validate_source


def engine_path(weights, image_size, batch_size=1, explicit=None) -> Path:
    weights = Path(weights).resolve()
    label = "" if batch_size == 1 else f".b{batch_size}"
    return Path(explicit).resolve() if explicit else weights.with_name(
        f"{weights.stem}.{size_label(image_size)}{label}.fp16.engine")


def static_camera_check(args, batch_size=1, explicit_engine=None):
    worker = Path(__file__).with_name("gst_camera_worker.py")
    result = subprocess.run(
        [args.gst_python, "-I", str(worker), "--check"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(
            "GStreamer/GI 环境检查失败：" + result.stderr[-1500:] + result.stdout[-1500:])
    payload = next((line for line in result.stdout.splitlines() if line.startswith("{")), None)
    if payload is None:
        raise RuntimeError("GStreamer 检查缺少结构化结果")
    report = dict(
        gstreamer=json.loads(payload),
        camera_connection_tested=False,
        cuda_inference_tested=False,
        batch_size=batch_size,
    )
    if args.backend == "tensorrt":
        path = engine_path(args.weights, args.imgsz, batch_size, explicit_engine)
        metadata = read_engine_metadata(path)
        tracking = validate_source(metadata, file_sha256(Path(args.weights).resolve()))
        if (tracking.get("max_batch", 1) != batch_size
                or image_shape(tracking["image_size"]) != image_shape(args.imgsz)):
            raise ValueError("engine 的固定 batch 或输入尺寸与参数不匹配")
        report["engine"] = dict(
            path=str(path),
            image_size=tracking["image_size"],
            batch_size=batch_size,
            source_fingerprint_valid=True,
            build_runtime=tracking["runtime"],
        )
    return report
