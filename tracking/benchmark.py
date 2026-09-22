"""Sequential isolated PT/candidate comparisons; identical frames and input shapes."""
import argparse
import copy
import json
from pathlib import Path
import statistics
import subprocess
import sys

import numpy as np

from .app import parser as tracking_parser


def command_for(args):
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / "track_video.py")]
    for action in tracking_parser()._actions:
        if action.dest == "help":
            continue
        value = getattr(args, action.dest)
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                command.append(action.option_strings[0])
        elif value is not None:
            if action.option_strings:
                command.append(action.option_strings[0])
            command.extend(str(v) for v in (value if isinstance(value, (tuple, list)) else [value]))
    return command


def iou(a, b):
    x = max(0, min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0]))
    y = max(0, min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    overlap = x*y
    return overlap/max(1, a[2]*a[3]+b[2]*b[3]-overlap)


def compare(left, right):
    if [r["frame_index"] for r in left] != [r["frame_index"] for r in right]:
        raise ValueError("Frame indices differ; comparison is invalid")
    if any(abs(a["timestamp"]-b["timestamp"]) > .001 for a, b in zip(left, right)):
        raise ValueError("Frame timestamps differ by more than 1 ms")
    overlaps, errors = [], []
    unmatched_left = unmatched_right = count_mismatch = 0
    for a, b in zip(left, right):
        aa, bb = a.get("detections", []), b.get("detections", [])
        count_mismatch += len(aa) != len(bb)
        pairs = sorted([(iou(x["bbox"], y["bbox"]), i, j)
                        for i, x in enumerate(aa) for j, y in enumerate(bb)
                        if x["class_id"] == y["class_id"]], reverse=True)
        used_a, used_b = set(), set()
        for overlap, i, j in pairs:
            if overlap < .5 or i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            overlaps.append(overlap)
            ax, ay, aw, ah = aa[i]["bbox"]
            bx, by, bw, bh = bb[j]["bbox"]
            errors.append(float(np.hypot(ax+aw/2-bx-bw/2, ay+ah/2-by-bh/2)))
        unmatched_left += len(aa)-len(used_a)
        unmatched_right += len(bb)-len(used_b)
    return dict(frames=len(left), matched_boxes=len(overlaps),
                mean_matched_iou=float(np.mean(overlaps)) if overlaps else None,
                mean_matched_center_difference_px=float(np.mean(errors)) if errors else None,
                unmatched_pt_boxes=unmatched_left, unmatched_candidate_boxes=unmatched_right,
                detection_count_mismatch_frames=count_mismatch,
                note="Same-class greedy IoU>=0.5 matching; consistency, not ground-truth accuracy.")


def main():
    parser = tracking_parser()
    parser.description = __doc__
    parser.set_defaults(max_frames=300, warmup=10, headless=True, no_video=True, mode="detect-benchmark")
    parser.add_argument("--candidate", choices=("onnx", "tensorrt"), default="tensorrt")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.mode != "detect-benchmark" or not args.weights:
        parser.error("Requires --weights, repeats>=1 and detect-benchmark mode")
    folder = Path(args.report_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    runs, comparisons = [], []
    for repeat in range(args.repeats):
        # Alternate order across repeats to reduce systematic cache/thermal ordering bias.
        backends = ("pt", args.candidate) if repeat % 2 == 0 else (args.candidate, "pt")
        records = {}
        for backend in backends:
            config = copy.copy(args)
            config.backend = backend
            if backend != "tensorrt":
                config.engine = None
            if backend != "onnx":
                config.onnx = None
            config.output = str(folder / f"{repeat+1}_{backend}")
            command = command_for(config)
            log = folder / f"{repeat+1}_{backend}.log"
            print(f"Run {repeat+1}/{args.repeats}: {backend}", flush=True)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"{backend} failed; see {log}. No fallback was used.")
            output = Path(config.output)
            report = json.loads((output / "performance.json").read_text())
            if report["environment"]["detector"]["backend"] != backend:
                raise RuntimeError("Actual model backend differs from requested backend")
            expected_format = "engine" if backend == "tensorrt" else backend
            if report["environment"]["detector"].get("actual_format") != expected_format:
                raise RuntimeError("Runtime model format differs from requested backend")
            if report["detection_calls"] != report["frames"]:
                raise RuntimeError("Benchmark must infer on every frame")
            runs.append(dict(repeat=repeat+1, backend=backend, output=str(output), performance=report))
            records[backend] = [json.loads(line) for line in (output / "track.jsonl").read_text().splitlines()]
        comparisons.append(compare(records["pt"], records[args.candidate]))
    medians = {backend: statistics.median(r["performance"]["stages"]["detector_ms"]["mean_ms"]
                                         for r in runs if r["backend"] == backend)
               for backend in ("pt", args.candidate)}
    report = dict(runs=runs, comparisons=comparisons, median_detector_mean_ms=medians,
                  detector_speedup=medians["pt"]/medians[args.candidate],
                  note="No drawing/video encoding. Includes data output. Hardware clocks/power must be held comparable externally.")
    (folder / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Detector speedup={report['detector_speedup']:.3f}; {folder / 'comparison.json'}")


if __name__ == "__main__":
    main()
