"""Compare synchronous and bounded asynchronous video output, including finalization."""
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

import cv2

from .app import parser as tracking_parser
from .benchmark import command_for


def inspect_video(path, expected_frames, expected_size, expected_fps):
    capture = cv2.VideoCapture(str(path))
    hashes = []
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open encoded video: {path}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isclose(fps, expected_fps, rel_tol=1e-4, abs_tol=.01):
            raise RuntimeError(f"Encoded FPS mismatch: {path}: {fps} != {expected_fps}")
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if [frame.shape[1], frame.shape[0]] != expected_size:
                raise RuntimeError(f"Encoded frame dimensions differ: {path}")
            hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
    finally:
        capture.release()
    if len(hashes) != expected_frames:
        raise RuntimeError(f"Decoded frame count differs: {path}: {len(hashes)} != {expected_frames}")
    return hashes


def main():
    parser = tracking_parser()
    parser.description = __doc__
    parser.set_defaults(headless=True, warmup=10, max_frames=300, log_every=0, encode_queue=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--report-dir", required=True)
    args = parser.parse_args()
    if (args.mode != "track" or args.no_video or not args.headless or args.repeats < 1
            or args.encode_queue not in (1, 2) or not args.max_frames or args.max_frames < 1):
        parser.error("Requires track mode, headless, saved video, repeats>=1, max-frames>0 and encode-queue=1/2")
    folder = Path(args.report_dir).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    runs = []
    for repeat in range(1, args.repeats + 1):
        for role in (("sync", "async") if repeat % 2 else ("async", "sync")):
            config = copy.copy(args)
            config.encode_queue = 0 if role == "sync" else args.encode_queue
            config.output = str(folder / f"{repeat}_{role}")
            print(f"Run {repeat}/{args.repeats}: {role}, queue={config.encode_queue}", flush=True)
            log = folder / f"{repeat}_{role}.log"
            command = command_for(config)
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Run failed; see {log}")
            output = Path(config.output)
            performance = json.loads((output / "performance.json").read_text())
            if performance["frames"] != args.max_frames:
                raise RuntimeError("Video ended before the requested number of frames")
            if performance["environment"]["encode_queue"] != config.encode_queue:
                raise RuntimeError("Actual output queue differs from requested queue")
            runs.append(dict(repeat=repeat, role=role, output=str(output), command=command,
                             performance=performance))
    # Decode and compare only after all timed runs, outside measured processing.
    comparisons = []
    for repeat in range(1, args.repeats + 1):
        pair = {r["role"]: r for r in runs if r["repeat"] == repeat}
        records, hashes = {}, {}
        for role, run in pair.items():
            output = Path(run["output"])
            records[role] = [json.loads(line) for line in (output / "track.jsonl").read_text().splitlines()]
            summary = json.loads((output / "summary.json").read_text())
            hashes[role] = inspect_video(output / "annotated.mp4", args.max_frames,
                                         run["performance"]["environment"]["output_size"], summary["fps"])
            expected_indices = list(range(args.start_frame, args.start_frame + args.max_frames))
            if [r["frame_index"] for r in records[role]] != expected_indices:
                raise RuntimeError("Tracking frame indices differ from the requested range")
        comparisons.append(dict(repeat=repeat, frames=args.max_frames,
                                tracking_records_identical=records["sync"] == records["async"],
                                decoded_video_pixels_identical=hashes["sync"] == hashes["async"],
                                differing_video_frames=[args.start_frame + i for i, (a, b) in
                                                        enumerate(zip(hashes["sync"], hashes["async"])) if a != b]))
    medians = {role: statistics.median(r["performance"]["pipeline_seconds_with_finalize"]
                                      for r in runs if r["role"] == role) for role in ("sync", "async")}
    report = dict(runs=runs, comparisons=comparisons, median_pipeline_seconds=medians,
                  speedup=medians["sync"] / medians["async"],
                  note="Same frames/model/output; alternating sequential processes. Times include queue/encoder drain, exclude initialization.")
    (folder / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Pipeline speedup={report['speedup']:.3f}; {folder / 'comparison.json'}")
    if not all(c["tracking_records_identical"] and c["decoded_video_pixels_identical"] for c in comparisons):
        raise RuntimeError("Output differences found; inspect comparison.json before accepting the optimization")


if __name__ == "__main__":
    main()
