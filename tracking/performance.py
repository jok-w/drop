"""Wall-time accounting. Nested detector and background stages are not additive."""
import platform
import sys
import time
from contextlib import contextmanager

import numpy as np


STAGES = ["read_wait_ms", "detector_ms", "preprocess_ms", "inference_ms", "postprocess_ms",
          "result_transfer_ms", "detector_overhead_ms", "tracking_ms", "draw_ms", "display_ms",
          "data_write_ms", "video_submit_ms", "frame_wall_ms", "background_prepare_ms",
          "sample_wait_ms", "shared_copy_ms", "color_convert_ms"]
TIMING_FIELDS = ["frame_index", "timestamp", "state", "detection_ran", "reason", "read_cached", *STAGES]


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return dict(count=0, total_ms=0., mean_ms=None, p50_ms=None, p95_ms=None, max_ms=None)
    data = np.asarray(values, dtype=float)
    return dict(count=len(data), total_ms=float(data.sum()), mean_ms=float(data.mean()),
                p50_ms=float(np.percentile(data, 50)), p95_ms=float(np.percentile(data, 95)),
                max_ms=float(data.max()))


class Performance:
    def __init__(self):
        self.started = time.perf_counter()
        self.sections = {}
        self.rows = []

    @contextmanager
    def section(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sections[name] = self.sections.get(name, 0.) + (time.perf_counter()-start)*1000

    def report(self, pipeline_seconds, environment):
        detected = [r for r in self.rows if r["detection_ran"]]
        return dict(
            schema_version=1, frames=len(self.rows), detection_calls=len(detected),
            startup_and_finalize_ms=self.sections,
            application_wall_seconds=time.perf_counter()-self.started,
            pipeline_seconds_with_finalize=pipeline_seconds,
            pipeline_fps_with_finalize=len(self.rows)/max(pipeline_seconds, 1e-9),
            stages={k: stats([r.get(k) for r in self.rows]) for k in STAGES},
            frame_groups={name: stats([r["frame_wall_ms"] for r in self.rows if test(r)])
                          for name, test in {
                              "detection": lambda r: r["detection_ran"],
                              "prediction": lambda r: r["state"] == "PREDICTED",
                              "lost": lambda r: r["state"] == "LOST",
                          }.items()},
            environment=dict(python=sys.version, platform=platform.platform(), **environment),
            notes=["Detector stages are nested inside detector_ms; do not sum them twice.",
                   "GPU library stage timings synchronize CUDA; they are not pure kernel latency.",
                   "Background preparation overlaps foreground processing; do not add it to wall time.",
                   "Video submit means encoder submission, not durable disk completion.",
                   "Finalize includes buffered file closes and encoder drain, not an fsync durability guarantee.",
                   "Initial cached frame read is in startup; pipeline includes its inference.",
                   "frame_wall_ms excludes the timing-row write and progress printing; pipeline includes both.",
                   "Application wall excludes writing this final report and console printing."])


def reader_snapshot(reader):
    return {key: getattr(reader, attr, None) for key, attr in
            [("background_prepare_ms", "prepare_ms"), ("sample_wait_ms", "pull_ms"),
             ("shared_copy_ms", "copy_ms"), ("color_convert_ms", "convert_ms")]}


def reader_delta(before, after):
    return {k: after[k]-v if v is not None and after[k] is not None else None for k, v in before.items()}
