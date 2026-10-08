"""Result serialization shared by offline, live, and batched pipelines."""
from __future__ import annotations

import csv
import json
from pathlib import Path


FIELDS = [
    "frame_index", "timestamp", "timestamp_source", "segment", "state", "source",
    "reason", "x", "y", "width", "height", "center_x", "center_y",
    "predicted_x", "predicted_y", "mahalanobis_squared", "candidate_bbox",
    "estimated_x", "estimated_y", "detection_ran", "detection_confidence",
    "class_id", "seconds_since_observation",
]


def csv_row(result):
    row = {key: result.get(key) for key in FIELDS}
    for key, value in zip(("x", "y", "width", "height"), result["bbox"] or [None] * 4):
        row[key] = value
    for key, value in zip(("center_x", "center_y"), result["center"] or [None] * 2):
        row[key] = value
    for key, value in zip(
            ("predicted_x", "predicted_y"), result["predicted_center"] or [None] * 2):
        row[key] = value
    row["candidate_bbox"] = json.dumps(result["candidate_bbox"]) if result["candidate_bbox"] else ""
    for key, value in zip(
            ("estimated_x", "estimated_y"), result.get("estimated_center") or [None] * 2):
        row[key] = value
    return row


class ResultSink:
    """Write the optional diagnostic files without coupling them to recognition."""

    def __init__(self, output: Path, extra_fields=()):
        self.output = Path(output)
        self.extra_fields = list(extra_fields)
        self._records = self._csv_file = None
        self._writer = None

    def __enter__(self):
        self._records = (self.output / "track.jsonl").open("w", encoding="utf-8", buffering=1)
        self._csv_file = (self.output / "track.csv").open(
            "w", encoding="utf-8-sig", newline="")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=FIELDS + self.extra_fields)
        self._writer.writeheader()
        return self

    def write(self, result):
        if self._writer is None:
            raise RuntimeError("ResultSink must be opened with a context manager")
        self._records.write(json.dumps(result, allow_nan=False) + "\n")
        row = csv_row(result)
        row.update({key: result.get(key) for key in self.extra_fields})
        self._writer.writerow(row)

    def flush(self):
        if self._csv_file is not None:
            self._csv_file.flush()
        if self._records is not None:
            self._records.flush()

    def __exit__(self, exc_type, exc, traceback):
        try:
            if self._records is not None:
                self._records.close()
        finally:
            if self._csv_file is not None:
                self._csv_file.close()
        self._records = self._csv_file = self._writer = None
