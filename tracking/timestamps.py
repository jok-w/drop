"""Timestamp normalization shared by file and archive readers."""
import math


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
