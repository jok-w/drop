"""Camera connection configuration; credentials never enter reports or argv."""
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class CameraConfig:
    host: str = "192.168.1.64"
    user: str = "admin"
    password: str = field(default="", repr=False)
    port: int = 554
    channel: int = 1
    stream: str = "main"
    latency_ms: int = 200
    timeout: float = 5.0

    def __post_init__(self):
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", self.host):
            raise ValueError("host 必须是主机名或 IPv4 地址，不含 URL/凭据")
        if not 1 <= self.port <= 65535 or self.channel < 1:
            raise ValueError("无效的摄像机端口或通道")
        if self.stream not in ("main", "sub", "third"):
            raise ValueError("未知码流")
        if not math.isfinite(self.timeout) or self.timeout <= 0 or self.latency_ms < 0:
            raise ValueError("超时必须为正数，缓冲时长不能为负数")

    @property
    def url(self):
        number = {"main": 1, "sub": 2, "third": 3}[self.stream]
        return f"rtsp://{self.host}:{self.port}/Streaming/Channels/{self.channel*100+number}"

    def public(self):
        return dict(url=self.url, transport="tcp", channel=self.channel,
                    stream=self.stream, latency_ms=self.latency_ms, timeout=self.timeout)

    @classmethod
    def from_args(cls, args):
        password = (Path(args.password_file).read_text().rstrip("\r\n") if args.password_file
                    else os.environ.get("HIKVISION_PASSWORD", ""))
        return cls(args.host, args.user, password, args.rtsp_port, args.channel,
                   args.stream, args.latency_ms, args.timeout)


@dataclass(frozen=True, slots=True)
class IngestConfig:
    """Capture policy without any dependency on argparse."""

    policy: str = "complete"
    seconds: float | None = None
    segment_seconds: float = 5.0
    min_free_mb: float = 512.0
    decode_width: int = 0
    gst_python: str = "/usr/bin/python3"

    def __post_init__(self):
        if self.policy not in ("complete", "latest"):
            raise ValueError("未知摄像机处理策略")
        for name in ("segment_seconds", "min_free_mb"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须为有限正数")
        if self.seconds is not None and (not math.isfinite(self.seconds) or self.seconds <= 0):
            raise ValueError("seconds 必须为有限正数")
        if self.decode_width < 0:
            raise ValueError("缩放宽度不能为负数")
        if self.decode_width and (
                self.policy != "latest" or self.decode_width < 2 or self.decode_width % 2):
            raise ValueError("decode-width 仅用于 latest，必须为正偶数")

    @classmethod
    def from_args(cls, args):
        return cls(
            policy=args.policy,
            seconds=args.seconds,
            segment_seconds=args.segment_seconds,
            min_free_mb=args.min_free_mb,
            decode_width=args.decode_width,
            gst_python=args.gst_python,
        )
