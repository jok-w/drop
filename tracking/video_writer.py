"""Jetson NVENC output through a bounded stdin pipe; no OpenCV GStreamer required."""
from __future__ import annotations

import logging
import math
import os
import platform
import select
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)
GST_ELEMENTS = ("fdsrc", "rawvideoparse", "nvvidconv", "nvv4l2h264enc",
                "h264parse", "qtmux", "filesink")
PIPE_BUFFER_BYTES = 1024 * 1024


def configure_pipe(fd: int) -> int:
    """Use fewer pipe fragments for large frames; keep the OS default if resizing is denied."""
    try:
        import fcntl
    except ImportError:
        return 65536
    try:
        fcntl.fcntl(fd, fcntl.F_SETPIPE_SZ, PIPE_BUFFER_BYTES)
    except OSError:
        logger.debug("无法扩大视频管道缓冲，使用系统默认容量", exc_info=True)
    try:
        return fcntl.fcntl(fd, fcntl.F_GETPIPE_SZ)
    except OSError:
        return 65536


def is_jetson() -> bool:
    return platform.system() == "Linux" and Path("/etc/nv_tegra_release").is_file()


def check_gstreamer() -> str:
    if platform.system() != "Linux":
        raise RuntimeError("GStreamer NVENC 输出仅支持 Jetson Linux")
    launch = shutil.which("gst-launch-1.0")
    inspect = shutil.which("gst-inspect-1.0")
    if not launch or not inspect:
        raise RuntimeError("缺少 gst-launch-1.0 或 gst-inspect-1.0")
    for element in GST_ELEMENTS:
        try:
            result = subprocess.run([inspect, element], capture_output=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise RuntimeError(f"检查 GStreamer 插件 {element} 失败：{error}") from error
        if result.returncode:
            raise RuntimeError(f"缺少可用 GStreamer 插件：{element}")
    return launch


def pipeline_command(launch: str, path: Path, fps: float, width: int,
                     height: int, bitrate: int) -> list[str]:
    rate = Fraction(fps).limit_denominator(100000)
    # Gst parses property strings itself, even though no shell is involved.
    location = str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')
    return [launch, "-q", "-e", "fdsrc", "fd=0", f"blocksize={PIPE_BUFFER_BYTES}", "!",
            "rawvideoparse", "format=bgrx", f"width={width}", f"height={height}",
            f"framerate={rate.numerator}/{rate.denominator}", "!",
            "nvvidconv", "!", "video/x-raw(memory:NVMM),format=NV12", "!",
            "nvv4l2h264enc", f"bitrate={bitrate}", "!", "h264parse", "!",
            "qtmux", "!", "filesink", f'location="{location}"', "sync=false"]


class GStreamerVideoWriter:
    """Submit every frame in order, drain EOS, then check the completed MP4 metadata."""

    def __init__(self, launch: str, path: Path, fps: float, width: int,
                 height: int, bitrate: int, timeout: float = 30.0) -> None:
        if not math.isfinite(fps) or fps <= 0 or min(width, height) < 2:
            raise ValueError("无效的输出帧率或尺寸")
        if width % 2 or height % 2 or bitrate <= 0:
            raise ValueError("硬件编码需要偶数宽高和正码率")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("编码器超时必须为有限正数")
        self.path, self.fps = path, fps
        self.width, self.height, self.timeout = width, height, timeout
        self.frames = 0
        self._cancelled = threading.Event()
        self._closed = False
        self._last_details = ""
        # A regular file cannot deadlock on verbose encoder stderr like a PIPE can.
        self._log = tempfile.TemporaryFile()
        self._process = None
        try:
            self._process = subprocess.Popen(
                pipeline_command(launch, path, fps, width, height, bitrate),
                stdin=subprocess.PIPE, stdout=self._log, stderr=subprocess.STDOUT,
                bufsize=0,
            )
            os.set_blocking(self._process.stdin.fileno(), False)
            self.pipe_buffer_bytes = configure_pipe(self._process.stdin.fileno())
        except BaseException:
            self.abort()
            raise

    def _details(self) -> str:
        if self._log.closed:
            return self._last_details
        self._log.seek(0, os.SEEK_END)
        self._log.seek(max(0, self._log.tell() - 4096))
        self._last_details = self._log.read().decode("utf-8", errors="replace").strip()
        return self._last_details

    def isOpened(self) -> bool:
        return not self._closed and self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        if not self.isOpened():
            raise RuntimeError(f"GStreamer 编码进程已退出：{self._details()}")
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError("硬件编码输入必须为指定尺寸的 uint8 BGR 图像")
        # BGRx has an unambiguous width*4 row stride (including widths not /4).
        # nvvidconv accepts BGRx directly and performs RGB -> NV12 on Jetson.
        data = memoryview(cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)).cast("B")
        fd = self._process.stdin.fileno()
        deadline = time.monotonic() + self.timeout
        try:
            while data:
                if self._cancelled.is_set():
                    raise OSError("视频写入已取消")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("向 GStreamer 提交视频帧超时")
                if self._process.poll() is not None:
                    raise BrokenPipeError("GStreamer 编码进程提前退出")
                if not select.select([], [fd], [], min(remaining, 0.5))[1]:
                    continue
                try:
                    written = os.write(fd, data[:self.pipe_buffer_bytes])
                except BlockingIOError:
                    continue
                if written <= 0:
                    raise BrokenPipeError("GStreamer 管道未接受数据")
                data = data[written:]
            self.frames += 1
        except (OSError, TimeoutError) as error:
            details = self._details()
            self.abort()
            raise RuntimeError(f"硬件编码写入失败：{error}\n{details}") from error

    def release(self) -> None:
        if self._closed:
            return
        try:
            # EOF causes fdsrc EOS: the encoder drains and qtmux writes the MP4 index.
            self._process.stdin.close()
            try:
                code = self._process.wait(timeout=self.timeout)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"GStreamer 编码器收尾超时：{self._details()}") from error
            if code != 0:
                raise RuntimeError(f"GStreamer 编码失败，退出码={code}：{self._details()}")
            if self.frames:
                self._verify_output()
        finally:
            self.abort()

    def _verify_output(self) -> None:
        capture = cv2.VideoCapture(str(self.path))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"硬件编码输出无法打开：{self.path}")
            count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = capture.get(cv2.CAP_PROP_FPS)
            width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
            values = (count, fps, width, height)
            if (not all(math.isfinite(v) for v in values)
                    or abs(count - self.frames) > 0.1
                    or (width, height) != (self.width, self.height)
                    # MP4 muxing can round a source rate such as 25.035 to
                    # 25.03; that difference does not indicate dropped frames.
                    or not math.isclose(fps, self.fps, rel_tol=1e-4, abs_tol=1e-2)):
                raise RuntimeError(
                    f"硬件输出元数据校验失败：提交={self.frames} 帧，文件={count} 帧，"
                    f"尺寸={width}x{height}，fps={fps}，预期={self.width}x{self.height}@{self.fps}")
            logger.info("硬件输出元数据校验通过：帧数=%s，尺寸=%sx%s，fps=%.6f",
                        self.frames, self.width, self.height, fps)
        finally:
            capture.release()

    def cancel_pending(self) -> None:
        """Thread-safe cancellation request; resource cleanup stays on the writer thread."""
        self._cancelled.set()

    def abort(self) -> None:
        """Bounded cleanup on failure; never hide the original processing exception."""
        if self._closed:
            return
        self._closed = True
        try:
            if self._process is not None:
                if self._process.poll() is None:
                    self._process.kill()
                self._process.wait(timeout=5)
                if self._process.stdin is not None:
                    self._process.stdin.close()
        except (OSError, subprocess.TimeoutExpired):
            logger.exception("清理 GStreamer 编码进程失败")
        finally:
            self._log.close()


class AsyncVideoWriter:
    """Single producer, bounded FIFO, one worker owning writes and finalization.

    Frames are copied on submission so callers may safely reuse their canvas.
    Only release() confirms that all submitted frames and the encoder have drained.
    """

    def __init__(self, writer, capacity=2, timeout=None):
        if capacity not in (1, 2):
            raise ValueError("异步输出队列容量必须是 1 或 2")
        self.writer, self.capacity = writer, capacity
        self.timeout = getattr(writer, "timeout", 30.0) if timeout is None else timeout
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("异步输出超时必须为有限正数")
        self._condition = threading.Condition()
        self._pending = deque()
        self._closing = self._aborted = self._done = False
        self._error = None
        self.submitted = self.completed = self.peak_queue_depth = 0
        self._write_times = []
        self._thread = threading.Thread(target=self._run, name="video-encoder", daemon=True)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f"异步视频输出失败：{self._error}") from self._error
        if self._aborted:
            raise RuntimeError("异步视频输出已取消")

    def write(self, frame):
        deadline = time.monotonic() + self.timeout
        with self._condition:
            while True:
                self._check()
                if self._closing or self._done:
                    raise RuntimeError("异步视频输出已关闭")
                if len(self._pending) < self.capacity:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("等待异步视频输出队列超时")
                self._condition.wait(min(remaining, .1))
            self._pending.append(frame.copy())
            self.submitted += 1
            self.peak_queue_depth = max(self.peak_queue_depth, len(self._pending))
            self._condition.notify_all()

    def _run(self):
        released = False
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(lambda: self._pending or self._closing or self._aborted)
                    if self._aborted:
                        return
                    if not self._pending:  # Closing, with the FIFO completely drained.
                        break
                    frame = self._pending.popleft()
                    self._condition.notify_all()
                started = time.perf_counter()
                self.writer.write(frame)
                duration = (time.perf_counter() - started) * 1000
                with self._condition:
                    self.completed += 1
                    self._write_times.append(duration)
                del frame
            self.writer.release()
            released = True
        except BaseException as error:
            with self._condition:
                self._error = error
                self._condition.notify_all()
        finally:
            if not released:
                try:
                    if hasattr(self.writer, "abort"):
                        self.writer.abort()
                    else:
                        self.writer.release()
                except BaseException as error:
                    with self._condition:
                        if self._error is None:
                            self._error = error
            with self._condition:
                self._pending.clear()
                self._done = True
                self._condition.notify_all()

    def release(self):
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        # At most capacity pending writes, one active write, and encoder drain.
        self._thread.join((self.capacity + 2) * self.timeout + 5)
        if self._thread.is_alive():
            self.abort()
            raise RuntimeError("异步视频输出收尾超时")
        with self._condition:
            self._check()
            if self.completed != self.submitted:
                raise RuntimeError(f"异步输出帧数不匹配：提交={self.submitted}，完成={self.completed}")

    def abort(self):
        with self._condition:
            if self._done:
                return
            self._aborted = True
            self._pending.clear()
            self._condition.notify_all()
        # Never close/kill the underlying writer concurrently with its write/release.
        if hasattr(self.writer, "cancel_pending"):
            self.writer.cancel_pending()
        self._thread.join(self.timeout + 5)
        if self._thread.is_alive():
            logger.error("异步视频输出线程未在超时内退出")

    def statistics(self):
        from .performance import stats
        with self._condition:
            return dict(queue_capacity=self.capacity, peak_queue_depth=self.peak_queue_depth,
                        submitted_frames=self.submitted, completed_frames=self.completed,
                        background_write_ms=stats(self._write_times),
                        note="Background writes overlap foreground processing; do not add their time to frame_wall_ms.")


def create_video_writer(path: Path, fps: float, width: int, height: int,
                        encoder: str, codec: str, bitrate: int):
    use_gst = encoder == "gstreamer" or (encoder == "auto" and is_jetson()
                                         and path.suffix.lower() == ".mp4")
    if use_gst:
        try:
            launch = check_gstreamer()
        except RuntimeError as error:
            if encoder == "gstreamer":
                raise
            logger.warning("硬件输出不可用，回退 OpenCV：%s", error)
        else:
            writer = GStreamerVideoWriter(launch, path, fps, width, height, bitrate)
            logger.info("视频输出：GStreamer / nvv4l2h264enc（NVENC），H.264，码率=%s bps", bitrate)
            return writer
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f"无法创建输出视频，请检查编码 {codec} 和路径：{path}")
    logger.info("视频输出：OpenCV，编码=%s", codec)
    return writer
