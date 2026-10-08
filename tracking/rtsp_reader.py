"""Supervised RTSP ingest with durable fragment queue and bounded live frames."""
import json
import mmap
import os
from pathlib import Path
import queue
import select
import socket
import sqlite3
import subprocess
import tempfile
import threading
import time

import cv2
import numpy as np

from .video_reader import receive_message


class SegmentStore:
    """Disk-backed backlog, so long capture sessions don't accumulate Python lists."""
    def __init__(self, path):
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS fragments (id INTEGER PRIMARY KEY, "
                        "path TEXT UNIQUE, payload TEXT, done INTEGER DEFAULT 0)")
        self.db.execute("CREATE INDEX IF NOT EXISTS pending_fragments ON fragments(done,id)")
        self.db.commit()

    def add(self, item):
        with self.lock:
            self.db.execute("INSERT OR IGNORE INTO fragments(path,payload) VALUES (?,?)",
                            (item["path"], json.dumps(item)))
            self.db.commit()

    def next(self):
        with self.lock:
            row = self.db.execute("SELECT id,payload FROM fragments WHERE done=0 ORDER BY id LIMIT 1").fetchone()
            return None if row is None else dict(json.loads(row[1]), database_id=row[0])

    def done(self, index):
        with self.lock:
            self.db.execute("UPDATE fragments SET done=1 WHERE id=?", (index,))
            self.db.commit()

    def counts(self):
        with self.lock:
            return dict(self.db.execute("SELECT done, count(*) FROM fragments GROUP BY done").fetchall())

    def close(self):
        self.db.close()


class JournalTail:
    def __init__(self, path):
        self.path, self.offset, self.partial = Path(path), 0, ""

    def read(self):
        if not self.path.exists():
            return []
        with self.path.open() as stream:
            stream.seek(self.offset)
            text = self.partial + stream.read()
            self.offset = stream.tell()
        lines = text.split("\n")
        self.partial = lines.pop()
        return [json.loads(line) for line in lines if line]


class Worker:
    def __init__(self, camera, folder, args):
        self.folder = folder
        folder.mkdir()
        self.control = self.shared = self.memory = self.process = self.log = None
        self.tail = JournalTail(folder / "events.jsonl")
        child = None
        read_fd = write_fd = None
        try:
            self.control, child = socket.socketpair()
            self.control.settimeout(2)
            read_fd, write_fd = os.pipe()
            config = dict(camera.public(), user=camera.user, password=camera.password,
                          folder=str(folder), mode=args.policy, decode_width=args.decode_width,
                          segment_seconds=args.segment_seconds,
                          min_free_bytes=int(args.min_free_mb*1024*1024), memory_bytes=3840*2160*4)
            command = [args.gst_python, "-I", str(Path(__file__).with_name("gst_camera_worker.py")),
                       "--config-fd", str(read_fd), "--control-fd", str(child.fileno())]
            fds = [read_fd, child.fileno()]
            if args.policy == "latest":
                self.memory = tempfile.TemporaryFile(dir="/dev/shm")
                self.memory.truncate(config["memory_bytes"])
                self.shared = mmap.mmap(self.memory.fileno(), config["memory_bytes"])
                command += ["--memory-fd", str(self.memory.fileno())]
                fds.append(self.memory.fileno())
            self.log = (folder / "worker.log").open("wb")
            self.process = subprocess.Popen(command, pass_fds=fds, stdin=subprocess.DEVNULL,
                                            stdout=self.log, stderr=self.log)
            os.close(read_fd)
            read_fd = None
            child.close()
            child = None
            with os.fdopen(write_fd, "w") as stream:
                write_fd = None
                json.dump(config, stream)
        except BaseException:
            if child is not None:
                child.close()
            for fd in (read_fd, write_fd):
                if fd is not None:
                    os.close(fd)
            self.close()
            raise

    def frame(self):
        if self.shared is None or not select.select([self.control], [], [], 0)[0]:
            return None
        message = receive_message(self.control)
        if message.get("event") != "frame":
            raise RuntimeError("Invalid live frame message")
        width, height = message["width"], message["height"]
        if min(width, height) < 2 or width*height*4 > len(self.shared):
            raise RuntimeError("Invalid live frame size")
        view = np.ndarray((height, width, 4), np.uint8, buffer=self.shared)
        try:
            image = cv2.cvtColor(view, cv2.COLOR_BGRA2BGR)
        finally:
            del view
        self.control.sendall(b"A")
        return image, message

    def request_stop(self):
        try:
            self.control.sendall(b"Q")
        except (OSError, AttributeError):
            pass

    def close(self):
        try:
            if self.process is not None and self.process.poll() is None:
                self.request_stop()
                try:
                    self.process.wait(timeout=7)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=3)
        finally:
            for name in ("control", "shared", "memory", "log"):
                obj = getattr(self, name)
                if obj is not None:
                    obj.close()
                    setattr(self, name, None)


class CameraIngest:
    def __init__(self, camera, folder, args):
        self.camera, self.folder, self.args = camera, folder, args
        self.store = SegmentStore(folder / "fragments.sqlite3")
        self.frames = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.finished = threading.Event()
        self.failure = None
        self.sessions = self.reconnects = self.archived = 0
        self.decoded_frames = self.closed_sessions = 0
        self.events = (folder / "connection.jsonl").open("a", buffering=1)
        self.thread = threading.Thread(target=self._run, name="camera-ingest", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _run(self):
        started = time.monotonic()
        delay = 1.
        worker = None
        try:
            while not self.stop_event.is_set():
                if self.args.seconds and time.monotonic()-started >= self.args.seconds:
                    break
                self.sessions += 1
                session = f"session_{self.sessions:05d}"
                worker = Worker(self.camera, self.folder / session, self.args)
                opened, source_size = {}, None
                stop_sent = False
                stop_at = None
                healthy = False
                fatal = False

                def poll_events():
                    nonlocal source_size, healthy, fatal
                    for event in worker.tail.read():
                        event["session_id"] = session
                        self.events.write(json.dumps(event)+"\n")
                        if event["event"] == "stream":
                            source_size = event["source_size"]
                            healthy = True
                        elif event["event"] == "fragment_opened":
                            opened[event["path"]] = event["running_ns"]
                        elif event["event"] == "fragment_closed":
                            if event["path"] not in opened or source_size is None:
                                raise RuntimeError("Fragment metadata is incomplete")
                            self.store.add(dict(path=event["path"], session_id=session,
                                                source_size=source_size, start_ns=opened.pop(event["path"]),
                                                end_ns=event["running_ns"],
                                                closed_monotonic=event["monotonic"]))
                            self.archived += 1
                        elif event["event"] == "stopped":
                            self.decoded_frames += event["decoded_frames"]
                            self.closed_sessions += 1
                        elif event["event"] == "error" and event["code"] in (
                                "disk_space_low", "unsupported_video_codec", "pipeline_setup_failed",
                                "frame_exceeds_shared_memory"):
                            self.failure = event["code"]
                            fatal = True

                while worker.process.poll() is None:
                    expired = self.args.seconds and time.monotonic()-started >= self.args.seconds
                    if (self.stop_event.is_set() or expired or fatal) and not stop_sent:
                        worker.request_stop()
                        stop_sent, stop_at = True, time.monotonic()
                    if stop_sent and time.monotonic()-stop_at > 8:
                        worker.process.kill()
                        self.failure = "camera_finalize_timeout"
                        break
                    poll_events()
                    if self.args.policy == "latest" and not stop_sent:
                        try:
                            item = worker.frame()
                        except (OSError, RuntimeError):
                            item = None
                            worker.request_stop()
                            stop_sent, stop_at = True, time.monotonic()
                        if item is not None:
                            item[1]["session_id"] = session
                            try:
                                self.frames.get_nowait()
                            except queue.Empty:
                                pass
                            self.frames.put_nowait(item)
                    time.sleep(.01)
                worker.close()
                poll_events()
                if not worker.tail.path.exists():
                    self.failure = "worker_startup_failed"
                    fatal = True
                self.events.write(json.dumps(dict(event="disconnected", session_id=session,
                                                 monotonic=time.monotonic(), unclosed_files=list(opened),
                                                 exit_code=worker.process.returncode))+"\n")
                worker = None
                if self.stop_event.is_set() or fatal or (self.args.seconds and time.monotonic()-started >= self.args.seconds):
                    break
                self.reconnects += 1
                self.events.write(json.dumps(dict(event="reconnecting", delay_seconds=delay))+"\n")
                remaining = self.args.seconds-(time.monotonic()-started) if self.args.seconds else delay
                self.stop_event.wait(max(0, min(delay, remaining)))
                # A valid stream alone is not proof of a stable connection.
                delay = min(30., delay*2) if not healthy else 1.
        except Exception as error:
            self.failure = type(error).__name__
            self.events.write(json.dumps(dict(event="supervisor_error", code=self.failure))+"\n")
        finally:
            try:
                if worker is not None:
                    worker.close()
            finally:
                self.events.close()
                self.finished.set()

    def close(self):
        self.stop()
        if self.thread.ident is None:
            self.events.close()
            self.finished.set()
        if self.thread.is_alive():
            self.thread.join(timeout=12)
        if self.thread.is_alive():
            raise RuntimeError("摄像机接收线程未结束")


def open_fragment(item, args):
    from .video_reader import create_video_capture
    return create_video_capture(Path(item["path"]), "gstreamer", 2, args.gst_python, 2)


class ArchiveReplay:
    """Reanalyze a closed capture into a NEW result folder, never append duplicates."""
    def __init__(self, source, folder):
        self.store = SegmentStore(folder / "fragments.sqlite3")
        path = (Path(source)/"fragments.sqlite3").resolve()
        if not path.is_file():
            self.store.close()
            raise ValueError("重放目录缺少 fragments.sqlite3")
        db = sqlite3.connect(path.as_uri()+"?mode=ro", uri=True)
        try:
            sessions = set()
            for payload, in db.execute("SELECT payload FROM fragments ORDER BY id"):
                item = json.loads(payload)
                if not Path(item["path"]).is_file():
                    raise ValueError("归档录像文件不存在")
                self.store.add(item)
                sessions.add(item["session_id"])
        except BaseException:
            self.store.close()
            raise
        finally:
            db.close()
        self.sessions, self.reconnects = len(sessions), 0
        self.decoded_frames = self.closed_sessions = 0
        self.archived = self.store.counts().get(0, 0)
        self.failure = None
        self.finished = threading.Event()
        self.finished.set()

    def start(self):
        pass

    def close(self):
        pass
