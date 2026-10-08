"""System-Python RTSP ingest. One connection, compressed archive, optional live NVDEC.

The parent receives events via an append-only journal and, in latest mode, owns
one shared frame until it acknowledges it. Password travels only through a pipe.
"""
import argparse
import json
import mmap
import os
from pathlib import Path
import select
import signal
import socket
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gst_decoder_worker import copy_bgrx, send_message


def elements(pipeline, Gst):
    iterator = pipeline.iterate_elements()
    while True:
        status, element = iterator.next()
        if status == Gst.IteratorResult.OK:
            yield element
        elif status == Gst.IteratorResult.RESYNC:
            iterator.resync()
        elif status == Gst.IteratorResult.DONE:
            return
        else:
            raise RuntimeError("Cannot enumerate GStreamer elements")


def build_video_branch(pipeline, Gst, codec, config, sample_ready, parsed):
    """Construct the same graph for static linking checks and real RTSP pads."""
    def make(factory, name=None):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Missing GStreamer element: {factory}")
        pipeline.add(element)
        return element

    def link(*elements):
        for left, right in zip(elements, elements[1:]):
            if not left.link(right):
                raise RuntimeError("Cannot link camera pipeline")

    scaling_filter = None
    depay = make("rtph264depay" if codec == "H264" else "rtph265depay")
    parser = "h264parse" if codec == "H264" else "h265parse"
    complete = config["mode"] == "complete"
    archive_format = "byte-stream" if complete else ("avc" if codec == "H264" else "hvc1")
    media_type = "video/x-h264" if codec == "H264" else "video/x-h265"
    tee = make("tee")
    archive_queue = make("queue")
    archive_queue.set_property("max-size-buffers", 0)
    archive_queue.set_property("max-size-bytes", 16*1024*1024)
    archive_queue.set_property("max-size-time", 2*Gst.SECOND)
    archive_parse = make(parser)
    archive_parse.set_property("config-interval", -1)
    archive_caps = make("capsfilter")
    archive_caps.set_property("caps", Gst.Caps.from_string(
        f"{media_type},stream-format={archive_format},alignment=au"))
    mux = make("splitmuxsink", "archive")
    if complete:
        # MPEG-TS accepts in-band H.264/H.265 parameter-set updates. Some live
        # cameras refine PPS during startup, which can leave MP4 without a
        # playable track because qtmux cannot renegotiate codec_data.
        mux.set_property("muxer-factory", "mpegtsmux")
    suffix = "ts" if complete else "mp4"
    mux.set_property("location", str(Path(config["folder"]) / f"raw_%05d.{suffix}"))
    mux.set_property("max-size-time", int(config["segment_seconds"]*Gst.SECOND))
    mux.set_property("max-size-bytes", 0)
    mux.set_property("async-finalize", True)
    # Parsing must happen after the tee: MP4 requires avc/hvc1 while the
    # Jetson decoder requires byte-stream. A shared parser cannot negotiate
    # both formats and fails the complete pipeline with not-negotiated.
    link(depay, tee)
    link(tee, archive_queue, archive_parse, archive_caps, mux)
    archive_parse.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, parsed)
    if config["mode"] == "latest":
        q = make("queue")
        # Never discard compressed reference pictures. Decoded samples are
        # replaced by sample_ready only in the explicit latest policy.
        q.set_property("max-size-buffers", 8)
        q.set_property("max-size-bytes", 0)
        q.set_property("max-size-time", 0)
        decode_parse = make(parser)
        decode_parse.set_property("config-interval", -1)
        decode_caps = make("capsfilter")
        decode_caps.set_property("caps", Gst.Caps.from_string(
            f"{media_type},stream-format=byte-stream,alignment=au"))
        decoder = make("nvv4l2decoder")
        convert = make("nvvidconv")
        convert.set_property("interpolation-method", 1)
        filt = make("capsfilter")
        filt.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGRx"))
        scaling_filter = filt
        sink = make("appsink")
        for key, value in [("emit-signals", True), ("sync", False),
                           ("max-buffers", 1), ("drop", True),
                           ("enable-last-sample", False), ("wait-on-eos", False)]:
            sink.set_property(key, value)
        sink.connect("new-sample", sample_ready)
        link(tee, q, decode_parse, decode_caps, decoder, convert, filt, sink)
    return depay, scaling_filter


def run(config, control_fd, memory_fd):
    import gi
    gi.require_version("Gst", "1.0")
    gi.require_version("GstVideo", "1.0")
    from gi.repository import Gst, GstVideo, GLib
    Gst.init(None)
    folder = Path(config["folder"])
    journal = (folder / "events.jsonl").open("a", buffering=1)
    lock = threading.Lock()

    def event(kind, **values):
        with lock:
            journal.write(json.dumps(dict(event=kind, monotonic=time.monotonic(),
                                          wall_time=time.time(), **values)) + "\n")

    control = socket.socket(fileno=control_fd)
    shared = mmap.mmap(memory_fd, config["memory_bytes"]) if memory_fd is not None else None
    loop = GLib.MainLoop()
    pipeline = Gst.Pipeline.new("camera")
    state = dict(linked=False, last_packet=time.monotonic(), latest=None, outstanding=False,
                 sequence=0, source_size=None, first=True, stopping=False, eos=False,
                 failed=False, stop_time=None, last_space_check=0.)
    latest_lock = threading.Lock()

    def make(factory, name=None):
        element = Gst.ElementFactory.make(factory, name)
        if element is None:
            raise RuntimeError(f"Missing GStreamer element: {factory}")
        pipeline.add(element)
        return element

    def fail(code):
        if not state["failed"]:
            # Do not serialize GStreamer error text: it may contain authentication data.
            event("error", code=code)
        state["failed"] = True
        stop()

    def stop(*unused):
        if not state["stopping"]:
            state["stopping"] = True
            state["stop_time"] = time.monotonic()
            # Network state changes may block while rtspsrc is connecting. Keep
            # the GLib watchdog running; the supervisor enforces a final bound.
            threading.Thread(target=pipeline.send_event, args=(Gst.Event.new_eos(),),
                             daemon=True).start()
        return False

    def sample_ready(sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.EOS
        with latest_lock:
            state["latest"] = (state["sequence"], sample, time.monotonic())
            state["sequence"] += 1
        return Gst.FlowReturn.OK

    def parsed(pad, info):
        state["last_packet"] = time.monotonic()
        caps = pad.get_current_caps()
        if caps:
            structure = caps.get_structure(0)
            width, height = structure.get_value("width"), structure.get_value("height")
            if width and height:
                size = [int(width), int(height)]
                if state["source_size"] is None:
                    state["source_size"] = size
                    event("stream", codec=state["codec"], source_size=size)
                    if config["mode"] == "latest" and config["decode_width"]:
                        w = min(size[0], config["decode_width"])
                        w = max(2, w//2*2)
                        h = max(2, round(size[1]*w/size[0])//2*2)
                        state["filter"].set_property("caps", Gst.Caps.from_string(
                            f"video/x-raw,format=BGRx,width={w},height={h}"))
                elif size != state["source_size"]:
                    GLib.idle_add(fail, "resolution_changed")
        return Gst.PadProbeReturn.OK

    def pad_added(source, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        s = caps.get_structure(0)
        if s.get_string("media") != "video" or state["linked"]:
            return
        codec = (s.get_string("encoding-name") or "").upper()
        if codec not in ("H264", "H265"):
            GLib.idle_add(fail, "unsupported_video_codec")
            return
        try:
            state["codec"] = codec
            depay, state["filter"] = build_video_branch(
                pipeline, Gst, codec, config, sample_ready, parsed)
            for child in elements(pipeline, Gst):
                child.sync_state_with_parent()
            if pad.link(depay.get_static_pad("sink")) != Gst.PadLinkReturn.OK:
                raise RuntimeError("RTSP pad link failed")
            state["linked"] = True
        except Exception:
            GLib.idle_add(fail, "pipeline_setup_failed")

    def message(bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            error, _ = msg.parse_error()
            event("gst_error", domain=str(error.domain), code=int(error.code))
            fail("gstreamer_error")
        elif msg.type == Gst.MessageType.EOS:
            state["eos"] = True
            event("eos")
            loop.quit()
        elif msg.type == Gst.MessageType.ELEMENT:
            s = msg.get_structure()
            if s and s.get_name() in ("splitmuxsink-fragment-opened", "splitmuxsink-fragment-closed"):
                event("fragment_opened" if s.get_name().endswith("opened") else "fragment_closed",
                      path=s.get_value("location"), running_ns=int(s.get_value("running-time")))

    def tick():
        now = time.monotonic()
        if select.select([control], [], [], 0)[0]:
            command = control.recv(16)
            if not command or b"Q" in command:
                stop()
            elif b"A" in command:
                state["outstanding"] = False
        if not state["stopping"] and now-state["last_packet"] > config["timeout"]:
            fail("stream_timeout")
        if now-state["last_space_check"] > 1:
            state["last_space_check"] = now
            v = os.statvfs(folder)
            if v.f_bavail*v.f_frsize < config["min_free_bytes"]:
                fail("disk_space_low")
        if state["stopping"]:
            if now-state["stop_time"] > 5:
                event("finalize_timeout")
                state["failed"] = True
                loop.quit()
            return True
        if shared is not None and not state["outstanding"]:
            with latest_lock:
                item = state["latest"]
                state["latest"] = None
            if item is not None:
                index, sample, received = item
                info = GstVideo.VideoInfo.new_from_caps(sample.get_caps())
                if info.width*info.height*4 > len(shared):
                    fail("frame_exceeds_shared_memory")
                    return True
                buffer = sample.get_buffer()
                pts = buffer.pts
                if pts == Gst.CLOCK_TIME_NONE:
                    fail("missing_video_pts")
                    return True
                stamp = sample.get_segment().to_running_time(Gst.Format.TIME, pts)
                if stamp == Gst.CLOCK_TIME_NONE:
                    fail("invalid_video_pts")
                    return True
                ok, mapped = buffer.map(Gst.MapFlags.READ)
                if not ok:
                    fail("frame_map_failed")
                    return True
                try:
                    meta = GstVideo.buffer_get_video_meta(buffer)
                    copy_bgrx(mapped.data, shared, info.width, info.height,
                              meta.stride[0] if meta else info.stride[0],
                              meta.offset[0] if meta else info.offset[0])
                finally:
                    buffer.unmap(mapped)
                send_message(control, dict(event="frame", index=index, pts_ns=int(stamp),
                                           width=info.width, height=info.height,
                                           source_size=state["source_size"],
                                           received_monotonic=received,
                                           fps=info.fps_n/info.fps_d if info.fps_d else 0))
                state["outstanding"] = True
        return True

    try:
        source = make("rtspsrc", "source")
        for key, value in [("location", config["url"]), ("user-id", config["user"]),
                           ("user-pw", config["password"]), ("protocols", 4),
                           ("latency", config["latency_ms"]),
                           ("tcp-timeout", int(config["timeout"]*1e6))]:
            source.set_property(key, value)
        source.connect("pad-added", pad_added)
        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", message)
        GLib.timeout_add(10, tick)
        GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGTERM, stop)
        event("connecting")
        if pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("RTSP startup failed")
        loop.run()
    except Exception:
        event("error", code="worker_exception")
        state["failed"] = True
    finally:
        pipeline.set_state(Gst.State.NULL)
        event("stopped", clean_eos=state["eos"], failed=state["failed"],
              decoded_frames=state["sequence"])
        journal.close()
        if shared is not None:
            shared.close()
        control.close()
    return int(state["failed"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--check", action="store_true")
    p.add_argument("--config-fd", type=int)
    p.add_argument("--control-fd", type=int)
    p.add_argument("--memory-fd", type=int)
    args = p.parse_args()
    if args.check:
        import gi
        gi.require_version("Gst", "1.0")
        gi.require_version("GstVideo", "1.0")
        from gi.repository import Gst, GstVideo
        Gst.init(None)
        factories = ("rtspsrc", "rtph264depay", "rtph265depay", "h264parse", "h265parse",
                     "capsfilter", "queue", "tee", "splitmuxsink", "mp4mux", "mpegtsmux",
                     "nvv4l2decoder", "nvvidconv", "appsink")
        available = {name: Gst.ElementFactory.find(name) is not None for name in factories}
        graphs = {}
        if all(available.values()):
            for codec in ("H264", "H265"):
                for mode in ("complete", "latest"):
                    graph = Gst.Pipeline.new(None)
                    try:
                        build_video_branch(graph, Gst, codec,
                            dict(folder="/tmp", segment_seconds=5, mode=mode),
                            lambda sink: Gst.FlowReturn.OK,
                            lambda pad, info: Gst.PadProbeReturn.OK)
                        # Exercise the exact iterator used when attaching dynamic RTSP pads.
                        graphs[f"{codec}_{mode}"] = sum(1 for _ in elements(graph, Gst))
                    finally:
                        graph.set_state(Gst.State.NULL)
        print(json.dumps(dict(gstreamer=Gst.version_string(), plugins=available, null_state_graphs=graphs)))
        raise SystemExit(0 if all(available.values()) else 1)
    if args.config_fd is None or args.control_fd is None:
        p.error("--config-fd and --control-fd are required unless --check")
    with os.fdopen(args.config_fd) as stream:
        config = json.load(stream)
    raise SystemExit(run(config, args.control_fd, args.memory_fd))
