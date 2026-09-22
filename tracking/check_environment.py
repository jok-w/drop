"""Read-only readiness checks for target Jetson deployment."""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess

from .runtime import configure_runtime, hardware_info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gst-python", default="/usr/bin/python3")
    parser.add_argument("--require-jetson", action="store_true")
    args = parser.parse_args()
    configure_runtime()
    report = dict(platform=platform.platform(), hardware=hardware_info(), checks={})
    from .video_writer import is_jetson, check_gstreamer
    from .video_reader import check_decoder
    from .model_artifacts import runtime_info
    report["jetson"] = is_jetson()
    def check(name, action):
        try:
            value = action()
            report["checks"][name] = dict(ok=True, details=value)
        except Exception as error:
            report["checks"][name] = dict(ok=False, error=str(error))
    check("tensorrt_runtime", lambda: runtime_info("0"))
    check("nvenc_plugins", check_gstreamer)
    check("nvdec_h264_plugins", lambda: check_decoder(args.gst_python, "h264parse"))
    check("nvdec_h265_plugins", lambda: check_decoder(args.gst_python, "h265parse"))
    if shutil.which("nvpmodel"):
        query = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True, timeout=10)
        report["power_mode_query"] = (query.stdout+query.stderr).strip()
    report["notes"] = ["Plugin checks do not replace actual video decode/encode tests.",
                       "TensorRT engines must be built and warmed on the target device."]
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.require_jetson and (not report["jetson"] or not all(v["ok"] for v in report["checks"].values())):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
