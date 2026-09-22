"""Package source, tests and built ONNX; exclude virtualenvs, inputs and experiment outputs."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    files = [root / name for name in ["track_video.py", "README.md", "JETSON_PERFORMANCE.md", "PERFORMANCE_VALIDATION.md",
             "pyproject.toml", "requirements.txt", "requirements-yolo.txt", "requirements-export.txt",
             "models/detect.pt", "models/detect.onnx"]]
    for directory in ("tracking", "tests", "tools"):
        files.extend(sorted((root / directory).glob("*.py")))
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise RuntimeError(f"Missing package inputs: {missing}")
    output = root / "outputs" / f"jetson_bundle_{datetime.now():%Y%m%d_%H%M%S}.zip"
    output.parent.mkdir(exist_ok=True)
    manifest = {}
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            name = path.relative_to(root).as_posix()
            data = path.read_bytes()
            archive.writestr("drop_tracked/"+name, data)
            manifest[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("drop_tracked/manifest.json", json.dumps(manifest, indent=2))
    print(output)


if __name__ == "__main__":
    main()
