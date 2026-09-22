"""Local runtime settings shared by inference and export commands."""
import os
from pathlib import Path


def configure_runtime():
    settings = Path(__file__).resolve().parents[1] / "outputs" / ".ultralytics"
    settings.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(settings))
    # Export/inference must not mutate dependencies halfway through a run.
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def hardware_info():
    result = {}
    release = Path("/etc/nv_tegra_release")
    if release.is_file():
        result["jetson_release"] = release.read_text().strip()
    try:
        import torch
        result.update(torch=torch.__version__, cuda=torch.version.cuda, cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            result["gpus"] = [{"name": torch.cuda.get_device_name(i),
                               "memory_bytes": torch.cuda.get_device_properties(i).total_memory,
                               "capability": list(torch.cuda.get_device_capability(i))}
                              for i in range(torch.cuda.device_count())]
    except ImportError:
        pass
    return result
