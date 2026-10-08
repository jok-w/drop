"""Fixed inference shapes in height, width order, shared by export and inference."""
from numbers import Integral

DEFAULT_IMAGE_SIZE = 640


def image_shape(size):
    values = [size] if isinstance(size, Integral) else size
    if not isinstance(values, (list, tuple)) or len(values) not in (1, 2):
        raise ValueError("imgsz 需要一个尺寸，或两个尺寸（高 宽）")
    if any(isinstance(v, bool) or not isinstance(v, Integral) or v < 32 or v % 32
           for v in values):
        raise ValueError("imgsz 的高和宽必须是正的 32 倍数")
    return (int(values[0]), int(values[-1]))


def canonical_size(size):
    height, width = image_shape(size)
    return height if height == width else [height, width]


def size_label(size):
    height, width = image_shape(size)
    return str(height) if height == width else f"{height}x{width}"
