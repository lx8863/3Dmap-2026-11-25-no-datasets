"""Serve the original front-camera sequence of scene_001."""

from functools import lru_cache
from pathlib import Path


SCENE = "scene_001"
CAMERA = "front-forward"
FRAME_ROOT = Path(__file__).resolve().parent.parent / "data" / "tandt_db" / SCENE / "images" / CAMERA
MAX_JPEG_BYTES = 8 * 1024 * 1024


@lru_cache(maxsize=1)
def read_frames(root, modified):
    timestamps = sorted(path.stem for path in root.glob("*.jpeg") if path.is_file())
    if not timestamps or not all(len(t) == 16 and t.isascii() and t.isdecimal() for t in timestamps):
        raise ValueError("找不到带时间戳的前视照片。")
    first = int(timestamps[0])
    return [{"timestamp": timestamp, "offsetMs": (int(timestamp) - first) / 1000} for timestamp in timestamps]


def timeline_manifest():
    frames = read_frames(FRAME_ROOT, FRAME_ROOT.stat().st_mtime_ns)
    return {"scene": SCENE, "camera": CAMERA, "frames": frames}


def timeline_image(index):
    frames = timeline_manifest()["frames"]
    if not 0 <= index < len(frames):
        raise FileNotFoundError(index)
    path = FRAME_ROOT / (frames[index]["timestamp"] + ".jpeg")
    if path.resolve(strict=True) != path:
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        image = stream.read(MAX_JPEG_BYTES + 1)
    if len(image) > MAX_JPEG_BYTES or not image.startswith(b"\xff\xd8"):
        raise ValueError("时间轴图像无效。")
    return image
