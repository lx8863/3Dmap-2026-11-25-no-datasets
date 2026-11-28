"""Accept photo uploads and prepare a COLMAP dataset for 3DGS."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import struct


MAX_HEADER = 32_768
MAX_IMAGE_BYTES = 32 * 1024**2
MAX_UPLOAD_BYTES = 512 * 1024**2
IMAGE_COUNTS = {"vggt": (2, 12), "colmap": (3, 128)}
MAX_ITERATIONS = 30_000


@dataclass(frozen=True)
class PhotoUpload:
    engine: str
    iterations: int
    sizes: list[int]


def read_upload(stream, length):
    if not 4 < length <= MAX_HEADER + MAX_UPLOAD_BYTES + 4:
        raise ValueError("照片总大小超过 512 MB，或上传为空。")
    prefix = stream.read(4)
    if len(prefix) != 4:
        raise ValueError("照片上传提前中断。")
    header_size = struct.unpack(">I", prefix)[0]
    if not 0 < header_size <= min(MAX_HEADER, length - 4):
        raise ValueError("上传设置无效。")
    header = stream.read(header_size)
    if len(header) != header_size:
        raise ValueError("照片上传提前中断。")
    options = json.loads(header)
    if not isinstance(options, dict):
        raise ValueError("上传设置无效。")
    engine = options.get("engine")
    if not isinstance(engine, str) or engine not in IMAGE_COUNTS:
        raise ValueError("请选择 VGGT 或 COLMAP。")
    iterations = options.get("iterations")
    if type(iterations) is not int or not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"请输入 1–{MAX_ITERATIONS:,} 的训练步数。")
    sizes = options.get("sizes")
    minimum, maximum = IMAGE_COUNTS[engine]
    if not isinstance(sizes, list) or not minimum <= len(sizes) <= maximum:
        raise ValueError(f"{engine.upper()} 需要 {minimum}–{maximum} 张照片。")
    if any(type(size) is not int or not 0 < size <= MAX_IMAGE_BYTES for size in sizes):
        raise ValueError("每张照片须小于或等于 32 MB。")
    if sum(sizes) > MAX_UPLOAD_BYTES or sum(sizes) != length - header_size - 4:
        raise ValueError("照片上传长度有误，或总大小超过 512 MB。")
    return PhotoUpload(engine, iterations, sizes)


def save_upload(stream, upload, folder):
    from PIL import Image, ImageOps, UnidentifiedImageError

    images = folder / "images"
    images.mkdir(parents=True)
    for index, size in enumerate(upload.sizes, 1):
        pending = folder / "incoming"
        with pending.open("wb") as target:
            remaining = size
            while remaining:
                block = stream.read(min(remaining, 1024**2))
                if not block:
                    raise ValueError("照片上传提前中断。")
                target.write(block)
                remaining -= len(block)
        try:
            with Image.open(pending, formats=("JPEG", "PNG", "WEBP")) as source:
                photo = ImageOps.exif_transpose(source)
                photo.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
                photo.convert("RGB").save(images / f"{index:04d}.jpg", quality=95, subsampling=0)
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
            raise ValueError(f"第 {index} 张照片无法读取。请选择有效的 JPG、PNG 或 WebP。") from exc
        finally:
            pending.unlink(missing_ok=True)


def calibrate_colmap(images, dataset):
    import pycolmap

    raw_models = dataset.parent / "colmap"
    raw_models.mkdir()
    database = raw_models / "features.db"
    pycolmap.extract_features(database, images, device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(database, device=pycolmap.Device.cpu)
    sparse = raw_models / "sparse"
    sparse.mkdir()
    models = pycolmap.incremental_mapping(database, images, sparse)
    if not models:
        raise ValueError("COLMAP 没有找到足够的重叠视角。请补拍相邻角度或尝试 VGGT。")
    index, best = max(models.items(), key=lambda item: item[1].num_reg_images())
    if best.num_reg_images() < 3 or best.num_points3D() == 0:
        raise ValueError("COLMAP 标定的有效照片少于 3 张，或未生成初始化点云。")
    print(f"COLMAP 已标定 {best.num_reg_images()} 张照片，准备训练数据。")
    dataset.mkdir()
    pycolmap.undistort_images(dataset, sparse / str(index), images)
    raw_sparse = dataset / "sparse"
    model = raw_sparse / "0"
    model.mkdir(exist_ok=True)
    for entry in raw_sparse.iterdir():
        if entry.is_file() and entry.suffix in {".bin", ".txt"}:
            shutil.move(entry, model / entry.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate photos using COLMAP.")
    parser.add_argument("images", type=Path)
    parser.add_argument("dataset", type=Path)
    options = parser.parse_args()
    try:
        calibrate_colmap(options.images, options.dataset)
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"COLMAP: {error}\n")
