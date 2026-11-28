#!/usr/bin/env python3
"""Serve the COLMAP workbench and run reconstructions on this machine."""

import argparse
import base64
from collections import deque
from contextlib import ExitStack
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from ipaddress import IPv4Interface, IPv4Network, ip_address
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import shlex
import shutil
import importlib.util
import signal
from statistics import median
import struct
import subprocess
import sys
import threading
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
import uuid

from dataset_timeline import timeline_image, timeline_manifest
from interactive_process import InteractiveProcess, InteractiveRenderError
from photo_jobs import read_upload, save_upload
from training import (
    cuda_environment, detect_gpus, make_training_plan, run_training, select_gpu,
    training_probe_command, training_progress,
)
from vggt_reconstruction import VGGT_CACHE, VGGT_REVISION, validate_inputs as validate_vggt_inputs


WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_GS_ROOT = WEB_ROOT.parent / "gaussian-splatting" / "gaussian-splatting"
PHOTO_JOBS_ROOT = WEB_ROOT.parent / "outputs" / "workbench"
SAMPLE_ROOT = WEB_ROOT / "sample"
LAN_PASSWORD_FILE = Path.home() / ".config" / "colmap-workbench" / "lan-password"
PRIVATE_LAN_RANGES = tuple(IPv4Network(cidr) for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16"))
FILES = {
    "/": ("studio.html", "text/html"),
    "/workbench": ("index.html", "text/html"),
    "/studio.css": ("studio.css", "text/css"),
    "/studio.js": ("studio.js", "text/javascript"),
    "/index.html": ("index.html", "text/html"),
    "/styles.css": ("styles.css", "text/css"),
    "/app.js": ("app.js", "text/javascript"),
    "/point-cloud.js": ("point-cloud.js", "text/javascript"),
    "/interactive-view.js": ("interactive-view.js", "text/javascript"),
    "/timeline.js": ("timeline.js", "text/javascript"),
}
STAGES = {
    "Feature extraction": "extract",
    "Feature matching": "match",
    "Sparse reconstruction": "sparse",
    "Dense reconstruction": "dense",
}
LEGACY_STAGES = {
    **STAGES,
    "Exhaustive feature matching": "match",
    "Sequential feature matching": "match",
    "Vocabulary tree feature matching": "match",
    "Loading database": "sparse",
}
CHOICES = {
    "dataType": ("individual", "video", "internet"),
    "quality": ("low", "medium", "high", "extreme"),
    "mapper": ("incremental", "global", "hierarchical"),
}
UINT64 = struct.Struct("<Q")
# ID, XYZ, RGB, reprojection error, observation count.
POINT_RECORD = struct.Struct("<QdddBBBdQ")
OBSERVATION_SIZE = struct.calcsize("<II")
MAX_PREVIEW_POINTS = 20_000
MAX_FOCUS_POINTS = 30_000
MAX_FOCUS_SOURCE_POINTS = 250_000
FOCUS_MIN_OPACITY = 0.1
MAX_VIEW_MANIFEST_BYTES = 16_384
MAX_VIEW_JPEG_BYTES = 8 * 1024 * 1024
VIEW_FILENAME = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9_.-]{0,158}\.jpe?g", re.IGNORECASE)
RENDER_IDLE_SECONDS = 45
RENDER_POSE_LIMITS = {
    "yaw": (-math.pi, math.pi),
    "pitch": (-1.25, 1.25),
    "zoom": (0.35, 4.0),
    "panX": (-1.5, 1.5),
    "panY": (-1.5, 1.5),
}


def executable_path(explicit_path):
    if explicit_path:
        path = shutil.which(os.path.expanduser(explicit_path))
        return os.path.abspath(path) if path else None
    source = WEB_ROOT.parent / "colmap"
    for path in (
        source / "install" / "bin" / "colmap",
        source / "build" / "src" / "colmap" / "exe" / "colmap",
        WEB_ROOT / ".colmap-env" / "bin" / "colmap",
    ):
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    path = shutil.which("colmap")
    if path:
        return os.path.abspath(path)
    fallback = WEB_ROOT / ".colmap-apt" / "colmap"
    return str(fallback) if fallback.is_file() and os.access(fallback, os.X_OK) else None


def colmap_output(executable, *arguments):
    try:
        result = subprocess.run([executable, *arguments], check=True, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout + result.stderr


def colmap_capabilities(executable):
    mappers = CHOICES["mapper"]
    if not executable:
        return None, None, None, mappers
    help_output = ""
    output = colmap_output(executable, "--version")
    if not output:
        help_output = colmap_output(executable, "help")
        output = help_output
    version_match = re.search(r"\bCOLMAP\s+([\d][\w.\-+]*)", output)
    version = version_match[1] if version_match else None

    automatic_help = colmap_output(executable, "automatic_reconstructor", "--help")
    if "--image_path" in automatic_help and "--workspace_path" in automatic_help:
        mapper_option = re.search(r"^\s*--mapper(?:\s|=)([^\n]*)", automatic_help, re.MULTILINE)
        if mapper_option is None:
            mappers = ("incremental",)
        elif (choices := re.search(r"\{([^}]*)\}", mapper_option[1])):
            supported = set(map(str.strip, choices[1].split(",")))
            mappers = tuple(mapper for mapper in mappers if mapper in supported) or mappers

    if re.search(r"\bwithout (CUDA|HIP|GPU)\b", output, re.IGNORECASE):
        return version, False, False, mappers
    if not re.search(r"\bwith (CUDA|HIP)\b", output, re.IGNORECASE):
        return version, None, None, mappers
    help_output = help_output or colmap_output(executable, "help")
    dense = None
    if help_output:
        dense = re.search(r"^\s*patch_match_stereo\s*$", help_output, re.MULTILINE) is not None
    return version, True, dense, mappers


def read_binary_points(path):
    size = path.stat().st_size
    points = []
    with path.open("rb") as model:
        header = model.read(UINT64.size)
        if len(header) != UINT64.size:
            raise ValueError("points3D.bin 缺少点数记录。")
        (count,) = UINT64.unpack(header)
        if count > (size - UINT64.size) // POINT_RECORD.size:
            raise ValueError("points3D.bin 的点数记录无效。")
        sample_count = min(count, MAX_PREVIEW_POINTS)
        sample_index = 0
        for index in range(count):
            record = model.read(POINT_RECORD.size)
            if len(record) != POINT_RECORD.size:
                raise ValueError("points3D.bin 的点记录不完整。")
            _, x, y, z, red, green, blue, _, track_size = POINT_RECORD.unpack(record)
            if index == sample_index * count // sample_count:
                sample_index += 1
                if all(map(math.isfinite, (x, y, z))):
                    points.append((x, y, z, red, green, blue))
            next_point = model.tell() + track_size * OBSERVATION_SIZE
            if next_point > size:
                raise ValueError("points3D.bin 的观测轨迹不完整。")
            model.seek(next_point)
    return count, points


def load_sparse_model(workspace):
    sparse = workspace / "sparse"
    if not sparse.is_dir():
        return None
    numbered = sorted(
        (entry for entry in sparse.iterdir() if entry.name.isascii() and entry.name.isdecimal()),
        key=lambda entry: int(entry.name),
    )
    for directory in [sparse, *numbered]:
        point_file = directory / "points3D.bin"
        image_file = directory / "images.bin"
        if not point_file.is_file() or not image_file.is_file():
            continue
        count, points = read_binary_points(point_file)
        if not points:
            continue
        with image_file.open("rb") as images:
            header = images.read(UINT64.size)
        if len(header) != UINT64.size:
            raise ValueError("images.bin 缺少注册视图数记录。")
        (cameras,) = UINT64.unpack(header)
        return {"count": count, "points": points, "cameras": cameras}
    return None


def dot_product(first, second):
    return sum(a * b for a, b in zip(first, second))


def unit_vector(vector):
    length = math.hypot(*vector)
    if not math.isfinite(length) or length <= 1e-12:
        return None
    return tuple(component / length for component in vector)


def camera_poses(cameras):
    poses = []
    for camera in cameras:
        if not isinstance(camera, dict):
            continue
        try:
            position = tuple(map(float, camera["position"]))
            rotation = tuple(tuple(map(float, row)) for row in camera["rotation"])
            if (
                len(position) != 3 or len(rotation) != 3 or any(len(row) != 3 for row in rotation)
                or not all(map(math.isfinite, (*position, *rotation[0], *rotation[1], *rotation[2])))
            ):
                continue
            # The camera-to-world basis stores right, down, and forward as columns.
            up = unit_vector(tuple(-row[1] for row in rotation))
            forward = unit_vector(tuple(row[2] for row in rotation))
        except (KeyError, TypeError, ValueError, OverflowError):
            continue
        if up is not None and forward is not None and abs(dot_product(up, forward)) < 0.2:
            poses.append((str(camera.get("img_name", "")), position, up, forward))
    return sorted(poses, key=lambda pose: pose[0])


def camera_ray_focus(poses):
    # Minimize the sum of squared distances to the camera viewing lines.
    equations = [[0.0] * 4 for _ in range(3)]
    for _, position, _, forward in poses:
        for row in range(3):
            for column in range(3):
                projection = float(row == column) - forward[row] * forward[column]
                equations[row][column] += projection
                equations[row][3] += projection * position[column]

    for column in range(3):
        pivot = max(range(column, 3), key=lambda row: abs(equations[row][column]))
        if abs(equations[pivot][column]) <= 1e-8 * len(poses):
            return None
        equations[column], equations[pivot] = equations[pivot], equations[column]
        divisor = equations[column][column]
        equations[column] = [value / divisor for value in equations[column]]
        for row in range(3):
            if row != column:
                factor = equations[row][column]
                equations[row] = [a - factor * b for a, b in zip(equations[row], equations[column])]

    center = tuple(row[3] for row in equations)
    if not all(map(math.isfinite, center)):
        return None
    in_front = sum(
        dot_product((target - origin for target, origin in zip(center, position)), forward) > 0
        for _, position, _, forward in poses
    )
    return center if 3 * in_front >= 2 * len(poses) else None


def camera_view_frame(cameras, points):
    poses = camera_poses(cameras)
    if not poses:
        return None
    _, eye, eye_up, eye_forward = poses[len(poses) // 5]
    combined_up = tuple(sum(pose[2][axis] for pose in poses) for axis in range(3))
    up = unit_vector(combined_up) if math.hypot(*combined_up) >= 0.2 * len(poses) else eye_up

    focus = camera_ray_focus(poses)
    radius = None
    if focus is not None:
        radius_estimate = 0.75 * median(math.dist(position, focus) for _, position, _, _ in poses)
        if math.isfinite(radius_estimate) and radius_estimate > 1e-12:
            radius = radius_estimate
    center = focus if radius is not None else tuple(median(point[axis] for point in points) for axis in range(3))
    aim = tuple(target - origin for target, origin in zip(center, eye))
    vertical = dot_product(aim, up)
    horizontal = tuple(component - vertical * axis for component, axis in zip(aim, up))
    horizontal_length = math.hypot(*horizontal)
    if not math.isfinite(vertical) or not math.isfinite(horizontal_length):
        return None
    forward = unit_vector(horizontal)
    if forward is None:
        vertical_forward = dot_product(eye_forward, up)
        forward = unit_vector(tuple(component - vertical_forward * axis for component, axis in zip(eye_forward, up)))
    if forward is None:
        return None
    right = (
        forward[1] * up[2] - forward[2] * up[1],
        forward[2] * up[0] - forward[0] * up[2],
        forward[0] * up[1] - forward[1] * up[0],
    )
    return {
        "center": center,
        "right": right,
        "forward": forward,
        "up": up,
        "elevation": math.atan2(-vertical, horizontal_length),
        "radius": radius,
    }


def gaussian_color(harmonics):
    # Invert the zero-degree spherical harmonics normalization used by 3DGS.
    return tuple(round(255 * min(1, max(0, 0.5 + 0.2820947918 * channel))) for channel in harmonics)


def sample_focused_points(stream, record_size, start, count, properties, frame):
    center = frame["center"]
    radius = frame["radius"]
    coordinates = [4 * properties.index(name) for name in (b"x", b"y", b"z")]
    harmonics = [4 * properties.index(name) for name in (b"f_dc_0", b"f_dc_1", b"f_dc_2")]
    opacity_offset = 4 * properties.index(b"opacity") if b"opacity" in properties else None
    opacity_threshold = math.log(FOCUS_MIN_OPACITY / (1 - FOCUS_MIN_OPACITY))
    read_float = struct.Struct("<f").unpack_from
    generator = random.Random(0)
    qualifying = 0
    points = []
    samples = min(count, MAX_FOCUS_SOURCE_POINTS)
    for sample in range(samples):
        stream.seek(start + (sample * count // samples) * record_size)
        record = stream.read(record_size)
        position = tuple(read_float(record, offset)[0] for offset in coordinates)
        if not math.dist(position, center) <= radius:
            continue
        alpha = 1.0
        if opacity_offset is not None:
            logit = read_float(record, opacity_offset)[0]
            if not logit >= opacity_threshold:
                continue
            alpha = 1 / (1 + math.exp(-logit))
        coefficients = tuple(read_float(record, offset)[0] for offset in harmonics)
        if not all(map(math.isfinite, coefficients)):
            continue
        qualifying += 1
        if len(points) < MAX_FOCUS_POINTS:
            points.append((*position, *gaussian_color(coefficients), alpha))
        else:
            replacement = generator.randrange(qualifying)
            if replacement < MAX_FOCUS_POINTS:
                points[replacement] = (*position, *gaussian_color(coefficients), alpha)
    return points


def load_training_model(ply, output_path):
    """Read sampled 3DGS centers and approximate colors from the saved PLY."""
    try:
        cameras = json.loads((output_path / "cameras.json").read_text(encoding="utf-8"))
        if not isinstance(cameras, list):
            cameras = []
    except (OSError, ValueError):
        cameras = []
    with ply.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("模型缺少 PLY 文件头。")
        if stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("点云预览支持小端二进制 3DGS PLY。")
        properties = []
        count = None
        for _ in range(512):
            fields = stream.readline().split()
            if fields == [b"end_header"]:
                break
            if not fields:
                raise ValueError("PLY 文件头不完整。")
            if fields[:2] == [b"element", b"vertex"]:
                count = int(fields[2])
            elif fields[:2] == [b"property", b"float"]:
                properties.append(fields[2])
            elif fields[0] in (b"element", b"property"):
                raise ValueError("PLY 包含点云预览不支持的元素。")
        else:
            raise ValueError("PLY 文件头超出预览支持的长度。")
        if count is None or count <= 0:
            raise ValueError("模型中没有高斯点。")

        indices = [properties.index(name) for name in (b"x", b"y", b"z", b"f_dc_0", b"f_dc_1", b"f_dc_2")]
        record = struct.Struct(f"<{len(properties)}f")
        start = stream.tell()
        if start + count * record.size > ply.stat().st_size:
            raise ValueError("PLY 的高斯点记录不完整。")
        points = []
        sample_count = min(count, MAX_PREVIEW_POINTS)
        for sample in range(sample_count):
            stream.seek(start + (sample * count // sample_count) * record.size)
            values = record.unpack(stream.read(record.size))
            x, y, z, red, green, blue = (values[index] for index in indices)
            if not all(map(math.isfinite, (x, y, z, red, green, blue))):
                continue
            points.append((x, y, z, *gaussian_color((red, green, blue))))
        if not points:
            raise ValueError("未找到可显示的高斯点。")
        frame = camera_view_frame(cameras, points)
        focus_points = []
        if frame is not None and frame["radius"] is not None:
            focus_points = sample_focused_points(stream, record.size, start, count, properties, frame)
    return {"count": count, "points": points, "cameras": len(cameras), "focusPoints": focus_points, "viewFrame": frame}


def training_views(directory, job_id):
    manifest = directory / "views.json"
    if not manifest.is_file():
        return [], {}
    if manifest.resolve(strict=True) != manifest:
        raise ValueError("渲染预览清单路径无效。")
    with manifest.open("rb") as stream:
        contents = stream.read(MAX_VIEW_MANIFEST_BYTES + 1)
    if len(contents) > MAX_VIEW_MANIFEST_BYTES:
        raise ValueError("渲染预览清单过大。")
    entries = json.loads(contents)
    if not isinstance(entries, list) or len(entries) > 3:
        raise ValueError("渲染预览清单格式无效。")

    views = []
    assets = {}
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or not entry["name"].strip():
            raise ValueError("渲染预览缺少视角名称。")
        view = {"name": entry["name"]}
        for source, target in (("render", "url"), ("reference", "referenceUrl")):
            filename = entry.get(source)
            if not isinstance(filename, str) or not VIEW_FILENAME.fullmatch(filename):
                raise ValueError("渲染预览图像文件名无效。")
            path = directory / filename
            if (
                not path.is_file() or path.resolve(strict=True) != path
                or not 0 < path.stat().st_size <= MAX_VIEW_JPEG_BYTES
            ):
                raise ValueError("渲染预览图像缺失或大小无效。")
            url = f"/api/jobs/{job_id}/views/{filename}"
            assets[url] = path
            view[target] = url
        views.append(view)
    return views, assets


def latest_training_model(output_path):
    candidates = []
    for directory in (output_path / "point_cloud").glob("iteration_*"):
        iteration = re.fullmatch(r"iteration_([0-9]+)", directory.name)
        model = directory / "point_cloud.ply"
        if iteration and model.is_file() and model.stat().st_size > 0:
            candidates.append((int(iteration[1]), model))
    if not candidates:
        raise ValueError(f"找不到训练结果 {output_path}/point_cloud/iteration_*/point_cloud.ply。")
    return max(candidates)


def directory_path(value, label):
    if not isinstance(value, str) or not value.strip() or "\0" in value:
        raise ValueError(f"请输入{label}。")
    try:
        return Path(value).expanduser().resolve()
    except RuntimeError as exc:
        raise ValueError(f"{label}的路径无效。") from exc


def reconstruction_options(payload, executable, mappers):
    if not isinstance(payload, dict):
        raise ValueError("任务设置格式无效。")
    images = directory_path(payload.get("imagePath"), "图像目录")
    workspace = directory_path(payload.get("workspacePath"), "工作区目录")
    if not images.is_dir():
        raise ValueError(f"图像目录 {images} 不存在。")
    if workspace == images or images in workspace.parents:
        raise ValueError("请将工作区放在图像目录以外。COLMAP 会递归扫描图像目录。")
    for name, accepted in CHOICES.items():
        if payload.get(name) not in accepted:
            raise ValueError(f"设置项 {name} 的值无效。")
    if payload["mapper"] not in mappers:
        raise ValueError("当前 COLMAP 版本的自动重建不支持所选建图算法。")
    for name in ("useGpu", "dense"):
        if not isinstance(payload.get(name), bool):
            raise ValueError(f"设置项 {name} 应为布尔值。")
    command = [
        executable,
        "automatic_reconstructor",
        "--image_path", str(images),
        "--workspace_path", str(workspace),
        "--data_type", payload["dataType"],
        "--quality", payload["quality"],
        "--use_gpu", str(int(payload["useGpu"])),
        "--dense", str(int(payload["dense"])),
    ]
    if mappers != ("incremental",):
        command.extend(["--mapper", payload["mapper"]])
    return command, images, workspace, payload["dense"]


def render_pose(payload):
    if not isinstance(payload, dict):
        raise ValueError("相机设置格式无效。")
    pose = {}
    for key, (minimum, maximum) in RENDER_POSE_LIMITS.items():
        value = payload.get(key)
        if type(value) not in (int, float) or not minimum <= value <= maximum:
            raise ValueError(f"相机设置项 {key} 超出范围。")
        pose[key] = value
    quality = payload.get("quality", "preview")
    if quality not in ("preview", "high"):
        raise ValueError("渲染画质设置无效。")
    pose["quality"] = quality
    return pose


class Job:
    def __init__(self, kind, output_path, dense=False, total=None, engine=None):
        self.id = uuid.uuid4().hex
        self.kind = kind
        self.dense = dense
        self.engine = engine
        self.output_path = output_path
        self.state = "running"
        self.stage = None
        self.logs = deque(maxlen=400)
        self.progress = {"iteration": 0, "total": total} if total else None
        self.model = None
        self.views = []
        self.view_assets = {}
        self.saved_model = None
        self.render_gpu = None
        self.interactive = False
        self.error = None

    def summary(self):
        model = None
        if self.model is not None:
            model = {
                "points": self.model["count"],
                "cameras": self.model["cameras"],
                "url": f"/api/jobs/{self.id}/model",
            }
        return {
            "id": self.id,
            "kind": self.kind,
            "engine": self.engine,
            "dense": self.dense,
            "outputPath": str(self.output_path),
            "state": self.state,
            "stage": self.stage,
            "logs": list(self.logs),
            "progress": self.progress,
            "model": model,
            "views": self.views,
            "interactive": self.interactive,
            "downloadUrl": f"/api/jobs/{self.id}/download.ply" if self.saved_model and self.state == "succeeded" else None,
            "error": self.error,
        }


class Workbench:
    def __init__(self, executable, gs_python, gs_root, gpus, *, vggt_python=None, vggt_root=None,
                 vggt_weights=None, photo_jobs_root=PHOTO_JOBS_ROOT, desktop=False):
        self.executable = executable
        self.photo_jobs_root = photo_jobs_root
        self.desktop = desktop
        self.colmap_version, self.colmap_gpu, self.colmap_dense, self.colmap_mappers = colmap_capabilities(executable)
        self.gs_python = gs_python
        self.gs_root = gs_root
        self.gpus = gpus
        self.vggt_python = vggt_python
        self.vggt_root = vggt_root
        self.vggt_weights = vggt_weights
        self.colmap_python = next(
            (python for python in (vggt_python, gs_python) if modules_available(python, "pycolmap")), None
        )
        self.gs_dependencies = modules_available(
            gs_python, "torch", "plyfile", "tqdm", "cv2", "PIL", "simple_knn", "diff_gaussian_rasterization"
        )
        self.job = None
        self.jobs = {"reconstruction": None, "training": None}
        self.training_result = None
        self.last_reconstruction = None
        self.process = None
        self.closing = False
        self.lock = threading.Lock()
        self.renderer_lock = threading.Lock()
        self.renderer = None
        self.renderer_job_id = None
        self.renderer_timer = None
        self.sample = None

    def photo_engine_error(self, engine):
        if engine == "vggt":
            return self.vggt_error()
        if not self.colmap_python:
            return "COLMAP 环境未就绪，请安装 PyCOLMAP。"
        return None

    def studio_training_error(self):
        if importlib.util.find_spec("PIL") is None:
            return "本地服务需要 Pillow 来读取照片。"
        return self.training_error() or (
            None if self.gs_dependencies else "3DGS 训练环境缺少 CUDA 扩展或 Python 依赖。"
        )

    def training_error(self):
        if not self.gpus:
            return "未检测到 NVIDIA GPU。"
        if self.gs_python is None:
            return "找不到 3DGS 的 Python 解释器。请检查 --gs-python。"
        if not (self.gs_root / "train.py").is_file():
            return "找不到 3DGS 的训练脚本。请检查 --gs-root。"
        return None

    def vggt_error(self):
        if not self.gpus:
            return "VGGT 重建需要 NVIDIA GPU。"
        if not self.vggt_python or not self.vggt_root:
            return "VGGT 环境未就绪。请参考 web/README.md 配置测试环境。"
        if not (self.vggt_root / "vggt" / "models" / "vggt.py").is_file():
            return "VGGT 环境未就绪。请参考 web/README.md 配置测试环境。"
        if not self.vggt_weights or not self.vggt_weights.is_file():
            return "缺少 VGGT 权重 model.safetensors。请参考 web/README.md。"
        return None

    def status(self):
        training_error = self.training_error()
        vggt_error = self.vggt_error()
        with self.lock:
            jobs = {kind: job.summary() if job is not None else None for kind, job in self.jobs.items()}
            training_result = None
            if self.training_result is self.jobs["training"]:
                training_result = jobs["training"]
            elif self.training_result is not None:
                training_result = self.training_result.summary()
            return {
                "available": self.executable is not None,
                "executable": self.executable,
                "colmapVersion": self.colmap_version,
                "colmapGpu": self.colmap_gpu,
                "colmapDense": self.colmap_dense,
                "colmapMappers": self.colmap_mappers,
                "gpus": self.gpus,
                "vggtAvailable": vggt_error is None,
                "vggtError": vggt_error,
                "studioTrainingError": self.studio_training_error(),
                "studioColmapError": self.photo_engine_error("colmap"),
                "trainingAvailable": training_error is None,
                "trainingError": training_error,
                "job": jobs[self.job.kind] if self.job is not None else None,
                "jobs": jobs,
                "trainingResult": training_result,
                "desktop": self.desktop,
                "sample": self.sample_preview(),
                "lastReconstruction": self.last_reconstruction,
            }

    def sample_preview(self):
        if not self.sample or not self.sample.views:
            return None
        return {
            "poster": self.sample.views[0]["url"],
            "renderUrl": f"/api/jobs/{self.sample.id}/render" if self.sample.interactive else None,
        }

    def init_sample(self):
        try:
            self.sample = self.read_saved_training(SAMPLE_ROOT, "vggt")
            self.sample.interactive = self.sample.interactive and self.gs_dependencies
        except (OSError, ValueError, RuntimeError):
            return

    def start_reconstruction(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("任务设置格式无效。")
        engine = payload.get("engine", "colmap")
        if engine == "vggt":
            self.start_vggt(payload)
            return
        if engine != "colmap":
            raise ValueError("请选择有效的重建引擎。")
        if not self.executable:
            raise ValueError("未找到 COLMAP 可执行文件。")
        command, images, workspace, dense = reconstruction_options(payload, self.executable, self.colmap_mappers)
        if payload["useGpu"] and self.colmap_gpu is False:
            raise ValueError("当前 COLMAP 构建仅支持 CPU 特征处理。")
        if dense and self.colmap_dense is False:
            raise ValueError("当前 COLMAP 构建无法进行 GPU 稠密重建。")
        job = Job("reconstruction", workspace, dense=dense, engine="colmap")
        if any((workspace / entry).exists() for entry in ("database.db", "sparse", "dense")):
            job.logs.append("COLMAP 会复用此工作区中的已有数据。更改图像或配置时可选择新的工作区。")
        self.start_job(job, self.run_reconstruction, command, images)

    def start_vggt(self, payload):
        if error := self.vggt_error():
            raise ValueError(error)
        images = directory_path(payload.get("imagePath"), "图像目录")
        workspace = directory_path(payload.get("workspacePath"), "工作区目录")
        validate_vggt_inputs(images, workspace)
        command, env = self.vggt_command(images, workspace)
        job = Job("reconstruction", workspace, engine="vggt")
        self.start_job(job, self.run_vggt, command, env)

    def vggt_command(self, images, workspace, gpu=None):
        command = [
            self.vggt_python, "-B", "-u", str(WEB_ROOT / "vggt_reconstruction.py"),
            "--vggt-root", str(self.vggt_root), "--weights", str(self.vggt_weights),
            "--images", str(images), "--output", str(workspace),
        ]
        return command, cuda_environment(gpu if gpu is not None else select_gpu(self.gpus))

    def start_photo_job(self, stream, upload):
        self.photo_jobs_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix=".upload-", dir=self.photo_jobs_root) as temporary:
            staged = Path(temporary) / "job"
            save_upload(stream, upload, staged)
            job = Job("training", self.photo_jobs_root, total=upload.iterations, engine=upload.engine)
            workspace = self.photo_jobs_root / job.id
            job.output_path = workspace / "model"
            self.start_job(job, self.run_photo_job, workspace, upload.iterations,
                           prepare=lambda: staged.rename(workspace))

    def run_photo_job(self, job, workspace, total):
        images, dataset = workspace / "images", workspace / "dataset"
        gpu = select_gpu(self.gpus)
        self.execute(
            job, training_probe_command(self.gs_python, self.gs_root), stage="prepare",
            cwd=self.gs_root, env=cuda_environment(gpu),
        )
        if job.engine == "vggt":
            command, env = self.vggt_command(images, dataset, gpu)
        else:
            command = [self.colmap_python, "-B", "-u", str(WEB_ROOT / "photo_jobs.py"), str(images), str(dataset)]
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        self.execute(job, command, stage="calibrate", env=env)
        plan = make_training_plan(
            {"sourceType": "dataset", "sourcePath": str(dataset), "modelPath": str(job.output_path),
             "iterations": total, "gpu": gpu["uuid"]},
            gs_python=self.gs_python, gs_root=self.gs_root, colmap=self.executable, gpus=self.gpus,
        )
        plan.command.extend(["--data_device", "cpu"])
        self.run_training(job, plan, probe=False)
        marker = workspace / ".result.json"
        marker.write_text(json.dumps({"engine": job.engine}), encoding="utf-8")
        marker.replace(workspace / "result.json")

    def restore_latest_photo_job(self):
        if not self.photo_jobs_root.is_dir():
            return
        manifests = sorted(self.photo_jobs_root.glob("*/result.json"),
                           key=lambda path: path.stat().st_mtime, reverse=True)
        for manifest in manifests:
            try:
                engine = json.loads(manifest.read_text(encoding="utf-8"))["engine"]
                if engine in ("vggt", "colmap"):
                    self.restore_training(manifest.parent / "model", engine=engine)
                    return
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                continue

    def start_training(self, payload):
        plan = make_training_plan(
            payload, gs_python=self.gs_python, gs_root=self.gs_root,
            colmap=self.executable, gpus=self.gpus,
        )
        job = Job("training", plan.output_path, total=plan.total)
        self.start_job(job, self.run_training, plan)

    def read_saved_training(self, output_path, engine=None):
        total, final_model = latest_training_model(output_path)
        job = Job("training", output_path, total=total, engine=engine)
        gpu = select_gpu(self.gpus) if self.gpus else None
        self.load_training_result(job, final_model, total, gpu=gpu, restored=True)
        self.load_training_views(job, output_path / "preview" / final_model.parent.name)
        job.state = "succeeded"
        job.stage = "complete"
        return job

    def restore_training(self, output_path, engine=None):
        job = self.read_saved_training(output_path, engine)
        with self.lock:
            self.job = job
            self.jobs[job.kind] = job
            self.training_result = job

    def start_job(self, job, target, *args, prepare=None):
        with self.lock:
            if self.closing or (self.job is not None and self.job.state == "running"):
                raise RuntimeError("已有任务在运行。")
            if prepare is not None:
                prepare()
            self.job = job
            self.jobs[job.kind] = job
            watcher = threading.Thread(target=self.monitor, args=(job, target, args), daemon=True)
            watcher.start()

    def execute(self, job, command, stage=None, cwd=None, env=None):
        with self.lock:
            if self.closing:
                raise RuntimeError("工作台服务已关闭。")
            if stage:
                job.stage = stage
            job.logs.append(f"$ {shlex.join(map(str, command))}")
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=True,
            )
            self.process = process

        try:
            with process.stdout as stream:
                for line in stream:
                    line = line.rstrip("\r\n")
                    with self.lock:
                        job.logs.append(line)
                        self.update_job_progress(job, line)
            exit_code = process.wait()
            if exit_code:
                raise subprocess.CalledProcessError(exit_code, command)
        finally:
            if process.poll() is None:
                stop_process(process)
            with self.lock:
                self.process = None

    def update_job_progress(self, job, line):
        if job.kind == "reconstruction" and job.engine != "vggt":
            legacy_stage = LEGACY_STAGES.get(line.strip())
            if legacy_stage:
                job.stage = legacy_stage
                return
            for heading, stage in STAGES.items():
                if line.endswith(f"=== {heading} ==="):
                    job.stage = stage
                    break
        elif job.stage == "training":
            iteration = training_progress(line)
            if iteration is not None:
                total = job.progress["total"]
                job.progress = {"iteration": min(iteration, total), "total": total}

    def run_reconstruction(self, job, command, images):
        job.output_path.mkdir(parents=True, exist_ok=True)
        self.execute(job, command)
        model = load_sparse_model(job.output_path)
        if model is None:
            raise ValueError("COLMAP 未生成可预览的稀疏点云。请查看运行日志。")
        with self.lock:
            job.model = model
            self.last_reconstruction = {
                "imagePath": str(images),
                "workspacePath": str(job.output_path),
            }
        if job.dense and not any((job.output_path / "dense").glob("*/fused.ply")):
            raise ValueError("已生成稀疏模型。稠密点云缺失，请检查日志中的 MVS 和 GPU 信息。")

    def run_vggt(self, job, command, env):
        self.execute(job, command, stage="vggt", env=env)
        model = load_sparse_model(job.output_path)
        if model is None:
            raise ValueError("VGGT 未生成可预览的点云。请查看运行日志。")
        with self.lock:
            job.model = model
            self.last_reconstruction = {
                "imagePath": str(job.output_path / "images"),
                "workspacePath": str(job.output_path),
                "sourceType": "dataset",
            }

    def run_training(self, job, plan, *, probe=True):
        def execute_step(stage, command, cwd, env):
            self.execute(job, command, stage=stage, cwd=cwd, env=env)

        run_training(plan, execute_step, probe=probe)
        final_model = plan.output_path / "point_cloud" / f"iteration_{plan.total}" / "point_cloud.ply"
        if not final_model.is_file():
            raise ValueError("3DGS 未写出最终训练模型。请查看运行日志。")
        self.load_training_result(job, final_model, plan.total, gpu=plan.gpu)

        previews = plan.output_path / "preview" / final_model.parent.name
        render_command = [
            plan.command[0], "-u", str(WEB_ROOT / "render_preview.py"),
            "--gs-root", str(plan.root), "--source", str(plan.dataset),
            "--model", str(final_model), "--output", str(previews),
        ]
        try:
            execute_step("preview", render_command, plan.root, plan.environment)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            with self.lock:
                job.logs.append(f"渲染预览生成失败（{exc}）。已保存训练模型。")
        else:
            self.load_training_views(job, previews, required=True)

    def load_training_result(self, job, final_model, total, gpu=None, restored=False):
        model = None
        preview_error = None
        try:
            model = load_training_model(final_model, job.output_path)
        except (OSError, ValueError, IndexError, struct.error) as exc:
            preview_error = f"无法预览已保存的点云（{exc}）。"
        with self.lock:
            job.progress = {"iteration": total, "total": total}
            job.model = model
            job.saved_model = final_model
            job.render_gpu = gpu.get("uuid") if gpu else None
            job.interactive = bool(
                job.render_gpu and self.gs_python and (WEB_ROOT / "render_interactive.py").is_file()
                and (self.gs_root / "train.py").is_file() and (job.output_path / "cameras.json").is_file()
            )
            if restored:
                job.logs.append(f"已加载训练结果 {final_model}")
            else:
                job.logs.append(f"模型已保存至 {final_model}")
            if preview_error:
                job.logs.append(preview_error)

    def load_training_views(self, job, directory, required=False):
        try:
            views, assets = training_views(directory, job.id)
            if required and not views:
                raise ValueError("未生成渲染预览图像。")
        except (OSError, ValueError, RuntimeError) as exc:
            with self.lock:
                job.logs.append(f"无法加载渲染预览（{exc}）。已保存训练模型。")
        else:
            with self.lock:
                job.views = views
                job.view_assets = assets

    def monitor(self, job, target, args):
        error = None
        try:
            with self.renderer_lock:
                self.stop_renderer()
            target(job, *args)
        except subprocess.CalledProcessError as exc:
            error = f"进程退出码 {exc.returncode}。请查看运行日志。"
            prefixes = {
                "calibrate": ("VGGT: ", "COLMAP: "),
                "vggt": ("VGGT: ",),
                "prepare": ("3DGS: ",),
            }.get(job.stage, ())
            if prefixes:
                error = next(
                    (line.split(": ", 1)[1] for line in reversed(job.logs) if line.startswith(prefixes)),
                    error,
                )
        except Exception as exc:
            error = str(exc) or type(exc).__name__
        with self.lock:
            job.error = error
            job.state = "failed" if error else "succeeded"
            if not error:
                job.stage = "complete"
                if job.kind == "training":
                    self.training_result = job

    def download_for(self, path):
        with self.lock:
            job = self.training_result
            if job and job.saved_model and path == f"/api/jobs/{job.id}/download.ply":
                return job.saved_model, f"3Dmap-{job.id[:8]}.ply"
        return None

    def model_for(self, path):
        with self.lock:
            for job in (*self.jobs.values(), self.training_result):
                if job is not None and path == f"/api/jobs/{job.id}/model":
                    return job.model
        return None

    def view_for(self, path):
        with self.lock:
            for job in (self.jobs["training"], self.training_result, self.sample):
                if job is not None and path in job.view_assets:
                    return job.view_assets[path]
        return None

    def render_training_frame(self, job_id, payload):
        pose = render_pose(payload)
        with self.renderer_lock:
            with self.lock:
                job = self.sample if self.sample and self.sample.id == job_id else self.training_result
                if job is None or job.id != job_id:
                    raise FileNotFoundError("找不到可交互渲染的训练结果。")
                if self.closing or (self.job is not None and self.job.state == "running"):
                    raise RuntimeError("任务运行期间暂停交互渲染。")
                if not job.interactive or not job.saved_model.is_file():
                    raise RuntimeError("此训练结果暂不可进行交互渲染。")
            if self.renderer is not None and (
                self.renderer_job_id != job_id or self.renderer.process.poll() is not None
            ):
                self.stop_renderer()
            try:
                if self.renderer is None:
                    command = [
                        self.gs_python, "-u", str(WEB_ROOT / "render_interactive.py"),
                        "--gs-root", str(self.gs_root), "--model", str(job.saved_model),
                        "--cameras", str(job.output_path / "cameras.json"),
                    ]
                    environment = dict(
                        os.environ,
                        CUDA_DEVICE_ORDER="PCI_BUS_ID",
                        CUDA_VISIBLE_DEVICES=job.render_gpu,
                        PYTHONDONTWRITEBYTECODE="1",
                    )
                    self.renderer = InteractiveProcess(command, self.gs_root, environment)
                    self.renderer.wait_until_ready()
                    self.renderer_job_id = job_id
                image = self.renderer.render(pose)
            except (InteractiveRenderError, OSError) as exc:
                self.stop_renderer()
                raise InteractiveRenderError(str(exc) or "交互渲染器启动失败。") from exc
            self.release_idle_renderer()
            return image

    def release_idle_renderer(self):
        if self.renderer_timer is not None:
            self.renderer_timer.cancel()

        def expire():
            with self.renderer_lock:
                if self.renderer_timer is timer:
                    self.stop_renderer()

        timer = threading.Timer(RENDER_IDLE_SECONDS, expire)
        timer.daemon = True
        self.renderer_timer = timer
        timer.start()

    def stop_renderer(self):
        if self.renderer_timer is not None:
            self.renderer_timer.cancel()
            self.renderer_timer = None
        if self.renderer is not None:
            process = self.renderer.process
            self.renderer = None
            self.renderer_job_id = None
            if process.poll() is None:
                stop_process(process)
            process.stdin.close()
            process.stdout.close()

    def close(self):
        with self.lock:
            self.closing = True
            process = self.process
        if process and process.poll() is None:
            stop_process(process)
        with self.renderer_lock:
            self.stop_renderer()


def stop_process(process):
    if os.name == "nt":
        for stop in (process.terminate, process.kill):
            try:
                stop()
                process.wait(timeout=3)
                return
            except ProcessLookupError:
                return
            except subprocess.TimeoutExpired:
                continue
        return
    for signum in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, signum)
            process.wait(timeout=3)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            continue
        return


def lan_interface(value):
    try:
        interface = IPv4Interface(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use a private IPv4 address in CIDR form, e.g. 192.168.1.10/24.") from exc
    if "/" not in value or not any(interface.network.subnet_of(network) for network in PRIVATE_LAN_RANGES):
        raise argparse.ArgumentTypeError("Use an address and subnet within a private IPv4 LAN.")
    return interface


def lan_authentication(path):
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        pass
    else:
        with os.fdopen(fd, "w", encoding="ascii") as file:
            file.write(secrets.token_urlsafe(24) + "\n")

    with path.open("rb") as file:
        if os.fstat(file.fileno()).st_mode & 0o077:
            raise ValueError(f"Set the permissions of {path} to 600 before starting LAN access.")
        password = file.read(1025).strip()
    if len(password) < 16 or len(password) > 1024 or any(byte < 33 or byte > 126 for byte in password):
        raise ValueError(f"LAN password in {path} must be 16 to 1024 printable ASCII characters without spaces.")
    return base64.b64encode(b"colmap:" + password).decode("ascii")


def configure_listener(server, workbench, hosts, network=None, authentication=None):
    port = server.server_port
    server.workbench = workbench
    server.valid_hosts = {f"{host}:{port}" for host in hosts}
    if port == 80:
        server.valid_hosts.update(hosts)
    server.lan_network = network
    server.lan_authentication = authentication


class Handler(BaseHTTPRequestHandler):
    timeout = 10

    def reply(self, status, body, content_type, headers=(), *, cache_control="no-store"):
        self.send_response(status)
        if not content_type.startswith("image/"):
            content_type += "; charset=utf-8"
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        for name, value in headers:
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def json(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.reply(status, body, "application/json")

    def authorized_request(self, path):
        host = self.headers.get("Host")
        origin = self.headers.get("Origin")
        network = self.server.lan_network
        cross_site = self.headers.get("Sec-Fetch-Site") == "cross-site"
        navigation = (
            self.command == "GET" and path in ("/", "/index.html", "/workbench")
            and self.headers.get("Sec-Fetch-Mode") == "navigate"
        )
        if (
            host not in self.server.valid_hosts
            or origin not in (None, f"http://{host}")
            or (cross_site and not navigation)
            or (network is not None and ip_address(self.client_address[0]) not in network)
        ):
            self.json(HTTPStatus.FORBIDDEN, {"error": "请通过工作台地址在允许的网络中访问。"})
            return False
        expected = self.server.lan_authentication
        if expected is not None:
            scheme, _, credential = self.headers.get("Authorization", "").partition(" ")
            if (
                scheme.lower() != "basic" or not credential.isascii()
                or not hmac.compare_digest(credential, expected)
            ):
                self.reply(
                    HTTPStatus.UNAUTHORIZED, "请输入工作台用户名和局域网口令。".encode(),
                    "text/plain", (("WWW-Authenticate", 'Basic realm="COLMAP Workbench", charset="UTF-8"'),),
                )
                return False
        return True

    def do_GET(self):
        path = urlsplit(self.path).path
        if not self.authorized_request(path):
            return
        if path == "/api/status":
            self.json(HTTPStatus.OK, self.server.workbench.status())
            return
        if path == "/api/timeline" or path.startswith("/api/timeline/"):
            self.get_timeline(path)
            return
        if path in FILES:
            filename, content_type = FILES[path]
            try:
                body = (WEB_ROOT / filename).read_bytes()
            except OSError:
                self.json(HTTPStatus.NOT_FOUND, {"error": "页面文件未找到。"})
            else:
                self.reply(HTTPStatus.OK, body, content_type)
            return
        download = self.server.workbench.download_for(path)
        if download is not None:
            self.send_ply(*download)
            return
        model = self.server.workbench.model_for(path)
        if model is not None:
            self.json(HTTPStatus.OK, model)
            return
        view = self.server.workbench.view_for(path)
        if view is not None:
            try:
                if view.resolve(strict=True) != view:
                    raise ValueError("Invalid preview path")
                with view.open("rb") as stream:
                    body = stream.read(MAX_VIEW_JPEG_BYTES + 1)
            except (OSError, ValueError, RuntimeError):
                pass
            else:
                if len(body) <= MAX_VIEW_JPEG_BYTES and body.startswith(b"\xff\xd8"):
                    self.reply(HTTPStatus.OK, body, "image/jpeg")
                    return
        self.json(HTTPStatus.NOT_FOUND, {"error": "资源未找到。"})

    def send_ply(self, path, filename):
        try:
            if path.resolve(strict=True) != path:
                raise OSError("Invalid saved model path")
            stream = path.open("rb")
            length = os.fstat(stream.fileno()).st_size
        except (OSError, RuntimeError):
            self.json(HTTPStatus.NOT_FOUND, {"error": "训练模型文件未找到。"})
            return
        with stream:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                shutil.copyfileobj(stream, self.wfile, 1024**2)
            except (ConnectionError, TimeoutError):
                pass

    def get_timeline(self, path):
        try:
            if path == "/api/timeline":
                self.json(HTTPStatus.OK, timeline_manifest())
                return
            match = re.fullmatch(r"/api/timeline/([0-9]{1,4})/reference", path)
            if match:
                body = timeline_image(int(match[1]))
                self.reply(HTTPStatus.OK, body, "image/jpeg", (("Cross-Origin-Resource-Policy", "same-origin"),),
                           cache_control="private, max-age=300")
                return
        except (OSError, RuntimeError, ValueError):
            pass
        self.json(HTTPStatus.NOT_FOUND, {"error": "时间轴帧未找到。"})

    def do_POST(self):
        if not self.authorized_request(self.path):
            return
        if self.path == "/api/photo-jobs":
            self.post_photos()
            return
        render_match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/render", self.path)
        if self.path not in ("/api/jobs", "/api/training") and render_match is None:
            self.json(HTTPStatus.NOT_FOUND, {"error": "资源未找到。"})
            return
        if self.headers.get_content_type() != "application/json":
            self.json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请求须使用 JSON。"})
            return
        workbench = self.server.workbench
        if render_match is not None:
            unavailable = None
        elif self.path == "/api/training":
            unavailable = workbench.training_error()
            start = workbench.start_training
        else:
            unavailable = None
            start = workbench.start_reconstruction
        if unavailable:
            self.json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": unavailable})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16_384:
                raise ValueError("任务设置长度无效。")
            payload = json.loads(self.rfile.read(length))
        except (OSError, ValueError) as exc:
            self.json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        if render_match is not None:
            self.render_frame(render_match[1], payload)
            return
        if self.path == "/api/jobs" and isinstance(payload, dict):
            engine = payload.get("engine", "colmap")
            unavailable = workbench.vggt_error() if engine == "vggt" else (
                "未找到 COLMAP 可执行文件。" if engine == "colmap" and not workbench.executable else None
            )
            if unavailable:
                self.json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": unavailable})
                return
        try:
            start(payload)
        except RuntimeError as exc:
            self.json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (OSError, ValueError) as exc:
            self.json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self.json(HTTPStatus.ACCEPTED, workbench.status())

    def post_photos(self):
        if self.headers.get_content_type() != "application/octet-stream":
            self.json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请从工作台提交照片。"})
            return
        workbench = self.server.workbench
        try:
            upload = read_upload(self.rfile, int(self.headers.get("Content-Length", "0")))
            unavailable = workbench.studio_training_error() or workbench.photo_engine_error(upload.engine)
            if unavailable:
                self.json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": unavailable})
                return
            workbench.start_photo_job(self.rfile, upload)
        except RuntimeError as exc:
            self.json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except (ImportError, OSError, ValueError) as exc:
            self.json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            self.json(HTTPStatus.ACCEPTED, workbench.status())

    def render_frame(self, job_id, payload):
        try:
            image = self.server.workbench.render_training_frame(job_id, payload)
        except FileNotFoundError as exc:
            self.json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
        except InteractiveRenderError as exc:
            self.json(HTTPStatus.BAD_GATEWAY, {"error": str(exc)})
        except RuntimeError as exc:
            self.json(HTTPStatus.CONFLICT, {"error": str(exc)})
        except ValueError as exc:
            self.json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        else:
            try:
                self.reply(HTTPStatus.OK, image, "image/jpeg")
            except (ConnectionError, TimeoutError):
                pass

    def log_message(self, format, *args):
        # Reconstruction output is captured in the workbench's own log.
        pass


def modules_available(python, *modules):
    if not python:
        return False
    if Path(python) == Path(sys.executable):
        return all(importlib.util.find_spec(name) is not None for name in modules)
    try:
        command = "from importlib.util import find_spec; import sys; sys.exit(not all(find_spec(n) for n in sys.argv[1:]))"
        result = subprocess.run([python, "-B", "-c", command, *modules], capture_output=True, timeout=8)
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def main():
    parser = argparse.ArgumentParser(description="Run the local COLMAP web workbench.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-root", type=Path, default=PHOTO_JOBS_ROOT)
    parser.add_argument("--desktop", action="store_true", help="Exit when the desktop launcher closes stdin.")
    parser.add_argument("--open", action="store_true", help="Open the studio in the default browser.")
    parser.add_argument("--lan", type=lan_interface, metavar="ADDRESS/PREFIX", help="Also listen on this private LAN interface with password authentication.")
    parser.add_argument("--colmap", help="Path to a compiled COLMAP executable.")
    parser.add_argument("--vggt-python", help="Python interpreter for VGGT.")
    parser.add_argument("--vggt-root", type=Path, default=VGGT_CACHE / f"vggt-{VGGT_REVISION}")
    parser.add_argument("--vggt-weights", type=Path, default=VGGT_CACHE / "model.safetensors")
    parser.add_argument("--gs-python", help="Python interpreter for 3DGS.")
    parser.add_argument("--gs-root", type=Path, default=DEFAULT_GS_ROOT, help="3DGS source directory.")
    parser.add_argument("--restore-training", type=Path, metavar="PATH", help="Load the latest completed 3DGS model in this output directory.")
    args = parser.parse_args()
    executable = executable_path(args.colmap)
    if args.colmap and executable is None:
        parser.error(f"COLMAP executable not found: {args.colmap}")

    def stop_server(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_server)
    vggt_venv = VGGT_CACHE / "venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    gs_python = (
        shutil.which(os.path.expanduser(args.gs_python)) if args.gs_python
        else shutil.which(str(WEB_ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")))
        or (shutil.which(str(vggt_venv)) if os.name == "nt" else None) or sys.executable
    )
    if gs_python is not None:
        gs_python = os.path.abspath(gs_python)
    vggt_python = shutil.which(os.path.expanduser(args.vggt_python or str(vggt_venv)))
    if args.vggt_python and not vggt_python:
        parser.error(f"VGGT Python executable not found: {args.vggt_python}")
    with ExitStack() as listeners:
        try:
            local = listeners.enter_context(ThreadingHTTPServer(("127.0.0.1", args.port), Handler))
            if args.lan is not None:
                address = str(args.lan.ip)
                lan = listeners.enter_context(ThreadingHTTPServer((address, local.server_port), Handler))
                authentication = lan_authentication(LAN_PASSWORD_FILE)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

        workbench = Workbench(
            executable, gs_python, args.gs_root.expanduser().resolve(), detect_gpus(),
            vggt_python=vggt_python, vggt_root=args.vggt_root.expanduser().resolve(),
            vggt_weights=args.vggt_weights.expanduser().resolve(),
            photo_jobs_root=args.data_root.expanduser().resolve(), desktop=args.desktop,
        )
        workbench.init_sample()
        if args.restore_training is not None:
            try:
                workbench.restore_training(args.restore_training.expanduser().resolve())
            except (OSError, ValueError, RuntimeError) as exc:
                parser.error(str(exc))
        if args.restore_training is None:
            workbench.restore_latest_photo_job()
        configure_listener(local, workbench, ("127.0.0.1", "localhost"))
        if args.lan is not None:
            configure_listener(lan, workbench, (address,), args.lan.network, authentication)
            threading.Thread(target=lan.serve_forever, daemon=True).start()
            listeners.callback(lan.shutdown)
            print(f"LAN: http://{address}:{lan.server_port}/ (user: colmap, password file: {LAN_PASSWORD_FILE})", flush=True)

        address = f"http://127.0.0.1:{local.server_port}/"
        print(f"COLMAP workbench: {address}", flush=True)
        if args.open:
            import webbrowser
            threading.Thread(target=webbrowser.open, args=(address,), daemon=True).start()
        if executable is None:
            print("COLMAP executable not found. Start again with --colmap /path/to/colmap to run jobs.", flush=True)
        if args.desktop:
            def close_with_launcher():
                sys.stdin.buffer.read()
                local.shutdown()
            threading.Thread(target=close_with_launcher, daemon=True).start()
        try:
            local.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            workbench.close()


if __name__ == "__main__":
    main()
